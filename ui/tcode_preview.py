"""T-code live preview viewer.

Opens next to a video and streams T-code commands to restim (UDP
localhost:12347 by default) so the user can feel signal adjustments
on the fly. restim drives the device — this viewer is a clock,
encoder, and UDP sender.

Layout:

    ┌──────────────────────────┬──────────────────────────┐
    │                          │ Signal folder [Browse]   │
    │                          │ Loaded: alpha, beta, ... │
    │        video             ├──────────────────────────┤
    │                          │ T-code channels          │
    │                          │  ☑ L0  alpha             │
    │                          │  ☑ L1  beta              │
    │                          │  ...                     │
    │                          ├──────────────────────────┤
    │                          │ Sync offset: ──●── 0.00s │
    │                          │ restim: ● [Test]         │
    │                          │ [ Start Streaming ]      │
    ├──────────────────────────┴──────────────────────────┤
    │ ▶ Play  ═══════●════════ 12.3 / 45.6 s   Video…    │
    └─────────────────────────────────────────────────────┘
"""

import copy
import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from funscript import Funscript
from processing.axis_markers import strip_axis_suffix
from processing.tcode_scheduler import TCodeScheduler
from processing.tcode_sender import DEFAULT_HOST, DEFAULT_PORT, TCodeUDPSender
from processing.tcode_stream import DEFAULT_AXIS_MAP
from ui.video_player_helper import VideoPlaybackMixin
from ui.media_source import (
    MediaConnectionState, VLC as ExternalVLCSource,
)


# Every channel we support, in the order shown in the UI.
# (tcode_axis, signal_name, short_description, default_enabled)
#
# Note: master volume lands on V1 (VOLUME_EXTERNAL) rather than V0
# (VOLUME_API). VOLUME_EXTERNAL is the factor explicitly intended for
# external-app drivers and bypasses restim's script-mapping override
# check — V0 can be silently hijacked by an auto-detected
# volume.funscript in the script tree, which meant our V0=0 mute
# didn't always stick. V1 → VOLUME_EXTERNAL always reaches the
# algorithm's multiplier, so zero means zero, reliably.
#
# The user must assign "V1" as the T-Code axis for VOLUME_EXTERNAL in
# restim's Preferences → Funscript axes (default is empty). The
# setup-check dialog below reminds them once per install.
_CHANNEL_ROWS = [
    ('L0', 'alpha',           'position α',                True),
    ('L1', 'beta',            'position β',                True),
    ('V1', 'volume',          'master volume (external)',  True),
    ('C0', 'frequency',       'carrier (500-1000 Hz)',     True),
    ('P0', 'pulse_frequency', 'pulse rate (0-100 Hz)',     True),
    ('P1', 'pulse_width',     'pulse width',               True),
    ('P3', 'pulse_rise_time', 'pulse rise',                True),
    ('E1', 'e1',              '4P intensity 1',            False),
    ('E2', 'e2',              '4P intensity 2',            False),
    ('E3', 'e3',              '4P intensity 3',            False),
    ('E4', 'e4',              '4P intensity 4',            False),
]

_ALL_AXIS_MAP = {axis: sig for axis, sig, _, _ in _CHANNEL_ROWS}

# Axes whose "neutral" value is zero — master volume and 4P intensities.
# When the user unchecks one of these mid-stream we send a single
# <axis>0000 packet so restim drops that electrode to silence instead
# of holding the last value it got. Position (L0/L1) and pulse-shape
# (C0/P0/P1/P3) axes are intentionally NOT in this set: zero for them
# means "bottom-left corner" or "minimum frequency," not silence.
_ZEROABLE_CHANNELS = frozenset({'V1', 'E1', 'E2', 'E3', 'E4'})


class TCodePreviewViewer(tk.Toplevel, VideoPlaybackMixin):
    """Live T-code preview window."""

    _log_prefix = '[tcode-preview]'
    # Default preview tick rate. 24 fps is cinematic standard and leaves
    # ~42 ms of per-tick budget for the video frame update and time
    # label refresh — enough headroom that normal ticks don't compete
    # with other Tk work. Override per-user via
    # config['ui']['tcode_preview_fps'] (5-60 range; values outside
    # are clamped). _TICK_MS is derived from _FPS at instance time.
    _FPS = 24
    _PULSE_MS = 240  # total duration of a test pulse burst

    def __init__(self, parent, main_window=None):
        super().__init__(parent)
        self.title("T-Code Live Preview — stream to restim")
        self.geometry("1150x820")
        self.minsize(800, 520)
        self.main_window = main_window

        # Resolve preview tick rate from config if set. Malformed values
        # fall through to the class default. _TICK_MS is derived so
        # _tick's after() delay stays consistent with the configured
        # FPS; changing self._FPS at runtime (via the UI spinner) will
        # cause the next tick to reschedule at the new rate.
        _cfg_fps = ((main_window.current_config or {})
                    .get('ui', {}).get('tcode_preview_fps')) \
            if main_window is not None else None
        if _cfg_fps is not None:
            try:
                self._FPS = max(5, min(60, int(_cfg_fps)))
            except (TypeError, ValueError):
                pass
        self._TICK_MS = int(1000 / self._FPS)

        # ── Signal buffer state ────────────────────────────────
        self._buffers = {}         # signal_name -> Funscript
        self._buffer_dir = None    # Path or None
        self._buffer_base = None   # basename string or None
        self._playhead_t = 0.0
        self._total_duration = 0.0
        self._playing = False
        self._last_tick_wall = None
        self._after_id = None

        # ── Streaming state ────────────────────────────────────
        self._sender = None
        self._scheduler = None

        # ── Video mixin attributes (required) ──────────────────
        self._video_path = None
        self._video_cap = None
        self._video_fps = 30.0
        self._video_duration = 0.0
        self._video_last_frame_time = -1.0
        self._video_photo = None
        # Time of the most recently *decoded* video frame, in seconds,
        # as reported by OpenCV. Scheduler uses this (rather than the
        # wall-clock-driven playhead) as the signal timebase when a
        # video is loaded. That locks T-code to what the user actually
        # sees, so slow decode no longer makes the stim run ahead of
        # the video. Fallback to playhead when no video is loaded.
        self._video_actual_t = None

        # ── External media source (restim-style HTTP sync) ──────
        # When the user picks a mode other than 'internal', an adapter
        # polls an external player (VLC via HTTP, etc.) for its
        # playback position. That position is fed into
        # _video_actual_t in _refresh_video_actual_time, so the T-code
        # scheduler tracks the external player's timeline instead of
        # the embedded video's. 'internal' (default) = existing
        # behaviour, embedded video.
        self._media_source_mode_var = tk.StringVar(value='internal')
        # VLC connection settings (seeded from config if present).
        _ext_cfg = ((main_window.current_config or {})
                    .get('external_media', {}) if main_window is not None
                    else {})
        self._vlc_ext_address_var = tk.StringVar(
            value=str(_ext_cfg.get('vlc_address', 'http://127.0.0.1:8080')))
        self._vlc_ext_password_var = tk.StringVar(
            value=str(_ext_cfg.get('vlc_password', '')))
        self._vlc_ext_status_var = tk.StringVar(value='(disabled)')
        self._external_source = None  # Instance of ExternalVLCSource or None
        self._external_source_unsubscribe = None
        self._external_status_refresh_id = None  # after-id for periodic UI refresh
        # Auto-load funscripts when VLC's loaded file changes. Default on
        # so the MultiFunPlayer-style "drag a video into VLC and funscripts
        # sync" workflow works out of the box.
        self._vlc_auto_load_var = tk.BooleanVar(value=True)
        # Remember the last file path we auto-loaded from so we don't
        # thrash on every poll — only reload when VLC reports a change.
        self._last_auto_loaded_path: str = ""

        # ── Async decode worker ─────────────────────────────────
        # The default VideoPlaybackMixin decodes on the UI thread.
        # For live preview we want the UI thread to never block on
        # decode (keeps the tick loop fast and scheduler accurate),
        # so we run decode in a background thread that owns the cap
        # and writes the latest decoded frame into a shared slot.
        # The UI's _update_video_frame just displays that slot.
        self._decode_thread = None
        self._decode_stop = threading.Event()
        self._decode_target_t = 0.0     # UI → worker: where to be
        self._decode_seek_target = None # UI → worker: hard-seek here
        self._decode_cap_lock = threading.Lock()
        self._latest_frame = None        # numpy BGR frame, worker writes
        self._latest_frame_t = None      # its video-time in seconds
        # Pre-built PIL image for the current decoded frame, already
        # resized to the display widget and RGB-converted. The decode
        # worker builds this right after storing _latest_frame so the
        # main thread only has to do the unavoidable Tk-bound
        # ImageTk.PhotoImage call. Moves ~8-10 ms of per-frame CPU
        # off the UI thread, which is what was making checkbox clicks
        # and other incidental interactions stutter video playback.
        self._latest_pil_image = None    # PIL.Image.Image or None
        # Widget size cache — main thread refreshes on each tick, worker
        # reads it to know what size to pre-resize to. Default matches
        # a sensible minimum so the first worker iteration produces
        # something reasonable before the main thread has ticked.
        self._video_widget_size = (640, 360)
        # Decode-rate cap. On high-res sources (1080p+, 4K), software
        # cv2 decode at 30 fps can peg 60-90 % of one CPU core; when
        # that coincides with Tk event handling (e.g. a checkbox
        # click's theme repaint) the main thread runs out of budget
        # and a video frame drops → visible stutter. Capping at
        # 30 Hz halves decode cost on 60 fps sources automatically
        # with no visible quality hit. Users tuning on 30 fps sources
        # can drop this toward 15-20 Hz via config.json (preview.
        # video_fps_cap) for extra relief at the cost of slight
        # playback judder in the preview pane (device output
        # unaffected). Read once from config; a relaunch of the
        # preview window picks up changes.
        preview_cfg = (main_window.current_config or {}).get(
            'preview', {}) if main_window is not None else {}
        try:
            fps_cap = float(preview_cfg.get('video_fps_cap', 30.0))
            if fps_cap <= 0:
                fps_cap = 30.0
        except (TypeError, ValueError):
            fps_cap = 30.0
        self._video_fps_cap = fps_cap

        # ── Video backend selection (cv2 vs VLC) ───────────────
        # cv2 = in-process software decode (default, back-compat).
        # vlc = native libvlc pipeline embedded into a Tk Frame via
        # set_nsobject — decode runs in VLC's own threads/native code,
        # so 30-60% of one CPU core stops being spent on decode inside
        # our Python process. When checkbox clicks or variant-process
        # subprocess spawn compete with decode for the main-thread
        # budget, that's the difference between smooth video and
        # visible stutter.
        #
        # Fallback: if VLC is requested but libvlc can't be loaded
        # (missing VLC.app, mismatched arch, etc.), silently drop back
        # to cv2 rather than leaving the user with a dead video pane.
        # Default to vlc: libvlc decodes natively (VideoToolbox HW
        # accel on macOS) and writes into a memory buffer we own, so
        # decode CPU drops out of our Python process entirely. The
        # main-thread cost is just the PhotoImage blit — same order
        # of magnitude as cv2's, minus the cv2 decode itself. Falls
        # back to cv2 automatically if libvlc fails to load.
        backend_name = str(
            preview_cfg.get('video_backend', 'vlc')).strip().lower()
        self._vlc_player = None
        self._use_vlc = False
        self._video_is_loaded = False  # unified flag for both backends
        if backend_name == 'vlc':
            try:
                from ui.vlc_video_backend import VLCVideoPlayer
                vp = VLCVideoPlayer()
                if vp.available:
                    self._vlc_player = vp
                    self._use_vlc = True
                    print(f"{self._log_prefix} using VLC video backend")
                else:
                    print(f"{self._log_prefix} VLC requested but libvlc "
                          f"unavailable — falling back to cv2")
            except Exception as e:
                print(f"{self._log_prefix} VLC import failed: {e} — "
                      f"falling back to cv2")

        # ── Tk vars ────────────────────────────────────────────
        # Load persisted offset from the main window's config if present.
        stored_offset = 0.0
        if main_window is not None:
            try:
                stored_offset = float(
                    (main_window.current_config or {})
                    .get('preview', {}).get('sync_offset_s', 0.0))
            except (ValueError, TypeError):
                stored_offset = 0.0
        self._sync_offset_var = tk.DoubleVar(value=stored_offset)
        # Plain-float mirror of the offset — the scheduler runs off the
        # main thread and cannot call tk var .get() directly. The main
        # thread updates _sync_offset_s_cached via a trace callback, and
        # also mirrors the value back into the main config so it sticks.
        self._sync_offset_s_cached = stored_offset
        self._sync_offset_var.trace_add(
            'write', lambda *_: self._refresh_offset_cache())
        self._video_offset_var = tk.DoubleVar(value=0.0)
        self._host_var = tk.StringVar(value=DEFAULT_HOST)
        self._port_var = tk.IntVar(value=DEFAULT_PORT)
        self._rate_var = tk.IntVar(value=60)
        self._scrub_var = tk.DoubleVar(value=0.0)
        self._channel_enabled = {
            axis: tk.BooleanVar(value=default)
            for axis, _sig, _desc, default in _CHANNEL_ROWS
        }
        # Plain-bool mirror of each checkbox so the scheduler thread
        # can read enable state without touching tk vars. Updated by
        # trace_add hooks below.
        self._channel_enabled_cached = {
            axis: bool(var.get())
            for axis, var in self._channel_enabled.items()
        }
        for axis, var in self._channel_enabled.items():
            var.trace_add('write',
                          lambda *_a, _axis=axis: self._refresh_channel_cache(_axis))
        self._signals_summary_var = tk.StringVar(value="(no signals loaded)")
        self._folder_var = tk.StringVar(value="(none)")
        self._conn_status_var = tk.StringVar(value="● unknown")
        self._time_label_var = tk.StringVar(value="0.00 / 0.00 s")

        # Variant bookkeeping. _variant_folders maps slot label → folder
        # path; _variant_var holds the currently-selected label.
        self._variant_folders: dict = {}
        self._variant_var = tk.StringVar(value='')
        self._variant_radio_frame = None
        self._variant_source_base = None  # stem of the parent funscript

        # One-shot flag: enable the E1-E4 checkboxes the first time we
        # load a folder that actually contains e1-e4 funscripts, so 4P
        # users don't have to click four boxes. Subsequent reloads /
        # variant switches leave the user's current state alone.
        self._autoloaded_4p = False
        self._reload_status_var = tk.StringVar(value="")
        self._reload_busy = False
        # Named _show_video_var so VideoPlaybackMixin._update_video_frame
        # gates decoding on it when the panel is hidden.
        self._show_video_var = tk.BooleanVar(value=True)
        # When the user pauses the video we force V1 off in the
        # effective enable filter so the scheduler stops emitting it —
        # otherwise paused video + still-streaming scheduler = constant
        # stim at the frozen-playhead value. Reset on play.
        self._video_paused_mute_v1 = False
        # Plain-bool mirror of Show-Video for the scheduler thread.
        # Tk vars must only be read from the main thread; the scheduler
        # reads this via _signal_clock every tick.
        self._show_video_cached = True
        self._show_video_var.trace_add(
            'write', lambda *_: self._refresh_show_video_cache())
        # Optional signal-to-video lock. When True, T-code samples at
        # the actual decoded-frame time (stim stays aligned with what
        # you see but plays at decode rate). When False, T-code samples
        # at the wall-clock playhead (stim plays at real-time regardless
        # of video speed — video may visibly desync). Default True;
        # flip off if your decoder can't keep up with the source FPS.
        self._lock_signal_to_video_var = tk.BooleanVar(value=True)
        self._lock_signal_to_video_cached = True
        self._lock_signal_to_video_var.trace_add(
            'write', lambda *_: self._refresh_lock_signal_cache())

        # Thread-safe hand-off for worker → UI. Workers push callables
        # here; the UI drains the queue on a periodic after() tick.
        # (Tk's .after() may only be called from the main thread.)
        self._ui_queue: "queue.Queue" = queue.Queue()
        self._ui_queue_after_id = None

        self._build_ui()
        self._drain_ui_queue()
        self._auto_load_on_open()
        self._tick_channel_values()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Auto-load on open ──────────────────────────────────────
    def _auto_load_on_open(self):
        """Pull video + signal variants out of the main window on open.

        Source of truth is ``main_window.input_files[0]``: the original
        funscript. Its parent holds the video and (optionally) a
        ``<base>_variants/`` directory with A/B/C/… subfolders. If there
        are no variants, signals load from the parent folder directly.
        """
        mw = self.main_window
        if mw is None:
            return
        input_files = getattr(mw, 'input_files', None) or []
        if not input_files:
            # No live input; try the legacy last_processed path as a
            # fallback for users who opened the viewer before processing.
            lpd = getattr(mw, 'last_processed_directory', None)
            lpf = getattr(mw, 'last_processed_filename', None)
            if lpd:
                try:
                    self._load_signals_from(lpd, lpf)
                except Exception as e:
                    print(f"{self._log_prefix} auto-load failed: {e}")
            return

        input_path = Path(input_files[0])
        parent = input_path.parent
        # Strip any axis marker so a fungen-style anchor
        # (capture_123.sway.funscript) keys the viewer to the clean
        # project basename (capture_123) — matching what the processor
        # writes for triplet outputs and the variant-folder name.
        base = strip_axis_suffix(input_path.stem)
        # FunGen saves as <vid>.raw.funscript — strip the trailing .raw
        # so variant scanning and signal loading key off the clean stem.
        if base.endswith('.raw'):
            base = base[:-4]
        self._variant_source_base = base

        # Video lookup: same stem, common container extensions.
        video_path = self._find_video_next_to(input_path)
        if video_path is not None:
            try:
                self._open_video(str(video_path))
            except Exception as e:
                print(f"{self._log_prefix} video open failed: {e}")

        # Variants → populate radios + load first one.
        variants = self._scan_variants(parent, base)
        self._variant_folders = variants
        self._refresh_variant_radios()
        if variants:
            first = sorted(variants)[0]
            self._variant_var.set(first)
            self._load_signals_from(variants[first], base)
        else:
            # Fall back to the parent folder — some projects don't
            # use the variant system and just have flat outputs there.
            try:
                self._load_signals_from(parent, base)
            except Exception as e:
                print(f"{self._log_prefix} parent-folder load failed: {e}")

    def _maybe_autoload_4p_channels(self, buffers):
        """Enable the E1-E4 checkboxes once, the first time a folder
        with 4P signals is loaded. Subsequent loads respect whatever
        state the user has set — we don't want to re-check a box the
        user deliberately disabled."""
        if self._autoloaded_4p:
            return
        e_keys = ('e1', 'e2', 'e3', 'e4')
        axis_keys = ('E1', 'E2', 'E3', 'E4')
        present = [k for k in e_keys if k in buffers]
        if not present:
            return  # no 4P output in this load — don't burn the flag
        for signal, axis in zip(e_keys, axis_keys):
            if signal in buffers and axis in self._channel_enabled:
                self._channel_enabled[axis].set(True)
        self._autoloaded_4p = True

    def _refresh_variant_radios(self):
        """Rebuild the variant-radio row from ``_variant_folders``.

        Called at open time and after a regenerate in case new variant
        subfolders appeared. Hides the label row when there are none.
        """
        frame = self._variant_radio_frame
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        if not self._variant_folders:
            return
        ttk.Label(frame, text="Variant:").pack(side=tk.LEFT)
        for slot in sorted(self._variant_folders):
            ttk.Radiobutton(
                frame, text=slot, value=slot,
                variable=self._variant_var,
                command=self._on_variant_selected).pack(side=tk.LEFT, padx=(4, 0))

    def _on_variant_selected(self):
        """Radio callback: swap signal buffers to the chosen variant.

        The atomic swap is the same mechanism the hot-reload uses —
        scheduler keeps streaming, playhead doesn't jump, restim gets
        the new signals on the next tick.
        """
        slot = self._variant_var.get()
        folder = self._variant_folders.get(slot)
        if folder is None or self._variant_source_base is None:
            return
        self._load_signals_from(folder, self._variant_source_base)

    def _find_video_next_to(self, funscript_path: Path):
        parent = funscript_path.parent
        base = funscript_path.stem
        # Build a priority list of base names to try, from most-specific
        # to least. FunGen raw outputs are named <vid>.raw.funscript or
        # <vid>.raw.sway.funscript — strip .raw suffix before axis suffix.
        bases = [base]
        axis_stripped = self._strip_axis_suffix(base)
        if axis_stripped != base:
            bases.insert(0, axis_stripped)
        # Strip a trailing .raw from any candidate that still has it.
        raw_stripped = set()
        for b in list(bases):
            if b.endswith('.raw'):
                raw_stripped.add(b[:-4])
        for b in raw_stripped:
            if b not in bases:
                bases.insert(0, b)
        for b in bases:
            for ext in ('.mp4', '.mov', '.mkv', '.m4v', '.avi', '.webm'):
                p = parent / f'{b}{ext}'
                if p.exists():
                    return p
        return None

    @staticmethod
    def _strip_axis_suffix(stem: str) -> str:
        """Delegate to the shared helper — same marker set everywhere
        (orderer, processor, viewer). See ``processing.axis_markers``.
        """
        return strip_axis_suffix(stem)

    def _scan_variants(self, parent: Path, base: str) -> dict:
        """Return {slot_name: folder_path} for ``<base>_variants/*``.

        Only includes subfolders that actually contain at least one
        ``<base>.*.funscript`` — empty or half-populated slots are
        hidden so the radios never offer a dead option.
        """
        root = parent / f'{base}_variants'
        if not root.is_dir():
            return {}
        out = {}
        for sub in root.iterdir():
            if sub.is_dir() and any(sub.glob(f'{base}.*.funscript')):
                out[sub.name] = sub
        return out

    # ── UI ─────────────────────────────────────────────────────
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        outer = ttk.Frame(self)
        outer.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        # Video gets all the extra space; the controls column is pinned
        # to a fixed width so it doesn't collapse when wrapped in a
        # scroll canvas (canvases have no natural width of their own).
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=0, minsize=340)
        outer.rowconfigure(0, weight=1)

        # Left: video panel + heatmap placeholder (stacked 7:1) + ctrls
        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        # Video + heatmap stack. Row weights 7:1 reserve 1/8 of the
        # vertical space below the video for a rectangular heatmap.
        # The heatmap itself is a placeholder frame for now; future
        # work will fill it with the actual visualization.
        video_stack = ttk.Frame(left)
        video_stack.grid(row=0, column=0, sticky='nsew')
        video_stack.columnconfigure(0, weight=1)
        video_stack.rowconfigure(0, weight=7)
        video_stack.rowconfigure(1, weight=1)

        # Both backends use a tk.Label — cv2 blits cv2-decoded frames,
        # VLC blits vmem-callback frames. The NSView embed path we
        # briefly tried was abandoned: libvlc's macosx vout SIGSEGVs
        # when attaching to a plain Tk Frame's NSView on this combo.
        self._video_widget = tk.Label(
            video_stack, bg='#181818', fg='#888',
            text="(drop a video file here, or click Browse)",
            anchor='center')
        self._video_widget.grid(row=0, column=0, sticky='nsew')

        # Heatmap placeholder. A plain Frame with a subtle label; the
        # actual heatmap renderer will replace the label's contents.
        self._heatmap_frame = tk.Frame(
            video_stack, bg='#0f0f0f', highlightthickness=0)
        self._heatmap_frame.grid(row=1, column=0, sticky='nsew',
                                 pady=(2, 0))
        self._heatmap_label = tk.Label(
            self._heatmap_frame, bg='#0f0f0f', fg='#555',
            text="heatmap", anchor='center')
        self._heatmap_label.pack(fill=tk.BOTH, expand=True)

        video_ctrls = ttk.Frame(left)
        video_ctrls.grid(row=1, column=0, sticky='ew', pady=(4, 0))
        self._play_btn = ttk.Button(
            video_ctrls, text="▶ Play", width=10, command=self._toggle_play)
        self._play_btn.pack(side=tk.LEFT)
        ttk.Button(video_ctrls, text="Browse video…",
                   command=self._browse_video).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(video_ctrls, text="Clear",
                   command=self._clear_video).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Checkbutton(video_ctrls, text="Show video",
                        variable=self._show_video_var,
                        command=self._toggle_video_panel).pack(
                            side=tk.LEFT, padx=(10, 0))
        self._video_path_lbl = ttk.Label(video_ctrls, text="(none)",
                                         foreground='#666')
        self._video_path_lbl.pack(side=tk.LEFT, padx=(10, 0))

        # Preview FPS control (right-aligned). Raising it makes the
        # video preview smoother but eats more per-tick budget;
        # lowering it gives each tick more headroom against Tk
        # preemption (combobox dropdowns, etc.) at the cost of
        # choppier visuals. Independent of video_fps_cap, which
        # controls the decoder's polling rate.
        self._fps_var = tk.IntVar(value=self._FPS)
        fps_spin = ttk.Spinbox(
            video_ctrls, from_=5, to=60, increment=1, width=4,
            textvariable=self._fps_var,
            command=self._on_fps_change)
        fps_spin.pack(side=tk.RIGHT, padx=(0, 4))
        ttk.Label(video_ctrls, text="FPS:").pack(side=tk.RIGHT,
                                                  padx=(8, 2))
        # Pick up typed changes too (not just arrow clicks).
        self._fps_var.trace_add(
            'write', lambda *_: self._on_fps_change())

        scrub_frame = ttk.Frame(left)
        scrub_frame.grid(row=2, column=0, sticky='ew', pady=(4, 0))
        scrub_frame.columnconfigure(0, weight=1)
        self._scrub_scale = ttk.Scale(
            scrub_frame, from_=0.0, to=1.0, orient='horizontal',
            variable=self._scrub_var, command=self._on_scrub)
        self._scrub_scale.grid(row=0, column=0, sticky='ew')
        ttk.Label(scrub_frame, textvariable=self._time_label_var,
                  width=18).grid(row=0, column=1, padx=(6, 0))

        # Right: controls (vertically scrollable — the streaming button
        # is at the bottom of the stack, and we don't want it clipped
        # when the user's screen is shorter than our ideal geometry).
        right_outer = ttk.Frame(outer)
        right_outer.grid(row=0, column=1, sticky='nsew')
        right = self._make_scrollable_right(right_outer)
        right.columnconfigure(0, weight=1)

        self._build_signal_panel(right, row=0)
        self._build_channel_panel(right, row=1)
        self._build_reload_panel(right, row=2)
        self._build_media_source_panel(right, row=3)
        self._build_sync_panel(right, row=4)
        self._build_restim_panel(right, row=5)

    def _make_scrollable_right(self, outer):
        """Wrap ``outer`` in a vertical-scrolling canvas and return the
        inner frame that subpanels grid into. Mirrors the pattern the
        main app uses for parameter tabs."""
        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient='vertical',
                                  command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor='nw')

        def _on_inner_configure(_event):
            canvas.configure(scrollregion=canvas.bbox('all'))

        def _on_canvas_configure(event):
            # Ignore the 1×1 spurious first Configure some ttk versions
            # emit; without this the inner window stays pinned at width
            # 1 until the user hovers.
            if event.width > 1:
                canvas.itemconfig(inner_id, width=event.width)

        inner.bind('<Configure>', _on_inner_configure)
        canvas.bind('<Configure>', _on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mouse-wheel scrolling when hovering the canvas area. macOS
        # sends small delta values; Linux uses Button-4/5.
        #
        # Guard with winfo_exists() + TclError suppression: bind_all
        # installs globally, so after this window is destroyed any
        # mousewheel event in the app still fires this callback against
        # a now-invalid canvas ("invalid command name ..."). Silently
        # drop those.
        def _on_mousewheel(event):
            try:
                if not canvas.winfo_exists():
                    return
                if event.num == 4:
                    canvas.yview_scroll(-1, 'units')
                elif event.num == 5:
                    canvas.yview_scroll(1, 'units')
                else:
                    canvas.yview_scroll(int(-event.delta / 60), 'units')
            except tk.TclError:
                pass
        for seq in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            canvas.bind_all(seq, _on_mousewheel, add='+')

        return inner

    def _build_signal_panel(self, parent, row):
        box = ttk.LabelFrame(parent, text="Signals")
        box.grid(row=row, column=0, sticky='ew', pady=(0, 6))
        box.columnconfigure(0, weight=1)

        top = ttk.Frame(box)
        top.grid(row=0, column=0, sticky='ew', padx=4, pady=4)
        top.columnconfigure(0, weight=1)
        ttk.Label(top, textvariable=self._folder_var,
                  foreground='#555', wraplength=280).grid(
                      row=0, column=0, sticky='ew')
        ttk.Button(top, text="Browse…", command=self._browse_signal_folder)\
            .grid(row=0, column=1, padx=(4, 0))

        # Variant radios: populated at open time by _auto_load_on_open
        # and refreshed after "Regenerate + Reload" when new variant
        # folders may have appeared. Hidden until at least one exists.
        self._variant_radio_frame = ttk.Frame(box)
        self._variant_radio_frame.grid(row=1, column=0, sticky='w',
                                       padx=6, pady=(0, 2))

        ttk.Label(box, textvariable=self._signals_summary_var,
                  foreground='#333').grid(
                      row=2, column=0, sticky='w', padx=6, pady=(0, 4))

    def _build_channel_panel(self, parent, row):
        box = ttk.LabelFrame(parent, text="T-code channels")
        box.grid(row=row, column=0, sticky='ew', pady=(0, 6))
        # Description column (1) takes all the slack so the value
        # column (2) is guaranteed to land on the right edge of the
        # box. Without this the value labels can be pushed off-screen
        # when the right panel is narrow.
        box.columnconfigure(1, weight=1)
        # Live value display for each channel — lets the user verify
        # what the viewer is actually sending, since restim's UI
        # doesn't read back C0/P0-P3 spinbox values even when T-code
        # is landing. Refreshed by _tick_channel_values at ~10 Hz.
        self._channel_value_vars = {
            axis: tk.StringVar(value='----')
            for axis, _sig, _desc, _ in _CHANNEL_ROWS
        }
        for i, (axis, sig, desc, _default) in enumerate(_CHANNEL_ROWS):
            var = self._channel_enabled[axis]
            ttk.Checkbutton(box, variable=var,
                            text=f"{axis}  {sig}").grid(
                                row=i, column=0, sticky='w', padx=6)
            ttk.Label(box, text=desc, foreground='#777').grid(
                row=i, column=1, sticky='w', padx=(4, 6))
            # tk.Label (not ttk) gives us reliable bg/fg and a bold
            # font so the live values stand out and can't get styled
            # away by whatever theme the user is running.
            val_lbl = tk.Label(
                box, textvariable=self._channel_value_vars[axis],
                foreground='#1565C0', background='#f0f0f0',
                font=('Menlo', 10, 'bold'),
                width=7, anchor='e', padx=3, pady=1,
                relief='flat', borderwidth=1)
            val_lbl.grid(row=i, column=2, sticky='e', padx=(0, 6), pady=1)

    def _build_reload_panel(self, parent, row):
        box = ttk.LabelFrame(parent, text="Hot reload")
        box.grid(row=row, column=0, sticky='ew', pady=(0, 6))
        box.columnconfigure(0, weight=1)
        box.columnconfigure(1, weight=1)

        self._reload_btn = ttk.Button(
            box, text="Reload files", command=self._reload_signals_async)
        self._reload_btn.grid(row=0, column=0, sticky='ew', padx=6, pady=4)
        self._regen_btn = ttk.Button(
            box, text="Regenerate + Reload",
            command=self._regenerate_and_reload_async)
        self._regen_btn.grid(row=0, column=1, sticky='ew', padx=6, pady=4)

        ttk.Label(box, textvariable=self._reload_status_var,
                  foreground='#555', wraplength=280).grid(
                      row=1, column=0, columnspan=2,
                      sticky='w', padx=6, pady=(0, 4))

    def _build_media_source_panel(self, parent, row):
        """Panel for selecting where playback timing comes from.

        Default 'Internal' = the embedded video backend (cv2 or libvlc)
        drives the timeline. 'External VLC (HTTP)' polls a running
        VLC app via its web interface — useful when the user prefers
        their own player for video and just wants T-code to follow.
        """
        box = ttk.LabelFrame(parent, text="Media source")
        box.grid(row=row, column=0, sticky='ew', pady=(0, 6))
        box.columnconfigure(1, weight=1)

        # Mode selector
        ttk.Label(box, text="Mode:").grid(
            row=0, column=0, padx=(6, 4), pady=4, sticky='w')
        mode_combo = ttk.Combobox(
            box, textvariable=self._media_source_mode_var,
            values=['internal', 'vlc_http'],
            state='readonly', width=14)
        mode_combo.grid(row=0, column=1, columnspan=2,
                        sticky='w', padx=(0, 6), pady=4)
        mode_combo.bind(
            '<<ComboboxSelected>>',
            lambda _e: self._on_media_source_mode_change())

        # VLC connection settings (visible even in internal mode so
        # users can set address/password before switching modes —
        # simpler than collapsible UI and low screen-space cost).
        ttk.Label(box, text="VLC URL:").grid(
            row=1, column=0, padx=(6, 4), pady=4, sticky='w')
        ttk.Entry(box, textvariable=self._vlc_ext_address_var,
                  width=28).grid(
            row=1, column=1, sticky='ew', padx=(0, 4), pady=4)

        ttk.Label(box, text="Password:").grid(
            row=2, column=0, padx=(6, 4), pady=4, sticky='w')
        ttk.Entry(box, textvariable=self._vlc_ext_password_var,
                  width=28, show='*').grid(
            row=2, column=1, sticky='ew', padx=(0, 4), pady=4)

        # Status indicator (updated from the polling thread, marshalled
        # back to the Tk thread via after()).
        ttk.Label(box, text="Status:").grid(
            row=3, column=0, padx=(6, 4), pady=4, sticky='w')
        ttk.Label(box, textvariable=self._vlc_ext_status_var,
                  foreground='#444').grid(
            row=3, column=1, sticky='w', padx=(0, 4), pady=4)

        # "Apply" button re-reads address/password and reconnects if
        # already in vlc_http mode. Separate from the mode combo so the
        # user can edit credentials without toggling modes.
        ttk.Button(box, text="Apply", width=8,
                   command=self._on_media_source_apply).grid(
            row=3, column=2, padx=(0, 6), pady=4)

        # Auto-load checkbox. When VLC reports a loaded file we try to
        # find matching funscripts next to it (same folder, same stem,
        # e.g. video.mp4 -> video.alpha.funscript, video.e1.funscript,
        # ...). The existing Browse button still works for manual
        # overrides; auto-load just removes the extra click when the
        # naming convention matches.
        ttk.Checkbutton(
            box, text="Auto-load funscripts from VLC's video",
            variable=self._vlc_auto_load_var).grid(
            row=4, column=0, columnspan=3, sticky='w',
            padx=6, pady=(2, 6))

        # "Open in VLC" convenience — launches the user's external VLC
        # with the project's video file pre-loaded, assuming VLC is
        # already configured with HTTP enabled (see Help). Derives the
        # video from the current buffer's base name + common container
        # extensions. Does nothing gracefully if no video is found.
        ttk.Button(box, text="Open video in VLC",
                   command=self._launch_external_vlc).grid(
            row=5, column=0, columnspan=3, sticky='ew',
            padx=6, pady=(0, 6))

    def _on_media_source_mode_change(self):
        """Handle a change in the media-source selector."""
        mode = self._media_source_mode_var.get()
        # Always tear down any existing external source before
        # creating a new one — simpler than mutating state.
        self._teardown_external_source()
        # Reset auto-load tracking so switching modes and back
        # re-checks for matching funscripts.
        self._last_auto_loaded_path = ""
        if mode == 'vlc_http':
            self._setup_external_vlc()
        else:
            self._vlc_ext_status_var.set('(disabled)')

    def _on_media_source_apply(self):
        """Re-read address/password and reconnect (if in vlc_http mode)."""
        # Persist to in-memory config so "Save Config" picks it up.
        try:
            cfg = self.main_window.current_config
            if cfg is not None:
                ext = cfg.setdefault('external_media', {})
                ext['vlc_address'] = self._vlc_ext_address_var.get()
                ext['vlc_password'] = self._vlc_ext_password_var.get()
        except AttributeError:
            pass
        if self._media_source_mode_var.get() == 'vlc_http':
            # Recreate the adapter with fresh settings.
            self._teardown_external_source()
            self._setup_external_vlc()

    def _setup_external_vlc(self):
        """Create and enable the external VLC adapter."""
        address = self._vlc_ext_address_var.get().strip()
        password = self._vlc_ext_password_var.get()
        try:
            self._external_source = ExternalVLCSource(
                address=address, password=password)
        except ImportError as e:
            # `requests` not installed in this env.
            self._vlc_ext_status_var.set(f'({e})')
            self._external_source = None
            return
        # Subscribe BEFORE enable so we catch the first status change.
        self._external_source_unsubscribe = (
            self._external_source.on_connection_changed(
                self._on_external_source_connection_changed))
        self._external_source.enable()
        self._vlc_ext_status_var.set('connecting...')
        # Also schedule a periodic UI refresh. The state machine only
        # fires callbacks on transitions, so if the first poll fails
        # (VLC unreachable) the UI would sit forever on "connecting..."
        # without this. Polling the adapter from the UI every 500 ms
        # surfaces errors and keeps the label fresh regardless.
        self._schedule_external_status_refresh()

    def _teardown_external_source(self):
        """Disable and drop any active external source adapter."""
        if self._external_status_refresh_id is not None:
            try:
                self.after_cancel(self._external_status_refresh_id)
            except Exception:
                pass
            self._external_status_refresh_id = None
        if self._external_source_unsubscribe is not None:
            try:
                self._external_source_unsubscribe()
            except Exception:
                pass
            self._external_source_unsubscribe = None
        if self._external_source is not None:
            try:
                self._external_source.disable()
            except Exception:
                pass
            self._external_source = None

    def _schedule_external_status_refresh(self):
        """Re-read the adapter state and reschedule. Runs on the Tk
        thread — safe to touch the status var."""
        if self._external_source is None:
            return
        self._refresh_external_source_status()
        try:
            self._external_status_refresh_id = self.after(
                500, self._schedule_external_status_refresh)
        except tk.TclError:
            self._external_status_refresh_id = None

    def _resolve_project_video_path(self):
        """Find the video associated with the currently-loaded project.

        Tries in priority order:
          1. main_window.input_files[0] — the original source funscript;
             its neighbouring video is what the user dropped in.
          2. self._buffer_dir + self._buffer_base — if signals are
             loaded from a _variants subfolder, walk up to the parent
             that holds the source files and look for a video there.
          3. self._video_path — if the embedded player already has
             one loaded, reuse that.

        Returns a pathlib.Path or None.
        """
        from pathlib import Path as _P
        mw = self.main_window
        if mw is not None:
            input_files = getattr(mw, 'input_files', None) or []
            if input_files:
                try:
                    v = self._find_video_next_to(_P(input_files[0]))
                    if v is not None:
                        return v
                except Exception:
                    pass
        # Fallback: walk up from the buffer dir. _variants/<slot>
        # layout means the real source folder is two levels up; flat
        # layout means it's the buffer dir itself.
        if self._buffer_dir is not None and self._buffer_base:
            candidates = [self._buffer_dir,
                          self._buffer_dir.parent,
                          self._buffer_dir.parent.parent]
            for folder in candidates:
                try:
                    p = folder / f"{self._buffer_base}.mp4"
                    v = self._find_video_next_to(p)
                    if v is not None:
                        return v
                except Exception:
                    pass
        if getattr(self, '_video_path', None):
            return _P(self._video_path)
        return None

    def _launch_external_vlc(self):
        """Open the project's video in the user's external VLC.

        Uses the OS's default 'open with VLC' handler on macOS, which
        picks up whatever HTTP-interface config VLC has in its
        preferences. No attempt to pass --extraintf / --http-password
        flags — those belong in VLC's persistent Preferences so the
        user doesn't need to remember them every launch.

        After launching, automatically switches Media Source mode to
        vlc_http IF it's currently Internal. If the user already set
        a mode explicitly, respect their choice.
        """
        import subprocess
        video = self._resolve_project_video_path()
        if video is None:
            messagebox.showinfo(
                "Open in VLC",
                "No video found for the current project.\n\n"
                "Load source files or signals first, or drop a video "
                "into the embedded panel, then try again.",
                parent=self)
            return

        try:
            # macOS-native path. 'open -a VLC <file>' respects VLC's
            # own configuration (HTTP interface, password, etc.) and
            # handles the case where VLC is already running (opens
            # the file as a playlist entry instead of relaunching).
            if sys.platform == 'darwin':
                subprocess.Popen(['open', '-a', 'VLC', str(video)])
            elif sys.platform == 'win32':
                # Windows: 'start' via cmd. Uses file association;
                # user may need to set VLC as default .mp4 handler,
                # or we can add a config for the full VLC path later.
                subprocess.Popen(
                    ['cmd', '/c', 'start', '', '/B', str(video)],
                    shell=False)
            else:
                # Linux / other: try vlc directly, fall back to
                # xdg-open if vlc isn't on PATH.
                try:
                    subprocess.Popen(['vlc', str(video)])
                except FileNotFoundError:
                    subprocess.Popen(['xdg-open', str(video)])
        except Exception as e:
            messagebox.showerror(
                "Open in VLC",
                f"Failed to launch VLC: {e}",
                parent=self)
            return

        # Auto-switch to vlc_http mode if the user hasn't picked a
        # mode yet. The polling adapter will start trying to connect;
        # once VLC is up and its HTTP interface is listening, status
        # flips from "not connected" to "playing — <file>".
        if self._media_source_mode_var.get() == 'internal':
            self._media_source_mode_var.set('vlc_http')
            self._on_media_source_mode_change()

    def _maybe_autoload_scripts_for(self, video_path: str) -> None:
        """Try to load funscripts matching a video file VLC just
        opened.

        Mirrors the logic of _auto_load_on_open but starts from the
        video path (as reported by VLC) rather than main_window's
        input_files. Specifically:

        1. Derive parent + base from the video path
           (e.g. ``.../test video/exp.mp4`` → parent=``.../test video``,
           base=``exp``).
        2. Check for a ``<base>_variants/`` subfolder. This is the
           standard funscript-tools output layout: processed signals
           (alpha/beta/e1..e4/pulse_*, etc.) live under
           ``_variants/A``, ``_variants/B``, and so on.
        3. If variants exist, pick one:
             - the user's currently-selected variant if it exists in
               the new video's variant set (sticky across videos);
             - otherwise the first alphabetically (usually A).
        4. If no variants folder, fall back to loading from the
           parent directly — some projects keep flat outputs there.

        Silent no-op if nothing resolves. The user can still Browse
        manually or disable auto-load via the checkbox.
        """
        from pathlib import Path as _P
        try:
            vp = _P(video_path)
        except (TypeError, ValueError):
            return
        parent = vp.parent
        if not parent.is_dir():
            return
        base = vp.stem
        if not base:
            return

        try:
            variants = self._scan_variants(parent, base)
        except Exception as e:
            print(f"{self._log_prefix} auto-load variant scan failed: {e}")
            variants = {}

        if variants:
            # Sticky selection: if user picked 'D' on the last video
            # and the new video also has a 'D' variant, load that.
            current = self._variant_var.get()
            chosen = current if current in variants else sorted(variants)[0]
            try:
                self._variant_source_base = base
                self._variant_folders = variants
                self._refresh_variant_radios()
                self._variant_var.set(chosen)
                self._load_signals_from(variants[chosen], base)
            except Exception as e:
                print(f"{self._log_prefix} auto-load from variant "
                      f"'{chosen}' failed: {e}")
            return

        # No variants folder — try a flat layout next to the video.
        # Only fire the loader if at least one matching file exists,
        # so a video with only source motion files (x/y/z/rz, no
        # processed signals) doesn't trigger a garbage load.
        matches = list(parent.glob(f"{base}.*.funscript"))
        # Filter out the source-motion files that should never be
        # loaded as processed signals. These are funscript-tools'
        # known axis-marker suffixes for raw triplet input.
        _SOURCE_ONLY_SUFFIXES = {'x', 'y', 'z', 'rz',
                                  'sway', 'heave', 'surge',
                                  'roll', 'twist', 'stroke'}
        signal_matches = [m for m in matches
                          if m.stem.rsplit('.', 1)[-1].lower()
                          not in _SOURCE_ONLY_SUFFIXES]
        if not signal_matches:
            return
        try:
            self._variant_source_base = base
            self._variant_folders = {}
            self._refresh_variant_radios()
            self._load_signals_from(str(parent), base)
        except Exception as e:
            print(f"{self._log_prefix} auto-load flat-layout "
                  f"failed for {video_path}: {e}")

    def _on_external_source_connection_changed(self):
        """Called from the polling thread when the external source's
        connection-state or file changes. Hop onto the Tk thread
        before touching any widget."""
        try:
            self.after(0, self._refresh_external_source_status)
        except tk.TclError:
            # Window closed; nothing to update.
            pass

    def _refresh_external_source_status(self):
        """Update the status label on the Tk thread. Also triggers the
        auto-load of matching funscripts if VLC's loaded file has
        changed since the last check."""
        if self._external_source is None:
            self._vlc_ext_status_var.set('(disabled)')
            return
        state = self._external_source.state()
        path = self._external_source.media_path()
        short = path.rsplit('/', 1)[-1] if path else ''

        # Auto-load funscripts next to the video when the reported
        # file path changes. Kept in the UI refresh loop so it runs
        # on the Tk thread — safe to call the existing loader.
        if (path
                and path != self._last_auto_loaded_path
                and self._vlc_auto_load_var.get()):
            self._last_auto_loaded_path = path
            self._maybe_autoload_scripts_for(path)

        if state == MediaConnectionState.NOT_CONNECTED:
            # Surface the underlying reason so the user knows whether
            # to check the URL, password, or that VLC is running.
            err = getattr(self._external_source, 'last_error', '') or ''
            if err:
                # Trim overlong error strings so the label doesn't
                # blow out the panel width.
                short_err = err if len(err) <= 60 else err[:57] + '...'
                self._vlc_ext_status_var.set(f'not connected — {short_err}')
            else:
                self._vlc_ext_status_var.set('not connected')
        elif state == MediaConnectionState.CONNECTED_BUT_NO_FILE_LOADED:
            self._vlc_ext_status_var.set('connected (no file)')
        elif state == MediaConnectionState.CONNECTED_AND_PAUSED:
            self._vlc_ext_status_var.set(f'paused — {short}')
        elif state == MediaConnectionState.CONNECTED_AND_PLAYING:
            self._vlc_ext_status_var.set(f'playing — {short}')
        else:
            self._vlc_ext_status_var.set(str(state))

    def _build_sync_panel(self, parent, row):
        box = ttk.LabelFrame(parent, text="Sync")
        box.grid(row=row, column=0, sticky='ew', pady=(0, 6))
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Offset (s):").grid(
            row=0, column=0, padx=(6, 4), pady=4, sticky='w')
        # ±3 s range covers typical restim pipeline delay (up to
        # several hundred ms) plus our own video display latency, with
        # headroom for codecs that decode slower than real time. The
        # entry field accepts any numeric value so extreme cases are
        # still reachable by typing.
        ttk.Scale(box, from_=-3.0, to=3.0, orient='horizontal',
                  variable=self._sync_offset_var).grid(
                      row=0, column=1, sticky='ew', padx=(0, 4))
        ttk.Entry(box, textvariable=self._sync_offset_var, width=7).grid(
            row=0, column=2, padx=(0, 6))

        # Test pulse: a visible + tactile marker for sync alignment.
        self._pulse_btn = ttk.Button(
            box, text="Test Pulse", command=self._fire_test_pulse)
        self._pulse_btn.grid(row=1, column=0, columnspan=2,
                             sticky='ew', padx=6, pady=(4, 4))
        self._pulse_marker = tk.Label(
            box, text='', width=10, background='#eee',
            foreground='white', font=('TkDefaultFont', 10, 'bold'))
        self._pulse_marker.grid(row=1, column=2, padx=(0, 6), pady=(4, 4))

        # Lock signal to video — when checked, T-code follows the
        # actual decoded-frame time (sync at cost of speed). When
        # unchecked, T-code follows wall-clock (real-time at cost of
        # visual sync). Toggle if the video plays slower than the
        # signal and you'd rather have real-time stim.
        lock_chk = ttk.Checkbutton(
            box, text="Lock stim to video (uncheck for real-time stim)",
            variable=self._lock_signal_to_video_var)
        lock_chk.grid(row=2, column=0, columnspan=3,
                      sticky='w', padx=6, pady=(2, 6))

    def _build_restim_panel(self, parent, row):
        box = ttk.LabelFrame(parent, text="restim")
        box.grid(row=row, column=0, sticky='ew')
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Host:").grid(row=0, column=0, padx=(6, 4), pady=2,
                                          sticky='w')
        ttk.Entry(box, textvariable=self._host_var, width=14).grid(
            row=0, column=1, sticky='w', pady=2)
        ttk.Label(box, text="Port:").grid(row=0, column=2, padx=(6, 4), pady=2,
                                          sticky='w')
        ttk.Entry(box, textvariable=self._port_var, width=7).grid(
            row=0, column=3, sticky='w', pady=2)

        ttk.Label(box, textvariable=self._conn_status_var).grid(
            row=1, column=0, columnspan=2, padx=6, pady=2, sticky='w')
        ttk.Button(box, text="Test", command=self._test_connection).grid(
            row=1, column=2, columnspan=2, padx=6, pady=2, sticky='ew')

        self._stream_btn = ttk.Button(
            box, text="▶ Start Streaming", command=self._toggle_streaming)
        self._stream_btn.grid(row=2, column=0, columnspan=4,
                              sticky='ew', padx=6, pady=(6, 6))

    # ── Signal loading ─────────────────────────────────────────
    def _browse_signal_folder(self):
        path = filedialog.askdirectory(parent=self, title="Signal folder")
        if not path:
            return
        self._load_signals_from(Path(path), None)

    def _load_signals_from(self, folder, base):
        """Scan ``folder`` for <base>.<suffix>.funscript files. If base
        is None, pick the most common basename among *.funscript files.
        """
        folder = Path(folder)
        if not folder.is_dir():
            return
        # Pick a basename if one wasn't supplied.
        if not base:
            names = [p.stem for p in folder.glob('*.funscript')]
            # Strip trailing ".<suffix>"
            candidates = {}
            for n in names:
                if '.' in n:
                    b = n.rsplit('.', 1)[0]
                    candidates[b] = candidates.get(b, 0) + 1
            base = max(candidates, key=candidates.get) if candidates else None
        if not base:
            self._signals_summary_var.set("(no funscripts found)")
            self._folder_var.set(str(folder))
            return

        buffers = self._load_buffers(folder, base)

        self._buffers = buffers
        self._buffer_dir = folder
        self._buffer_base = base
        self._folder_var.set(f"{folder}  —  base: {base}")
        if buffers:
            self._signals_summary_var.set(
                f"Loaded {len(buffers)}: " + ", ".join(sorted(buffers)))
        else:
            # No processed signals matched. Check whether the folder
            # has only source motion files (raw x/y/z/rz input), in
            # which case the user hasn't run Process yet. Point them
            # at the fix rather than just reporting "Loaded 0".
            _SOURCE_SUFFIXES = {'x', 'y', 'z', 'rz', 'sway', 'heave',
                                'surge', 'roll', 'twist', 'stroke'}
            source_files = []
            # Plain <base>.funscript is also a source file (stroke).
            if (folder / f"{base}.funscript").exists():
                source_files.append(f"{base}.funscript")
            for suffix in _SOURCE_SUFFIXES:
                p = folder / f"{base}.{suffix}.funscript"
                if p.exists():
                    source_files.append(p.name)
            if source_files:
                self._signals_summary_var.set(
                    f"Source files present ({len(source_files)}) but "
                    f"no processed signals — click \"Process All Files\" "
                    f"in the main window, then Reload here.")
            else:
                self._signals_summary_var.set(
                    f"Loaded 0 — no signals found for base \"{base}\".")

        # Auto-enable E1-E4 checkboxes the first time we see 4P signals
        # so users with a 4-electrode setup don't have to tick four
        # boxes. Once toggled (user or auto), we leave it alone.
        self._maybe_autoload_4p_channels(buffers)

        # Duration = max of loaded buffers.
        if buffers:
            self._total_duration = max(
                (float(fs.x[-1]) if len(fs.x) else 0.0)
                for fs in buffers.values())
        else:
            self._total_duration = 0.0

        # Preserve the current playhead across reloads and variant
        # switches so switching A→B mid-playback keeps the video and
        # audio aligned at the same moment. Only clamp into range and
        # fall back to 0 if we have no duration yet.
        if self._total_duration > 0:
            self._playhead_t = max(
                0.0, min(self._playhead_t, self._total_duration))
        else:
            self._playhead_t = 0.0
        self._update_scrub_range()
        self._update_time_label()

    # ── Playback (playhead clock) ──────────────────────────────
    def _update_scrub_range(self):
        try:
            self._scrub_scale.config(to=max(self._total_duration, 0.001))
        except tk.TclError:
            pass
        try:
            self._scrub_var.set(self._playhead_t)
        except tk.TclError:
            pass

    def _tick_channel_values(self):
        """Periodically refresh the per-channel value display. Runs on
        the main thread at ~10 Hz — fast enough to see changing values,
        slow enough to skip a chunk of PIL / Tk work the streamer
        doesn't care about. Samples the same way the scheduler would,
        so the number shown is exactly what's going out on the wire
        (or would be, if the channel were checked).
        """
        try:
            from processing.tcode_stream import sample_at
            t = self._signal_clock()
            for axis, signal_name, _desc, _ in _CHANNEL_ROWS:
                fs = self._buffers.get(signal_name)
                if fs is None:
                    self._channel_value_vars[axis].set('—')
                    continue
                val = sample_at(fs, t)
                intval = max(0, min(9999, int(round(val * 9999))))
                enabled = self._channel_enabled_cached.get(axis, False)
                # Dim muted channels with a dash prefix so the eye
                # tells them apart from live ones at a glance.
                if enabled:
                    self._channel_value_vars[axis].set(f"{intval:04d}")
                else:
                    self._channel_value_vars[axis].set(f"·{intval:04d}")
        except Exception as e:
            # Never let a display error kill the loop.
            print(f"{self._log_prefix} channel-value tick failed: {e}")
        self._channel_value_after_id = self.after(
            100, self._tick_channel_values)

    @staticmethod
    def _fmt_mmss(seconds: float) -> str:
        """Format seconds as MM:SS (or H:MM:SS for long videos)."""
        if seconds < 0 or seconds != seconds:  # guard NaN
            seconds = 0.0
        total = int(seconds)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _update_time_label(self):
        self._time_label_var.set(
            f"{self._fmt_mmss(self._playhead_t)} / "
            f"{self._fmt_mmss(self._total_duration)}")

    def _on_scrub(self, _val=None):
        """Scale callback: seek the playhead. Runs whether or not the
        clock is playing — if the user drags mid-playback, playback
        just picks up from the new position on the next tick.

        During playback ``_tick`` also writes the current playhead into
        ``_scrub_var``, which re-enters this callback; that's fine,
        the write is a self-fixed-point and produces no visible work.
        """
        try:
            t = float(self._scrub_var.get())
        except (tk.TclError, ValueError):
            return
        new_t = max(0.0, min(t, self._total_duration))
        # Tick-driven updates set scrub_var to _playhead_t exactly, so
        # this guard keeps us from re-forcing the video on every tick.
        if abs(new_t - self._playhead_t) < 1e-6:
            return
        self._playhead_t = new_t
        self._update_time_label()
        self._update_video_frame(force=True)
        self._refresh_video_actual_time()

    def _on_fps_change(self):
        """Update the preview tick rate when the FPS spinbox changes.

        Writes through to self._FPS (used by _tick's after() scheduling
        at each cycle) and threads the new value into the in-memory
        config so clicking Save Config in the main window persists it.
        Malformed values are ignored rather than crashing.
        """
        try:
            new_fps = int(self._fps_var.get())
        except (tk.TclError, ValueError):
            return
        new_fps = max(5, min(60, new_fps))
        if new_fps != self._FPS:
            self._FPS = new_fps
            # Keep the legacy _TICK_MS attribute in sync so any external
            # code reading it (shouldn't exist, but defensive) sees a
            # consistent value.
            self._TICK_MS = int(1000 / new_fps)
        try:
            cfg = self.main_window.current_config
            if cfg is not None:
                cfg.setdefault('ui', {})['tcode_preview_fps'] = new_fps
        except AttributeError:
            pass

    def _toggle_play(self):
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if self._total_duration <= 0.0:
            messagebox.showinfo(
                "No signals",
                "Load a signal folder before playing.",
                parent=self)
            return
        # Resync video to the current playhead before we start ticking.
        # This is a cheap way to correct any drift that accumulated
        # during a prior play session: each pause/play cycle snaps
        # video back to the playhead time. Uses force=True so it
        # goes down the cap.set() hard-seek path.
        self._update_video_frame(force=True)
        self._refresh_video_actual_time()
        # Release the pause-time V1 mute so the scheduler's next tick
        # emits the real volume-at-playhead value and the device picks
        # up stim naturally on resume.
        self._video_paused_mute_v1 = False
        self._playing = True
        self._play_btn.config(text="⏸ Pause")
        self._last_tick_wall = time.monotonic()
        self._tick()

    def _stop_playback(self):
        self._playing = False
        self._play_btn.config(text="▶ Play")
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        # Silence the device on pause — "paused video = constant hum"
        # is surprising; the natural expectation is that the stim
        # stops when the video does. We do three things:
        # (a) set the pause-mute flag so the scheduler's enable filter
        #     drops V1 on every tick (prevents re-emitting the frozen
        #     playhead's V1 value),
        # (b) send one immediate V1=0 packet,
        # (c) send a second V1=0 on a short delay so it wins any race
        #     with an in-flight scheduler tick. The scheduler runs in
        #     another thread at ~60 Hz and may have already built the
        #     current frame (with a live V1 value) before step (a)
        #     landed. That live packet can arrive at restim AFTER our
        #     immediate zero, leaving volume frozen at the paused level.
        #     The delayed resend lands ≥1 scheduler tick later, by which
        #     time the mute flag is guaranteed to have been observed
        #     and no further V1 packets are in flight.
        # _start_playback clears the flag so resume is clean.
        self._video_paused_mute_v1 = True
        if (self._sender is not None
                and self._scheduler is not None
                and self._scheduler.is_running
                and self._channel_enabled_cached.get('V1', False)):
            self._send_zero_out('V1')
            self.after(40, self._force_v1_zero_if_still_paused)

    def _force_v1_zero_if_still_paused(self):
        """Belt-and-suspenders V1 zero resend ~40 ms after pause, to win
        the race against any scheduler tick that was mid-flight when the
        mute flag flipped. Aborts if playback has resumed in the meantime."""
        if not self._video_paused_mute_v1:
            return
        if (self._sender is not None
                and self._scheduler is not None
                and self._scheduler.is_running
                and self._channel_enabled_cached.get('V1', False)):
            self._send_zero_out('V1')

    def _tick(self):
        if not self._playing:
            return
        now = time.monotonic()
        dt = now - (self._last_tick_wall or now)
        self._last_tick_wall = now
        # Cap per-tick advance so Tk preemption (e.g. ttk.Combobox
        # dropdown open/close on macOS, which can block ~100-200 ms)
        # produces a brief slow-mo rather than a visible jump. Pulled
        # at tick time rather than stored because self._FPS can be
        # live-adjusted via the UI spinner.
        max_dt = 1.5 * (1.0 / self._FPS)
        if dt > max_dt:
            dt = max_dt
        self._playhead_t += dt
        if self._total_duration > 0 and self._playhead_t >= self._total_duration:
            self._playhead_t = self._total_duration
            self._stop_playback()
        try:
            self._scrub_var.set(self._playhead_t)
        except tk.TclError:
            pass
        self._update_time_label()
        self._update_video_frame()
        self._refresh_video_actual_time()
        # Reschedule at the current rate (picks up live FPS changes).
        self._after_id = self.after(int(1000 / self._FPS), self._tick)

    # ── Async decode worker ─────────────────────────────────────
    def _start_decode_worker(self):
        if self._decode_thread is not None and self._decode_thread.is_alive():
            return
        self._decode_stop.clear()
        self._decode_thread = threading.Thread(
            target=self._decode_loop, name='tcode-video-decoder',
            daemon=True)
        self._decode_thread.start()

    def _stop_decode_worker(self):
        if self._decode_thread is None:
            return
        self._decode_stop.set()
        self._decode_thread.join(timeout=1.0)
        self._decode_thread = None

    def _preprocess_frame_for_display(self, frame):
        """Worker-side: convert a freshly decoded BGR numpy frame to a
        PIL image already sized to the current display widget. Moves
        cv2.resize + cv2.cvtColor + Image.fromarray off the UI thread.
        Returns a PIL.Image, or None if prerequisites aren't met.

        Safe to call from the decode worker: only reads
        ``self._video_widget_size`` (atomic tuple swap from main) and
        uses cv2 / PIL in isolation.
        """
        if frame is None:
            return None
        try:
            import cv2
            from PIL import Image
        except ImportError:
            return None
        try:
            h, w = frame.shape[:2]
            wid_w, wid_h = self._video_widget_size
            resized = frame
            if wid_w > 20 and wid_h > 20:
                scale = min(wid_w / w, wid_h / h)
                # Don't upscale above native — wastes time and adds no
                # information (the PhotoImage copy is the same cost).
                if scale < 1.0:
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    resized = cv2.resize(
                        frame, (new_w, new_h),
                        interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        except Exception as e:
            print(f"{self._log_prefix} preprocess failed: {e}")
            return None

    def _decode_loop(self):
        """Background decode loop. Owns ``self._video_cap`` through the
        cap lock, advances it toward ``self._decode_target_t``, and
        writes the latest decoded frame into ``self._latest_frame``.

        Three cases:
        * Big backward jump or very large forward jump: hard seek via
          ``cap.set``.
        * Minor forward lag: ``grab()`` to skip any frames we've fallen
          behind on, then ``grab() + retrieve()`` to decode the one we
          actually show.
        * Ahead of target (worker decoded faster than wall-clock):
          sleep briefly so we don't burn CPU spinning past the target.
        """
        try:
            import cv2
        except ImportError:
            return
        while not self._decode_stop.is_set():
            cap = self._video_cap
            if cap is None:
                if self._decode_stop.wait(0.05):
                    break
                continue
            try:
                with self._decode_cap_lock:
                    if self._video_cap is None:
                        continue
                    cap = self._video_cap
                    native_fps = self._video_fps or 30.0
                    # Effective decode rate: min of video's native fps
                    # and the user-configurable cap. A lower cap makes
                    # the "behind target" threshold larger, so the
                    # worker decodes fewer frames per second — most of
                    # the skipped work happens via cheap cap.grab()
                    # rather than full retrieve+decode.
                    fps = min(native_fps, float(self._video_fps_cap))
                    frame_period = 1.0 / max(fps, 1.0)
                    # Pending seek request from UI (scrub / pause-play
                    # resync) takes priority.
                    seek = self._decode_seek_target
                    if seek is not None:
                        self._decode_seek_target = None
                        cap.set(cv2.CAP_PROP_POS_MSEC, seek * 1000.0)
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            self._latest_frame = frame
                            self._latest_frame_t = (
                                cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                            # Pre-build the display-ready PIL image so
                            # the main thread's next tick doesn't pay
                            # the resize/cvtColor/PIL cost itself.
                            self._latest_pil_image = (
                                self._preprocess_frame_for_display(frame))
                        continue

                    target = float(self._decode_target_t)
                    try:
                        actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    except Exception:
                        actual = 0.0
                    diff = target - actual

                    if diff < -0.1 or diff > 2.0:
                        # Big jump in either direction → hard seek.
                        # Cheaper than grabbing hundreds of frames.
                        cap.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0)
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            self._latest_frame = frame
                            self._latest_frame_t = (
                                cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                            self._latest_pil_image = (
                                self._preprocess_frame_for_display(frame))
                        continue

                    if diff > frame_period * 0.5:
                        # We're behind the wall-clock target. Grab-skip
                        # any extra frames we've fallen behind on (cheap),
                        # then grab+retrieve the one we'll actually show.
                        skip = max(0,
                                   min(int((diff - frame_period)
                                           / frame_period), 10))
                        for _ in range(skip):
                            cap.grab()
                        ok = cap.grab()
                        if ok:
                            ok, frame = cap.retrieve()
                            if ok and frame is not None:
                                self._latest_frame = frame
                                self._latest_frame_t = (
                                    cap.get(cv2.CAP_PROP_POS_MSEC)
                                    / 1000.0)
                                self._latest_pil_image = (
                                    self._preprocess_frame_for_display(frame))
            except Exception as e:
                print(f"{self._log_prefix} decode error: {e}")

            # Pacing: if we've caught up (or got ahead), sleep a short
            # while so we don't spin. The tick time of the playhead
            # drives target_t forward, so we'll wake naturally.
            try:
                actual_t = float(self._latest_frame_t or 0.0)
            except Exception:
                actual_t = 0.0
            lead = actual_t - float(self._decode_target_t)
            if lead > 0.0:
                # Ahead of wall-clock → sleep until the playhead catches up.
                if self._decode_stop.wait(min(lead, 0.05)):
                    break
            else:
                # Behind: loop immediately for another grab. Tiny sleep
                # so we don't saturate a core if something else goes wrong.
                if self._decode_stop.wait(0.001):
                    break

    def _display_latest_frame(self):
        """Main-thread: blit the worker's pre-built PIL image to the
        video widget. All CPU-heavy work (resize, color convert, PIL
        construction) already happened in the decode worker — the
        only main-thread cost here is the unavoidable Tk-bound
        ImageTk.PhotoImage call plus label.config.

        Also refreshes the cached widget size the worker reads, so a
        window resize gets picked up on the next decoded frame."""
        if not self._show_video_cached:
            return
        # Refresh widget size cache for the worker. Calling winfo_*
        # from main thread is guaranteed safe; the worker just reads
        # the stored tuple (atomic in CPython).
        try:
            wid_w = max(1, self._video_widget.winfo_width())
            wid_h = max(1, self._video_widget.winfo_height())
            self._video_widget_size = (wid_w, wid_h)
        except tk.TclError:
            pass

        pil_img = self._latest_pil_image
        if pil_img is not None:
            try:
                from PIL import ImageTk
            except ImportError:
                return
            try:
                self._video_photo = ImageTk.PhotoImage(pil_img)
                self._video_widget.config(
                    image=self._video_photo, text='')
                return
            except Exception as e:
                print(f"{self._log_prefix} display failed: {e}")
                return

        # Fallback path: worker hasn't produced a PIL image yet (very
        # first frame, or preprocess failed). Do the work on main so
        # the viewer still shows something rather than staying blank.
        frame = self._latest_frame
        if frame is None:
            return
        try:
            import cv2
            from PIL import Image, ImageTk
        except ImportError:
            return
        try:
            h, w = frame.shape[:2]
            wid_w, wid_h = self._video_widget_size
            if wid_w > 20 and wid_h > 20:
                scale = min(wid_w / w, wid_h / h)
                if scale < 1.0:
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    frame = cv2.resize(
                        frame, (new_w, new_h),
                        interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            self._video_photo = ImageTk.PhotoImage(img)
            self._video_widget.config(image=self._video_photo, text='')
        except Exception as e:
            print(f"{self._log_prefix} display fallback failed: {e}")

    def _open_video(self, path):
        """Open a video. VLC backend hands the path to libvlc and lets
        it draw natively into our Tk frame; cv2 backend wraps the
        mixin's open, adding the decode-worker startup + cap lock.

        The cap lock only matters for cv2: without it the worker could
        try to grab() on a cap the main thread is replacing, which
        segfaults in OpenCV on some backends. VLC owns its own media
        internally; no locking needed on our side."""
        if self._use_vlc and self._vlc_player is not None:
            self._video_path = path
            try:
                self._video_path_lbl.config(
                    text=os.path.basename(path), foreground='#222')
            except tk.TclError:
                pass
            # vmem callbacks write into our own buffer, so there's no
            # NSView embed step — we can load synchronously. Duration
            # and fps become valid after libvlc parses the media,
            # which we poll for briefly below.
            self._vlc_player.load(path)
            # Start the player paused-at-zero so the first unlock
            # callback fires and fills our display buffer with frame 0
            # for immediate display. Without this, nothing shows until
            # the user hits Play.
            self._vlc_player.play()
            self._vlc_player.pause()
            self._vlc_player.set_time(0.0)
            dur = 0.0
            fps = 30.0
            for _ in range(15):
                d = self._vlc_player.get_duration()
                if d and d > 0:
                    dur = d
                    break
                try:
                    self.update()
                except tk.TclError:
                    return
                time.sleep(0.02)
            try:
                fps = self._vlc_player.get_fps() or 30.0
            except Exception:
                fps = 30.0
            self._video_duration = dur
            self._video_fps = fps if fps > 0 else 30.0
            self._video_last_frame_time = -1.0
            self._video_is_loaded = True
            try:
                self._update_scrub_range()
            except Exception:
                pass
            self._update_video_frame(force=True)
            return

        with self._decode_cap_lock:
            super()._open_video(path)
        if self._video_cap is not None:
            self._video_is_loaded = True
            self._start_decode_worker()

    def _clear_video(self):
        """Tear down video state. VLC branch pauses the player and
        resets duration/path; cv2 branch stops the decode worker and
        releases the cap under lock."""
        if self._use_vlc and self._vlc_player is not None:
            try:
                self._vlc_player.pause()
            except Exception:
                pass
            self._video_path = None
            self._video_duration = 0.0
            self._video_last_frame_time = -1.0
            self._video_actual_t = None
            self._video_is_loaded = False
            try:
                self._video_path_lbl.config(text="(none)",
                                            foreground='#666')
            except tk.TclError:
                pass
            return

        self._stop_decode_worker()
        self._latest_frame = None
        self._latest_frame_t = None
        self._latest_pil_image = None
        with self._decode_cap_lock:
            super()._clear_video()
        self._video_is_loaded = False

    def _update_video_frame(self, force=False):
        """Drive the video backend from the current playhead.

        VLC branch: mirror play/pause + seek only on force. VLC
        advances its own internal timeline at real-time, so per-tick
        seeks would cause visible stutter — we only resync on scrub,
        play, or pause.

        cv2 branch: feed the decode worker a target time and blit
        whatever frame it most recently produced."""
        if self._use_vlc and self._vlc_player is not None:
            if not self._video_is_loaded:
                return
            # Respect the Show-Video gate — pause native decode when
            # the pane is hidden so libvlc doesn't burn cycles on
            # frames the user can't see.
            if hasattr(self, '_show_video_var') and not bool(
                    self._show_video_var.get()):
                try:
                    self._vlc_player.pause()
                except Exception:
                    pass
                return
            try:
                offset = float(self._video_offset_var.get())
            except (tk.TclError, ValueError):
                offset = 0.0
            video_t = max(0.0, float(self._playhead_t) + offset)
            if self._video_duration > 0:
                video_t = min(video_t, self._video_duration - 1e-3)
            # Mirror transport. Only seek on force (scrub or
            # pause-play resync) — libvlc advances its own timeline
            # in real-time, so per-tick seeks would cause stutter.
            if self._playing:
                if force:
                    self._vlc_player.set_time(video_t)
                if not self._vlc_player.is_playing():
                    self._vlc_player.play()
            else:
                if self._vlc_player.is_playing():
                    self._vlc_player.pause()
                if force:
                    self._vlc_player.set_time(video_t)
            self._video_last_frame_time = video_t
            # Pull the newest decoded frame out of the vmem double
            # buffer and blit it. Returns None when no new frame has
            # landed since the last call, in which case we just keep
            # the current label contents.
            img = self._vlc_player.snapshot_pil()
            if img is not None:
                try:
                    from PIL import Image, ImageTk
                    # Downscale to the widget if the preview pane is
                    # smaller than our 960x540 decode buffer — keeps
                    # PhotoImage copies cheap on narrow layouts.
                    wid_w = max(1, self._video_widget.winfo_width())
                    wid_h = max(1, self._video_widget.winfo_height())
                    if wid_w > 20 and wid_h > 20:
                        scale = min(wid_w / img.width, wid_h / img.height)
                        if scale < 1.0:
                            new_w = max(1, int(img.width * scale))
                            new_h = max(1, int(img.height * scale))
                            img = img.resize(
                                (new_w, new_h), Image.BILINEAR)
                    self._video_photo = ImageTk.PhotoImage(img)
                    self._video_widget.config(
                        image=self._video_photo, text='')
                except Exception as e:
                    print(f"{self._log_prefix} VLC blit failed: {e}")
            return

        if self._video_cap is None:
            return
        try:
            offset = float(self._video_offset_var.get())
        except (tk.TclError, ValueError):
            offset = 0.0
        t_signal = float(self._playhead_t)
        video_t = t_signal + offset
        if video_t < 0:
            video_t = 0.0
        if self._video_duration > 0:
            video_t = min(video_t, self._video_duration - 1e-3)

        # Ask the worker to aim for this time. ``force=True`` (scrub
        # or pause-play resync) triggers an explicit seek so the worker
        # skips the grab-forward path.
        if force:
            self._decode_seek_target = video_t
        self._decode_target_t = video_t
        self._display_latest_frame()

    def _refresh_video_actual_time(self):
        """Mirror the active playback source's current position into
        ``_video_actual_t`` for the signal clock.

        Source priority:
          1. External media source (e.g. VLC HTTP adapter) if the user
             selected one via the Media Source panel AND it's connected
             with a file loaded. Otherwise fall through.
          2. Embedded libvlc (``self._use_vlc``): read
             ``_vlc_player.get_time()`` — already threadsafe and cheap.
          3. Embedded cv2 decode worker: picks up
             ``self._latest_frame_t`` (worker writes, UI reads; never
             call ``cap.get()`` from here).

        When the external source is selected but not yet connected or
        has no file loaded, we leave ``_video_actual_t`` as-is rather
        than clearing it, so the scheduler doesn't glitch during brief
        disconnects.
        """
        # External source takes precedence when enabled and connected.
        if (self._external_source is not None
                and self._external_source.is_media_loaded()):
            self._video_actual_t = float(
                self._external_source.map_timestamp(time.time()))
            return

        if not self._video_is_loaded:
            self._video_actual_t = None
            return
        if self._use_vlc and self._vlc_player is not None:
            t = self._vlc_player.get_time()
            if t is not None:
                self._video_actual_t = float(t)
            return
        t = self._latest_frame_t
        if t is not None:
            self._video_actual_t = float(t)

    def _refresh_show_video_cache(self):
        try:
            self._show_video_cached = bool(self._show_video_var.get())
        except tk.TclError:
            pass

    def _refresh_lock_signal_cache(self):
        try:
            self._lock_signal_to_video_cached = bool(
                self._lock_signal_to_video_var.get())
        except tk.TclError:
            pass

    def _signal_clock(self):
        """What time should T-code be sampled at this tick?

        Called from the scheduler thread, so it must NOT touch any
        tk vars — reads cached plain-bool mirrors instead.

        Source priority:

        1. External media source (VLC HTTP, etc.) — when an adapter is
           active and has a file loaded, its ``map_timestamp`` is the
           ground truth. This is the restim-style flow: start streaming
           and T-code follows the external player automatically, with
           no local Play-button press needed. The adapter polls at
           10 Hz; between polls ``map_timestamp`` interpolates via
           wall-clock, so the scheduler sees a smooth timeline.

        2. Embedded video (cv2 or libvlc) with "Lock signal to video"
           enabled — use the actual decoded-frame time so stim stays
           aligned with what you see (at the cost of running at
           decode rate).

        3. Local wall-clock playhead — real-time regardless of video
           speed; used when the lock is off or no video is loaded.
        """
        # Priority 1: external source.
        ext = self._external_source
        if ext is not None and ext.is_media_loaded():
            return ext.map_timestamp(time.time())

        # Priority 2: embedded-video lock path.
        if (self._lock_signal_to_video_cached
                and self._show_video_cached
                and self._video_is_loaded
                and self._video_actual_t is not None):
            return self._video_actual_t

        # Priority 3: local wall-clock playhead.
        return self._playhead_t

    # ── Streaming ──────────────────────────────────────────────
    def _test_connection(self):
        host = self._host_var.get().strip() or DEFAULT_HOST
        try:
            port = int(self._port_var.get())
        except (tk.TclError, ValueError):
            port = DEFAULT_PORT
        probe = TCodeUDPSender(host, port)
        up = probe.probe(timeout=0.3)
        probe.close()
        if up:
            self._conn_status_var.set(f"● connected  {host}:{port}")
        else:
            self._conn_status_var.set(f"● no response  {host}:{port}")

    def _toggle_streaming(self):
        if self._scheduler is not None and self._scheduler.is_running:
            self._stop_streaming()
        else:
            self._start_streaming()

    def _start_streaming(self):
        if not self._buffers:
            messagebox.showinfo(
                "No signals",
                "Load a signal folder before streaming.",
                parent=self)
            return
        host = self._host_var.get().strip() or DEFAULT_HOST
        try:
            port = int(self._port_var.get())
        except (tk.TclError, ValueError):
            port = DEFAULT_PORT

        self._sender = TCodeUDPSender(host, port)

        # Reset VOLUME_API (V0) to 1.0 — we don't stream V0 anymore
        # (master goes through V1/VOLUME_EXTERNAL), but a previous
        # Test Pulse in an older build could have left API stuck at
        # 0, which silently zeros the whole volume product
        # (master × api × inactivity × external). One explicit write
        # undoes that, and it's idempotent when API is already 1.0.
        try:
            self._sender.send(b'V09999\n')
        except Exception:
            pass

        # Sync restim's state with the current checkbox intent: any
        # zeroable channel (V1 / E1-E4) that is CURRENTLY UNCHECKED
        # gets an explicit zero-out packet now, so restim doesn't keep
        # driving an electrode we aren't going to feed. Without this,
        # if the user unchecked channels before hitting Start Streaming,
        # restim holds whatever values it had from a prior session.
        for axis in _ZEROABLE_CHANNELS:
            if not self._channel_enabled_cached.get(axis, False):
                self._send_zero_out(axis)

        # Setup-check dialog on the first Start Streaming per install.
        # Covers the three restim-side gates (Funscript Kit T-code
        # names, script-mapping, Mouse pattern). Fires regardless of
        # which channels are enabled because the V1/VOLUME_EXTERNAL
        # setup applies to every stream, not just 4P ones.
        if not self._is_4p_warning_dismissed():
            self._maybe_warn_4p_setup()

        # Pass the full axis map; the scheduler filters per-tick via
        # enabled_fn so the user can flip checkboxes on/off mid-stream
        # and have it take effect within one tick.
        if not any(v for v in self._channel_enabled_cached.values()):
            messagebox.showwarning(
                "No channels",
                "Enable at least one T-code channel before streaming.",
                parent=self)
            self._sender.close()
            self._sender = None
            return

        self._scheduler = TCodeScheduler(
            sender=self._sender,
            # Signal clock prefers the decoded-video-frame time over
            # the wall-clock playhead, so T-code tracks what the user
            # is actually seeing rather than running ahead when decode
            # lags. Falls back to playhead when no video is loaded.
            time_fn=self._signal_clock,
            buffers_fn=lambda: self._buffers,
            axis_map=dict(_ALL_AXIS_MAP),
            offset_s_fn=lambda: self._sync_offset_s_cached,
            enabled_fn=lambda: self._effective_enabled_cached(),
            rate_hz=max(10, int(self._rate_var.get() or 60)),
        )
        self._scheduler.start()
        self._stream_btn.config(text="■ Stop Streaming")
        self._conn_status_var.set(f"● streaming  {host}:{port}")

    def _stop_streaming(self):
        if self._scheduler is not None:
            self._scheduler.stop()
            self._scheduler = None
        if self._sender is not None:
            self._sender.close()
            self._sender = None
        self._stream_btn.config(text="▶ Start Streaming")
        self._conn_status_var.set("● stopped")

    # ── Test pulse ──────────────────────────────────────────────
    def _fire_test_pulse(self):
        """Inject a short synthetic V0 bump so the user has a known
        tactile + visual event to align the offset against.

        The pulse temporarily pauses the scheduler, streams a half-sine
        V0 ramp (0 → 0.75 → 0) over ``_PULSE_MS`` milliseconds, then
        resumes. Other channels hold their last value during the pulse.
        """
        if self._sender is None:
            messagebox.showinfo(
                "Not streaming",
                "Start streaming first, then fire a test pulse.",
                parent=self)
            return
        if getattr(self, '_pulse_active', False):
            return
        self._pulse_active = True
        try:
            self._pulse_btn.config(state='disabled')
            self._pulse_marker.config(text='● PULSE', background='#d32f2f')
        except tk.TclError:
            pass
        # Remember pre-pulse pause state so the test pulse doesn't
        # inadvertently resume a scheduler the user intentionally paused.
        self._pulse_was_paused = (
            self._scheduler is not None and self._scheduler.is_paused)
        if self._scheduler is not None:
            self._scheduler.pause()

        def burst():
            import math as _m
            # 20 steps of ~12 ms each → ~240 ms total.
            steps = 20
            step_ms = self._PULSE_MS / steps
            peak = 0.75
            # Pulse on V1 (VOLUME_EXTERNAL) — the same axis our stream
            # drives. Using V0 would set VOLUME_API and leave it stuck
            # at 0 after the pulse, which then zeros the volume product
            # (api × external × master × inactivity) forever because we
            # don't stream V0 anymore.
            for i in range(steps):
                # Half-sine from 0 → peak → 0
                frac = i / (steps - 1)
                v = int(round(peak * _m.sin(_m.pi * frac) * 9999))
                try:
                    self._sender.send(f'V1{v:04d}\n'.encode('ascii'))
                except Exception:
                    break
                time.sleep(step_ms / 1000.0)
            self._post_to_ui(self._end_test_pulse)

        threading.Thread(target=burst, daemon=True).start()

    def _end_test_pulse(self):
        self._pulse_active = False
        try:
            self._pulse_btn.config(state='normal')
            self._pulse_marker.config(text='', background='#eee')
        except tk.TclError:
            pass
        # Only resume the scheduler if it wasn't already paused before
        # the pulse — otherwise we'd clobber the user's pause.
        if (self._scheduler is not None
                and not getattr(self, '_pulse_was_paused', False)):
            self._scheduler.resume()

    def _is_4p_warning_dismissed(self) -> bool:
        mw = self.main_window
        if mw is None:
            return False
        # Flag name bumped from 'e1e4_warning_dismissed' to
        # 'setup_dismissed_v2' because the dialog content materially
        # changed (added V1/VOLUME_EXTERNAL guidance + expanded to cover
        # non-4P streams). Anyone who dismissed the old narrow version
        # should see the new advice once.
        return bool(
            (mw.current_config or {}).get('preview', {})
            .get('setup_dismissed_v2', False))

    def _remember_4p_dismiss(self):
        mw = self.main_window
        if mw is None:
            return
        cfg = mw.current_config
        cfg.setdefault('preview', {})['setup_dismissed_v2'] = True

    def _maybe_warn_4p_setup(self):
        """Modal advisory dialog with a 'Don't show again' checkbox.

        Dismiss state is stored in ``main_window.current_config`` under
        ``preview.e1e4_warning_dismissed`` so it persists across runs
        as soon as the user saves the config. UDP is fire-and-forget,
        so we can't *verify* restim's kit prefs — this is advisory only.

        Three gates in restim have to line up for our stream to actually
        drive the device; we list all of them because users routinely
        hit (2) or (3) after doing (1).
        """
        dialog = tk.Toplevel(self)
        dialog.title("restim setup check")
        dialog.transient(self)
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame,
                  text="For T-code to cleanly drive restim, three things "
                       "need to be set on restim's side:",
                  wraplength=440, justify=tk.LEFT).pack(
                      anchor='w', pady=(0, 8))

        items = [
            ("1. Funscript Kit T-code names",
             "Preferences → Funscript / T-Code. Set these T-Code axis "
             "entries: INTENSITY_A/B/C/D → E1, E2, E3, E4 (for 4P); "
             "VOLUME_EXTERNAL → V1 (we use external override for "
             "master volume because it's immune to script-mapping "
             "conflicts — V0 can be silently hijacked by an "
             "auto-detected volume.funscript, V1 cannot)."),
            ("2. No conflicting script mapping (external media mode only)",
             "If you're using MPC-HC / HereSphere / VLC / Kodi sync, open "
             "restim's script-mapping tree (on the Media page) and clear "
             "any auto-detected funscripts under POSITION_ALPHA/BETA "
             "and INTENSITY_A/B/C/D. A loaded funscript silently wins "
             "over T-code there. In Internal media mode (restim's own "
             "playback) this is a no-op — the script mapping is "
             "ignored, so skip this gate. Our master-volume path (V1 "
             "→ VOLUME_EXTERNAL) bypasses script mapping regardless."),
            ("3. Pattern selector = 'Mouse'",
             "3-phase/4-phase tab. Any other pattern (Spirograph, Circles, "
             "Random Walk, etc.) writes to positions and intensities at "
             "60 Hz, overwriting your stream. 'Mouse' is read-only."),
        ]
        for title, body in items:
            ttk.Label(frame, text=title,
                      font=('TkDefaultFont', 10, 'bold')).pack(
                          anchor='w', pady=(2, 0))
            ttk.Label(frame, text=body, wraplength=440,
                      justify=tk.LEFT, foreground='#444').pack(
                          anchor='w', pady=(0, 4))

        dismiss_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Don't show this again",
                        variable=dismiss_var).pack(
                            anchor='w', pady=(8, 8))

        def on_ok():
            if dismiss_var.get():
                self._remember_4p_dismiss()
            dialog.destroy()

        ttk.Button(frame, text="OK", command=on_ok).pack(anchor='e')
        dialog.bind('<Return>', lambda _e: on_ok())
        dialog.grab_set()
        dialog.focus()
        # Block start_streaming until the user closes the dialog so the
        # scheduler doesn't start a beat ahead of the user's decision.
        self.wait_window(dialog)

    def _toggle_video_panel(self):
        """Show/hide the video widget; the mixin's _update_video_frame
        already short-circuits decoding while ``_show_video_var`` is
        False, so toggling this avoids the cv2 + PIL work too."""
        if self._show_video_var.get():
            try:
                self._video_widget.grid()
            except tk.TclError:
                pass
            self._update_video_frame(force=True)
        else:
            try:
                self._video_widget.grid_remove()
            except tk.TclError:
                pass

    def _effective_enabled_cached(self):
        """Scheduler-side enabled filter: combines the user's channel
        checkbox state with any overrides set by pause or other temporary
        muting. Called on every scheduler tick, so keep it cheap."""
        if self._video_paused_mute_v1:
            return {**self._channel_enabled_cached, 'V1': False}
        return self._channel_enabled_cached

    def _refresh_channel_cache(self, axis):
        """Trace callback: mirror one channel's checkbox into the plain
        dict the scheduler thread reads. Honors mid-stream toggles so
        unchecking a channel stops its packets within one tick.

        For intensity axes (V0, E1-E4) a True→False transition also
        fires one zero-out packet so restim drops that electrode to
        silence instead of holding the last value. Position and
        pulse-shape axes intentionally hold — zero would be a
        positional corner or a minimum frequency, not "off."
        """
        try:
            new_state = bool(self._channel_enabled[axis].get())
        except (tk.TclError, KeyError):
            return
        old_state = self._channel_enabled_cached.get(axis, False)
        self._channel_enabled_cached[axis] = new_state
        if (old_state and not new_state
                and axis in _ZEROABLE_CHANNELS
                and self._sender is not None):
            self._send_zero_out(axis)

    def _send_zero_out(self, axis: str):
        """Send a single zero-value packet with a gentle 100 ms fade.

        Longer than the scheduler's per-tick interval on purpose — when
        the user unchecks a channel mid-stream (or we sync state at
        stream start), a 20 ms cut can still register as a pop. 100 ms
        is long enough to feel like a fade, short enough to feel
        immediate."""
        if self._sender is None:
            return
        try:
            self._sender.send(f'{axis}0000I100\n'.encode('ascii'))
        except Exception:
            pass

    def _refresh_offset_cache(self):
        """Trace callback: mirror the DoubleVar into a plain float that
        the scheduler thread can safely read without touching tk state,
        and persist it in main_window.current_config so it survives a
        close."""
        try:
            v = float(self._sync_offset_var.get() or 0.0)
        except (tk.TclError, ValueError):
            return
        self._sync_offset_s_cached = v
        if self.main_window is not None:
            self.main_window.current_config.setdefault(
                'preview', {})['sync_offset_s'] = v

    # ── Cross-thread UI dispatch ──────────────────────────────
    def _post_to_ui(self, fn):
        """Thread-safe: schedule ``fn`` to run on the UI thread.

        Worker threads can't call ``self.after()`` — Tkinter only
        accepts scheduling from the main thread. Instead, workers push
        callables onto ``_ui_queue`` and the main thread drains it.
        """
        self._ui_queue.put(fn)

    def _drain_ui_queue(self):
        """Main-thread loop: run any callbacks that workers queued up."""
        try:
            while True:
                fn = self._ui_queue.get_nowait()
                try:
                    fn()
                except Exception as e:
                    print(f"{self._log_prefix} UI callback failed: {e}")
        except queue.Empty:
            pass
        # 30 ms poll — fast enough for reload feedback, cheap on the
        # event loop. Store the after id so _on_close can cancel it.
        self._ui_queue_after_id = self.after(30, self._drain_ui_queue)

    # ── Hot reload ─────────────────────────────────────────────
    def _reload_signals_async(self):
        """Reload the signal folder off the UI thread and atomically swap
        the scheduler's active buffers. Keeps streaming uninterrupted —
        the scheduler continues reading at the same playhead, picking up
        the new buffers on its next tick.
        """
        if self._reload_busy:
            return
        if self._buffer_dir is None:
            self._reload_status_var.set("No signal folder loaded.")
            return
        folder = self._buffer_dir
        base = self._buffer_base
        self._set_reload_busy(True, "Reloading files…")

        def worker():
            try:
                new_buffers = self._load_buffers(folder, base)
                self._post_to_ui(lambda: self._apply_reload(new_buffers, None))
            except Exception as e:
                msg = str(e)
                self._post_to_ui(lambda: self._apply_reload({}, msg))

        threading.Thread(target=worker, daemon=True).start()

    def _regenerate_and_reload_async(self):
        """Run the main window's processor with its current config, then
        reload the fresh files. Scheduler keeps streaming through the
        whole cycle — the swap happens on the UI thread once regeneration
        completes."""
        if self._reload_busy:
            return
        mw = self.main_window
        if mw is None:
            messagebox.showinfo(
                "No source", "Regenerate needs the main window's "
                "processor — open this viewer from the main app.",
                parent=self)
            return
        input_files = getattr(mw, 'input_files', None) or []
        if not input_files:
            messagebox.showinfo(
                "No input",
                "Load an input funscript in the main window first.",
                parent=self)
            return

        # Snapshot config and input paths on the UI thread so the worker
        # sees a stable view even if the user keeps editing.
        try:
            mw.parameter_tabs.update_config(mw.current_config)
        except Exception:
            pass
        config_snapshot = copy.deepcopy(mw.current_config)
        input_file = input_files[0]  # one-file regenerate for now

        # Pin the output directory to wherever we loaded signals from —
        # usually a variant subfolder like <scene>_variants/A/ —
        # regardless of what file_management.mode says. Without this,
        # regenerating while viewing variant A would dump fresh output
        # into the parent folder next to the video.
        if self._buffer_dir is not None:
            fm = config_snapshot.setdefault('file_management', {})
            fm['mode'] = 'central'
            fm['central_folder_path'] = str(self._buffer_dir)

        self._set_reload_busy(True, f"Regenerating {Path(input_file).name}…")

        def worker():
            err = None
            try:
                from processor import RestimProcessor
                proc = RestimProcessor(config_snapshot)
                ok = proc.process(input_file, progress_callback=None)
                if not ok:
                    err = "processor returned failure"
            except Exception as e:
                err = str(e)
            if err:
                self._post_to_ui(lambda: self._apply_reload({}, err))
                return
            # Output directory: central-mode override (we set this above
            # from self._buffer_dir when available), otherwise the input
            # file's own folder.
            fm = config_snapshot.get('file_management', {}) or {}
            if fm.get('mode') == 'central' and fm.get('central_folder_path'):
                out_dir = Path(fm['central_folder_path'])
            else:
                out_dir = Path(input_file).parent
            new_base = Path(input_file).stem
            try:
                new_buffers = self._load_buffers(out_dir, new_base)
                self._post_to_ui(lambda: self._apply_reload(
                    new_buffers, None, out_dir=out_dir, base=new_base))
            except Exception as e:
                msg = str(e)
                self._post_to_ui(lambda: self._apply_reload({}, msg))

        threading.Thread(target=worker, daemon=True).start()

    def _load_buffers(self, folder, base):
        """Worker-thread helper: read funscript files into a new dict.

        Returns a freshly-constructed dict so the caller can swap it in
        atomically by a single reference assignment.
        """
        folder = Path(folder)
        out = {}
        for _axis, signal_name, _desc, _ in _CHANNEL_ROWS:
            path = folder / f"{base}.{signal_name}.funscript"
            if path.exists():
                try:
                    out[signal_name] = Funscript.from_file(path)
                except Exception as e:
                    print(f"{self._log_prefix} failed to load {path}: {e}")
        return out

    def _apply_reload(self, new_buffers, error_msg, out_dir=None, base=None):
        """Main-thread callback: swap buffers and update UI."""
        if error_msg:
            self._set_reload_busy(False, f"Reload failed: {error_msg}")
            return
        if not new_buffers:
            self._set_reload_busy(False, "Reload: no files found.")
            return
        # Atomic reference swap. The scheduler thread reads self._buffers
        # via a lambda and will see the new dict on its next tick.
        self._buffers = new_buffers
        if out_dir is not None:
            self._buffer_dir = out_dir
        if base is not None:
            self._buffer_base = base
        self._total_duration = max(
            (float(fs.x[-1]) if len(fs.x) else 0.0)
            for fs in new_buffers.values())
        self._signals_summary_var.set(
            f"Loaded {len(new_buffers)}: " + ", ".join(sorted(new_buffers)))
        self._update_scrub_range()
        self._update_time_label()

        # If the main window has an input funscript, rescan for variants —
        # a regenerate may have produced a new variant subfolder that we
        # should expose in the radio row.
        mw = self.main_window
        if mw is not None:
            input_files = getattr(mw, 'input_files', None) or []
            if input_files and self._variant_source_base:
                parent = Path(input_files[0]).parent
                self._variant_folders = self._scan_variants(
                    parent, self._variant_source_base)
                self._refresh_variant_radios()

        stamp = time.strftime("%H:%M:%S")
        self._set_reload_busy(
            False, f"Reloaded {len(new_buffers)} signals at {stamp}.")

    def _set_reload_busy(self, busy, status):
        self._reload_busy = busy
        state = 'disabled' if busy else 'normal'
        try:
            self._reload_btn.config(state=state)
            self._regen_btn.config(state=state)
        except tk.TclError:
            pass
        self._reload_status_var.set(status)

    # ── Shutdown ───────────────────────────────────────────────
    def _on_close(self):
        self._stop_playback()
        self._stop_streaming()
        self._stop_decode_worker()
        self._clear_video()
        # Stop any polling external media source so its daemon thread
        # doesn't outlive the window.
        self._teardown_external_source()
        # Release libvlc resources so the Instance+MediaPlayer don't
        # linger in the process after the window closes.
        if self._vlc_player is not None:
            try:
                self._vlc_player.release()
            except Exception:
                pass
            self._vlc_player = None
        if self._ui_queue_after_id is not None:
            try:
                self.after_cancel(self._ui_queue_after_id)
            except Exception:
                pass
            self._ui_queue_after_id = None
        if getattr(self, '_channel_value_after_id', None) is not None:
            try:
                self.after_cancel(self._channel_value_after_id)
            except Exception:
                pass
            self._channel_value_after_id = None
        try:
            self.destroy()
        except Exception:
            pass
