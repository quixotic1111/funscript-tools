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
from processing.tcode_scheduler import TCodeScheduler
from processing.tcode_sender import DEFAULT_HOST, DEFAULT_PORT, TCodeUDPSender
from processing.tcode_stream import DEFAULT_AXIS_MAP
from ui.video_player_helper import VideoPlaybackMixin


# Every channel we support, in the order shown in the UI.
# (tcode_axis, signal_name, short_description, default_enabled)
_CHANNEL_ROWS = [
    ('L0', 'alpha',           'position α',           True),
    ('L1', 'beta',            'position β',           True),
    ('V0', 'volume',          'master amplitude',     True),
    ('C0', 'frequency',       'carrier (500-1000 Hz)', True),
    ('P0', 'pulse_frequency', 'pulse rate (0-100 Hz)', True),
    ('P1', 'pulse_width',     'pulse width',          True),
    ('P3', 'pulse_rise_time', 'pulse rise',           True),
    ('E1', 'e1',              '4P intensity 1',       False),
    ('E2', 'e2',              '4P intensity 2',       False),
    ('E3', 'e3',              '4P intensity 3',       False),
    ('E4', 'e4',              '4P intensity 4',       False),
]

_ALL_AXIS_MAP = {axis: sig for axis, sig, _, _ in _CHANNEL_ROWS}


class TCodePreviewViewer(tk.Toplevel, VideoPlaybackMixin):
    """Live T-code preview window."""

    _log_prefix = '[tcode-preview]'
    _TICK_MS = 33    # ~30 Hz UI loop — playhead + video frame
    _PULSE_MS = 240  # total duration of a test pulse burst

    def __init__(self, parent, main_window=None):
        super().__init__(parent)
        self.title("T-Code Live Preview — stream to restim")
        self.geometry("1100x720")
        self.minsize(800, 520)
        self.main_window = main_window

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
        self._signals_summary_var = tk.StringVar(value="(no signals loaded)")
        self._folder_var = tk.StringVar(value="(none)")
        self._conn_status_var = tk.StringVar(value="● unknown")
        self._time_label_var = tk.StringVar(value="0.00 / 0.00 s")
        self._reload_status_var = tk.StringVar(value="")
        self._reload_busy = False

        # Thread-safe hand-off for worker → UI. Workers push callables
        # here; the UI drains the queue on a periodic after() tick.
        # (Tk's .after() may only be called from the main thread.)
        self._ui_queue: "queue.Queue" = queue.Queue()
        self._ui_queue_after_id = None

        self._build_ui()
        self._drain_ui_queue()

        # Auto-load signals from the main window's last-processed folder.
        mw = self.main_window
        if mw is not None and getattr(mw, 'last_processed_directory', None):
            try:
                self._load_signals_from(
                    mw.last_processed_directory,
                    mw.last_processed_filename)
            except Exception as e:
                print(f"{self._log_prefix} auto-load failed: {e}")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ─────────────────────────────────────────────────────
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        outer = ttk.Frame(self)
        outer.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        # Left: video panel
        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        self._video_widget = tk.Label(
            left, bg='#181818', fg='#888',
            text="(drop a video file here, or click Browse)",
            anchor='center')
        self._video_widget.grid(row=0, column=0, sticky='nsew')

        video_ctrls = ttk.Frame(left)
        video_ctrls.grid(row=1, column=0, sticky='ew', pady=(4, 0))
        self._play_btn = ttk.Button(
            video_ctrls, text="▶ Play", width=10, command=self._toggle_play)
        self._play_btn.pack(side=tk.LEFT)
        ttk.Button(video_ctrls, text="Browse video…",
                   command=self._browse_video).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(video_ctrls, text="Clear",
                   command=self._clear_video).pack(side=tk.LEFT, padx=(6, 0))
        self._video_path_lbl = ttk.Label(video_ctrls, text="(none)",
                                         foreground='#666')
        self._video_path_lbl.pack(side=tk.LEFT, padx=(10, 0))

        scrub_frame = ttk.Frame(left)
        scrub_frame.grid(row=2, column=0, sticky='ew', pady=(4, 0))
        scrub_frame.columnconfigure(0, weight=1)
        self._scrub_scale = ttk.Scale(
            scrub_frame, from_=0.0, to=1.0, orient='horizontal',
            variable=self._scrub_var, command=self._on_scrub)
        self._scrub_scale.grid(row=0, column=0, sticky='ew')
        ttk.Label(scrub_frame, textvariable=self._time_label_var,
                  width=18).grid(row=0, column=1, padx=(6, 0))

        # Right: controls
        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky='nsew')
        right.columnconfigure(0, weight=1)

        self._build_signal_panel(right, row=0)
        self._build_channel_panel(right, row=1)
        self._build_reload_panel(right, row=2)
        self._build_sync_panel(right, row=3)
        self._build_restim_panel(right, row=4)

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
        ttk.Label(box, textvariable=self._signals_summary_var,
                  foreground='#333').grid(
                      row=1, column=0, sticky='w', padx=6, pady=(0, 4))

    def _build_channel_panel(self, parent, row):
        box = ttk.LabelFrame(parent, text="T-code channels")
        box.grid(row=row, column=0, sticky='ew', pady=(0, 6))
        for i, (axis, sig, desc, _default) in enumerate(_CHANNEL_ROWS):
            var = self._channel_enabled[axis]
            ttk.Checkbutton(box, variable=var,
                            text=f"{axis}  {sig}").grid(
                                row=i, column=0, sticky='w', padx=6)
            ttk.Label(box, text=desc, foreground='#777').grid(
                row=i, column=1, sticky='w', padx=(4, 6))

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

    def _build_sync_panel(self, parent, row):
        box = ttk.LabelFrame(parent, text="Sync")
        box.grid(row=row, column=0, sticky='ew', pady=(0, 6))
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Offset (s):").grid(
            row=0, column=0, padx=(6, 4), pady=4, sticky='w')
        ttk.Scale(box, from_=-1.0, to=1.0, orient='horizontal',
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
        self._signals_summary_var.set(
            f"Loaded {len(buffers)}: " + ", ".join(sorted(buffers)))

        # Duration = max of loaded buffers.
        if buffers:
            self._total_duration = max(
                (float(fs.x[-1]) if len(fs.x) else 0.0)
                for fs in buffers.values())
        else:
            self._total_duration = 0.0

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

    def _update_time_label(self):
        self._time_label_var.set(
            f"{self._playhead_t:6.2f} / {self._total_duration:6.2f} s")

    def _on_scrub(self, _val=None):
        if self._playing:
            return  # scrubbing during play is ignored; paused scrub seeks
        try:
            t = float(self._scrub_var.get())
        except (tk.TclError, ValueError):
            return
        self._playhead_t = max(0.0, min(t, self._total_duration))
        self._update_time_label()
        self._update_video_frame(force=True)

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

    def _tick(self):
        if not self._playing:
            return
        now = time.monotonic()
        dt = now - (self._last_tick_wall or now)
        self._last_tick_wall = now
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
        self._after_id = self.after(self._TICK_MS, self._tick)

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

        # Warn about E1-E4 needing kit-preferences setup in restim if the
        # user has enabled any of them (skipped if previously dismissed).
        if any(self._channel_enabled[a].get()
               for a in ('E1', 'E2', 'E3', 'E4')):
            if not self._is_4p_warning_dismissed():
                self._maybe_warn_4p_setup()

        axis_map = {a: s for a, s in _ALL_AXIS_MAP.items()
                    if self._channel_enabled[a].get()}
        if not axis_map:
            messagebox.showwarning(
                "No channels",
                "Enable at least one T-code channel before streaming.",
                parent=self)
            self._sender.close()
            self._sender = None
            return

        self._scheduler = TCodeScheduler(
            sender=self._sender,
            time_fn=lambda: self._playhead_t,
            buffers_fn=lambda: self._buffers,
            axis_map=axis_map,
            offset_s_fn=lambda: self._sync_offset_s_cached,
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
            for i in range(steps):
                # Half-sine from 0 → peak → 0
                frac = i / (steps - 1)
                v = int(round(peak * _m.sin(_m.pi * frac) * 9999))
                try:
                    self._sender.send(f'V0{v:04d}\n'.encode('ascii'))
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
        return bool(
            (mw.current_config or {}).get('preview', {})
            .get('e1e4_warning_dismissed', False))

    def _remember_4p_dismiss(self):
        mw = self.main_window
        if mw is None:
            return
        cfg = mw.current_config
        cfg.setdefault('preview', {})['e1e4_warning_dismissed'] = True

    def _maybe_warn_4p_setup(self):
        """Modal advisory dialog with a 'Don't show again' checkbox.

        Dismiss state is stored in ``main_window.current_config`` under
        ``preview.e1e4_warning_dismissed`` so it persists across runs
        as soon as the user saves the config. UDP is fire-and-forget,
        so we can't *verify* restim's kit prefs — this is advisory only.
        """
        dialog = tk.Toplevel(self)
        dialog.title("4P channels")
        dialog.transient(self)
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            frame,
            text=("E1-E4 are enabled. restim will ignore them unless "
                  "you've assigned T-code names E1..E4 to the "
                  "INTENSITY_A..D axes in Preferences → Funscript axes."),
            wraplength=380, justify=tk.LEFT).pack(pady=(0, 10))
        dismiss_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Don't show this again",
                        variable=dismiss_var).pack(anchor='w', pady=(0, 10))

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
            # Determine the new output directory: central-mode override
            # or the input file's own folder.
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
        self._clear_video()
        if self._ui_queue_after_id is not None:
            try:
                self.after_cancel(self._ui_queue_after_id)
            except Exception:
                pass
            self._ui_queue_after_id = None
        try:
            self.destroy()
        except Exception:
            pass
