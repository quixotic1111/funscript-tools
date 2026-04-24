"""VLC-based video backend using libvlc's ``vmem`` callback API.

Why callback-vmem instead of native NSView embed: libvlc's macosx
vout crashes (SIGSEGV) when asked to attach to a plain Tk Frame's
NSView on this Python/Tk/VLC combination — the vout tries to install
a CALayer into a view that isn't layer-backed by default, and the
render thread faults. The callback path sidesteps the platform vout
entirely: libvlc's decoder writes decoded frames into a plain BGRA
buffer we provide, on libvlc's own threads, and we blit from that
buffer onto a tk.Label with PhotoImage at tick time.

The win vs cv2:
    - decode runs in libvlc's native threads (+ VideoToolbox HW accel
      on macOS) — the GIL isn't touched during decode
    - no Python-side cv2.VideoCapture / resize / cvtColor pass; the
      only CPU we spend is the final PhotoImage blit

Trade-off: decode resolution is fixed at open time (libvlc needs to
know the buffer size up front). We pick a display-ish resolution
(960×540) so the preview pane looks fine without burning memory on a
full-4K BGRA buffer. If the preview is resized much larger, upscale
quality degrades — not worth fixing until someone asks.

Public API:

    player = VLCMemVideoPlayer()
    if player.available:
        player.load("/path/to/video.mp4")
        player.play()
        player.set_time(12.5)      # seconds
        t = player.get_time()      # seconds, may be None
        dur = player.get_duration()
        img = player.snapshot_pil()  # PIL.Image or None
        player.pause()
        player.release()

The old ``attach_to_widget`` method is gone — vmem doesn't embed into
a native view, so it was dead code in this path.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
import threading
from pathlib import Path
from typing import Optional


_BUNDLED_VLC_CANDIDATES = [
    str(Path(__file__).resolve().parent.parent / "VLC" / "VLC.app"),
    "/Applications/VLC.app",
]


def _find_vlc_app() -> Optional[Path]:
    for candidate in _BUNDLED_VLC_CANDIDATES:
        p = Path(candidate)
        if p.is_dir() and (p / "Contents" / "MacOS").is_dir():
            return p
    return None


def _configure_vlc_env_macos() -> Optional[Path]:
    """Pre-load libvlc.dylib and set VLC_PLUGIN_PATH from a bundled
    VLC.app. Must run before ``import vlc``."""
    app = _find_vlc_app()
    if app is None:
        return None
    libdir = app / "Contents" / "MacOS" / "lib"
    plugins = app / "Contents" / "MacOS" / "plugins"
    libvlc = libdir / "libvlc.dylib"
    libvlccore = libdir / "libvlccore.dylib"
    if not libvlc.exists():
        return None
    os.environ.setdefault("VLC_PLUGIN_PATH", str(plugins))
    try:
        ctypes.CDLL(str(libvlccore))
    except OSError:
        pass
    try:
        ctypes.CDLL(str(libvlc))
    except OSError:
        return None
    return app


# ctypes signatures for the vmem callbacks. libvlc calls these from
# its decoder thread on every frame, so they MUST be fast and
# GIL-safe. Python callback overhead is minimal here — we're just
# flipping a pointer and a flag.

_LOCK_T = ctypes.CFUNCTYPE(
    ctypes.c_void_p,                        # return: picture id
    ctypes.c_void_p,                        # opaque
    ctypes.POINTER(ctypes.c_void_p))        # planes[0] out

_UNLOCK_T = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,                        # opaque
    ctypes.c_void_p,                        # picture id
    ctypes.POINTER(ctypes.c_void_p))        # planes

_DISPLAY_T = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,                        # opaque
    ctypes.c_void_p)                        # picture id


class VLCMemVideoPlayer:
    """libvlc wrapper that renders to a memory buffer via vmem.

    The decoder fills an internal BGRA buffer at its own pace; the
    main thread pulls PIL Images out via ``snapshot_pil()``. That
    call returns None when no new frame has landed since the last
    fetch, so the UI can skip a redraw.
    """

    # Fixed decode resolution. Larger = sharper preview at the cost
    # of a bigger ctypes buffer and more per-frame bytes to ship. The
    # preview pane is usually 600-900 px wide; 960x540 is comfortably
    # above that so we're not visibly lossy.
    _W = 960
    _H = 540
    _BPP = 4   # BGRA
    _PITCH = _W * _BPP
    _BUF_LEN = _PITCH * _H

    def __init__(self):
        self.available: bool = False
        self._instance = None
        self._player = None
        self._vlc_mod = None
        self._vlc_app_path: Optional[Path] = None

        # Double-buffered frames: libvlc writes into ``buf_decode``,
        # main thread reads ``buf_display``. On unlock, we swap the
        # two under ``_frame_lock`` so the next decode doesn't stomp
        # the frame the UI is about to PIL-wrap. A monotonic counter
        # tells the UI "this is a new frame, redraw" vs "same frame
        # as last time, skip".
        self._buf_a = (ctypes.c_ubyte * self._BUF_LEN)()
        self._buf_b = (ctypes.c_ubyte * self._BUF_LEN)()
        self._buf_decode_ptr = ctypes.cast(self._buf_a, ctypes.c_void_p).value
        self._buf_display_ptr = ctypes.cast(self._buf_b, ctypes.c_void_p).value
        self._buf_decode_obj = self._buf_a   # keeps ref alive
        self._buf_display_obj = self._buf_b
        self._frame_counter = 0
        self._last_consumed_counter = -1
        self._frame_lock = threading.Lock()

        # Strong refs to the CFUNCTYPE wrappers — if we let Python GC
        # them libvlc will call freed memory and crash.
        self._lock_cb = _LOCK_T(self._on_lock)
        self._unlock_cb = _UNLOCK_T(self._on_unlock)
        self._display_cb = _DISPLAY_T(self._on_display)

        self._load_lib()

    # ── setup ────────────────────────────────────────────────────
    def _load_lib(self):
        if platform.system() == "Darwin":
            self._vlc_app_path = _configure_vlc_env_macos()
        try:
            import vlc  # type: ignore
        except Exception as e:
            print(f"[vlc-backend] python-vlc not importable: {e}",
                  file=sys.stderr)
            return
        try:
            # --vout=none — we handle display via vmem callbacks, so
            # we don't want libvlc to also try to open a window.
            # --no-audio — audio belongs to the device/restim pipeline,
            # not the preview.
            # --quiet — silence libvlc's info chatter on stderr.
            args = [
                "--quiet",
                "--no-audio",
                "--no-video-title-show",
            ]
            self._instance = vlc.Instance(args)
            if self._instance is None:
                raise RuntimeError("vlc.Instance() returned None")
            self._player = self._instance.media_player_new()
            if self._player is None:
                raise RuntimeError("media_player_new() returned None")
            self._vlc_mod = vlc

            # Register the callbacks + buffer format BEFORE the first
            # media is loaded. libvlc reads these at play() time.
            self._player.video_set_callbacks(
                self._lock_cb, self._unlock_cb, self._display_cb, None)
            self._player.video_set_format(
                b"RV32", self._W, self._H, self._PITCH)

            self.available = True
        except Exception as e:
            print(f"[vlc-backend] init failed: {e}", file=sys.stderr)
            self._instance = None
            self._player = None
            self.available = False

    # ── vmem callbacks (decoder thread) ──────────────────────────
    def _on_lock(self, opaque, planes):
        """Hand libvlc a pointer to write the next decoded frame."""
        with self._frame_lock:
            planes[0] = self._buf_decode_ptr
        # picture id — arbitrary non-zero value; we use the buffer
        # pointer itself so unlock can verify it hasn't drifted.
        return self._buf_decode_ptr

    def _on_unlock(self, opaque, picture, planes):
        """Frame decode complete. Swap decode↔display under the lock
        so the main thread reads a stable buffer on its next tick."""
        with self._frame_lock:
            # Swap pointers.
            self._buf_decode_ptr, self._buf_display_ptr = (
                self._buf_display_ptr, self._buf_decode_ptr)
            self._buf_decode_obj, self._buf_display_obj = (
                self._buf_display_obj, self._buf_decode_obj)
            self._frame_counter += 1

    def _on_display(self, opaque, picture):
        # vmem's display step is optional; everything real happens in
        # unlock. Left as a no-op so libvlc has the callback it wants.
        pass

    # ── transport ────────────────────────────────────────────────
    def load(self, path: str):
        """Swap in a new media file. Does NOT auto-play — caller
        controls play/pause explicitly."""
        if not self.available or self._instance is None or self._player is None:
            return
        try:
            media = self._instance.media_new(path)
            self._player.set_media(media)
            try:
                media.parse_with_options(1, 500)  # PARSE_LOCAL
            except Exception:
                pass
        except Exception as e:
            print(f"[vlc-backend] load failed: {e}", file=sys.stderr)

    def play(self):
        if not self.available or self._player is None:
            return
        try:
            self._player.play()
        except Exception as e:
            print(f"[vlc-backend] play failed: {e}", file=sys.stderr)

    def pause(self):
        if not self.available or self._player is None:
            return
        try:
            self._player.set_pause(1)
        except Exception as e:
            print(f"[vlc-backend] pause failed: {e}", file=sys.stderr)

    def is_playing(self) -> bool:
        if not self.available or self._player is None:
            return False
        try:
            return bool(self._player.is_playing())
        except Exception:
            return False

    def set_time(self, seconds: float):
        if not self.available or self._player is None:
            return
        try:
            self._player.set_time(int(max(0.0, seconds) * 1000.0))
        except Exception as e:
            print(f"[vlc-backend] set_time failed: {e}", file=sys.stderr)

    def get_time(self) -> Optional[float]:
        if not self.available or self._player is None:
            return None
        try:
            ms = self._player.get_time()
            if ms is None or ms < 0:
                return None
            return ms / 1000.0
        except Exception:
            return None

    def get_duration(self) -> Optional[float]:
        if not self.available or self._player is None:
            return None
        try:
            ms = self._player.get_length()
            if ms is None or ms <= 0:
                return None
            return ms / 1000.0
        except Exception:
            return None

    def get_fps(self) -> float:
        if not self.available or self._player is None:
            return 30.0
        try:
            fps = self._player.get_fps()
            if fps and fps > 0:
                return float(fps)
        except Exception:
            pass
        return 30.0

    # ── frame access (main thread) ───────────────────────────────
    def snapshot_pil(self):
        """Return a PIL Image of the most recently decoded frame, or
        None if no new frame has landed since the last call.

        Called from the UI thread on every tick. Does ONE buffer copy
        (~2 MB) + one PIL.Image.frombytes — no PIL resize, no
        cv2.cvtColor. The BGRA → RGBA reorder happens via PIL's
        raw-mode specifier (``BGRA``), which is zero-copy.
        """
        if not self.available:
            return None
        try:
            from PIL import Image
        except ImportError:
            return None
        with self._frame_lock:
            if self._frame_counter == self._last_consumed_counter:
                return None
            # Copy the display buffer out from under the lock so the
            # decoder doesn't race us. bytes() of a ctypes array is
            # the shortest path to an immutable snapshot.
            buf = bytes(self._buf_display_obj)
            self._last_consumed_counter = self._frame_counter
        # PIL accepts "RV32" aka BGRA via the raw-mode "BGRA".
        img = Image.frombytes("RGBA", (self._W, self._H), buf, "raw", "BGRA")
        return img

    # ── cleanup ──────────────────────────────────────────────────
    def release(self):
        try:
            if self._player is not None:
                try:
                    self._player.stop()
                except Exception:
                    pass
                try:
                    self._player.release()
                except Exception:
                    pass
                self._player = None
            if self._instance is not None:
                try:
                    self._instance.release()
                except Exception:
                    pass
                self._instance = None
        finally:
            self.available = False


# Back-compat alias. Callers that imported the old class name still
# work; they get the vmem-based implementation instead.
VLCVideoPlayer = VLCMemVideoPlayer
