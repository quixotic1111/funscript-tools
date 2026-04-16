"""Fixed-rate scheduler that streams T-code frames to restim.

Runs a background thread that ticks at ``rate_hz`` (default 60). On
each tick it reads the current signal time via a user-supplied
``time_fn``, samples the active signal buffers (also user-supplied),
encodes a T-code frame, and fires it through a
:class:`TCodeUDPSender`.

The scheduler is deliberately agnostic about video playback, signal
generation, and the UI. The caller provides:

    time_fn        -> float   : current playhead time in seconds
    buffers_fn     -> dict    : {signal_name: Funscript} to sample
    offset_s_fn    -> float   : user sync offset applied as
                                sample_time = time_fn() + offset_s_fn()
    enabled_fn     -> dict    : {axis: bool} per-channel mute

Start/stop/pause control the thread. Pause simply stops sending —
restim will hold the last value it received, which is usually what
you want when the user hits pause on the video. Stop sends one final
zeroing frame (V0=0) as a safety shutoff so a disconnected scheduler
doesn't leave the device pinned at the last amplitude.
"""

import logging
import threading
import time
from typing import Callable, Dict, Optional

from funscript import Funscript

from processing.tcode_stream import (
    DEFAULT_AXIS_MAP,
    encode_frame_bytes,
)

logger = logging.getLogger(__name__)


BuffersFn = Callable[[], Dict[str, Funscript]]
TimeFn = Callable[[], float]
OffsetFn = Callable[[], float]
EnabledFn = Callable[[], Optional[Dict[str, bool]]]


class TCodeScheduler:
    """Drives a sender at a fixed rate from a user-supplied clock."""

    def __init__(self,
                 sender,
                 time_fn: TimeFn,
                 buffers_fn: BuffersFn,
                 axis_map: Optional[Dict[str, str]] = None,
                 offset_s_fn: Optional[OffsetFn] = None,
                 enabled_fn: Optional[EnabledFn] = None,
                 rate_hz: float = 60.0):
        self.sender = sender
        self.time_fn = time_fn
        self.buffers_fn = buffers_fn
        self.axis_map = axis_map or DEFAULT_AXIS_MAP
        self.offset_s_fn = offset_s_fn or (lambda: 0.0)
        self.enabled_fn = enabled_fn or (lambda: None)
        self.rate_hz = rate_hz
        self._period = 1.0 / float(rate_hz)

        self._stop = threading.Event()
        self._paused = threading.Event()  # set → paused (skip sends)
        self._thread: Optional[threading.Thread] = None
        self._tick_count = 0  # for diagnostics

    # ── lifecycle ────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._paused.clear()
        self._tick_count = 0
        self._thread = threading.Thread(
            target=self._run, name='TCodeScheduler', daemon=True)
        self._thread.start()

    def stop(self, safety_zero: bool = True) -> None:
        """Halt the thread and (by default) send one V0=0 frame so the
        device goes quiet instead of holding the last amplitude."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if safety_zero:
            try:
                self.sender.send(b'V00000\n')
            except Exception:
                pass

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    @property
    def tick_count(self) -> int:
        return self._tick_count

    # ── run loop ─────────────────────────────────────────────────
    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            # Sleep until the scheduled next_tick, but wake early if
            # stop() is called. Using Event.wait lets us react to shutdown
            # without burning a full period waiting.
            now = time.monotonic()
            sleep_for = next_tick - now
            if sleep_for > 0:
                if self._stop.wait(sleep_for):
                    break

            # Fire this tick unless paused.
            if not self._paused.is_set():
                self._fire_tick()

            # Schedule next tick from the planned time, not from now —
            # this keeps long-run drift at zero. If we fell so far behind
            # that catching up would spam (e.g. GIL hiccup, GC pause),
            # resync to the current clock instead of chewing through
            # missed ticks.
            next_tick += self._period
            behind = time.monotonic() - next_tick
            if behind > self._period * 5:
                next_tick = time.monotonic() + self._period

    def _fire_tick(self) -> None:
        self._tick_count += 1
        try:
            t = float(self.time_fn()) + float(self.offset_s_fn() or 0.0)
            buffers = self.buffers_fn() or {}
            payload = encode_frame_bytes(
                buffers, t, self.axis_map, self.enabled_fn())
            if payload:
                self.sender.send(payload)
        except Exception:
            # Never let a bad callback kill the scheduler — log and keep going.
            logger.exception("TCodeScheduler tick failed")
