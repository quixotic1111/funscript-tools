"""
One-Euro filter — adaptive low-pass that kills jitter without the
lag-ringing artifacts a plain EMA produces on heavy settings.

Reference: Casiez, Roussel, Vogel, "1€ Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems" (CHI 2012).

Idea: use a fast-tunable low-pass (cutoff_hz) but make the cutoff
itself adaptive to estimated signal velocity. When the signal is
nearly still, cutoff is low → aggressive smoothing → jitter killed.
When the signal moves fast, cutoff rises proportional to |velocity|
→ filter becomes near-transparent → genuine motion pulses pass
through without visible lag. Two knobs:

  min_cutoff_hz : baseline cutoff at zero velocity. Lower = more
    smoothing when idle. 1.0 Hz is a reasonable default for the
    50 Hz funscript signals this pipeline processes; drop to
    0.3 Hz for very twitchy trackers, raise to 3.0 Hz if the
    filter feels laggy on slow intentional motion.
  beta : velocity-to-cutoff gain. Higher = more responsive to
    fast motion (less smoothing during action). 0.05 is a
    conservative default that doesn't over-smooth; raise toward
    0.1-0.2 if the output feels dull on fast strokes.

A third parameter, d_cutoff_hz, filters the velocity estimate
itself — kept at 1.0 Hz because the velocity low-pass rarely
needs tuning and exposing it just adds noise to the UI surface.
Override via the function kwarg when you do need it.

This implementation handles non-uniform time grids (dt varies per
step) even though the spatial pipeline currently feeds a uniform
50 Hz grid — keeps the code drop-in for other callers.
"""

import math
import time
from typing import Sequence

import numpy as np


# How often the per-sample loop yields the GIL via time.sleep(0). Lets
# other Python threads (notably the T-code scheduler's tick loop) run
# even when one_euro_filter is chewing through a long signal. 2048
# picks up roughly every 40 ms of filtering on a 50 Hz signal —
# enough to keep the scheduler's 20 ms ticks from accumulating drift
# without measurably slowing the loop.
_GIL_YIELD_INTERVAL = 2048


def _low_pass_alpha(cutoff_hz: float, dt: float) -> float:
    """Classical first-order low-pass alpha coefficient for an EMA
    stepped by `dt` seconds with the given -3dB cutoff. Kept for
    external callers; the inner loop inlines this formula to avoid
    per-sample function-call overhead."""
    if cutoff_hz <= 0.0 or dt <= 0.0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


def one_euro_filter(
    t: Sequence[float],
    x: Sequence[float],
    *,
    min_cutoff_hz: float = 1.0,
    beta: float = 0.05,
    d_cutoff_hz: float = 1.0,
) -> np.ndarray:
    """Apply the One-Euro adaptive low-pass filter to a 1D signal.

    Args:
        t: Timestamps in seconds, same length as x. Monotonically
            increasing. Uniform grids are fine; non-uniform grids
            work too (dt is recomputed per sample).
        x: Samples to filter.
        min_cutoff_hz: Baseline cutoff at zero velocity.
        beta: Velocity-to-cutoff gain (dimensionless). Higher =
            filter becomes more transparent on fast motion.
        d_cutoff_hz: Low-pass cutoff for the velocity estimate.

    Returns:
        Filtered samples as a float64 numpy array of the same
        length as x.

    Notes:
        First sample passes through unfiltered (no prior state
        to smooth against). Second sample onward is filtered.

        Performance: dt and the velocity-filter alpha (dx_alpha)
        are pre-computed once as numpy arrays since both depend
        only on `t` and the fixed `d_cutoff_hz`. The adaptive
        alpha on the signal itself stays recursive (cutoff reacts
        to the filtered velocity) but the inner loop inlines the
        alpha formula and reads from pre-computed arrays — no
        per-sample function calls, no float() casts.

        GIL yielding: every 2048 iterations the loop calls
        time.sleep(0), which cooperatively yields so background
        threads (T-code scheduler, etc.) can run between chunks.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    n = x_arr.shape[0]
    if n == 0:
        return x_arr.copy()
    if t_arr.shape[0] != n:
        raise ValueError(
            f"t and x must be the same length (got {t_arr.shape[0]} "
            f"and {n})"
        )

    out = np.empty(n, dtype=np.float64)
    out[0] = float(x_arr[0])
    if n == 1:
        return out

    # Pre-compute per-sample dt (vectorized np.diff).
    dt_arr = np.diff(t_arr)

    # Pre-compute dx_alpha (velocity low-pass coefficient). Depends
    # only on dt and the constant d_cutoff_hz — fully vectorizable.
    # Guards: dt<=0 → alpha=1 (coast; the main loop also handles
    # this case below so the array value is never actually read).
    if d_cutoff_hz > 0.0:
        tau_d = 1.0 / (2.0 * math.pi * float(d_cutoff_hz))
        with np.errstate(divide='ignore', invalid='ignore'):
            dx_alpha_arr = 1.0 / (1.0 + tau_d / dt_arr)
        dx_alpha_arr = np.where(dt_arr > 0.0, dx_alpha_arr, 1.0)
    else:
        dx_alpha_arr = np.ones_like(dt_arr)

    # Recursive core. Inlined _low_pass_alpha for the adaptive branch
    # avoids 259k+ function calls per filter pass on a 3 min / 60 Hz
    # clip. Reads from pre-computed dt_arr and dx_alpha_arr by
    # integer index — no Python-level arithmetic on the timestamps.
    prev_x_filt = out[0]
    prev_dx_filt = 0.0
    tau_factor = 1.0 / (2.0 * math.pi)

    for i in range(1, n):
        dt_i = dt_arr[i - 1]
        if dt_i <= 0.0:
            out[i] = prev_x_filt
            continue

        x_i = x_arr[i]
        dx_raw = (x_i - prev_x_filt) / dt_i
        dx_alpha = dx_alpha_arr[i - 1]
        dx_filt = dx_alpha * dx_raw + (1.0 - dx_alpha) * prev_dx_filt

        # Adaptive cutoff rises with |velocity|. Inline alpha formula:
        #   cutoff = min_cutoff_hz + beta * |dx_filt|
        #   tau = 1 / (2π · cutoff)
        #   alpha = 1 / (1 + tau / dt)
        cutoff = min_cutoff_hz + beta * (
            dx_filt if dx_filt >= 0.0 else -dx_filt)
        if cutoff <= 0.0:
            alpha = 1.0
        else:
            tau = tau_factor / cutoff
            alpha = 1.0 / (1.0 + tau / dt_i)

        x_filt = alpha * x_i + (1.0 - alpha) * prev_x_filt
        out[i] = x_filt
        prev_x_filt = x_filt
        prev_dx_filt = dx_filt

        # Cooperatively yield the GIL every N iterations so the
        # T-code scheduler (or any other real-time-ish background
        # thread) can run between chunks.
        if i & (_GIL_YIELD_INTERVAL - 1) == 0:
            time.sleep(0)

    return out
