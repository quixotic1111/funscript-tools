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
from typing import Sequence

import numpy as np


def _low_pass_alpha(cutoff_hz: float, dt: float) -> float:
    """Classical first-order low-pass alpha coefficient for an EMA
    stepped by `dt` seconds with the given -3dB cutoff."""
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

    prev_x_filt = out[0]
    prev_dx_filt = 0.0

    for i in range(1, n):
        dt = float(t_arr[i] - t_arr[i - 1])
        if dt <= 0.0:
            # Degenerate timestep — coast on previous filtered
            # value rather than dividing by zero or going
            # backwards in time.
            out[i] = prev_x_filt
            continue

        # Velocity estimate, filtered by a plain low-pass at
        # d_cutoff_hz so spikes in the raw derivative don't
        # yank the adaptive cutoff around.
        dx_raw = (float(x_arr[i]) - prev_x_filt) / dt
        dx_alpha = _low_pass_alpha(d_cutoff_hz, dt)
        dx_filt = dx_alpha * dx_raw + (1.0 - dx_alpha) * prev_dx_filt

        # Adaptive cutoff rises with |velocity|. At rest the
        # cutoff is min_cutoff (heavy smoothing); during motion
        # the filter becomes near-transparent.
        cutoff = min_cutoff_hz + beta * abs(dx_filt)
        alpha = _low_pass_alpha(cutoff, dt)
        x_filt = alpha * float(x_arr[i]) + (1.0 - alpha) * prev_x_filt

        out[i] = x_filt
        prev_x_filt = x_filt
        prev_dx_filt = dx_filt

    return out
