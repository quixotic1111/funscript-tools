"""
Input sharpener — restore high-frequency content + push the signal
toward its extremes. The complement of the input smoothing stage:
where smoothing flattens jitter, sharpening reintroduces transient
energy and bimodal distribution.

Motivation: the Mask-Moments tracker produces smooth mask-centroid
motion. Compared to a sharp-tracker source like Quad, its output
has ~3× less high-frequency energy and spends more time dwelling
in the middle of the value range instead of hitting the extremes.
Downstream the spatial projection reads this as "mild gradual
modulation" vs "punchy discrete hits." Reshaping the source signal
to have Quad-like transient energy + bimodal distribution closes
that gap without changing the tracker.

Two stages, applied in order:

  1. Pre-emphasis (unsharp mask) — add a scaled copy of the high-
     pass component back to the signal. `y = x + k * (x - lp(x))`
     with an EMA low-pass at 3 Hz cutoff. Boosts frequencies above
     the cutoff by (1 + k); doesn't touch DC.
  2. Saturation (soft waveshaper) — tanh-based push toward the
     [0, 1] extremes. `y = 0.5 + 0.5 * tanh(g * (x - 0.5)) / tanh(g * 0.5)`.
     Preserves endpoints exactly, makes the value distribution
     bimodal instead of uniform-ish. At g = 0, identity;
     at g = 4, strong push.

Both are bounded and clip-safe; output is always clamped to
[0, 1] at the end.
"""

import math
from typing import Sequence

import numpy as np


def sharpen_signal(
    x: Sequence[float],
    *,
    pre_emphasis: float = 0.0,
    saturation: float = 0.0,
    pre_emphasis_cutoff_hz: float = 3.0,
    sample_rate_hz: float = 50.0,
) -> np.ndarray:
    """Apply pre-emphasis + saturation to a 1D signal.

    Args:
        x: Input samples in [0, 1]. Non-1D inputs are flattened.
        pre_emphasis: High-frequency boost amount. 0 = identity,
            1 = 2× boost above cutoff, 2 = 3× boost, etc. Above
            ~3 the output typically clips heavily and the user is
            effectively just limiting — rarely useful.
        saturation: Soft-clip strength toward [0, 1] extremes.
            0 = identity, 1 = mild push, 4 = strong (near-square-
            wave at large inputs). Endpoints always preserved.
        pre_emphasis_cutoff_hz: Low-pass cutoff used by the unsharp
            mask. Frequencies above this get boosted. 3 Hz matches
            the spectral break between Mask-Moments (most energy
            <3 Hz) and Quad (significant energy >3 Hz).
        sample_rate_hz: Samples per second of x. Used to convert
            the cutoff frequency into an EMA alpha.

    Returns:
        Float64 numpy array of the same length as x, clamped to
        [0, 1].
    """
    arr = np.asarray(x, dtype=np.float64).flatten()
    n = arr.shape[0]
    if n < 2:
        return arr.copy()

    out = arr.copy()

    # Pre-emphasis via unsharp mask: out = x + k * (x - lp(x)).
    # The EMA low-pass is a first-order filter with -3 dB at the
    # configured cutoff frequency; the high-pass component (x - lp)
    # is added back to the signal scaled by `pre_emphasis`.
    if pre_emphasis > 0.0:
        fc = max(0.1, float(pre_emphasis_cutoff_hz))
        fs = max(1.0, float(sample_rate_hz))
        alpha = 1.0 - math.exp(-2.0 * math.pi * fc / fs)
        lp = np.empty(n, dtype=np.float64)
        lp[0] = arr[0]
        for i in range(1, n):
            lp[i] = alpha * arr[i] + (1.0 - alpha) * lp[i - 1]
        hp = arr - lp
        out = arr + float(pre_emphasis) * hp

    # Saturation: soft clip via tanh. The denominator tanh(g*0.5)
    # normalizes the curve so that x = 0 → y = 0 and x = 1 → y = 1
    # exactly — the waveshaper only redistributes values in
    # between, pushing them toward the endpoints without altering
    # the overall range.
    if saturation > 0.01:
        g = float(saturation)
        denom = math.tanh(g * 0.5)
        if denom > 1e-6:
            centered = out - 0.5
            shaped = 0.5 * np.tanh(g * centered) / denom
            out = 0.5 + shaped

    return np.clip(out, 0.0, 1.0)
