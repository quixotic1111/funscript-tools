"""
Shared output-shaping toolkit for electrode-intensity pipelines.

Both spatial projection kernels (compute_spatial_intensities for the
trochoid 2D-curve projector, compute_linear_intensities_3d for the XYZ
Linear 3D projector) produce a dict of per-electrode intensity arrays.
The shaping tools here operate generically on that dict shape, regardless
of which projection produced it:

    apply_cross_electrode_normalize(out, mode)
        Rebalance so either (per_frame) the cross-electrode sum is 1 at
        every sample, or (energy_preserve) the time-averaged total
        energy stays flat.

    apply_one_euro_per_electrode(out, t_sec, min_cutoff_hz, beta)
        Velocity-adaptive low-pass filter applied independently to each
        electrode. Kills high-frequency content from sharp sharpness +
        busy curve / tracker jitter without introducing the lag a fixed
        low-pass would.

    apply_per_electrode_gain(out, gains)
        Multiplicative per-channel gain / trim. Last-stage rebalancing
        for physical-device variation across electrodes.

All three helpers take and return Dict[str, np.ndarray] and never
mutate the input. Intended to be called as post-stages inside the
projection kernels (or downstream of them) — they don't know or care
about projection geometry.
"""

from typing import Any, Dict, Sequence, Union

import numpy as np

from .one_euro_filter import one_euro_filter


VALID_NORMALIZE_MODES = ('clamped', 'per_frame', 'energy_preserve')


def apply_cross_electrode_normalize(
    out: Dict[str, np.ndarray],
    mode: str,
) -> Dict[str, np.ndarray]:
    """
    Cross-electrode balancing.

    Args:
        out: Dict of per-electrode arrays, typically named 'e1'...'eN'.
            Values must be equal-length 1D numpy arrays.
        mode: One of VALID_NORMALIZE_MODES.
            - 'clamped': no-op. Returned dict has the same arrays (a
              shallow copy — callers shouldn't rely on identity).
            - 'per_frame': rescale so Σ e_i(t) = 1 at every sample.
              Frames where the raw sum is ~0 stay at 0 on all channels.
            - 'energy_preserve': rescale all channels by a time-varying
              factor so Σ e_i(t) equals the time-average of Σ e_i(t)
              across the signal. Flattens global energy swings without
              forcing sum-to-1. Callers should final-clip to [0, 1] if
              they want a hard range ceiling (this helper does not).

    Returns:
        Fresh dict with the same keys; each array may or may not be a
        new object depending on the mode.

    Raises:
        ValueError for unrecognized mode.
    """
    if mode not in VALID_NORMALIZE_MODES:
        raise ValueError(
            f"mode must be one of {VALID_NORMALIZE_MODES}, got {mode!r}")
    if mode == 'clamped' or not out:
        return dict(out)

    keys = list(out.keys())
    stack = np.stack([out[k] for k in keys], axis=0)
    totals = stack.sum(axis=0)
    safe = totals > 1e-9
    safe_totals = np.where(safe, totals, 1.0)

    new_out: Dict[str, np.ndarray] = {}
    if mode == 'per_frame':
        for k in keys:
            new_out[k] = np.where(safe, out[k] / safe_totals, 0.0)
    else:  # 'energy_preserve'
        finite_totals = totals[np.isfinite(totals)]
        target = float(finite_totals.mean()) if finite_totals.size else 0.0
        if target <= 1e-9:
            return dict(out)
        scale = np.where(safe, target / safe_totals, 0.0)
        for k in keys:
            new_out[k] = out[k] * scale
    return new_out


def apply_per_electrode_gain(
    out: Dict[str, np.ndarray],
    gains: Union[Sequence[float], Dict[str, float], None],
) -> Dict[str, np.ndarray]:
    """
    Per-electrode multiplicative gain / trim.

    Applies `out[k] *= gains[k]` for each electrode. Physical devices
    often need per-channel level trim because output coils, skin
    contact, and felt sensation vary across electrodes; this is the
    last chance to rebalance before the final [0, 1] clip.

    Args:
        out: Dict of per-electrode arrays, typically keyed 'e1'...'eN'.
        gains: Either
            - a sequence (list/tuple/array) of floats, taken positionally
              as gains for e1, e2, ... in that order. Shorter sequence =
              trailing electrodes get gain 1.0. Longer is truncated.
            - a dict mapping electrode key → float. Missing keys default
              to 1.0 (unity gain, no change).
            - None → return a shallow copy unchanged.

    Returns:
        Fresh dict with each array scaled by its gain. Callers should
        final-clip to [0, 1] if gains can push values above 1.
    """
    if gains is None or not out:
        return dict(out)

    keys = list(out.keys())
    if isinstance(gains, dict):
        gain_map = {k: float(gains.get(k, 1.0)) for k in keys}
    else:
        seq = list(gains)
        gain_map = {}
        for i, k in enumerate(keys):
            gain_map[k] = float(seq[i]) if i < len(seq) else 1.0

    new_out: Dict[str, np.ndarray] = {}
    for k in keys:
        g = gain_map[k]
        if g == 1.0:
            new_out[k] = out[k]
        else:
            new_out[k] = out[k] * g
    return new_out


def apply_one_euro_per_electrode(
    out: Dict[str, np.ndarray],
    t_sec: Sequence[float],
    min_cutoff_hz: float = 1.0,
    beta: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Per-electrode One-Euro adaptive low-pass.

    Applies processing.one_euro_filter.one_euro_filter to each value in
    `out` independently. Kills coil-ramp-rate discontinuities without
    introducing visible lag on genuine motion pulses.

    Args:
        out: Dict of per-electrode arrays. All arrays must share length.
        t_sec: Timestamps in seconds, same length as each electrode
            array. Non-uniform grids are fine.
        min_cutoff_hz: Baseline cutoff at zero velocity. Lower = heavier
            smoothing on still/slow signals. 1.0 Hz is a conservative
            default for 50 Hz electrode outputs.
        beta: Velocity-to-cutoff gain. Higher = filter becomes more
            transparent on fast intensity changes. 0.05 follows the
            reference paper.

    Returns:
        Fresh dict with each array replaced by the filtered version.
        If t_sec length doesn't match the electrode arrays, returns
        the input dict unchanged and prints a warning — callers should
        treat this as "smoothing silently skipped."
    """
    if not out:
        return dict(out)

    t_arr = np.asarray(t_sec, dtype=float)
    any_key = next(iter(out))
    n = len(out[any_key])
    if len(t_arr) != n:
        print(f"[output_shaping] t_sec length {len(t_arr)} != electrode "
              f"length {n}; skipping smoother.")
        return dict(out)

    new_out: Dict[str, np.ndarray] = {}
    for k, arr in out.items():
        new_out[k] = one_euro_filter(
            t_arr, arr,
            min_cutoff_hz=float(min_cutoff_hz),
            beta=float(beta))
    return new_out
