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

    apply_soft_knee_limiter(out, threshold, ceiling=1.0)
        Smooth tanh-based limiter. Values below threshold pass through
        unchanged; values above threshold are compressed asymptotically
        toward ceiling. Prevents the hard-clip artifacts that occur
        when gains > 1 or other boosts push peaks past 1.0.

    compute_velocity_weight(arrays, t_sec, ...) → np.ndarray
        Per-frame [0, 1] weight from the magnitude of the signal(s)'
        time derivative. Single-axis: |d arr/dt|. Multi-axis: root-sum-
        squared of per-axis velocities. Smoothed, percentile-normalized,
        raised to a response curve, and mixed with a floor.

    apply_velocity_weight(out, weight)
        Multiply each electrode by the per-frame weight. Holds → quiet,
        fast motion → full intensity. Scalar (same weight applies to
        every electrode).

All helpers take and return Dict[str, np.ndarray] and never mutate the
input. Intended to be called as post-stages inside the projection
kernels (or downstream of them) — they don't know or care about
projection geometry.
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


def resolve_per_electrode_scalar(
    value: Union[float, Sequence[float], None],
    n: int,
    default: float = 1.0,
    floor: float = 0.01,
) -> list:
    """
    Expand a scalar-or-sequence knob to a length-n list with a floor.

    Used by both projection kernels to accept either a single number
    (broadcast to all electrodes) or a per-electrode sequence without
    forcing callers to branch on the type. Short sequences are padded
    with `default`; long sequences are truncated to n entries. Non-
    numeric / NaN / non-positive values are clipped to `floor` so the
    kernel never takes a 0 or negative exponent.

    Examples:
        resolve_per_electrode_scalar(2.0, 4) -> [2.0, 2.0, 2.0, 2.0]
        resolve_per_electrode_scalar([1, 2, 3, 4], 4) -> [1.0, 2.0, 3.0, 4.0]
        resolve_per_electrode_scalar([2, 4], 4) -> [2.0, 4.0, 1.0, 1.0]
        resolve_per_electrode_scalar(None, 3) -> [1.0, 1.0, 1.0]
    """
    if value is None:
        return [max(float(floor), float(default))] * int(n)
    if isinstance(value, (int, float, np.floating, np.integer)):
        v = float(value)
        return [max(float(floor), v)] * int(n)
    try:
        seq = list(value)
    except TypeError:
        return [max(float(floor), float(default))] * int(n)
    out = []
    for i in range(int(n)):
        if i < len(seq):
            try:
                v = float(seq[i])
                if not np.isfinite(v) or v < floor:
                    v = max(float(floor), float(default))
                out.append(v)
            except (TypeError, ValueError):
                out.append(max(float(floor), float(default)))
        else:
            out.append(max(float(floor), float(default)))
    return out


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


def apply_soft_knee_limiter(
    out: Dict[str, np.ndarray],
    threshold: float,
    ceiling: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Soft-knee tanh-based limiter.

    Below `threshold`, samples pass through unchanged. Above the
    threshold, samples curve smoothly toward `ceiling` via tanh so
    peaks get rounded off instead of hard-clipped. The net effect:
    gains > 1 or `energy_preserve` overshoots get squeezed musically
    instead of snapped flat at 1.

    Formula (per sample, where x is raw intensity):
        if x <= threshold:   y = x
        else:                y = threshold + (ceiling - threshold) *
                                 tanh((x - threshold) / (ceiling - threshold))

    tanh maps [0, ∞) → [0, 1), so the compressed region is bounded
    by `ceiling` from above and transitions smoothly from linear
    pass-through at the threshold.

    Args:
        out: Dict of per-electrode arrays.
        threshold: Knee position in (0, ceiling). Common values 0.7–0.95.
            Lower = earlier compression = more "limited" feel.
        ceiling: Absolute upper bound the output asymptotes to.
            Default 1.0 matches the downstream clip.

    Returns:
        Fresh dict with each array limited. Values may still exceed
        `ceiling` by a tiny numerical margin; callers should retain
        the final clip as a safety net.

    Raises:
        ValueError if threshold is not in (0, ceiling).
    """
    if not out:
        return dict(out)
    threshold = float(threshold)
    ceiling = float(ceiling)
    if not (0.0 < threshold < ceiling):
        raise ValueError(
            f"threshold must be in (0, ceiling); got "
            f"threshold={threshold}, ceiling={ceiling}")

    headroom = ceiling - threshold
    new_out: Dict[str, np.ndarray] = {}
    for k, arr in out.items():
        over = arr - threshold
        compressed = threshold + headroom * np.tanh(over / headroom)
        new_out[k] = np.where(arr <= threshold, arr, compressed)
    return new_out


def compute_velocity_weight(
    arrays: Sequence[np.ndarray],
    t_sec: Sequence[float],
    *,
    floor: float = 0.0,
    response: float = 1.0,
    smoothing_hz: float = 3.0,
    normalization_percentile: float = 0.99,
    gate_threshold: float = 0.05,
) -> np.ndarray:
    """
    Compute a per-frame [0, 1] weight from the magnitude of the input
    signal(s)' time derivative.

    Single 1D input: weight ~ |d arr/dt|.
    Multi-axis input: weight ~ sqrt(Σ (d arr_i/dt)^2) — Euclidean
    velocity magnitude across axes (natural extension for XYZ triplets).

    Pipeline:
      1. Per-axis numerical derivatives via np.gradient.
      2. Root-sum-squared across axes.
      3. Low-pass filtered at `smoothing_hz` (raw velocity is noisy).
      4. Normalized by the `normalization_percentile` of filtered values
         (so one-sample spikes don't flatten the range).
      5. Gated: samples whose normalized speed falls below
         `gate_threshold` are forced to 0. Kills the residual micro-
         velocity from tracker noise that would otherwise survive as
         "light touch" on held positions when floor = 0.
      6. Raised to `response` power (1 = linear, higher = sharper).
      7. Mixed with `floor`: weight = floor + (1 - floor) * shaped.
         Floor 0 → silent on holds (combined with gate), 0.3 → 30%
         baseline on holds regardless of gate.

    Args:
        arrays: One or more 1D signal arrays, all same length.
        t_sec: Timestamps in seconds, same length. Non-uniform fine.
        floor: Minimum weight; 0 = holds go fully silent (when paired
            with a non-zero gate_threshold).
        response: Exponent on the normalized speed. 1 = linear.
        smoothing_hz: Low-pass cutoff on raw velocity magnitude.
        normalization_percentile: Percentile used as the "full speed"
            reference so one-sample spikes don't collapse the dynamic
            range. 0.99 typical.
        gate_threshold: Minimum normalized speed to let through. Speeds
            below this value are zeroed BEFORE the floor mix so
            floor=0 produces actual silence on holds instead of the
            residual tracker-noise bleed a bare linear gate would
            leak. 0.05 (5% of peak) is a conservative default that
            kills micro-jitter without cutting genuine slow motion.
            Set to 0 to disable the gate entirely (old behavior).

    Returns:
        1D numpy array in [0, 1], same length as each input.
    """
    if not arrays:
        return np.ones(0)
    lens = {len(a) for a in arrays}
    if len(lens) != 1:
        raise ValueError(f"all input arrays must be same length; got {lens}")
    n = lens.pop()
    t = np.asarray(t_sec, dtype=float)
    if len(t) != n:
        raise ValueError(
            f"t_sec length {len(t)} != input length {n}")
    if n < 2:
        return np.full(n, float(floor))

    # Derivatives per axis; np.gradient handles non-uniform t.
    dt = np.gradient(t)
    mag_sq = np.zeros(n, dtype=float)
    for a in arrays:
        da = np.gradient(np.asarray(a, dtype=float), t)
        mag_sq += da * da
    speed = np.sqrt(np.maximum(mag_sq, 0.0))
    # Replace non-finite (e.g. from duplicate timestamps) with 0.
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)

    # Low-pass via a single-pole EMA parameterized by cutoff_hz × dt.
    if smoothing_hz > 0.0:
        smoothed = np.empty(n, dtype=float)
        smoothed[0] = speed[0]
        for i in range(1, n):
            local_dt = float(dt[i]) if dt[i] > 0 else 1e-3
            tau = 1.0 / (2.0 * np.pi * float(smoothing_hz))
            alpha = 1.0 / (1.0 + tau / local_dt)
            smoothed[i] = alpha * speed[i] + (1.0 - alpha) * smoothed[i - 1]
        speed = smoothed

    # Normalize by percentile. Fall back to max if percentile computes 0.
    p = float(np.clip(normalization_percentile, 0.5, 1.0))
    peak = float(np.quantile(speed, p)) if speed.size else 0.0
    if peak < 1e-9:
        peak = float(speed.max()) if speed.size else 1.0
    if peak < 1e-9:
        return np.full(n, float(floor))
    norm = np.clip(speed / peak, 0.0, 1.0)
    # Hard gate below the threshold — kills residual micro-velocity
    # from tracker noise so floor=0 produces actual silence on holds.
    gate = float(np.clip(gate_threshold, 0.0, 1.0))
    if gate > 0.0:
        norm = np.where(norm < gate, 0.0, norm)
    if response != 1.0:
        norm = norm ** float(response)

    floor = float(np.clip(floor, 0.0, 1.0))
    return floor + (1.0 - floor) * norm


def apply_velocity_weight(
    out: Dict[str, np.ndarray],
    weight: Union[Sequence[float], None],
) -> Dict[str, np.ndarray]:
    """
    Multiply every electrode by the per-frame velocity weight.

    Args:
        out: Dict of per-electrode arrays.
        weight: 1D array same length as each electrode array, OR None.
            If None, return a shallow copy unchanged.

    Returns:
        Fresh dict. Arrays not copied if weight is None.
    """
    if weight is None or not out:
        return dict(out)
    w = np.asarray(weight, dtype=float)
    any_key = next(iter(out))
    if len(w) != len(out[any_key]):
        print(f"[output_shaping] velocity weight length {len(w)} "
              f"!= electrode length {len(out[any_key])}; skipping.")
        return dict(out)
    return {k: arr * w for k, arr in out.items()}


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
