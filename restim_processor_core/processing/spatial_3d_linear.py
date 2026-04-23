"""
Spatial 3D Linear — XYZ → N-electrode projection.

Takes three 1D spatial signals (X, Y, Z, each in [0, 1]) and projects the
per-frame 3D point onto a straight line of N electrodes along the X axis.
Per-electrode intensity is the clamped Euclidean proximity from the
signal point to that electrode, raised to a sharpness exponent.

This is the kernel the "Spatial 3D (X,Y,Z triplet)" mode in the main
window uses. The surrounding pipeline (input smoothing, noise gate,
envelope, per-channel defaults) lives in `processor.py` and the UI
tuning panel in `ui/main_window.py`.

Separate from `processing/trochoid_spatial.py`, which houses a different
kernel (`compute_spatial_intensities`) that drives 4 electrodes from a
1D funscript via a 2D parametric curve. The two pipelines are mutually
exclusive at processor time — enabling Spatial 3D Linear disables the
Trochoid Spatial path for E1..EN generation.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .output_shaping import (
    apply_cross_electrode_normalize,
    apply_one_euro_per_electrode,
    apply_per_electrode_gain,
    apply_soft_knee_limiter,
    apply_velocity_weight,
    resolve_per_electrode_scalar,
    VALID_NORMALIZE_MODES,
)


# Unit-cube diagonal — worst-case distance from any point in [0, 1]^3
# to any electrode. Raw distances are normalized against it so intensity
# lies in [0, 1] before the sharpness exponent.
_UNIT_CUBE_DIAG = float(np.sqrt(3.0))

VALID_FALLOFF_SHAPES = (
    'linear', 'gaussian', 'raised_cosine', 'inverse_square',
)


def _apply_falloff(d: np.ndarray, shape: str, scale: float) -> np.ndarray:
    """
    Convert 3D distance to raw per-electrode intensity per the chosen
    falloff shape. `scale` is the characteristic distance each shape
    interprets as its knee / sigma / radius — callers supply
    width * effective_diag so the semantic is "same geometry across
    shapes."
    """
    scale = max(float(scale), 1e-9)
    if shape == 'linear':
        return np.clip(1.0 - d / scale, 0.0, 1.0)
    if shape == 'gaussian':
        # Bell with sigma = scale. Peak 1 at d=0, 0.607 at d=sigma,
        # 0.135 at d=2·sigma. Never hits exact zero — asymptotic.
        return np.exp(-(d * d) / (2.0 * scale * scale))
    if shape == 'raised_cosine':
        # Flat-top Hann-like window. Zero slope at d=0 and at d=scale;
        # exactly 0 for d >= scale.
        return np.where(
            d < scale,
            0.5 * (1.0 + np.cos(np.pi * d / scale)),
            0.0)
    if shape == 'inverse_square':
        # Physical analog (light/gravity falloff). 1 at d=0, 0.5 at
        # d=scale, 0.2 at d=2·scale. Asymptotic tail.
        return 1.0 / (1.0 + (d / scale) ** 2)
    raise ValueError(
        f"unknown falloff shape {shape!r}; expected one of "
        f"{VALID_FALLOFF_SHAPES}")


def compute_linear_intensities_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_electrodes: int = 4,
    electrode_x: np.ndarray = None,
    center_yz: Tuple[float, float] = (0.5, 0.5),
    sharpness: float = 1.0,
    normalize: str = 'clamped',
    t_sec: Optional[Sequence[float]] = None,
    output_smoothing_enabled: bool = False,
    output_smoothing_min_cutoff_hz: float = 1.0,
    output_smoothing_beta: float = 0.05,
    electrode_gain=None,
    output_limiter_enabled: bool = False,
    output_limiter_threshold: float = 0.85,
    velocity_weight: Optional[Sequence[float]] = None,
    y_weight: float = 1.0,
    z_weight: float = 1.0,
    falloff_shape: str = 'linear',
    falloff_width: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Per-electrode intensity from three spatial scripts onto a straight
    electrode line along the X axis.

    Electrodes sit at (electrode_x[i], center_yz[0], center_yz[1]).
    Raw per-electrode intensity is 1 − d / √3 (clamped to [0, 1]) raised
    to `sharpness`, where d is 3D Euclidean distance from the signal
    point to that electrode and √3 is the unit-cube diagonal. With a
    linear geometry Y and Z are symmetric: they collapse into a shared
    radial proximity from the line — by design.

    Args:
        x, y, z: 1D arrays of equal length, each in [0, 1].
        n_electrodes: Number of electrodes along the line.
        electrode_x: Explicit X positions (length n). Defaults to
            linspace(0.1, 0.9, n_electrodes).
        center_yz: (Y, Z) position of the electrode line.
        sharpness: Falloff exponent. Accepts either a scalar
            (broadcast to every electrode — the historic behavior)
            or a sequence with one entry per electrode for
            per-channel control (useful to accentuate a primary
            electrode while leaving others softer). Values below
            0.01 are clipped to avoid zero/negative exponents.
        normalize: Cross-electrode balancing applied after the raw
            proximity calc. See processing.output_shaping.
            VALID_NORMALIZE_MODES for the full set.
            - 'clamped' (default): raw per-electrode, clipped to [0, 1].
            - 'per_frame': rebalance so Σ e_i(t) = 1 at every sample.
            - 'energy_preserve': rescale so total energy is flat across
              the signal; no sum-to-1 ceiling.
        t_sec: Optional timestamps in seconds, same length as x/y/z.
            Required when `output_smoothing_enabled` is True. Ignored
            otherwise.
        output_smoothing_enabled: When True, apply a One-Euro adaptive
            low-pass to each electrode after normalization. Kills coil-
            ramp-rate discontinuities without adding lag on genuine
            motion pulses. Default False (back-compat).
        output_smoothing_min_cutoff_hz: Baseline cutoff at zero
            velocity. Lower = more smoothing on still signals. 1.0 Hz
            is the reference default.
        output_smoothing_beta: Velocity-to-cutoff gain. Higher = filter
            becomes more transparent on fast changes. 0.05 is
            conservative.
        electrode_gain: Optional per-electrode multiplicative gain /
            trim applied after normalize + smoothing, before the final
            [0, 1] clip. Accepts a list (positional) or dict (keyed
            e1..eN). Missing or None = unity. Useful to balance
            physical-device channel differences.
        output_limiter_enabled: When True, apply a soft-knee tanh
            limiter after electrode_gain and before the final clip.
            Rounds off peaks that exceed the threshold instead of
            hard-clipping them at 1.0 — avoids the crunchy artifacts
            that gain > 1 or energy_preserve overshoots can produce.
            Default False.
        output_limiter_threshold: Knee position in (0, 1). Samples
            below this value pass through unchanged; samples above
            are compressed asymptotically toward 1.0. 0.85 is a
            reasonable default — lower = more limited/compressed,
            higher = more transparent. Default 0.85.
        velocity_weight: Optional per-frame [0, 1] array applied as a
            scalar multiplier to every electrode after smoothing,
            before per-electrode gain. Meant to carry the output of
            processing.output_shaping.compute_velocity_weight so
            held positions quiet down naturally. None = no gating.
        y_weight, z_weight: Per-axis multipliers inside the distance
            calc. Default 1.0 each → pre-existing rotation-symmetric
            behavior where Y and Z collapse into a shared radial
            proximity. Set differently to break that symmetry and
            make Y and Z behave as independent physical axes. 0 on
            an axis removes it entirely (1D or 2D kernel via the
            same code path).
        falloff_shape: Which distance-to-intensity function to use.
            See VALID_FALLOFF_SHAPES.
            - 'linear' (default): 1 − d/(width·diag), clamped.
              Hard-edge falloff matching pre-feature behavior.
            - 'gaussian': bell-curve, smooth and asymptotic.
            - 'raised_cosine': flat peak, smooth zero at cutoff.
            - 'inverse_square': physical-feel, long tail.
        falloff_width: Scale applied to the effective unit-cube
            diagonal to produce the characteristic distance each shape
            interprets as its knee / sigma / radius. 1.0 (default) =
            full diagonal — for linear, matches the historic formula
            exactly. Lower values tighten the falloff; higher values
            broaden it.

    Returns:
        Dict {'e1': array, ...} of length n_electrodes. Arrays share the
        length of x and lie in [0, 1].
    """
    if normalize not in VALID_NORMALIZE_MODES:
        raise ValueError(
            f"normalize must be one of {VALID_NORMALIZE_MODES}, "
            f"got {normalize!r}")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if not (len(x) == len(y) == len(z)):
        raise ValueError(
            f"x, y, z must share length; got {len(x)}, {len(y)}, {len(z)}")

    n = int(n_electrodes)
    if n < 1:
        raise ValueError(f"n_electrodes must be >= 1, got {n}")

    if electrode_x is None:
        electrode_x = np.linspace(0.1, 0.9, n)
    else:
        electrode_x = np.asarray(electrode_x, dtype=float)
        if len(electrode_x) != n:
            raise ValueError(
                f"electrode_x length {len(electrode_x)} != "
                f"n_electrodes {n}")

    cy, cz = float(center_yz[0]), float(center_yz[1])
    sharpness_list = resolve_per_electrode_scalar(
        sharpness, n, default=1.0, floor=0.01)
    wy, wz = float(y_weight), float(z_weight)

    # When the Y/Z axis weights are non-unity, the maximum distance
    # from any point in [0, 1]^3 to an electrode shifts from the
    # unit-cube diagonal. Recompute the diagonal so intensity still
    # reaches exactly 0 at the worst-case corner:
    #   max distance = sqrt(max_dx² + wy² · max_dy² + wz² · max_dz²)
    # With signal and electrodes in [0, 1], max_d on each axis is at
    # most 1 (signal at 0, electrode at 1 or vice versa). electrode_x
    # actually lies in [0.1, 0.9] by default, but using 1.0 is a safe
    # upper bound.
    effective_diag = float(np.sqrt(1.0 + wy * wy + wz * wz)) or _UNIT_CUBE_DIAG

    if falloff_shape not in VALID_FALLOFF_SHAPES:
        raise ValueError(
            f"falloff_shape must be one of {VALID_FALLOFF_SHAPES}, "
            f"got {falloff_shape!r}")
    scale = float(falloff_width) * effective_diag

    out: Dict[str, np.ndarray] = {}
    for i in range(n):
        dx = x - float(electrode_x[i])
        dy = (y - cy) * wy
        dz = (z - cz) * wz
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        raw = _apply_falloff(d, falloff_shape, scale)
        out[f'e{i + 1}'] = raw ** sharpness_list[i]

    # Cross-electrode balancing (clamped / per_frame / energy_preserve).
    out = apply_cross_electrode_normalize(out, normalize)

    # Optional per-electrode One-Euro post-stage.
    if output_smoothing_enabled:
        if t_sec is None:
            print("[spatial_3d_linear] output_smoothing_enabled=True but "
                  "t_sec was not provided; skipping smoother.")
        else:
            out = apply_one_euro_per_electrode(
                out, t_sec,
                min_cutoff_hz=output_smoothing_min_cutoff_hz,
                beta=output_smoothing_beta)

    # Velocity weight — time-varying scalar gate driven by input motion.
    # Applied after smoothing so quick transitions in the weight don't
    # get smoothed away, and before gain so per-channel gain shapes
    # the already-gated signal.
    out = apply_velocity_weight(out, velocity_weight)

    # Per-electrode gain/trim — last linear shaping stage.
    out = apply_per_electrode_gain(out, electrode_gain)

    # Soft-knee limiter — sits after gain so boosts get smoothly rolled
    # off instead of hard-clipping at the final clip. Retains that
    # clip as a safety net for any numerical overshoot past the tanh
    # asymptote.
    if output_limiter_enabled:
        out = apply_soft_knee_limiter(
            out, threshold=output_limiter_threshold, ceiling=1.0)

    for key, arr in out.items():
        out[key] = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(out[key], 0.0, 1.0, out=out[key])

    return out
