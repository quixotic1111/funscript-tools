"""
Spatial 3D Curve — 1D input → 3D parametric curve → N electrodes in 3D.

A third projector alongside the existing two:
  - Trochoid Spatial  (1D → 2D curve → 4 angular electrodes)
  - Spatial 3D Linear (XYZ triplet → N electrodes along a line)
  - Spatial 3D Curve  (1D → 3D curve → N electrodes in 3D)

The pattern mirrors Trochoid Spatial but operates in three dimensions:
the 1D input position parameterizes a 3D curve (helix, trefoil knot,
3D Lissajous, spherical spiral, …), and each (x, y, z) on that curve
projects onto N electrodes arranged in 3D space (tetrahedral, ring,
custom). The resulting per-electrode intensity arrays then flow
through the shared output-shaping toolkit (normalize, One-Euro
smoothing, velocity weight, per-electrode gain, soft-knee limiter,
solo/mute mask) the same way the other kernels do.

Use cases this opens up that neither Trochoid nor Linear 3D cover:
  - Non-planar sensation traces (e.g., movement along a knot instead
    of around a 2D rose).
  - Spatially asymmetric arrays (tetrahedral inscribed in a device
    chamber, ring around a circumference at a specific axial plane).
  - 3D Lissajous audition — pick (a, b, c, phi, psi) and hear the
    coherence / chaos of the resulting pattern.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from funscript import Funscript
from .output_shaping import (
    apply_cross_electrode_normalize,
    apply_one_euro_per_electrode,
    apply_per_electrode_gain,
    apply_soft_knee_limiter,
    apply_solo_mute_mask,
    apply_velocity_weight,
    compute_velocity_weight,
    resolve_per_electrode_scalar,
    VALID_NORMALIZE_MODES,
)
from .spatial_3d_linear import _apply_falloff, VALID_FALLOFF_SHAPES


# ============================================================================
# Curve families — 1D parameter theta → (x, y, z) trace
# ============================================================================

CURVE_FAMILIES_3D = (
    'helix', 'trefoil_knot', 'torus_knot', 'lissajous_3d',
    'spherical_spiral',
)

# Per-family defaults. `theta_max_pi` is the upper bound of the
# natural parameter range in units of π — families that need longer
# ranges to trace the full curve (e.g., torus knots with large q)
# can set this higher than 2.0.
CURVE_FAMILY_DEFAULTS_3D: Dict[str, Dict[str, Any]] = {
    'helix': {
        'description': (
            'Ascending helix — spiral along the z-axis. '
            'r = radius, h = total height, turns = full revolutions.'),
        'params': {'r': 0.6, 'h': 1.2, 'turns': 3.0},
        'theta_max_pi': 2.0,
    },
    'trefoil_knot': {
        'description': (
            'Trefoil knot — simplest non-trivial knot. scale = uniform '
            'size multiplier.'),
        'params': {'scale': 0.25},
        'theta_max_pi': 2.0,
    },
    'torus_knot': {
        'description': (
            'Torus knot (p, q) — winds p times around the torus axis '
            'and q times around the tube. (2, 3) = trefoil, '
            '(3, 2) = same knot other way. '
            'R = major radius, r = minor radius.'),
        'params': {'R': 1.0, 'r': 0.4, 'p': 2.0, 'q': 3.0,
                   'scale': 0.4},
        'theta_max_pi': 2.0,
    },
    'lissajous_3d': {
        'description': (
            'Lissajous 3D — x, y, z sinusoids with independent '
            'frequencies (a, b, c) and phases (phi, psi). '
            'Rational frequency ratios give closed curves; '
            'irrational give space-filling motion.'),
        'params': {'A': 1.0, 'B': 1.0, 'C': 1.0,
                   'a': 3.0, 'b': 2.0, 'c': 5.0,
                   'phi': 1.5708, 'psi': 0.0,
                   'scale': 0.7},
        'theta_max_pi': 2.0,
    },
    'spherical_spiral': {
        'description': (
            'Spherical spiral — wraps along a sphere from pole to '
            'pole, with `c` full longitudinal loops per pole-to-pole '
            'pass.'),
        'params': {'c': 5.0, 'scale': 0.85},
        'theta_max_pi': 2.0,
    },
}


def _eval_helix(theta, r, h, turns):
    angle = theta * float(turns)
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    # Center around z=0 so the curve straddles the electrode array.
    z = h * (theta / (2.0 * np.pi) - 0.5)
    return x, y, z


def _eval_trefoil_knot(theta, scale):
    x = scale * (np.sin(theta) + 2.0 * np.sin(2.0 * theta))
    y = scale * (np.cos(theta) - 2.0 * np.cos(2.0 * theta))
    z = scale * -np.sin(3.0 * theta)
    return x, y, z


def _eval_torus_knot(theta, R, r, p, q, scale):
    tube = R + r * np.cos(q * theta)
    x = scale * tube * np.cos(p * theta)
    y = scale * tube * np.sin(p * theta)
    z = scale * r * np.sin(q * theta)
    return x, y, z


def _eval_lissajous_3d(theta, A, B, C, a, b, c, phi, psi, scale):
    x = scale * A * np.sin(a * theta + phi)
    y = scale * B * np.sin(b * theta)
    z = scale * C * np.sin(c * theta + psi)
    return x, y, z


def _eval_spherical_spiral(theta, c, scale):
    # cos(θ)·cos(c·θ), cos(θ)·sin(c·θ), −sin(θ) — wraps on a unit sphere.
    x = scale * np.cos(theta) * np.cos(c * theta)
    y = scale * np.cos(theta) * np.sin(c * theta)
    z = scale * -np.sin(theta)
    return x, y, z


_CURVE_EVALUATORS_3D = {
    'helix': _eval_helix,
    'trefoil_knot': _eval_trefoil_knot,
    'torus_knot': _eval_torus_knot,
    'lissajous_3d': _eval_lissajous_3d,
    'spherical_spiral': _eval_spherical_spiral,
}


def _coerce_params_3d(family: str, params: Dict[str, Any]) -> Dict[str, Any]:
    spec = CURVE_FAMILY_DEFAULTS_3D.get(family, {})
    defaults = spec.get('params', {})
    out = {}
    for k, default_v in defaults.items():
        v = params.get(k, default_v) if params else default_v
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = float(default_v)
    return out


def curve_xyz_3d(
    theta: np.ndarray, family: str, params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a 3D parametric curve from the requested family."""
    if family not in _CURVE_EVALUATORS_3D:
        raise ValueError(
            f"family must be one of {CURVE_FAMILIES_3D}, got {family!r}")
    coerced = _coerce_params_3d(family, params or {})
    return _CURVE_EVALUATORS_3D[family](theta, **coerced)


def get_family_theta_max(family: str) -> float:
    spec = CURVE_FAMILY_DEFAULTS_3D.get(family)
    if not spec:
        raise ValueError(f"unknown 3D curve family: {family!r}")
    return float(spec['theta_max_pi']) * float(np.pi)


# ============================================================================
# Electrode arrangements — return (N, 3) positions in the unit sphere
# ============================================================================

ELECTRODE_ARRANGEMENTS_3D = ('tetrahedral', 'ring', 'custom')


def _tetrahedral_positions(n: int) -> np.ndarray:
    """
    Regular tetrahedron vertices inscribed in the unit sphere for
    N=4. For N=3, equilateral triangle at z=0. For N != 3,4, fall
    back to a ring (can't inscribe a regular polytope arbitrarily).
    """
    if n == 4:
        s = 1.0 / math.sqrt(3.0)
        return np.array([
            [+s, +s, +s],
            [+s, -s, -s],
            [-s, +s, -s],
            [-s, -s, +s],
        ], dtype=float)
    if n == 3:
        angles = np.radians([0.0, 120.0, 240.0])
        return np.stack(
            [np.cos(angles), np.sin(angles), np.zeros(3)], axis=1)
    return _ring_positions(n)


def _ring_positions(n: int) -> np.ndarray:
    """N electrodes on the unit circle at z=0, equally spaced in angle."""
    if n < 1:
        return np.zeros((0, 3), dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack(
        [np.cos(angles), np.sin(angles), np.zeros(n)], axis=1)


def electrode_positions_3d(
    arrangement: str, n: int,
    custom_positions: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    """
    Return an (n, 3) float64 array of electrode positions.

    Args:
        arrangement: one of ELECTRODE_ARRANGEMENTS_3D.
        n: number of electrodes.
        custom_positions: required when arrangement == 'custom'. Must
            be coercible to shape (n, 3).

    Raises:
        ValueError for unknown arrangement or mismatched custom shape.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if arrangement == 'tetrahedral':
        return _tetrahedral_positions(n)
    if arrangement == 'ring':
        return _ring_positions(n)
    if arrangement == 'custom':
        if custom_positions is None:
            raise ValueError(
                "custom arrangement requires custom_positions")
        arr = np.asarray(custom_positions, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] != 3:
            raise ValueError(
                f"custom_positions shape {arr.shape} != ({n}, 3)")
        return arr
    raise ValueError(
        f"unknown arrangement {arrangement!r}; "
        f"expected one of {ELECTRODE_ARRANGEMENTS_3D}")


# ============================================================================
# Main kernel
# ============================================================================

def compute_3d_curve_intensities(
    input_y: np.ndarray,
    family: str = 'helix',
    params: Optional[Dict[str, Any]] = None,
    n_electrodes: int = 4,
    electrode_arrangement: str = 'tetrahedral',
    electrode_positions_3d_custom: Optional[Sequence[Sequence[float]]] = None,
    sharpness: Union[float, Sequence[float]] = 1.0,
    cycles_per_unit: float = 1.0,
    theta_offset: float = 0.0,
    close_on_loop: bool = False,
    normalize: str = 'clamped',
    falloff_shape: str = 'linear',
    falloff_width: float = 1.0,
    t_sec: Optional[Sequence[float]] = None,
    output_smoothing_enabled: bool = False,
    output_smoothing_min_cutoff_hz: float = 1.0,
    output_smoothing_beta: float = 0.05,
    electrode_gain=None,
    output_limiter_enabled: bool = False,
    output_limiter_threshold: float = 0.85,
    velocity_weight: Optional[Sequence[float]] = None,
    electrode_solo=None,
    electrode_mute=None,
) -> Dict[str, np.ndarray]:
    """
    Compute per-electrode intensity from a 1D input driving a 3D curve.

    Args:
        input_y: 1D array of input positions in [0, 1]. Drives the
            curve parameter θ = theta_max · cycles_per_unit · input_y
            + theta_offset.
        family: which 3D curve family to parameterize. See
            CURVE_FAMILIES_3D.
        params: family-specific parameter dict (uses defaults for
            missing / invalid keys).
        n_electrodes: number of output channels.
        electrode_arrangement: 'tetrahedral', 'ring', or 'custom'.
            Tetrahedral is a regular tetrahedron inscribed in the
            unit sphere (N=4) / equilateral triangle at z=0 (N=3) /
            ring fallback otherwise. Ring is N equally spaced points
            on the unit circle at z=0. Custom takes the
            `electrode_positions_3d_custom` array.
        electrode_positions_3d_custom: (N, 3) positions when
            arrangement == 'custom'.
        sharpness: scalar or per-electrode sequence. Exponent on the
            raw falloff-based intensity.
        cycles_per_unit: how many full curve traversals per 0→1 input
            sweep. Same semantic as Trochoid Spatial.
        theta_offset: radians added to θ before curve evaluation —
            rotates the starting point of the trace.
        close_on_loop: snap cycles_per_unit to the nearest integer
            (≥ 1) so θ(input=0) and θ(input=1) land on equivalent
            points.
        normalize / falloff_shape / falloff_width / t_sec /
        output_smoothing_* / electrode_gain / output_limiter_* /
        velocity_weight / electrode_solo / electrode_mute:
            identical semantics to compute_linear_intensities_3d's
            kwargs; all dispatch through the shared
            processing.output_shaping helpers.

    Returns:
        Dict {'e1', 'e2', ...} of per-electrode intensity arrays in
        [0, 1], same length as input_y.
    """
    if family not in CURVE_FAMILIES_3D:
        raise ValueError(
            f"family must be one of {CURVE_FAMILIES_3D}, got {family!r}")
    if normalize not in VALID_NORMALIZE_MODES:
        raise ValueError(
            f"normalize must be one of {VALID_NORMALIZE_MODES}, "
            f"got {normalize!r}")
    if falloff_shape not in VALID_FALLOFF_SHAPES:
        raise ValueError(
            f"falloff_shape must be one of {VALID_FALLOFF_SHAPES}, "
            f"got {falloff_shape!r}")

    input_y = np.clip(np.asarray(input_y, dtype=float), 0.0, 1.0)
    n = int(n_electrodes)
    if n < 1:
        raise ValueError(f"n_electrodes must be >= 1, got {n}")

    # Electrode positions in normalized space (unit sphere / unit circle).
    positions = electrode_positions_3d(
        electrode_arrangement, n, electrode_positions_3d_custom)

    # Drive θ from the input, applying stitching controls.
    theta_max = get_family_theta_max(family)
    effective_cycles = float(cycles_per_unit)
    if close_on_loop:
        effective_cycles = max(1.0, float(round(effective_cycles)))
    theta = theta_max * effective_cycles * input_y + float(theta_offset)

    # Evaluate the 3D curve and normalize to a unit-radius reference
    # so electrode distances are comparable across very different
    # family scales.
    x, y, z = curve_xyz_3d(theta, family, params or {})
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not finite.all():
        x = np.where(finite, x, 0.0)
        y = np.where(finite, y, 0.0)
        z = np.where(finite, z, 0.0)
    radii = np.sqrt(x * x + y * y + z * z)
    rmax = float(radii.max()) if radii.size else 1.0
    if rmax < 1e-12:
        rmax = 1.0
    xn = x / rmax
    yn = y / rmax
    zn = z / rmax

    # With both curve point and electrodes on the unit sphere, the
    # maximum possible distance is 2 (antipodal). Use that as the
    # reference diagonal so `falloff_width = 1` means "opposite-side
    # electrode → 0 raw intensity" in the linear shape.
    effective_diag = 2.0
    scale = float(falloff_width) * effective_diag

    sharpness_list = resolve_per_electrode_scalar(
        sharpness, n, default=1.0, floor=0.01)

    out: Dict[str, np.ndarray] = {}
    for i in range(n):
        ex, ey, ez = positions[i]
        dx = xn - float(ex)
        dy = yn - float(ey)
        dz = zn - float(ez)
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        raw = _apply_falloff(d, falloff_shape, scale)
        out[f'e{i + 1}'] = raw ** sharpness_list[i]

    # Output shaping pipeline — identical ordering to the Linear 3D
    # kernel: normalize → 1€ smooth → velocity weight → gain →
    # limiter → solo/mute → final clip.
    out = apply_cross_electrode_normalize(out, normalize)
    if output_smoothing_enabled:
        if t_sec is None:
            print("[spatial_3d_curve] output_smoothing_enabled=True "
                  "but t_sec was not provided; skipping smoother.")
        else:
            out = apply_one_euro_per_electrode(
                out, t_sec,
                min_cutoff_hz=output_smoothing_min_cutoff_hz,
                beta=output_smoothing_beta)
    out = apply_velocity_weight(out, velocity_weight)
    out = apply_per_electrode_gain(out, electrode_gain)
    if output_limiter_enabled:
        out = apply_soft_knee_limiter(
            out, threshold=output_limiter_threshold, ceiling=1.0)
    out = apply_solo_mute_mask(out, electrode_solo, electrode_mute)

    for key, arr in out.items():
        out[key] = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(out[key], 0.0, 1.0, out=out[key])

    return out


def generate_3d_curve_funscripts(
    main_funscript: Funscript,
    family: str = 'helix',
    params: Optional[Dict[str, Any]] = None,
    n_electrodes: int = 4,
    electrode_arrangement: str = 'tetrahedral',
    electrode_positions_3d_custom: Optional[Sequence[Sequence[float]]] = None,
    sharpness: Union[float, Sequence[float]] = 1.0,
    cycles_per_unit: float = 1.0,
    theta_offset: float = 0.0,
    close_on_loop: bool = False,
    normalize: str = 'clamped',
    falloff_shape: str = 'linear',
    falloff_width: float = 1.0,
    densify_hz: float = 60.0,
    output_smoothing_enabled: bool = False,
    output_smoothing_min_cutoff_hz: float = 1.0,
    output_smoothing_beta: float = 0.05,
    electrode_gain=None,
    output_limiter_enabled: bool = False,
    output_limiter_threshold: float = 0.85,
    velocity_weight_enabled: bool = False,
    velocity_weight_floor: float = 0.0,
    velocity_weight_response: float = 1.0,
    velocity_weight_smoothing_hz: float = 3.0,
    velocity_weight_normalization_percentile: float = 0.99,
    velocity_weight_gate_threshold: float = 0.05,
    electrode_solo=None,
    electrode_mute=None,
) -> Dict[str, Funscript]:
    """
    Build per-electrode Funscripts from a main 1D signal driving a 3D
    curve onto 3D electrode positions. Mirrors trochoid_spatial's
    generate_spatial_funscripts structure (including densify_hz
    resampling so high cycles / long curves don't alias away).
    """
    t_in = np.asarray(main_funscript.x, dtype=float)
    y_in = np.asarray(main_funscript.y, dtype=float)
    if len(t_in) >= 2 and float(densify_hz) > 0.0:
        duration = float(t_in[-1] - t_in[0])
        if duration > 0.0:
            n = max(2, int(np.ceil(duration * float(densify_hz))) + 1)
            t_out = np.linspace(float(t_in[0]), float(t_in[-1]), n)
            y_for_mapping = np.clip(
                np.interp(t_out, t_in, y_in), 0.0, 1.0)
        else:
            t_out = t_in.copy()
            y_for_mapping = y_in
    else:
        t_out = t_in.copy()
        y_for_mapping = y_in

    # Build the velocity weight array from the 1D driver (same as
    # trochoid's generate helper does).
    vw_array = None
    if velocity_weight_enabled and len(y_for_mapping) >= 2:
        vw_array = compute_velocity_weight(
            [np.asarray(y_for_mapping, dtype=float)],
            np.asarray(t_out, dtype=float),
            floor=velocity_weight_floor,
            response=velocity_weight_response,
            smoothing_hz=velocity_weight_smoothing_hz,
            normalization_percentile=velocity_weight_normalization_percentile,
            gate_threshold=velocity_weight_gate_threshold,
        )

    intensities = compute_3d_curve_intensities(
        y_for_mapping,
        family=family,
        params=params,
        n_electrodes=n_electrodes,
        electrode_arrangement=electrode_arrangement,
        electrode_positions_3d_custom=electrode_positions_3d_custom,
        sharpness=sharpness,
        cycles_per_unit=cycles_per_unit,
        theta_offset=theta_offset,
        close_on_loop=close_on_loop,
        normalize=normalize,
        falloff_shape=falloff_shape,
        falloff_width=falloff_width,
        t_sec=np.asarray(t_out, dtype=float),
        output_smoothing_enabled=output_smoothing_enabled,
        output_smoothing_min_cutoff_hz=output_smoothing_min_cutoff_hz,
        output_smoothing_beta=output_smoothing_beta,
        electrode_gain=electrode_gain,
        output_limiter_enabled=output_limiter_enabled,
        output_limiter_threshold=output_limiter_threshold,
        velocity_weight=vw_array,
        electrode_solo=electrode_solo,
        electrode_mute=electrode_mute,
    )

    out: Dict[str, Funscript] = {}
    for key, arr in intensities.items():
        out[key] = Funscript(
            t_out.copy(), arr,
            metadata=dict(main_funscript.metadata))
    return out


def get_default_config() -> Dict[str, Any]:
    """Default config block for spatial_3d_curve."""
    return {
        'enabled': False,
        'family': 'helix',
        'n_electrodes': 4,
        'electrode_arrangement': 'tetrahedral',
        'sharpness': 1.0,
        'cycles_per_unit': 1.0,
        'theta_offset': 0.0,
        'close_on_loop': False,
        'normalize': 'clamped',
        'falloff_shape': 'linear',
        'falloff_width': 1.0,
        'params_by_family': {
            fam: dict(spec['params'])
            for fam, spec in CURVE_FAMILY_DEFAULTS_3D.items()
        },
    }


def list_curve_families_3d() -> Dict[str, Dict[str, Any]]:
    """Return a copy of CURVE_FAMILY_DEFAULTS_3D for UI consumption."""
    out = {}
    for name, spec in CURVE_FAMILY_DEFAULTS_3D.items():
        out[name] = {
            'description': spec['description'],
            'params': dict(spec['params']),
            'theta_max_pi': spec['theta_max_pi'],
        }
    return out
