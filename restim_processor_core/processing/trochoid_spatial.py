"""
Trochoid Spatial Mapping.

Drives N electrodes (default 4 at 0/90/180/270°) from a 1D input signal
by:

    1. Parameterizing a 2D curve (hypotrochoid, rose, butterfly, ...)
       by the input position.
    2. For each (x, y) on the curve, computing per-electrode intensity
       via a directional / distance / amplitude rule.

Result: each electrode gets its own intensity track over time, and
together they paint a "rotating sensation" pattern around the
electrode array as the input sweeps.

This is an alternative to the response-curve-based motion-axis
generation in motion_axis_generation.py — it produces the same kind of
output (e1/e2/e3/e4 intensity funscripts) by a fundamentally different
mechanism (spatial projection vs. per-axis curve mapping).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from funscript import Funscript
from .trochoid_quantization import (
    curve_xy, get_family_theta_max, FAMILY_DEFAULTS,
)
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


VALID_MAPPINGS = (
    'directional', 'distance', 'amplitude', 'tangent_directional',
    'blend',
)

# Mapping modes that 'blend' can mix. Order is stable for UI layout.
BLEND_COMPONENT_MODES = (
    'directional', 'tangent_directional', 'distance', 'amplitude',
)

# Kept as a module-level alias for back-compat with callers that
# imported the 2D-specific name; same values as the shared constant.
VALID_NORMALIZE_2D = VALID_NORMALIZE_MODES


def _mode_directional(xn, yn, rn, electrode_angles_deg, sharpness):
    # `sharpness` here is always a list of length N (caller pre-resolves).
    path_angle = np.arctan2(yn, xn)
    out: Dict[str, np.ndarray] = {}
    for i, angle_deg in enumerate(electrode_angles_deg):
        ang = np.radians(angle_deg)
        cos_val = np.clip(np.cos(path_angle - ang), 0.0, 1.0)
        out[f'e{i + 1}'] = cos_val ** sharpness[i]
    return out


def _mode_tangent_directional(xn, yn, rn, electrode_angles_deg, sharpness):
    tx = np.gradient(xn)
    ty = np.gradient(yn)
    valid = np.sqrt(tx * tx + ty * ty) > 1e-12
    path_angle = np.arctan2(ty, tx)
    out: Dict[str, np.ndarray] = {}
    for i, angle_deg in enumerate(electrode_angles_deg):
        ang = np.radians(angle_deg)
        cos_val = np.clip(np.cos(path_angle - ang), 0.0, 1.0)
        intensity = cos_val ** sharpness[i]
        out[f'e{i + 1}'] = np.where(valid, intensity, 0.0)
    return out


def _mode_distance(xn, yn, rn, electrode_angles_deg, sharpness):
    out: Dict[str, np.ndarray] = {}
    for i, angle_deg in enumerate(electrode_angles_deg):
        ang = np.radians(angle_deg)
        ex, ey = np.cos(ang), np.sin(ang)
        d = np.sqrt((xn - ex) ** 2 + (yn - ey) ** 2)
        intensity = np.clip(1.0 - d / 2.0, 0.0, 1.0)
        out[f'e{i + 1}'] = intensity ** sharpness[i]
    return out


def _mode_amplitude(xn, yn, rn, electrode_angles_deg, sharpness):
    path_angle = np.arctan2(yn, xn)
    out: Dict[str, np.ndarray] = {}
    for i, angle_deg in enumerate(electrode_angles_deg):
        ang = np.radians(angle_deg)
        cos_val = np.clip(np.cos(path_angle - ang), 0.0, 1.0)
        out[f'e{i + 1}'] = rn * (cos_val ** sharpness[i])
    return out


_MODE_FUNCS = {
    'directional': _mode_directional,
    'tangent_directional': _mode_tangent_directional,
    'distance': _mode_distance,
    'amplitude': _mode_amplitude,
}

# Default electrode angles in degrees (compass-like layout):
DEFAULT_ELECTRODE_ANGLES_DEG = (0.0, 90.0, 180.0, 270.0)


def compute_spatial_intensities(
    input_y: np.ndarray,
    family: str,
    params: Dict[str, Any],
    electrode_angles_deg: Tuple[float, ...] = DEFAULT_ELECTRODE_ANGLES_DEG,
    mapping: str = 'directional',
    sharpness: float = 1.0,
    cycles_per_unit: float = 1.0,
    normalize: str = 'clamped',
    theta_offset: float = 0.0,
    close_on_loop: bool = False,
    t_sec: Optional[np.ndarray] = None,
    smoothing_enabled: bool = False,
    smoothing_min_cutoff_hz: float = 1.0,
    smoothing_beta: float = 0.05,
    blend_directional: float = 0.0,
    blend_tangent_directional: float = 0.0,
    blend_distance: float = 0.0,
    blend_amplitude: float = 0.0,
    electrode_gain=None,
    output_limiter_enabled: bool = False,
    output_limiter_threshold: float = 0.85,
    velocity_weight: Optional[np.ndarray] = None,
    electrode_solo=None,
    electrode_mute=None,
) -> Dict[str, np.ndarray]:
    """
    Compute per-electrode intensity arrays from a 1D input signal.

    Args:
        input_y: 1D array of input positions in [0, 1].
        family: Curve family ('hypo', 'epi', 'rose', etc.).
        params: Family-specific curve parameters.
        electrode_angles_deg: Compass angles for the electrodes (one per
            output channel). Default: (0, 90, 180, 270).
        mapping: Which projection rule to use.
            - 'directional': cosine of the electrode angle vs the angle
              from origin to the curve point (i.e., *position* angle).
              Electrode lights when the curve *is located* in its
              direction; doesn't depend on which way the pen is moving.
            - 'distance': proximity of the curve point to an electrode
              seated on the unit circle.
            - 'amplitude': directional × normalized radius — combines
              reach and direction.
            - 'tangent_directional': cosine of the electrode angle vs
              the pen's instantaneous direction of travel (the tangent
              to the sampled path). Electrode lights when the pen is
              *moving toward* it, regardless of where it sits.
            - 'blend': weighted combination of the four modes above,
              controlled by `blend_directional`, `blend_tangent_directional`,
              `blend_distance`, `blend_amplitude`. Weights are applied
              raw (no internal normalization) — the user can over-drive
              past 1.0 or fade to silence with all zeros.
        sharpness: Exponent applied to the cosine in directional,
            tangent_directional, and amplitude modes. Higher = more
            selective. 1.0 = soft, 4.0 = sharp. Accepts either a
            scalar (broadcast to every electrode) or a sequence with
            one entry per electrode for per-channel control.
        cycles_per_unit: How many full curve cycles per 0→1 input change.
            Higher = faster electrode flicker per stroke.
        theta_offset: Radians added to `theta` before the curve is
            evaluated. Rotates the starting point of the pen's path
            around the curve — useful for phase-aligning multi-channel
            exports or picking which part of a multi-lobe family the
            signal enters first. Default 0.
        close_on_loop: When True, silently replaces `cycles_per_unit`
            with max(1, round(cycles_per_unit)) for this call only so
            that θ at input=0 and θ at input=1 differ by an integer
            multiple of the family's natural theta_max. For families
            whose curve is periodic in θ (rose, lissajous with rational
            a/b, butterfly over its 12π span) this produces exact
            endpoint coincidence — the stitching click between looping
            strokes goes away. For hypo/epi with irrational (R-r)/r
            ratios the θ-integer alignment still helps but the curve
            itself may not close at theta_max, so the effect is
            partial. The user's configured value is not mutated.
            Default False.
        t_sec: Optional timestamps in seconds, same length as input_y.
            Required when `smoothing_enabled` is True — the One-Euro
            filter needs dt per sample. Ignored when smoothing is off.
        smoothing_enabled: When True, apply a One-Euro adaptive low-pass
            to each electrode's intensity before clipping. Kills the
            audible-rate discontinuities that high sharpness × high
            cycles_per_unit can produce, without introducing the lag a
            fixed low-pass would add on fast motion. Default False.
        smoothing_min_cutoff_hz: Baseline cutoff at zero velocity.
            Lower = heavier smoothing on held/slow signals. 1.0 Hz is
            a reasonable default for 50 Hz input grids. Default 1.0.
        smoothing_beta: Velocity-to-cutoff gain. Higher = filter becomes
            more transparent on fast intensity changes. 0.05 is the
            reference paper's conservative default. Default 0.05.
        blend_directional, blend_tangent_directional, blend_distance,
        blend_amplitude: Per-mode weights used only when mapping='blend'.
            Each sub-mode's per-electrode intensity is scaled by its
            weight and summed. Ignored for other mappings. Defaults are
            all 0.0, so setting mapping='blend' without setting weights
            yields silence (forces intentional configuration).
        electrode_gain: Optional per-electrode multiplicative gain /
            trim applied after normalize + smoothing, before the final
            [0, 1] clip. Accepts a list (positional) or dict (keyed
            e1..eN). Missing or None = unity.
        output_limiter_enabled: When True, apply a soft-knee tanh
            limiter after electrode_gain. Default False.
        output_limiter_threshold: Knee position in (0, 1). 0.85
            default. Lower = more limited, higher = more transparent.
        velocity_weight: Optional per-frame [0, 1] array applied as a
            scalar gate to every electrode after smoothing, before
            gain. Let holds go quiet. None = no gating.
        normalize: Cross-electrode balancing applied after the per-mode
            intensity calc.
            - 'clamped' (default): raw per-electrode, just clipped to
              [0, 1]. Preserves the mapping's natural dynamics — total
              energy can swing as some electrodes fall silent.
            - 'per_frame': rebalance so Σ e_i(t) = 1 at every t.
              Relative cross-electrode shape preserved; kills temporal
              energy swings but forces the output into a [0, 1/N]
              ceiling. Zero-sum frames (no electrode firing) are left
              at zero.
            - 'energy_preserve': scale all channels by a time-varying
              factor so Σ e_i(t) equals the time-average of Σ e_i(t)
              across the signal. Flattens total energy without the
              sum-to-1 ceiling. Clipped to [0, 1] so dead-zone frames
              (where the raw sum is tiny) don't blow up past unity.

    Returns:
        Dict {'e1': array, 'e2': array, ...} matching the order of
        electrode_angles_deg. Each array is the same length as input_y
        and lies in [0, 1].
    """
    if mapping not in VALID_MAPPINGS:
        raise ValueError(
            f"mapping must be one of {VALID_MAPPINGS}, got {mapping!r}")
    if normalize not in VALID_NORMALIZE_2D:
        raise ValueError(
            f"normalize must be one of {VALID_NORMALIZE_2D}, "
            f"got {normalize!r}")

    input_y = np.asarray(input_y, dtype=float)
    input_y = np.clip(input_y, 0.0, 1.0)

    # Drive the curve parameter from the input. theta_max keeps the sweep
    # extent consistent across families (butterfly traces over 12π naturally,
    # most others over 2π).
    theta_max = get_family_theta_max(family)
    effective_cycles = float(cycles_per_unit)
    if close_on_loop:
        # Round toward the nearest integer ≥ 1 so input=0 and input=1
        # land on the same curve point. Guarantees stroke-loop closure.
        effective_cycles = max(1.0, float(round(effective_cycles)))
    theta = theta_max * effective_cycles * input_y + float(theta_offset)
    x, y = curve_xy(theta, family, params)

    # Normalize the curve into a unit-radius reference so the angular /
    # distance math is comparable across very different families.
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.all():
        x = np.where(finite, x, 0.0)
        y = np.where(finite, y, 0.0)
    radii = np.sqrt(x * x + y * y)
    rmax = float(np.max(radii)) if len(radii) else 1.0
    if rmax < 1e-12:
        rmax = 1.0
    xn = x / rmax
    yn = y / rmax
    rn = radii / rmax  # in [0, 1]

    # sharpness accepts scalar (broadcast) or sequence (per-electrode).
    # Helpers below always receive a length-N list and index by i.
    sharpness_list = resolve_per_electrode_scalar(
        sharpness, len(electrode_angles_deg),
        default=1.0, floor=0.01)

    if mapping == 'blend':
        weights = {
            'directional': float(blend_directional),
            'tangent_directional': float(blend_tangent_directional),
            'distance': float(blend_distance),
            'amplitude': float(blend_amplitude),
        }
        out: Dict[str, np.ndarray] = {
            f'e{i + 1}': np.zeros_like(xn)
            for i in range(len(electrode_angles_deg))
        }
        for mode_name, w in weights.items():
            if abs(w) < 1e-12:
                continue
            sub = _MODE_FUNCS[mode_name](
                xn, yn, rn, electrode_angles_deg, sharpness_list)
            for k in out:
                out[k] = out[k] + w * sub[k]
    else:
        out = _MODE_FUNCS[mapping](
            xn, yn, rn, electrode_angles_deg, sharpness_list)

    # Cross-electrode balancing and optional post-stage smoothing — both
    # delegated to the shared output-shaping toolkit so the Linear 3D
    # kernel can apply the identical transforms.
    out = apply_cross_electrode_normalize(out, normalize)
    if smoothing_enabled:
        if t_sec is None:
            print("[trochoid_spatial] smoothing_enabled=True but t_sec "
                  "was not provided; skipping smoother.")
        else:
            out = apply_one_euro_per_electrode(
                out, t_sec,
                min_cutoff_hz=smoothing_min_cutoff_hz,
                beta=smoothing_beta)
    out = apply_velocity_weight(out, velocity_weight)
    out = apply_per_electrode_gain(out, electrode_gain)
    if output_limiter_enabled:
        out = apply_soft_knee_limiter(
            out, threshold=output_limiter_threshold, ceiling=1.0)
    out = apply_solo_mute_mask(out, electrode_solo, electrode_mute)

    # Final sanitation
    for key, arr in out.items():
        out[key] = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(out[key], 0.0, 1.0, out=out[key])

    return out


def generate_spatial_funscripts(
    main_funscript: Funscript,
    family: str,
    params: Dict[str, Any],
    electrode_angles_deg: Tuple[float, ...] = DEFAULT_ELECTRODE_ANGLES_DEG,
    mapping: str = 'directional',
    sharpness: float = 1.0,
    cycles_per_unit: float = 1.0,
    densify_hz: float = 60.0,
    normalize: str = 'clamped',
    theta_offset: float = 0.0,
    close_on_loop: bool = False,
    smoothing_enabled: bool = False,
    smoothing_min_cutoff_hz: float = 1.0,
    smoothing_beta: float = 0.05,
    blend_directional: float = 0.0,
    blend_tangent_directional: float = 0.0,
    blend_distance: float = 0.0,
    blend_amplitude: float = 0.0,
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
    Build per-electrode Funscript outputs from the main signal.

    When `densify_hz > 0`, the input y is linearly resampled to that
    rate before the curve mapping is evaluated. This matters when
    `cycles_per_unit` is high or the family has many lobes: between
    two sparse input samples the curve can traverse a significant
    arc, producing intensity variation that would otherwise be
    aliased away. Setting densify_hz = 0 preserves the old behavior
    (output shares the input's timestamps exactly).
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

    # Funscript.x is already in seconds (see funscript.py load path).
    _vw = None
    if velocity_weight_enabled and len(y_for_mapping) >= 2:
        _vw = compute_velocity_weight(
            [np.asarray(y_for_mapping, dtype=float)],
            np.asarray(t_out, dtype=float),
            floor=velocity_weight_floor,
            response=velocity_weight_response,
            smoothing_hz=velocity_weight_smoothing_hz,
            normalization_percentile=velocity_weight_normalization_percentile,
            gate_threshold=velocity_weight_gate_threshold,
        )
    intensities = compute_spatial_intensities(
        y_for_mapping, family, params, electrode_angles_deg,
        mapping, sharpness, cycles_per_unit,
        normalize=normalize,
        theta_offset=theta_offset,
        close_on_loop=close_on_loop,
        t_sec=np.asarray(t_out, dtype=float),
        smoothing_enabled=smoothing_enabled,
        smoothing_min_cutoff_hz=smoothing_min_cutoff_hz,
        smoothing_beta=smoothing_beta,
        blend_directional=blend_directional,
        blend_tangent_directional=blend_tangent_directional,
        blend_distance=blend_distance,
        blend_amplitude=blend_amplitude,
        electrode_gain=electrode_gain,
        output_limiter_enabled=output_limiter_enabled,
        output_limiter_threshold=output_limiter_threshold,
        velocity_weight=_vw,
        electrode_solo=electrode_solo,
        electrode_mute=electrode_mute,
    )
    out = {}
    for key, arr in intensities.items():
        out[key] = Funscript(t_out.copy(), arr,
                             metadata=dict(main_funscript.metadata))
    return out


def get_default_config() -> Dict[str, Any]:
    """Default config block for trochoid_spatial."""
    return {
        'enabled': False,
        'family': 'hypo',
        'mapping': 'directional',
        'sharpness': 1.0,
        'cycles_per_unit': 1.0,
        'normalize': 'clamped',
        'theta_offset': 0.0,
        'close_on_loop': False,
        'smoothing_enabled': False,
        'smoothing_min_cutoff_hz': 1.0,
        'smoothing_beta': 0.05,
        'blend_directional': 0.0,
        'blend_tangent_directional': 0.0,
        'blend_distance': 0.0,
        'blend_amplitude': 0.0,
        'electrode_gain': [1.0, 1.0, 1.0, 1.0],
        'output_limiter_enabled': False,
        'output_limiter_threshold': 0.85,
        'velocity_weight_enabled': False,
        'velocity_weight_floor': 0.0,
        'velocity_weight_response': 1.0,
        'velocity_weight_smoothing_hz': 3.0,
        'velocity_weight_normalization_percentile': 0.99,
        'velocity_weight_gate_threshold': 0.05,
        'electrode_angles_deg': list(DEFAULT_ELECTRODE_ANGLES_DEG),
        'params_by_family': {
            fam: dict(spec['params'])
            for fam, spec in FAMILY_DEFAULTS.items()
        },
    }
