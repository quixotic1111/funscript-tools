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
    'directional', 'distance', 'amplitude', 'distance_amplitude',
    'tangent_directional', 'blend', 'lobe_gate',
)

# Mapping modes that 'blend' can mix. Order is stable for UI layout.
BLEND_COMPONENT_MODES = (
    'directional', 'tangent_directional', 'distance', 'amplitude',
    'distance_amplitude',
)

# Kept as a module-level alias for back-compat with callers that
# imported the 2D-specific name; same values as the shared constant.
VALID_NORMALIZE_2D = VALID_NORMALIZE_MODES

# What drives θ from the 1D input signal. Each option produces a
# monotonic or near-monotonic signal in [0, 1] that is then scaled
# by `theta_max * cycles_per_unit` to get θ.
VALID_INPUT_DRIVERS = ('position', 'arc_length', 'hilbert_envelope')


VALID_ANGULAR_FALLOFFS = ('cos_n', 'von_mises', 'raised_cosine')
VALID_DISTANCE_FALLOFFS = (
    'linear', 'gaussian', 'raised_cosine', 'inverse_square',
)

# Effective distance "scale" used by distance-based modes. 2.0 is the
# maximum 2D distance from any point on the unit-radius disc to an
# electrode seated on the unit circle — matches the current `d/2.0`
# normalization baked into _mode_distance / _mode_distance_amplitude.
_DISTANCE_FALLOFF_SCALE = 2.0


def _angular_falloff(delta_cos: np.ndarray, shape: str, n: float) -> np.ndarray:
    """Angular falloff from `delta_cos = cos(θ − θ_e)` (full [-1, 1]).

    `shape` selects the functional form; `n` is the width/selectivity
    exponent (same semantics as `sharpness` — higher = narrower lobe).

    All shapes produce values in [0, 1], peaking at delta_cos=1 (pen
    aligned with electrode) and approaching 0 as alignment goes away.

    - `cos_n` (default, current behavior): clip(cos, 0, 1) ** n.
      Hard cutoff at ±π/2: no activation behind the electrode.
    - `raised_cosine`: ((1 + cos) / 2) ** n. Peaks at 1 when aligned,
      reaches exactly 0 at ±π. Soft tails, no hard cutoff — a back-
      facing electrode gets a small but nonzero bleed.
    - `von_mises`: exp(n · (cos − 1)). Circular Gaussian. Peak 1 at
      alignment, decays exponentially as alignment loosens. Natural
      FWHM knob — doubling n roughly halves the half-power width.
    """
    n = max(float(n), 0.01)
    if shape == 'cos_n':
        return np.clip(delta_cos, 0.0, 1.0) ** n
    if shape == 'raised_cosine':
        return (0.5 * (1.0 + delta_cos)) ** n
    if shape == 'von_mises':
        return np.exp(n * (delta_cos - 1.0))
    raise ValueError(
        f"angular_falloff must be one of {VALID_ANGULAR_FALLOFFS}, "
        f"got {shape!r}")


def _distance_falloff(d: np.ndarray, shape: str, n: float,
                      scale: float = _DISTANCE_FALLOFF_SCALE) -> np.ndarray:
    """Distance-to-intensity falloff. `scale` is the characteristic
    distance each shape interprets as its knee / σ / radius.

    - `linear` (default, current): clip(1 − d/scale, 0, 1) ** n.
      Hard cutoff at d = scale.
    - `gaussian`: exp(−(d/scale)²) ** n. Bell, asymptotic tail.
    - `raised_cosine`: flat-top Hann. 1 at d=0, exactly 0 at d≥scale,
      zero slope at both endpoints.
    - `inverse_square`: 1/(1 + (d/scale)²) ** n. Physical-feel, long
      tail (0.5 at d=scale, 0.2 at 2·scale).
    """
    n = max(float(n), 0.01)
    scale = max(float(scale), 1e-9)
    if shape == 'linear':
        return np.clip(1.0 - d / scale, 0.0, 1.0) ** n
    if shape == 'gaussian':
        return np.exp(-(d * d) / (scale * scale)) ** n
    if shape == 'raised_cosine':
        base = np.where(d < scale,
                        0.5 * (1.0 + np.cos(np.pi * d / scale)),
                        0.0)
        return base ** n
    if shape == 'inverse_square':
        return (1.0 / (1.0 + (d / scale) ** 2)) ** n
    raise ValueError(
        f"distance_falloff must be one of {VALID_DISTANCE_FALLOFFS}, "
        f"got {shape!r}")


def _compute_driver_signal(input_y: np.ndarray, driver: str) -> np.ndarray:
    """Reparameterize the raw input into the signal that drives θ.

    Args:
        input_y: 1D array clipped to [0, 1].
        driver: One of VALID_INPUT_DRIVERS.
          - 'position' (default): identity — θ tracks input position.
          - 'arc_length': cumulative sum of |Δy|, normalized so the
            final sample lands at 1. Makes θ advance by distance
            traveled, timing-independent; one full sweep per unit of
            total path length. Feels the same on fast vs slow strokes
            that cover equal travel.
          - 'hilbert_envelope': normalized magnitude of the analytic
            signal (via Hilbert transform) of the demeaned input.
            Decouples θ from absolute position: θ is driven by the
            amplitude of oscillation instead. Quiet held positions
            give θ near the envelope floor; peaks of oscillation drive
            θ near 1.
    """
    if driver == 'position':
        return input_y
    if driver == 'arc_length':
        if len(input_y) < 2:
            return np.zeros_like(input_y)
        diffs = np.abs(np.diff(input_y))
        total = float(diffs.sum())
        if total < 1e-9:
            return np.zeros_like(input_y)
        cum = np.concatenate([[0.0], np.cumsum(diffs) / total])
        return np.clip(cum, 0.0, 1.0)
    if driver == 'hilbert_envelope':
        if len(input_y) < 4:
            return input_y.copy()
        try:
            from scipy.signal import hilbert
            demeaned = input_y - float(np.mean(input_y))
            analytic = hilbert(demeaned)
            env = np.abs(analytic)
        except Exception as e:
            print(f"[trochoid_spatial] hilbert_envelope failed ({e}); "
                  f"falling back to position driver.")
            return input_y
        peak = float(env.max())
        if peak < 1e-9:
            return np.zeros_like(input_y)
        return np.clip(env / peak, 0.0, 1.0)
    raise ValueError(
        f"input_driver must be one of {VALID_INPUT_DRIVERS}, got {driver!r}")


def _mode_directional(xn, yn, rn, electrode_angles_deg, sharpness,
                      angular_falloff='cos_n', distance_falloff='linear'):
    # `sharpness` here is always a list of length N (caller pre-resolves).
    path_angle = np.arctan2(yn, xn)
    out: Dict[str, np.ndarray] = {}
    for i, angle_deg in enumerate(electrode_angles_deg):
        ang = np.radians(angle_deg)
        delta_cos = np.cos(path_angle - ang)
        out[f'e{i + 1}'] = _angular_falloff(
            delta_cos, angular_falloff, sharpness[i])
    return out


def _mode_tangent_directional(xn, yn, rn, electrode_angles_deg, sharpness,
                              angular_falloff='cos_n',
                              distance_falloff='linear'):
    tx = np.gradient(xn)
    ty = np.gradient(yn)
    valid = np.sqrt(tx * tx + ty * ty) > 1e-12
    path_angle = np.arctan2(ty, tx)
    out: Dict[str, np.ndarray] = {}
    for i, angle_deg in enumerate(electrode_angles_deg):
        ang = np.radians(angle_deg)
        delta_cos = np.cos(path_angle - ang)
        intensity = _angular_falloff(
            delta_cos, angular_falloff, sharpness[i])
        out[f'e{i + 1}'] = np.where(valid, intensity, 0.0)
    return out


def _mode_distance(xn, yn, rn, electrode_angles_deg, sharpness,
                   angular_falloff='cos_n', distance_falloff='linear'):
    out: Dict[str, np.ndarray] = {}
    for i, angle_deg in enumerate(electrode_angles_deg):
        ang = np.radians(angle_deg)
        ex, ey = np.cos(ang), np.sin(ang)
        d = np.sqrt((xn - ex) ** 2 + (yn - ey) ** 2)
        out[f'e{i + 1}'] = _distance_falloff(
            d, distance_falloff, sharpness[i])
    return out


def _mode_amplitude(xn, yn, rn, electrode_angles_deg, sharpness,
                    angular_falloff='cos_n', distance_falloff='linear'):
    path_angle = np.arctan2(yn, xn)
    out: Dict[str, np.ndarray] = {}
    for i, angle_deg in enumerate(electrode_angles_deg):
        ang = np.radians(angle_deg)
        delta_cos = np.cos(path_angle - ang)
        out[f'e{i + 1}'] = rn * _angular_falloff(
            delta_cos, angular_falloff, sharpness[i])
    return out


def _mode_distance_amplitude(xn, yn, rn, electrode_angles_deg, sharpness,
                             angular_falloff='cos_n',
                             distance_falloff='linear'):
    """Proximity × reach: distance falloff weighted by normalized radius.

    Parallel to `amplitude` (which is direction × reach); here the
    angular term is replaced by the unit-circle-distance term.
    Emphasizes "how close is the pen to the electrode AND how far
    from the origin is it" — bright when the curve reaches close to
    an electrode, quiet when the curve stays near the origin or far
    from that electrode's post.
    """
    out: Dict[str, np.ndarray] = {}
    for i, angle_deg in enumerate(electrode_angles_deg):
        ang = np.radians(angle_deg)
        ex, ey = np.cos(ang), np.sin(ang)
        d = np.sqrt((xn - ex) ** 2 + (yn - ey) ** 2)
        prox = _distance_falloff(d, distance_falloff, sharpness[i])
        out[f'e{i + 1}'] = rn * prox
    return out


def _detect_lobes(rn):
    """Detect lobe peaks in rn over the sampled trajectory.

    Returns a list of (peak_idx, left_valley_idx, right_valley_idx).
    Same prominence rule as `trochoid_viewer._recompute_lobes`: at
    least 8% of the radius dynamic range. Falls back to the global
    maximum when no prominent peak survives the filter.
    """
    n = len(rn)
    if n < 5:
        return []
    r = np.asarray(rn, dtype=float)
    r_max = float(r.max())
    r_min = float(r.min())
    if r_max - r_min < 1e-9:
        return []
    min_prom = 0.08 * (r_max - r_min)
    d = np.diff(r)
    sign = np.sign(d)
    last = 1.0
    for i in range(len(sign)):
        if sign[i] == 0:
            sign[i] = last
        else:
            last = sign[i]
    peaks_idx = []
    valleys_idx = [0]
    for i in range(1, len(sign)):
        if sign[i - 1] > 0 and sign[i] < 0:
            peaks_idx.append(i)
        elif sign[i - 1] < 0 and sign[i] > 0:
            valleys_idx.append(i)
    valleys_idx.append(n - 1)
    lobes = []
    for p in peaks_idx:
        lv = max([v for v in valleys_idx if v < p], default=0)
        rv = min([v for v in valleys_idx if v > p], default=n - 1)
        prom = r[p] - max(r[lv], r[rv])
        if prom >= min_prom:
            lobes.append((p, lv, rv))
    if not lobes:
        lobes = [(int(np.argmax(r)), 0, n - 1)]
    return lobes


def _mode_lobe_gate(xn, yn, rn, electrode_angles_deg, sharpness,
                    angular_falloff='cos_n', distance_falloff='linear'):
    """Event-driven mapping: each electrode fires while the pen is
    inside a lobe whose peak angle is closest to that electrode's
    angle. Within a lobe the output is a smooth cosine bump from
    valley to valley, scaled by the lobe's peak radius and raised
    to `sharpness[i]` (higher → narrower pulse).

    Unlike the cosine-based modes this produces discrete pulses tied
    to the curve's geometry rather than continuous angular gradients.
    """
    n = len(xn)
    n_el = len(electrode_angles_deg)
    out = {f'e{i + 1}': np.zeros(n, dtype=float) for i in range(n_el)}
    lobes = _detect_lobes(rn)
    if not lobes:
        return out

    r_max = float(np.asarray(rn).max()) if n else 1.0
    if r_max < 1e-12:
        r_max = 1.0
    el_angs = np.radians([float(a) for a in electrode_angles_deg])

    for p, lv, rv in lobes:
        peak_angle = float(np.arctan2(yn[p], xn[p]))
        diffs = np.abs(((el_angs - peak_angle + np.pi)
                        % (2 * np.pi)) - np.pi)
        k = int(np.argmin(diffs))
        left = max(int(lv), 0)
        right = min(int(rv), n - 1)
        if right <= left:
            continue
        idx = np.arange(left, right + 1)
        bump = np.zeros(right - left + 1, dtype=float)
        mask_l = idx <= p
        mask_r = idx > p
        if mask_l.any():
            if p > left:
                u = (idx[mask_l] - left) / float(p - left)
                bump[mask_l] = np.sin(u * np.pi / 2.0) ** 2
            else:
                bump[mask_l] = 1.0
        if mask_r.any():
            if right > p:
                u = 1.0 - (idx[mask_r] - p) / float(right - p)
                bump[mask_r] = np.sin(u * np.pi / 2.0) ** 2
        amplitude = float(rn[p] / r_max)
        values = (amplitude * bump) ** sharpness[k]
        # Accumulate with max so overlapping lobes mapped to the
        # same electrode don't double up past 1.
        slice_view = out[f'e{k + 1}'][left:right + 1]
        np.maximum(slice_view, values, out=slice_view)
    return out


_MODE_FUNCS = {
    'directional': _mode_directional,
    'tangent_directional': _mode_tangent_directional,
    'distance': _mode_distance,
    'amplitude': _mode_amplitude,
    'distance_amplitude': _mode_distance_amplitude,
    'lobe_gate': _mode_lobe_gate,
}

# Default electrode angles in degrees (compass-like layout):
DEFAULT_ELECTRODE_ANGLES_DEG = (0.0, 90.0, 180.0, 270.0)


def suggest_electrode_angles(
    family: str,
    params: Dict[str, Any],
    n_electrodes: int = 4,
    mapping: str = 'directional',
    sharpness: float = 1.0,
    cycles_per_unit: float = 1.0,
    n_samples: int = 2000,
    rotation_step_deg: float = 2.5,
    spacing: str = 'uniform',
    angular_falloff: str = 'cos_n',
    distance_falloff: str = 'linear',
) -> Tuple[List[float], float]:
    """
    Search for the electrode angle layout that best separates the
    N output channels for a given curve + cycles configuration.

    Scoring: per-channel intensity vectors are computed over a dense
    0→1 input sweep; the objective is the sum of absolute off-diagonal
    Pearson correlations across channels. Lower = less crosstalk
    between channels = more distinct per-electrode output.

    Args:
        family, params, mapping, sharpness, cycles_per_unit: Same as
            `compute_spatial_intensities`. Use the same values that
            will be used downstream — a suggestion is only valid for
            the config it was computed against.
        n_electrodes: Number of electrodes (default 4).
        n_samples: Input sweep resolution for scoring. Higher = slower
            but more stable. 2000 is a reasonable default.
        rotation_step_deg: Granularity of the rotation search. 2.5°
            gives 144 candidates across 360°/N for N=4, cheap enough
            to run interactively.
        spacing: 'uniform' — keep equi-spaced layout, search only over
            the global rotation offset. For rotationally-symmetric
            curves this is a no-op; for curves like rose(k=5) with
            k not a multiple of N it can meaningfully reduce
            crosstalk.

    Returns:
        (best_angles_deg, best_score). `best_angles_deg` is a list of
        length `n_electrodes`. `best_score` is the objective value at
        that layout (lower is better).
    """
    if spacing != 'uniform':
        raise ValueError(
            f"only spacing='uniform' is supported, got {spacing!r}")
    n = int(n_electrodes)
    if n < 2:
        return [0.0] * n, 0.0
    sweep = np.linspace(0.0, 1.0, int(n_samples))
    # Rotation search range: the layout has n-fold symmetry under
    # rotation by 360/n, so only search that window.
    span = 360.0 / n
    n_steps = max(4, int(round(span / float(rotation_step_deg))))
    rotations = np.linspace(0.0, span, n_steps, endpoint=False)

    best_score = float('inf')
    best_angles: List[float] = [float(i * span) for i in range(n)]
    for rot in rotations:
        angles = tuple(float(rot + i * span) for i in range(n))
        try:
            out = compute_spatial_intensities(
                sweep, family, params,
                electrode_angles_deg=angles,
                mapping=mapping, sharpness=sharpness,
                cycles_per_unit=cycles_per_unit,
                angular_falloff=angular_falloff,
                distance_falloff=distance_falloff)
        except Exception as e:
            print(f"[suggest_angles] rotation {rot:.1f}° failed: {e}")
            continue
        stack = np.stack([out[f'e{i + 1}'] for i in range(n)], axis=0)
        # Drop channels that are flat (std=0) to avoid NaN in corrcoef.
        stds = stack.std(axis=1)
        valid = stds > 1e-9
        if valid.sum() < 2:
            continue
        stack_v = stack[valid]
        corr = np.corrcoef(stack_v)
        iu = np.triu_indices(corr.shape[0], k=1)
        score = float(np.sum(np.abs(corr[iu])))
        if score < best_score:
            best_score = score
            best_angles = [float(a % 360.0) for a in angles]
    return best_angles, best_score


def compute_field_grid(
    mapping: str,
    electrode_angles_deg: Tuple[float, ...] = DEFAULT_ELECTRODE_ANGLES_DEG,
    sharpness: float = 1.0,
    grid_res: int = 120,
    extent: float = 1.3,
    angular_falloff: str = 'cos_n',
    distance_falloff: str = 'linear',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the per-electrode field over a 2D grid in the unit-circle
    plane. Used by UI previews to visualize "where does the pen need to
    go to excite this electrode."

    For modes that depend only on position (`directional`, `distance`,
    `amplitude`, `distance_amplitude`) the field is well-defined per
    pixel. For mapping modes that depend on trajectory state
    (`tangent_directional`, `lobe_gate`) or user weights (`blend`),
    the function falls back to the `directional` field as a reference
    view — callers should treat the overlay as indicative in those
    cases.

    Args:
        mapping: Any value from VALID_MAPPINGS. Non-positional modes
            fall back to 'directional'.
        electrode_angles_deg: Compass angles for the electrodes.
        sharpness: Scalar exponent (applied uniformly across the grid
            regardless of per-electrode overrides — the grid is a
            schematic view).
        grid_res: Number of samples per axis. Total work is O(grid_res²).
        extent: Half-size of the axis-aligned square region to sample,
            in unit-radius coordinates. 1.3 matches the preview window.

    Returns:
        (xs, ys, field) where xs/ys are 1D coordinate arrays of length
        grid_res and field has shape (n_electrodes, grid_res, grid_res).
        Values are clipped to [0, 1].
    """
    xs = np.linspace(-extent, extent, grid_res)
    ys = np.linspace(-extent, extent, grid_res)
    X, Y = np.meshgrid(xs, ys)
    sharp = max(0.01, float(sharpness))
    n_el = len(electrode_angles_deg)
    field = np.zeros((n_el, grid_res, grid_res), dtype=float)

    positional = mapping in (
        'directional', 'distance', 'amplitude', 'distance_amplitude')
    effective = mapping if positional else 'directional'

    if effective in ('distance', 'distance_amplitude'):
        R = np.sqrt(X * X + Y * Y)
        for i, ang_deg in enumerate(electrode_angles_deg):
            a = np.radians(float(ang_deg))
            ex, ey = np.cos(a), np.sin(a)
            d = np.sqrt((X - ex) ** 2 + (Y - ey) ** 2)
            prox = _distance_falloff(d, distance_falloff, sharp)
            if effective == 'distance_amplitude':
                field[i] = np.clip(R, 0.0, 1.0) * prox
            else:
                field[i] = prox
    else:
        path_angle = np.arctan2(Y, X)
        R = np.sqrt(X * X + Y * Y)
        for i, ang_deg in enumerate(electrode_angles_deg):
            a = np.radians(float(ang_deg))
            delta_cos = np.cos(path_angle - a)
            cos_val = _angular_falloff(delta_cos, angular_falloff, sharp)
            if effective == 'amplitude':
                field[i] = np.clip(R, 0.0, 1.0) * cos_val
            else:
                field[i] = cos_val
    np.clip(field, 0.0, 1.0, out=field)
    return xs, ys, field


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
    blend_distance_amplitude: float = 0.0,
    electrode_gain=None,
    output_limiter_enabled: bool = False,
    output_limiter_threshold: float = 0.85,
    velocity_weight: Optional[np.ndarray] = None,
    electrode_solo=None,
    electrode_mute=None,
    input_driver: str = 'position',
    angular_falloff: str = 'cos_n',
    distance_falloff: str = 'linear',
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
            - 'distance_amplitude': distance × normalized radius —
              proximity weighted by reach. Parallel to 'amplitude' but
              the angular term is replaced by unit-circle proximity.
              Emphasizes "the curve reaches near this electrode" while
              staying quiet when the pen sits near the origin.
            - 'tangent_directional': cosine of the electrode angle vs
              the pen's instantaneous direction of travel (the tangent
              to the sampled path). Electrode lights when the pen is
              *moving toward* it, regardless of where it sits.
            - 'blend': weighted combination of the positional modes
              above, controlled by `blend_directional`,
              `blend_tangent_directional`, `blend_distance`,
              `blend_amplitude`, `blend_distance_amplitude`. Weights
              are applied raw (no internal normalization) — the user
              can over-drive past 1.0 or fade to silence with all
              zeros.
            - 'lobe_gate': event-driven. Peaks (lobes) of the
              trajectory's radius are detected and each one assigned
              to the electrode whose angle is closest to the lobe's
              peak angle. Inside an assigned lobe the electrode fires
              as a cosine bump (valley → peak → valley) scaled by
              peak radius; elsewhere it's silent. Discrete pulses
              rather than continuous-cosine flow.
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
        angular_falloff: Shape of the angular-domain falloff used by
            directional / tangent_directional / amplitude modes (and
            their contributions in blend). One of
            VALID_ANGULAR_FALLOFFS.
            - 'cos_n' (default): clip(cos(Δθ), 0, 1)^n — current
              behavior. Hard cutoff behind the electrode.
            - 'raised_cosine': ((1 + cos(Δθ))/2)^n. Soft tail around
              the full circle; small bleed even behind the electrode.
            - 'von_mises': exp(n·(cos−1)). Circular Gaussian; n gives
              a natural FWHM knob.
        distance_falloff: Shape of the distance-domain falloff used by
            distance / distance_amplitude modes (and their
            contributions in blend). One of VALID_DISTANCE_FALLOFFS.
            - 'linear' (default): clip(1 − d/scale, 0, 1)^n — current
              behavior. Hard cutoff at scale.
            - 'gaussian': exp(−(d/scale)²)^n. Smooth asymptotic tail.
            - 'raised_cosine': flat-top Hann, zero slope at d=0 and
              at d=scale; exactly 0 beyond.
            - 'inverse_square': 1/(1+(d/scale)²)^n. Long tail.
        input_driver: How the raw input signal maps to θ. One of
            VALID_INPUT_DRIVERS.
            - 'position' (default): θ tracks position linearly —
              strokes sweep the curve forward then backward.
            - 'arc_length': θ advances by cumulative travel distance.
              Monotonic (no back-sweeping), timing-independent — each
              unit of total path length produces one full θ sweep
              regardless of stroke shape or tempo.
            - 'hilbert_envelope': θ is driven by the amplitude envelope
              of the input (Hilbert transform of the demeaned signal,
              normalized). Held positions keep θ quiet; oscillation
              amplitude drives θ up. Decouples θ from absolute position.
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
    if input_driver not in VALID_INPUT_DRIVERS:
        raise ValueError(
            f"input_driver must be one of {VALID_INPUT_DRIVERS}, "
            f"got {input_driver!r}")
    if angular_falloff not in VALID_ANGULAR_FALLOFFS:
        raise ValueError(
            f"angular_falloff must be one of {VALID_ANGULAR_FALLOFFS}, "
            f"got {angular_falloff!r}")
    if distance_falloff not in VALID_DISTANCE_FALLOFFS:
        raise ValueError(
            f"distance_falloff must be one of {VALID_DISTANCE_FALLOFFS}, "
            f"got {distance_falloff!r}")

    input_y = np.asarray(input_y, dtype=float)
    input_y = np.clip(input_y, 0.0, 1.0)
    driver_signal = _compute_driver_signal(input_y, input_driver)

    # Drive the curve parameter from the input. theta_max keeps the sweep
    # extent consistent across families (butterfly traces over 12π naturally,
    # most others over 2π).
    theta_max = get_family_theta_max(family)
    effective_cycles = float(cycles_per_unit)
    if close_on_loop:
        # Round toward the nearest integer ≥ 1 so input=0 and input=1
        # land on the same curve point. Guarantees stroke-loop closure.
        # Note: only truly closes the loop when input_driver='position',
        # since arc_length/hilbert_envelope reparameterize the signal.
        effective_cycles = max(1.0, float(round(effective_cycles)))
    theta = theta_max * effective_cycles * driver_signal + float(theta_offset)
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
            'distance_amplitude': float(blend_distance_amplitude),
        }
        out: Dict[str, np.ndarray] = {
            f'e{i + 1}': np.zeros_like(xn)
            for i in range(len(electrode_angles_deg))
        }
        for mode_name, w in weights.items():
            if abs(w) < 1e-12:
                continue
            sub = _MODE_FUNCS[mode_name](
                xn, yn, rn, electrode_angles_deg, sharpness_list,
                angular_falloff=angular_falloff,
                distance_falloff=distance_falloff)
            for k in out:
                out[k] = out[k] + w * sub[k]
    else:
        out = _MODE_FUNCS[mapping](
            xn, yn, rn, electrode_angles_deg, sharpness_list,
            angular_falloff=angular_falloff,
            distance_falloff=distance_falloff)

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
    blend_distance_amplitude: float = 0.0,
    electrode_gain=None,
    output_limiter_enabled: bool = False,
    output_limiter_threshold: float = 0.85,
    input_driver: str = 'position',
    angular_falloff: str = 'cos_n',
    distance_falloff: str = 'linear',
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
        blend_distance_amplitude=blend_distance_amplitude,
        electrode_gain=electrode_gain,
        output_limiter_enabled=output_limiter_enabled,
        output_limiter_threshold=output_limiter_threshold,
        velocity_weight=_vw,
        electrode_solo=electrode_solo,
        electrode_mute=electrode_mute,
        input_driver=input_driver,
        angular_falloff=angular_falloff,
        distance_falloff=distance_falloff,
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
        'blend_distance_amplitude': 0.0,
        'input_driver': 'position',
        'angular_falloff': 'cos_n',
        'distance_falloff': 'linear',
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
