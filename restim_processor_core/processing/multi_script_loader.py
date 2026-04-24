"""
Shared loader for multi-script (X/Y/Z) processing.

Takes up to three funscript paths and resamples them onto a common
uniform time grid so downstream projections (e.g. the linear-array
3D spatial mapping in trochoid_spatial) can treat them as one signal
(X(t), Y(t), Z(t)).

Missing slots are filled with a constant (default 0.5, the neutral
midpoint for a normalized funscript). That way 1- or 2-script inputs
still produce a usable signal: flat on the unfilled axes.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from funscript import Funscript
from processing.one_euro_filter import one_euro_filter
from processing.input_sharpen import sharpen_signal


def load_dof_scripts(
    path_x: Optional[str],
    path_y: Optional[str] = None,
    path_z: Optional[str] = None,
    path_rz: Optional[str] = None,
    hz: float = 50.0,
    fill_value: float = 0.5,
    max_samples: int = 50_000,
    *,
    input_smoothing_enabled: bool = False,
    input_smoothing_min_cutoff_hz: float = 1.0,
    input_smoothing_beta: float = 0.05,
    input_smoothing_d_cutoff_hz: float = 1.0,
    input_sharpen_enabled: bool = False,
    input_sharpen_pre_emphasis: float = 1.0,
    input_sharpen_saturation: float = 1.0,
    input_sharpen_pre_emphasis_cutoff_hz: float = 3.0,
    noise_gate_enabled: bool = False,
    noise_gate_threshold: float = 0.05,
    noise_gate_window_s: float = 0.5,
    noise_gate_attack_s: float = 0.02,
    noise_gate_release_s: float = 0.3,
    noise_gate_rest_level: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    Load up to four funscripts (XYZ + optional rz roll) and resample
    onto a shared uniform time grid.

    Args:
        path_x, path_y, path_z, path_rz: Paths to .funscript files.
            Any may be None or non-existent — those axes are filled
            with `fill_value`. rz is absolute roll around the shaft
            axis, 0–100 integer like any funscript; 50 = neutral.
        hz: Output sample rate (samples per second).
        fill_value: Value used for missing axes.
        max_samples: Hard cap on output length (safety).
        input_smoothing_enabled: When True, each axis is passed
            through a One-Euro adaptive low-pass filter AFTER the
            resample. Reduces high-frequency tracker jitter
            without the lag-ringing artifacts a plain EMA
            produces at heavy settings. OFF by default so
            existing workflows are unchanged.
        input_smoothing_min_cutoff_hz: Baseline cutoff at zero
            velocity (Hz). Lower = more smoothing at rest.
        input_smoothing_beta: Velocity-to-cutoff gain
            (dimensionless). Higher = more responsive to fast
            motion (filter becomes more transparent).
        input_smoothing_d_cutoff_hz: Low-pass cutoff applied to
            the filter's internal velocity estimate. Rarely
            needs tuning; kept at 1 Hz by default.

    Returns:
        (t, x, y, z, rz) — five numpy arrays of equal length. t spans
        from the earliest first-sample to the latest last-sample
        across the loaded scripts.

    Raises:
        ValueError: if no path resolves to a non-empty funscript.
    """
    paths = [path_x, path_y, path_z, path_rz]
    loaded = []
    for p in paths:
        fs = None
        if p and Path(p).is_file():
            try:
                fs = Funscript.from_file(p)
            except Exception as e:
                print(f"[multi_script_loader] failed to load {p}: {e}")
                fs = None
        loaded.append(fs)

    valid = [fs for fs in loaded if fs is not None and len(fs.x) > 0]
    if not valid:
        raise ValueError(
            "at least one of path_x / path_y / path_z / path_rz must "
            "be a valid non-empty funscript")

    t_min = min(float(fs.x[0]) for fs in valid)
    t_max = max(float(fs.x[-1]) for fs in valid)
    if t_max <= t_min:
        raise ValueError("input scripts span zero duration")

    dt = 1.0 / max(1e-3, float(hz))
    n = int(np.ceil((t_max - t_min) / dt)) + 1
    n = min(n, int(max_samples))
    t = np.linspace(t_min, t_max, n)

    axes = []
    for fs in loaded:
        if fs is None or len(fs.x) == 0:
            axes.append(np.full_like(t, float(fill_value)))
        else:
            axes.append(np.interp(
                t,
                np.asarray(fs.x, dtype=float),
                np.asarray(fs.y, dtype=float)))

    # Noise gate applied per-axis-combined AFTER resample but
    # BEFORE smoothing/sharpening. Gating first keeps the smoother
    # from spreading jitter across the gate boundary, and the
    # sharpener can't amplify noise that's already been squelched.
    # All axes are gated off a SINGLE envelope derived from the
    # per-axis maximum rolling peak-to-peak, so the 3D trajectory
    # stays faithful — a quiet section collapses X/Y/Z/rz toward
    # rest together instead of warping the position.
    if noise_gate_enabled and len(axes) > 0:
        from processing.noise_gate import gate_uniform_signals_combined
        dt = float(t[1] - t[0]) if len(t) >= 2 else 1.0 / float(hz)
        if dt > 0.0:
            axes = gate_uniform_signals_combined(
                axes, dt,
                threshold=noise_gate_threshold,
                window_s=noise_gate_window_s,
                attack_s=noise_gate_attack_s,
                release_s=noise_gate_release_s,
                rest_level=noise_gate_rest_level,
            )

    # One-Euro filter applied per axis AFTER resample. Filters
    # tracker jitter (Mask-Moments' flickering Otsu threshold,
    # LK sub-pixel noise, etc.) before the signal hits the spatial
    # projection. Adaptive cutoff means fast intentional motion
    # passes through near-transparently while idle micro-jitter
    # gets crushed — avoids the lag-ringing a heavy fixed EMA
    # produces at similar smoothing strength. Skipped entirely
    # when the axis is the constant-fill case (nothing to
    # smooth).
    if input_smoothing_enabled:
        # t is in seconds already (constructed above via
        # np.linspace over script timestamps which funscript-loader
        # reports in seconds — if the upstream convention changes
        # to ms, adjust here).
        for i, arr in enumerate(axes):
            # Constant-value arrays: no signal to filter.
            if np.all(arr == arr[0]):
                continue
            axes[i] = one_euro_filter(
                t, arr,
                min_cutoff_hz=input_smoothing_min_cutoff_hz,
                beta=input_smoothing_beta,
                d_cutoff_hz=input_smoothing_d_cutoff_hz,
            )

    # Input sharpener runs AFTER the smoother so it operates on
    # already-denoised signal (no risk of amplifying jitter). Pre-
    # emphasis + saturation reshape the signal to have Quad-like
    # transient energy + bimodal distribution, closing the feel
    # gap between smooth-tracker sources (Mask-Moments) and sharp-
    # tracker sources downstream at the spatial projection.
    if input_sharpen_enabled:
        for i, arr in enumerate(axes):
            if np.all(arr == arr[0]):
                continue
            axes[i] = sharpen_signal(
                arr,
                pre_emphasis=input_sharpen_pre_emphasis,
                saturation=input_sharpen_saturation,
                pre_emphasis_cutoff_hz=(
                    input_sharpen_pre_emphasis_cutoff_hz),
                sample_rate_hz=float(hz),
            )

    return t, axes[0], axes[1], axes[2], axes[3]


def load_xyz_triplet(
    path_x: Optional[str],
    path_y: Optional[str] = None,
    path_z: Optional[str] = None,
    hz: float = 50.0,
    fill_value: float = 0.5,
    max_samples: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible 3-axis wrapper around load_dof_scripts.

    Returns (t, x, y, z) — without the rz channel. New code should
    call load_dof_scripts directly.
    """
    t, x, y, z, _rz = load_dof_scripts(
        path_x, path_y, path_z, None,
        hz=hz, fill_value=fill_value, max_samples=max_samples)
    return t, x, y, z
