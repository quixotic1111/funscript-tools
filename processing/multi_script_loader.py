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


def load_dof_scripts(
    path_x: Optional[str],
    path_y: Optional[str] = None,
    path_z: Optional[str] = None,
    path_rz: Optional[str] = None,
    hz: float = 50.0,
    fill_value: float = 0.5,
    max_samples: int = 50_000,
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
