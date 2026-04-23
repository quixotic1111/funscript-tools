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

from typing import Dict, Tuple

import numpy as np


# Unit-cube diagonal — worst-case distance from any point in [0, 1]^3
# to any electrode. Raw distances are normalized against it so intensity
# lies in [0, 1] before the sharpness exponent.
_UNIT_CUBE_DIAG = float(np.sqrt(3.0))


def compute_linear_intensities_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_electrodes: int = 4,
    electrode_x: np.ndarray = None,
    center_yz: Tuple[float, float] = (0.5, 0.5),
    sharpness: float = 1.0,
    normalize: str = 'clamped',
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
        sharpness: Falloff exponent. 1 = linear, >1 = steeper.
        normalize: 'clamped' (default) returns raw per-electrode
            intensity; 'per_frame' divides each frame by its
            cross-electrode sum so each time step's energies sum to 1.

    Returns:
        Dict {'e1': array, ...} of length n_electrodes. Arrays share the
        length of x and lie in [0, 1].
    """
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
    sharpness = max(0.01, float(sharpness))

    out: Dict[str, np.ndarray] = {}
    for i in range(n):
        dx = x - float(electrode_x[i])
        dy = y - cy
        dz = z - cz
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        intensity = np.clip(1.0 - d / _UNIT_CUBE_DIAG, 0.0, 1.0) ** sharpness
        out[f'e{i + 1}'] = intensity

    if normalize == 'per_frame':
        stack = np.stack([out[f'e{i + 1}'] for i in range(n)], axis=0)
        totals = stack.sum(axis=0)
        safe = totals > 1e-9
        for i in range(n):
            out[f'e{i + 1}'] = np.where(safe, out[f'e{i + 1}'] / totals, 0.0)
    elif normalize != 'clamped':
        raise ValueError(
            f"normalize must be 'clamped' or 'per_frame', "
            f"got {normalize!r}")

    for key, arr in out.items():
        out[key] = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(out[key], 0.0, 1.0, out=out[key])

    return out
