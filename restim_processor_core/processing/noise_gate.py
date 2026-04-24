"""Activity-based noise gate for funscript signals.

Computes a rolling peak-to-peak amplitude over a centered look-back
window. When the amplitude drops below `threshold`, the signal is
considered quiet and the gate closes — pulling the output toward a
rest level. Transitions are smoothed via asymmetric attack/release
time constants so the gate opens fast on new motion and closes
gradually after motion ends, avoiding clicks.

Applied as a pre-pipeline stage in processor._execute_pipeline so
trochoid quantization and every downstream file inherit the gated
signal. Preserves original timestamps — only `y` is modified.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

from funscript import Funscript


def _alpha_from_tau(tau_s, dt):
    if tau_s <= 0.0 or dt <= 0.0:
        return 1.0
    return 1.0 - float(np.exp(-dt / tau_s))


def _rolling_peak_to_peak(y, half):
    """Centered-window peak-to-peak on a uniform grid.

    For each index i, returns max(y[i-half:i+half+1]) - min(same).
    Plain loop — n at 50 Hz on a 10-minute clip is 30k, negligible
    against the rest of the pipeline.
    """
    n = len(y)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = y[lo:hi]
        out[i] = float(window.max() - window.min())
    return out


def _gate_envelope_from_p2p(p2p, threshold, attack_s, release_s, dt):
    """Soft-threshold `p2p` then pass through an asymmetric EMA.

    Returns the gate multiplier in [0, 1], same length as `p2p`.
    """
    edge = 0.1 * threshold
    lo_edge = max(0.0, threshold - edge)
    hi_edge = threshold + edge
    if hi_edge > lo_edge:
        gate_raw = np.clip(
            (p2p - lo_edge) / (hi_edge - lo_edge), 0.0, 1.0)
    else:
        gate_raw = (p2p >= threshold).astype(np.float64)

    alpha_open = _alpha_from_tau(attack_s, dt)
    alpha_close = _alpha_from_tau(release_s, dt)
    out = np.empty_like(gate_raw)
    out[0] = gate_raw[0]
    for i in range(1, len(gate_raw)):
        target = gate_raw[i]
        prev = out[i - 1]
        alpha = alpha_open if target > prev else alpha_close
        out[i] = prev + alpha * (target - prev)
    return out


def gate_uniform_signals_combined(
    signals,
    dt,
    threshold=0.05,
    window_s=0.5,
    attack_s=0.02,
    release_s=0.3,
    rest_level=0.5,
):
    """Gate multiple axes synchronously off their combined activity.

    For callers whose input is already on a uniform time grid (e.g.
    multi_script_loader after resample). Computes per-axis rolling
    peak-to-peak, takes the max across axes, builds a single gate
    envelope, and applies it identically to all inputs. Keeps the
    relative 3D position faithful — a quiet section collapses *all*
    axes toward rest together rather than warping the trajectory.

    Args:
        signals: list/tuple of 1D numpy arrays, equal length, each in
            [0, 1]. Constant-valued arrays are ignored in the combined
            activity metric (their p2p is 0).
        dt: timestep of the uniform grid in seconds.
        threshold, window_s, attack_s, release_s, rest_level: see
            `apply_noise_gate`.

    Returns:
        List of gated arrays, one per input axis (same order, same
        shape). Combined gate envelope is available via the caller
        running the same math; we don't return it to keep the API
        simple.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    arrays = [np.asarray(s, dtype=np.float64) for s in signals]
    n = len(arrays[0]) if arrays else 0
    for arr in arrays:
        if len(arr) != n:
            raise ValueError("all signals must be the same length")
    if n < 2:
        return [a.copy() for a in arrays]

    grid_hz = 1.0 / dt
    window_len = max(3, int(round(window_s * grid_hz)))
    half = window_len // 2

    # Per-axis rolling p2p; skip constant-fill axes (all-equal → p2p = 0).
    p2ps = []
    for arr in arrays:
        if np.all(arr == arr[0]):
            p2ps.append(np.zeros(n, dtype=np.float64))
        else:
            p2ps.append(_rolling_peak_to_peak(arr, half))
    combined = np.maximum.reduce(p2ps) if p2ps else np.zeros(n)

    gate = _gate_envelope_from_p2p(
        combined, threshold, attack_s, release_s, dt)
    out = []
    for arr in arrays:
        y_out = rest_level + gate * (arr - rest_level)
        y_out = np.clip(y_out, 0.0, 1.0)
        out.append(y_out)
    return out


def apply_noise_gate(
    funscript,
    threshold=0.05,
    window_s=0.5,
    attack_s=0.02,
    release_s=0.3,
    rest_level=0.5,
    grid_hz=50.0,
):
    """Gate quiet sections of a funscript toward `rest_level`.

    Args:
        funscript: input Funscript (y in [0, 1]).
        threshold: peak-to-peak amplitude below which the gate closes
            (0.0-0.5). 0.05 = 5% of full scale.
        window_s: width of the centered window used to estimate local
            peak-to-peak amplitude.
        attack_s: time constant for gate opening (seconds). Short so
            the gate snaps open on resumed motion.
        release_s: time constant for gate closing. Longer so the gate
            closes gradually, avoiding clicks.
        rest_level: target value for the signal when the gate is
            fully closed. 0.5 = neutral center.
        grid_hz: internal uniform sample rate for the rolling-window
            computation. 50 Hz is plenty for typical funscripts.

    Returns:
        New Funscript with the same x timestamps and gated y values.
        Metadata is preserved.
    """
    x = np.asarray(funscript.x, dtype=np.float64)
    y = np.asarray(funscript.y, dtype=np.float64)
    md = dict(funscript.metadata) if funscript.metadata else {}

    n = len(x)
    if n < 2:
        return Funscript(x.copy(), y.copy(), md)

    t0, t1 = float(x[0]), float(x[-1])
    duration = t1 - t0
    if duration <= 0.0:
        return Funscript(x.copy(), y.copy(), md)

    # Resample to a uniform grid so the rolling window is a fixed
    # index count (and so np.interp can round-trip the gate signal
    # back to the funscript's original timestamps).
    dt = 1.0 / float(grid_hz)
    n_grid = int(np.ceil(duration * grid_hz)) + 1
    t_grid = t0 + np.arange(n_grid) * dt
    if t_grid[-1] > t1:
        t_grid[-1] = t1
    y_grid = np.interp(t_grid, x, y)

    # Rolling peak-to-peak over a centered window
    window_len = max(3, int(round(window_s * grid_hz)))
    half = window_len // 2
    p2p = _rolling_peak_to_peak(y_grid, half)

    gate_smoothed = _gate_envelope_from_p2p(
        p2p, threshold, attack_s, release_s, dt)

    # Back to original timestamps and apply: signal is pulled toward
    # rest_level as the gate closes.
    gate_at_x = np.interp(x, t_grid, gate_smoothed)
    y_out = rest_level + gate_at_x * (y - rest_level)
    y_out = np.clip(y_out, 0.0, 1.0)

    return Funscript(x.copy(), y_out, md)
