"""
Dynamic-range compressor for multi-electrode spatial output.

Purpose: flatten the loud-quiet amplitude cycles ("LFO feel") that
the distance-based spatial projection inherently produces when the
3D signal traces a cyclic path through the electrode array. Mask-
Moments-class trackers produce smooth centroid motion that maps to
smooth intensity envelopes — rhythmic "grabbing ↔ mild" swings at
stroke rate. Compression lifts the quiet half of that cycle toward
the loud half while preserving temporal structure (no smearing) and
relative spatial balance (quieter electrodes stay relatively quieter
than louder ones).

Design choices:
  * GLOBAL envelope (max across electrodes) drives gain reduction.
    Per-electrode compression would equalize channels and destroy
    the spatial distinction between near and far electrodes — not
    what we want. Global compression preserves per-frame electrode
    balance and only flattens the across-frames loudness cycle.
  * Hard-knee for simplicity — soft-knee buys little here because
    our input range is [0, 1] and the envelope typically swings
    across the whole range on every stroke cycle.
  * Peak envelope follower with asymmetric attack/release. Peak
    tracking (not RMS) because the intensity values are already a
    kind of instantaneous envelope from the projection — we don't
    need another smoothing stage before the compressor.
  * Optional makeup gain. Defaults to 1.0 (none) because with the
    near-unity peaks from the projection, makeup often causes
    clipping. Users can tune it upward if they want more apparent
    loudness after compression.
"""

import math
from typing import Dict

import numpy as np


def _attack_release_alpha(time_ms: float, sample_rate_hz: float) -> float:
    """EMA alpha for the given attack/release time at a sample rate.

    Returns 1.0 for instant response (time_ms <= 0), and a value in
    (0, 1) for finite time constants. The alpha is the WEIGHT of the
    new sample in each step: env = alpha * new + (1 - alpha) * prev.
    """
    if time_ms <= 0.0 or sample_rate_hz <= 0.0:
        return 1.0
    tau_s = time_ms / 1000.0
    return 1.0 - math.exp(-1.0 / max(tau_s * sample_rate_hz, 1e-6))


def _envelope_follower(
    x: np.ndarray,
    attack_alpha: float,
    release_alpha: float,
) -> np.ndarray:
    """Peak envelope follower with asymmetric attack/release.

    When the input rises above the current envelope, attack_alpha
    drives a fast chase upward. When it falls below, release_alpha
    drives a slower decay. Classic compressor sidechain envelope.
    """
    n = x.shape[0]
    env = np.empty(n, dtype=np.float64)
    e = 0.0
    for i in range(n):
        v = float(abs(x[i]))
        if v > e:
            e = attack_alpha * v + (1.0 - attack_alpha) * e
        else:
            e = release_alpha * v + (1.0 - release_alpha) * e
        env[i] = e
    return env


def compress_intensities(
    intensities: Dict[str, np.ndarray],
    *,
    threshold: float = 0.4,
    ratio: float = 3.0,
    attack_ms: float = 10.0,
    release_ms: float = 150.0,
    makeup: float = 1.0,
    sample_rate_hz: float = 50.0,
) -> Dict[str, np.ndarray]:
    """Apply global-envelope compression to a dict of electrode
    intensity time series.

    Args:
        intensities: dict of {electrode_name: np.ndarray in [0, 1]}.
            All arrays must be the same length. Unmodified inputs
            are tolerated (returned unchanged if 0 or 1 electrodes).
        threshold: Gain reduction engages when envelope exceeds
            this level. 0.4 is a reasonable starting point for
            intensity signals — most of the stroke rhythm happens
            in [0.2, 0.9].
        ratio: Compression ratio. For every unit above threshold in
            the input envelope, only 1/ratio makes it to the output.
            4:1 is standard "vocal-style" compression; 10:1 is
            heavy limiting territory.
        attack_ms: Envelope rise time constant. Shorter = clamps
            peaks faster (catches transients) but can cause pumping
            if too short. 10 ms is a musical default for 50 Hz
            signal. For the pipeline's 1 Hz LFO target, attack
            doesn't matter much — release does.
        release_ms: Envelope fall time. This is the knob that
            actually determines how fast quiet moments come back up
            to near-threshold. Shorter = more pumping; longer =
            smoother sustain. 150 ms gives a stroke-rate-appropriate
            recovery curve.
        makeup: Linear gain applied after compression. 1.0 = no
            makeup. Crank up if the output feels quieter after
            compression, but watch for clipping at high ratios.
        sample_rate_hz: Samples per second of the input arrays.
            Used to convert attack_ms / release_ms into EMA alphas.

    Returns:
        New dict of {electrode_name: compressed intensity array}.
        Original dict is not modified.
    """
    if not intensities:
        return dict(intensities)

    # Stack electrodes into a (n_electrodes, n_samples) matrix so
    # we can take a per-frame max for the sidechain source.
    keys = list(intensities.keys())
    mat = np.stack(
        [np.asarray(intensities[k], dtype=np.float64) for k in keys],
        axis=0,
    )
    n_elec, n_samples = mat.shape
    if n_samples == 0:
        return {k: np.asarray(intensities[k], dtype=np.float64).copy()
                for k in keys}

    # Global sidechain envelope = max across electrodes per frame.
    # Use max (not mean) because any one electrode hitting a peak
    # is what the user perceives as "grabby," and that peak should
    # drive gain reduction even when others are quieter.
    sidechain = np.max(mat, axis=0)

    attack_alpha = _attack_release_alpha(attack_ms, sample_rate_hz)
    release_alpha = _attack_release_alpha(release_ms, sample_rate_hz)
    env = _envelope_follower(sidechain, attack_alpha, release_alpha)

    # Gain reduction: hard-knee compressor.
    # Above threshold: output_level = threshold + (env - threshold) / ratio
    # Below threshold: output_level = env (no compression)
    # Gain = output_level / env, clipped to <= 1 (no expansion).
    threshold = max(1e-3, min(1.0, float(threshold)))
    ratio = max(1.0, float(ratio))
    safe_env = np.where(env > 1e-6, env, 1e-6)
    over = env - threshold
    compressed_level = np.where(
        over > 0.0,
        threshold + over / ratio,
        env,
    )
    gain = np.clip(compressed_level / safe_env, 0.0, 1.0)

    # Apply same gain to every electrode (global compression
    # preserves per-frame spatial balance).
    makeup = max(0.0, float(makeup))
    out_mat = mat * gain[np.newaxis, :] * makeup
    out_mat = np.clip(out_mat, 0.0, 1.0)

    return {keys[i]: out_mat[i, :].copy() for i in range(n_elec)}
