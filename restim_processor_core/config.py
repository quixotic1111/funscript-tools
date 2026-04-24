import copy
import json
import os
from pathlib import Path
from typing import Dict, Any


DEFAULT_CONFIG = {
    "general": {
        "rest_level": 0.4,
        "ramp_up_duration_after_rest": 1.0,
        "speed_window_size": 5,
        "accel_window_size": 3
    },
    "speed": {
        "interpolation_interval": 0.1,
        "normalization_method": "max"
    },
    "alpha_beta_generation": {
        "points_per_second": 25,
        "algorithm": "top-right-left",
        "min_distance_from_center": 0.1,
        "speed_threshold_percent": 50,
        "direction_change_probability": 0.1,
        "min_stroke_amplitude": 0.0,
        "point_density_scale": 1.0
    },
    "prostate_generation": {
        # Prostate alpha/beta generation is the heaviest stage in the
        # pipeline (~73% of runtime on a typical Spatial 3D triplet).
        # Default to off; users who want prostate files can enable it
        # in their local config.json. Turn it back on here if you're
        # restoring the historical default behaviour.
        "generate_prostate_files": False,
        "generate_from_inverted": True,
        "algorithm": "tear-shaped",
        "points_per_second": 25,
        "min_distance_from_center": 0.5
    },
    "frequency": {
        "pulse_freq_min": 0.40,
        "pulse_freq_max": 0.95,
        "frequency_ramp_combine_ratio": 2,
        "pulse_frequency_combine_ratio": 3,
        "direction_bias": 0.0,
        "direction_polarity": "up_higher",
        "direction_smoothing_s": 0.3
    },
    "volume": {
        "volume_ramp_combine_ratio": 20.0,
        "prostate_volume_multiplier": 1.5,
        "prostate_rest_level": 0.7,
        "ramp_percent_per_hour": 15
    },
    "pulse": {
        "pulse_width_min": 0.1,
        "pulse_width_max": 0.45,
        "pulse_width_combine_ratio": 3,
        "beta_mirror_threshold": 0.5,
        "pulse_rise_min": 0.00,
        "pulse_rise_max": 0.80,
        "pulse_rise_combine_ratio": 2
    },
    "advanced": {
        "enable_pulse_frequency_inversion": False,
        "enable_volume_inversion": False,
        "enable_frequency_inversion": False
    },
    "noise_gate": {
        # Activity-based gate applied to main_funscript BEFORE every
        # other pipeline stage. Computes a rolling peak-to-peak over
        # `window_s` seconds; when p2p < `threshold`, the signal is
        # pulled toward `rest_level` (smoothed by attack/release time
        # constants so the gate doesn't click).
        "enabled": False,
        "threshold": 0.05,
        "window_s": 0.5,
        "attack_s": 0.02,
        "release_s": 0.3,
        "rest_level": 0.5
    },
    "trochoid_quantization": {
        "enabled": False,
        "n_points": 23,
        "projection": "radius",
        "family": "hypo",
        "deduplicate_holds": False,
        "params_by_family": {
            "hypo": {"R": 5.0, "r": 3.0, "d": 2.0},
            "epi": {"R": 5.0, "r": 3.0, "d": 2.0},
            "rose": {"a": 1.0, "k": 5.0},
            "lissajous": {"A": 1.0, "B": 1.0, "a": 3.0, "b": 2.0,
                          "delta": 1.5708},
            "butterfly": {"scale": 1.0},
            "superformula": {"a": 1.0, "b": 1.0, "m": 6.0,
                             "n1": 1.0, "n2": 7.0, "n3": 8.0},
            "custom": {"x_expr": "sin(3*t)", "y_expr": "cos(2*t)"}
        }
    },
    "trochoid_spatial": {
        "enabled": False,
        "family": "hypo",
        "mapping": "directional",
        "sharpness": 1.0,
        "cycles_per_unit": 1.0,
        "normalize": "clamped",
        "theta_offset": 0.0,
        "close_on_loop": False,
        "smoothing_enabled": False,
        "smoothing_min_cutoff_hz": 1.0,
        "smoothing_beta": 0.05,
        "blend_directional": 0.0,
        "blend_tangent_directional": 0.0,
        "blend_distance": 0.0,
        "blend_amplitude": 0.0,
        "electrode_gain": [1.0, 1.0, 1.0, 1.0],
        "output_limiter_enabled": False,
        "output_limiter_threshold": 0.85,
        "velocity_weight_enabled": False,
        "velocity_weight_floor": 0.0,
        "velocity_weight_response": 1.0,
        "velocity_weight_smoothing_hz": 3.0,
        "velocity_weight_normalization_percentile": 0.99,
        "velocity_weight_gate_threshold": 0.05,
        "electrode_angles_deg": [0.0, 90.0, 180.0, 270.0],
        "params_by_family": {
            "hypo": {"R": 5.0, "r": 3.0, "d": 2.0},
            "epi": {"R": 5.0, "r": 3.0, "d": 2.0},
            "rose": {"a": 1.0, "k": 5.0},
            "lissajous": {"A": 1.0, "B": 1.0, "a": 3.0, "b": 2.0,
                          "delta": 1.5708},
            "butterfly": {"scale": 1.0},
            "superformula": {"a": 1.0, "b": 1.0, "m": 6.0,
                             "n1": 1.0, "n2": 7.0, "n3": 8.0},
            "custom": {"x_expr": "sin(3*t)", "y_expr": "cos(2*t)"}
        }
    },
    "spatial_3d_linear": {
        # When enabled, the batch drop zone is reinterpreted: the first
        # three dropped scripts become X, Y, Z of a single 3D signal,
        # and the processor emits one set of e1..e4 funscripts produced
        # by projecting the 3D signal onto a straight-line electrode
        # array. Defaults match the Animation Viewer's Linear 3D
        # tuning-panel defaults so previews and processor output agree.
        "enabled": False,
        "n_electrodes": 4,
        "sharpness": 1.0,
        "normalize": "clamped",   # or "per_frame" / "energy_preserve"
        "center_yz": [0.5, 0.5],  # shaft line position in Y, Z
        # Per-axis weights applied inside the distance calc:
        #   d = sqrt(dx² + (y_weight · dy)² + (z_weight · dz)²)
        # 1.0 each (default) preserves the historic behavior where Y
        # and Z collapse into a shared radial proximity (rotation-
        # symmetric). Different values break that symmetry so Y and
        # Z behave as independent physical axes. 0 on an axis removes
        # it entirely from the projection — handy when a tracker
        # channel isn't meaningful (1D via 0,0; 2D via 0 on one).
        "y_weight": 1.0,
        "z_weight": 1.0,
        # Per-electrode X positions along the shaft (0 = base, 1 = tip).
        # 4 slots even when n_electrodes = 3 so the UI stays populated;
        # processor reads only the first n entries. Default matches
        # np.linspace(0.1, 0.9, 4). Match these to your device's
        # physical electrode spacing when it isn't evenly distributed.
        "electrode_x_positions": [0.1, 0.367, 0.633, 0.9],
        # Distance-to-intensity falloff shape. Pairs with
        # `falloff_width` which scales the characteristic knee /
        # sigma / radius:
        #   linear          — hard-edge 1-d/(w·diag) (legacy default).
        #   gaussian        — smooth bell, asymptotic tail.
        #   raised_cosine   — flat peak + zero-slope cutoff.
        #   inverse_square  — physical-feel, slow tail.
        # Sharpness still applies as a post-exponent so it keeps its
        # meaning across shapes.
        "falloff_shape": "linear",
        "falloff_width": 1.0,
        # Per-electrode sharpness override. When `enabled` is True,
        # the kernel uses the 4 values in `per_electrode_sharpness`
        # instead of the scalar `sharpness` above. Useful to
        # accentuate one electrode (e.g., center-heavy) while
        # keeping the others softer. Values < 0.01 are floored.
        "per_electrode_sharpness_enabled": False,
        "per_electrode_sharpness": [1.0, 1.0, 1.0, 1.0],
        # DAW-style solo/mute for per-electrode listening during
        # tuning. 4 bools each. Mute silences that channel; Solo on
        # any channel restricts output to soloed channels only. Both
        # persist in the config so processor runs respect them —
        # use the "Clear S/M" UI button to reset before saving a
        # variant meant for shipping.
        "electrode_solo": [False, False, False, False],
        "electrode_mute": [False, False, False, False],
    },
    # Spatial 3D Curve — third projector alongside trochoid_spatial
    # and spatial_3d_linear. Takes a single 1D main input, parameterizes
    # a 3D curve (helix / trefoil / torus knot / 3D Lissajous / spherical
    # spiral), and projects each (x, y, z) onto N electrodes arranged
    # in 3D (tetrahedral default). Mutually exclusive with trochoid
    # and wave modes.
    "spatial_3d_curve": {
        "enabled": False,
        "family": "helix",
        "n_electrodes": 4,
        "electrode_arrangement": "tetrahedral",
        "sharpness": 1.0,
        "cycles_per_unit": 1.0,
        "theta_offset": 0.0,
        "close_on_loop": False,
        "normalize": "clamped",
        "falloff_shape": "linear",
        "falloff_width": 1.0,
        # Per-family params — edit these in the UI, they persist here.
        "params_by_family": {
            "helix": {"r": 0.6, "h": 1.2, "turns": 3.0},
            "trefoil_knot": {"scale": 0.25},
            "torus_knot": {"R": 1.0, "r": 0.4, "p": 2.0, "q": 3.0,
                           "scale": 0.4},
            "lissajous_3d": {"A": 1.0, "B": 1.0, "C": 1.0,
                             "a": 3.0, "b": 2.0, "c": 5.0,
                             "phi": 1.5708, "psi": 0.0, "scale": 0.7},
            "spherical_spiral": {"c": 5.0, "scale": 0.85},
        },
        # Shared shaping knobs (same semantics as trochoid / linear 3D):
        "output_smoothing_enabled": False,
        "output_smoothing_min_cutoff_hz": 1.0,
        "output_smoothing_beta": 0.05,
        "electrode_gain": [1.0, 1.0, 1.0, 1.0],
        "output_limiter_enabled": False,
        "output_limiter_threshold": 0.85,
        "velocity_weight_enabled": False,
        "velocity_weight_floor": 0.0,
        "velocity_weight_response": 1.0,
        "velocity_weight_smoothing_hz": 3.0,
        "velocity_weight_normalization_percentile": 0.99,
        "velocity_weight_gate_threshold": 0.05,
        "electrode_solo": [False, False, False, False],
        "electrode_mute": [False, False, False, False],
        # Flat-default values for parameter channels the device expects
        # to exist. Emitted as 2-point funscripts (start + end, same
        # value) so restim/playback has a valid file even though the
        # 3D pipeline doesn't derive per-frame modulation for these.
        # Users who want richer modulation should hand-author their own
        # frequency/pulse_* scripts and place them next to the outputs.
        "default_frequency": 0.5,
        "default_pulse_frequency": 0.5,
        "default_pulse_width": 0.5,
        "default_pulse_rise_time": 0.5,
        # Blend the flat `default_frequency` with per-frame |v| (speed_y).
        # 0.0 = flat default (matches prior behavior), 1.0 = fully driven
        # by |v|. Intermediate values linearly interpolate. Pulse shape
        # channels stay flat. Revisit once we've heard the output.
        "frequency_speed_mix": 0.0,
        # Release envelope on `speed_y` (the signal fueling
        # frequency_speed_mix). Asymmetric leaky integrator: instant
        # attack, exponential decay toward 0 when motion slows. Gives
        # intensity a natural tail instead of snapping dead at every
        # pause. τ in seconds; 0.0 = no decay (previous behavior);
        # 0.3 = ~37% remaining 300 ms after motion stops; 1.0 = long
        # hold. Only active when frequency_speed_mix > 0.
        "release_tau_s": 0.0,
        # Minimum value for speed_y after the release envelope. Acts
        # like a rest-level floor on the motion-derived carrier so the
        # device never drops fully quiet during pauses. 0.0 = off
        # (previous behavior — speed_y can decay to 0); 0.3 = always
        # at least 30% of full speed-driven intensity. Only audible
        # when frequency_speed_mix > 0.
        "speed_floor": 0.0,
        # One-Euro adaptive low-pass applied to the raw X/Y/Z/rz
        # input signals AFTER the resample to 50 Hz, BEFORE the
        # spatial projection. Kills high-frequency tracker jitter
        # (Mask-Moments Otsu flicker, LK sub-pixel noise, …) so
        # the electrodes don't receive noisy derivatives. Off by
        # default — enable when raw tracker output is visibly
        # jumpy on the output side. Start with defaults; lower
        # min_cutoff_hz if still jittery at rest, raise beta if
        # the filter feels laggy during fast intentional motion.
        # Activity-based noise gate applied per-axis-combined AFTER
        # the resample to 50 Hz but BEFORE input_smoothing and
        # input_sharpen. When the combined (max across X/Y/Z/rz)
        # rolling peak-to-peak falls below `threshold`, all axes are
        # pulled together toward `rest_level` (keeping 3D trajectory
        # coherent rather than warping it). Smoothed by asymmetric
        # attack/release so transitions don't click. Off by default.
        "noise_gate": {
            "enabled": False,
            "threshold": 0.05,
            "window_s": 0.5,
            "attack_s": 0.02,
            "release_s": 0.3,
            "rest_level": 0.5,
        },
        "input_smoothing": {
            "enabled": False,
            "min_cutoff_hz": 1.0,
            "beta": 0.05,
            "d_cutoff_hz": 1.0,
        },
        # Output smoothing — One-Euro adaptive low-pass applied to the
        # electrode intensity outputs AFTER the cross-electrode
        # normalize, inside compute_linear_intensities_3d. Complements
        # input_smoothing (which cleans X/Y/Z BEFORE projection); this
        # cleans E1..EN AFTER. Kills coil-ramp-rate discontinuities
        # without adding lag on genuine motion pulses. Off by default —
        # enable when the electrode outputs feel clicky on device.
        "output_smoothing": {
            "enabled": False,
            "min_cutoff_hz": 1.0,
            "beta": 0.05,
        },
        # Per-electrode multiplicative gain / trim. Applied AFTER
        # output_smoothing, BEFORE the final [0, 1] clip. Four slots
        # for 4-electrode devices; extra entries beyond n_electrodes
        # are ignored, missing trailing entries default to 1.0.
        # Typical range 0.0 (mute) to 2.0 (double, will clip at 1.0).
        # Use for physical-device channel balancing when one coil
        # runs hot or cold relative to the others.
        "electrode_gain": [1.0, 1.0, 1.0, 1.0],
        # Soft-knee tanh limiter applied AFTER electrode_gain, BEFORE
        # the final [0, 1] clip. Rounds off peaks above `threshold`
        # so gains > 1 or energy_preserve overshoots get compressed
        # smoothly instead of hard-clipped. Pair with electrode_gain
        # > 1 or with aggressive normalize modes to avoid crunchy
        # clipping artifacts.
        "output_limiter": {
            "enabled": False,
            "threshold": 0.85,
        },
        # Velocity-weighted intensity. Per-frame [0, 1] gate derived
        # from |d(X,Y,Z)/dt| magnitude. Applied AFTER output_smoothing,
        # BEFORE electrode_gain. Holds go quiet, fast motion stays at
        # full intensity. Good for "touch-while-moving" feel. Off by
        # default (preserves existing steady-state output on holds).
        #   floor: minimum weight when motionless (0 = fully silent,
        #          0.3 = 30% on holds).
        #   response: exponent on the normalized speed (1 = linear,
        #             higher = more aggressive).
        #   smoothing_hz: low-pass cutoff on the raw velocity so single-
        #                 sample spikes don't dominate.
        #   normalization_percentile: which quantile of the filtered
        #                             speed defines "full motion" so
        #                             one-off spikes don't flatten the
        #                             dynamic range. 0.99 typical.
        "velocity_weight": {
            "enabled": False,
            "floor": 0.0,
            "response": 1.0,
            "smoothing_hz": 3.0,
            "normalization_percentile": 0.99,
            # Hard cutoff below this normalized-speed level so
            # tracker-noise micro-velocity doesn't leak through as
            # "light touch" when floor = 0. 0.05 = 5% of peak speed,
            # conservative. Raise toward 0.15 for noisy trackers; 0
            # disables the gate entirely.
            "gate_threshold": 0.05,
        },
        # Input sharpener applied per-axis AFTER resample and
        # AFTER input_smoothing, BEFORE the spatial projection.
        # Two stages: pre-emphasis (unsharp-mask high-frequency
        # boost) followed by saturation (tanh soft-clip toward
        # [0, 1] extremes). Designed to close the gap between
        # smooth-tracker sources (Mask-Moments mask centroid) and
        # sharp-tracker sources (Quad's rigid-body points) — adds
        # the transient energy + bimodal distribution that reads
        # as "punchy" in the downstream projection. Off by
        # default; enable when the input signal is a low-pass
        # version of the motion you want (smooth trackers, heavily
        # EMA'd sources).
        "input_sharpen": {
            "enabled": False,
            "pre_emphasis": 1.0,
            "saturation": 1.0,
            "pre_emphasis_cutoff_hz": 3.0,
        },
        # Dynamic-range compressor on the electrode intensities.
        # Applied AFTER the spatial projection, BEFORE the output
        # Butterworth smoothing. Flattens the "mild ↔ grabbing"
        # loudness cycles that distance-based projection produces
        # when a smooth centroid traces a cyclic path through the
        # electrode array. Global-envelope (max across electrodes
        # drives gain reduction applied uniformly) so the per-
        # frame spatial balance between channels is preserved —
        # only the cross-frames loudness cycle gets compressed.
        "compression": {
            "enabled": False,
            "threshold": 0.4,
            "ratio": 3.0,
            "attack_ms": 10.0,
            "release_ms": 150.0,
            "makeup": 1.0,
        },
        # Geometric mapping: optionally blend the flat `default_pulse_*`
        # values with per-frame signals derived from the 3D geometry.
        # Each mix is 0.0 = flat default (matches prior behavior) to
        # 1.0 = fully geometry-driven. Sources:
        #   pulse_width_radial_mix — radial distance from shaft axis
        #     (Y,Z plane distance from `center_yz`), normalized so
        #     the YZ-corner maps to 1.
        #   pulse_rise_azimuth_mix — (cos(atan2(z-cz, y-cy)) + 1) / 2,
        #     smooth and wrap-free (sign info collapses, but rise-time
        #     is symmetric anyway).
        #   pulse_frequency_vradial_mix — dr/dt, percentile-normalized
        #     and centered at 0.5 (outward = >0.5, inward = <0.5).
        # All off by default. Enable one at a time on device.
        "geometric_mapping": {
            "pulse_width_radial_mix": 0.0,
            "pulse_rise_azimuth_mix": 0.0,
            "pulse_frequency_vradial_mix": 0.0,
            "vradial_normalization_percentile": 0.99,
            # 4-DoF roll modulator (requires a .rz.funscript in the
            # drop). Drives pulse_frequency via angular velocity
            # dω/dt of the absolute roll signal, normalized the same
            # way as dr/dt. When both vradial_mix and omega_mix are
            # > 0, the two modulations sum into pulse_frequency and
            # the result is clipped to [0, 1] — wobble and twist
            # both contribute.
            "pulse_frequency_omega_mix": 0.0,
            "omega_normalization_percentile": 0.99,
            # Symmetric EMA smoothing on the three geometric source
            # signals (radial, azimuth, dr/dt) before they blend into
            # the pulse channels. Kills jitter from small wobbles
            # without adding phase lag. τ in seconds; 0.0 = no
            # smoothing (instant); 0.1 = ~100 ms settling; 0.3 = calm.
            "hold_tau_s": 0.0,
        },
        # EXPERIMENTAL — reverb-analog effects at envelope rate. Not
        # audio; "reverb" means "sum delayed + attenuated copies back
        # into the signal." Four effects, each off by default. Run
        # after smoothing, before dedup (so dedup doesn't collapse
        # the reverb tails). Cannot add energy to a dead signal.
        #
        # Safety: feedback coefficients are clamped to < 0.95 to
        # prevent runaway. Outputs are clipped to [0, 1].
        "reverb": {
            "enabled": False,
            # 1. Single-tap feedback delay on volume_y. Produces
            #    discrete echoes; feedback > 0.5 self-sustains.
            "volume_tail": {
                "mix": 0.0,        # 0=dry, 1=fully-wet
                "delay_ms": 200.0,
                "feedback": 0.4,
            },
            # 2. FIR multi-tap comb on volume_y. Incommensurate delays
            #    give a dense Schroeder-like tail with no single
            #    echo audible. Unconditionally stable (no feedback).
            "volume_multitap": {
                "mix": 0.0,
                "delays_ms": [83.0, 127.0, 191.0, 307.0],
                "gains":     [0.40, 0.30, 0.22, 0.15],
            },
            # 3. Cross-electrode bleed: each E gets a delayed copy
            #    of its neighbors' envelopes summed in. Creates a
            #    spatial "movement through the array" even when the
            #    source position is stationary. No audio equivalent.
            "cross_electrode": {
                "mix": 0.0,
                "delay_ms": 100.0,
                "feedback": 0.0,   # per-electrode self-sustain
            },
            # 4. Single-tap feedback delay on the per-frame pulse_width
            #    signal. Only audible when PW × radial mix > 0
            #    (otherwise pulse_width is a flat 2-point funscript).
            "pulse_width_tail": {
                "mix": 0.0,
                "delay_ms": 150.0,
                "feedback": 0.3,
            },
        },
        # Volume ramp: linear rise across the whole clip, multiplied
        # into the max-E envelope. `ramp_percent_total` is the total
        # percent increase from clip start to clip end — e.g. 40 means
        # the clip opens at 60% and rises linearly to 100% before a
        # fade-out on the last sample. Short-clip friendly (unlike the
        # 1D pipeline's `volume.ramp_percent_per_hour` which is rate-
        # calibrated for multi-hour sessions and barely moves on short
        # previews). Set to 0 to disable the ramp entirely.
        "ramp_percent_total": 40.0,
        # Speed normalization: divide |v|(t) by the (99th-percentile,
        # unclipped-to-avoid-single-spike-dominance) value before
        # clipping into [0, 1]. Higher = less aggressive normalization
        # (raw speed stays closer to its absolute magnitude); 1.0 =
        # normalize to the peak. Clip always happens at [0, 1].
        "speed_normalization_percentile": 0.99,
        # Butterworth low-pass smoothing applied to E1..En only.
        # Reduces high-frequency transitions that manifest as device
        # flicker. OFF by default — enable explicitly once you've
        # landed on a cutoff that removes flicker without dulling
        # the intended dynamics.
        "smoothing": {
            "enabled": False,
            "cutoff_hz": 8.0,
            "order": 2,
        },
        # Deduplicate hold-runs in E1..En after smoothing. Samples in
        # the interior of a constant-within-tolerance run are dropped
        # (first + last are kept) so the device's linear interp
        # doesn't slope across the held window. OFF by default.
        "deduplicate_holds": {
            "enabled": False,
            "tolerance": 0.005,  # ~0.5 % of full scale
        },
    },
    "traveling_wave": {
        "enabled": False,
        "direction": "bounce",
        "envelope_mode": "input",
        "wave_speed_hz": 1.0,
        "wave_width": 0.18,
        "speed_mod": 0.0,
        "sharpness": 1.0,
        "velocity_window_s": 0.10,
        "noise_gate": 0.10,
        "exclusive": False,
        "electrode_positions": [0.85, 0.65, 0.45, 0.25]
    },
    "variants": {
        "active": "A",
        "slots": {
            "A": {"label": "A", "enabled": True, "config": {}},
            "B": {"label": "B", "enabled": False, "config": {}},
            "C": {"label": "C", "enabled": False, "config": {}},
            "D": {"label": "D", "enabled": False, "config": {}}
        }
    },
    "file_management": {
        "mode": "local",  # "local" or "central"
        "central_folder_path": "",
        "create_backups": True,
        "zip_output": False
    },
    "options": {
        "normalize_volume": True,
        "delete_intermediary_files": True,
        "overwrite_existing_files": False
    },
    "ui": {
        "dark_mode": False
    },
    "positional_axes": {
        "mode": "motion_axis",  # kept for backward compat; use generate_legacy/generate_motion_axis flags instead
        "generate_legacy": True,       # Motion Axis (3P) tab: generate legacy alpha/beta scripts
        "generate_motion_axis": True,  # Motion Axis (4P) tab: generate E1-E4 scripts
        "phase_shift": {
            "enabled": False,
            "delay_percentage": 10.0,  # Percentage of segment duration (0-100)
            "min_segment_duration": 0.25  # Minimum time between extremes in seconds
        },
        "motion_axis_phase_shift": {
            "enabled": False,
            "delay_percentage": 10.0,
            "min_segment_duration": 0.25
        },
        "e1": {
            "enabled": True,
            "curve": {
                "name": "Linear",
                "description": "Direct 1:1 mapping",
                "control_points": [(0.0, 0.0), (1.0, 1.0)]
            },
            "smoothing": {"enabled": False, "cutoff_hz": 8.0, "order": 2}
        },
        "e2": {
            "enabled": True,
            "curve": {
                "name": "Ease In",
                "description": "Gradual start, strong finish",
                "control_points": [(0.0, 0.0), (0.5, 0.2), (1.0, 1.0)]
            },
            "smoothing": {"enabled": False, "cutoff_hz": 8.0, "order": 2}
        },
        "e3": {
            "enabled": True,
            "curve": {
                "name": "Ease Out",
                "description": "Strong start, gradual finish",
                "control_points": [(0.0, 0.0), (0.5, 0.8), (1.0, 1.0)]
            },
            "smoothing": {"enabled": False, "cutoff_hz": 8.0, "order": 2}
        },
        "e4": {
            "enabled": True,
            "curve": {
                "name": "Bell Curve",
                "description": "Emphasis on middle range",
                "control_points": [(0.0, 0.0), (0.25, 0.3), (0.5, 1.0), (0.75, 0.3), (1.0, 0.0)]
            },
            "smoothing": {"enabled": False, "cutoff_hz": 8.0, "order": 2}
        }
    }
}

# Parameter validation ranges
PARAMETER_RANGES = {
    "general": {
        "rest_level": (0.0, 1.0),
        "ramp_up_duration_after_rest": (0.0, 10.0),
        "speed_window_size": (1, 30),
        "accel_window_size": (1, 10)
    },
    "speed": {
        "interpolation_interval": (0.01, 1.0)
    },
    "alpha_beta_generation": {
        "points_per_second": (1, 100),
        "min_distance_from_center": (0.1, 0.9),
        "speed_threshold_percent": (0, 100),
        "direction_change_probability": (0.0, 1.0),
        "min_stroke_amplitude": (0.0, 1.0),
        "point_density_scale": (0.25, 2.0)
    },
    "prostate_generation": {
        "points_per_second": (1, 100),
        "min_distance_from_center": (0.3, 0.9)
    },
    "frequency": {
        "pulse_freq_min": (0.0, 1.0),
        "pulse_freq_max": (0.0, 1.0),
        "frequency_ramp_combine_ratio": (1, 10),
        "pulse_frequency_combine_ratio": (1, 10),
        "direction_bias": (0.0, 0.5),
        "direction_smoothing_s": (0.0, 2.0)
    },
    "volume": {
        "volume_ramp_combine_ratio": (10.0, 40.0),
        "prostate_volume_multiplier": (1.0, 3.0),
        "prostate_rest_level": (0.0, 1.0),
        "ramp_percent_per_hour": (0, 40)
    },
    "pulse": {
        "pulse_width_min": (0.0, 1.0),
        "pulse_width_max": (0.0, 1.0),
        "pulse_width_combine_ratio": (1, 10),
        "beta_mirror_threshold": (0.0, 0.5),
        "pulse_rise_min": (0.0, 1.0),
        "pulse_rise_max": (0.0, 1.0),
        "pulse_rise_combine_ratio": (1, 10)
    },
    "noise_gate": {
        "threshold": (0.0, 0.5),
        "window_s": (0.05, 3.0),
        "attack_s": (0.0, 1.0),
        "release_s": (0.0, 5.0),
        "rest_level": (0.0, 1.0)
    },
    "trochoid_quantization": {
        "n_points": (2, 256),
        "R": (0.001, 1000.0),
        "r": (0.001, 1000.0),
        "d": (0.0, 1000.0)
    },
    "positional_axes": {
        # Note: Individual axis validation handled by motion_axis_generation module
        # Phase shift parameter ranges
        "phase_shift": {
            "delay_percentage": (0.0, 100.0),
            "min_segment_duration": (0.1, 5.0)
        }
    }
}


class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = DEFAULT_CONFIG.copy()
        self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file, falling back to defaults if file doesn't exist."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    self.config = self._merge_configs(DEFAULT_CONFIG, loaded_config)
                    self._migrate_trochoid_config()
                    self.validate_config()
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration.")
                self.config = DEFAULT_CONFIG.copy()
        return self.config

    def _migrate_trochoid_config(self):
        """Lift legacy flat trochoid keys (curve_type, R, r, d) into the new
        family / params_by_family structure. No-op once migration has run."""
        tq = self.config.get('trochoid_quantization')
        if not isinstance(tq, dict):
            return
        legacy_curve = tq.pop('curve_type', None)
        legacy_R = tq.pop('R', None)
        legacy_r = tq.pop('r', None)
        legacy_d = tq.pop('d', None)
        if legacy_curve in ('hypo', 'epi'):
            # Direct assignment: the merged-in default 'family' may already be
            # 'hypo'; the legacy curve_type wins on first migration.
            tq['family'] = legacy_curve
            params = tq.setdefault('params_by_family', {}).setdefault(
                legacy_curve, {})
            if legacy_R is not None:
                params['R'] = float(legacy_R)
            if legacy_r is not None:
                params['r'] = float(legacy_r)
            if legacy_d is not None:
                params['d'] = float(legacy_d)

    def save_config(self) -> bool:
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config file: {e}")
            return False

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update configuration with new values."""
        try:
            self.config = self._merge_configs(self.config, new_config)
            self.validate_config()
            return True
        except ValueError as e:
            print(f"Invalid configuration: {e}")
            return False

    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = DEFAULT_CONFIG.copy()

    def validate_config(self):
        """Validate configuration values against allowed ranges."""
        for section, params in PARAMETER_RANGES.items():
            if section not in self.config:
                continue

            # Special handling for nested positional_axes.phase_shift
            if section == 'positional_axes' and 'phase_shift' in params:
                phase_shift_ranges = params['phase_shift']
                phase_shift_config = self.config.get('positional_axes', {}).get('phase_shift', {})

                for param, (min_val, max_val) in phase_shift_ranges.items():
                    if param in phase_shift_config:
                        value = phase_shift_config[param]
                        if not (min_val <= value <= max_val):
                            raise ValueError(f"Parameter positional_axes.phase_shift.{param} = {value} is outside valid range [{min_val}, {max_val}]")
                continue  # Skip normal processing for positional_axes

            for param, range_tuple in params.items():
                if param not in self.config[section]:
                    continue

                # range_tuple should be a tuple of (min_val, max_val)
                if not isinstance(range_tuple, tuple) or len(range_tuple) != 2:
                    continue  # Skip if not a valid range tuple

                min_val, max_val = range_tuple
                value = self.config[section][param]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"Parameter {section}.{param} = {value} is outside valid range [{min_val}, {max_val}]")

        # Additional validation
        freq_config = self.config.get('frequency', {})
        if 'pulse_freq_min' in freq_config and 'pulse_freq_max' in freq_config:
            if freq_config['pulse_freq_min'] >= freq_config['pulse_freq_max']:
                raise ValueError("pulse_freq_min must be less than pulse_freq_max")

        pulse_config = self.config.get('pulse', {})
        if 'pulse_width_min' in pulse_config and 'pulse_width_max' in pulse_config:
            if pulse_config['pulse_width_min'] >= pulse_config['pulse_width_max']:
                raise ValueError("pulse_width_min must be less than pulse_width_max")

        if 'pulse_rise_min' in pulse_config and 'pulse_rise_max' in pulse_config:
            if pulse_config['pulse_rise_min'] >= pulse_config['pulse_rise_max']:
                raise ValueError("pulse_rise_min must be less than pulse_rise_max")

    def _merge_configs(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries.

        Deep-copies values pulled from `base` so that subsequent mutations of
        the returned config (e.g. by migration helpers) do not leak back into
        the original DEFAULT_CONFIG dictionary.
        """
        result = copy.deepcopy(base)

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result