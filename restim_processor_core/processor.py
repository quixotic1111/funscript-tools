import os
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Optional, Dict, Any

from funscript import Funscript
from processing.speed_processing import convert_to_speed
from processing.basic_transforms import (
    invert_funscript, map_funscript, limit_funscript,
    normalize_funscript, mirror_up_funscript
)
from processing.combining import combine_funscripts, apply_direction_bias
from processing.special_generators import (
    make_volume_ramp, make_volume_ramp_per_clip)
from processing.funscript_1d_to_2d import generate_alpha_beta_from_main
from processing.funscript_prostate_2d import generate_alpha_beta_prostate_from_main
from processing.motion_axis_generation import (
    generate_motion_axes, copy_existing_axis_files, validate_motion_axis_config,
    apply_modulation,
)
from processing.phase_shift_generation import generate_all_phase_shifted_funscripts
from processing.trochoid_quantization import (
    quantize_to_curve, deduplicate_holds,
    FAMILY_DEFAULTS as _CURVE_FAMILY_DEFAULTS,
)
from processing.trochoid_spatial import generate_spatial_funscripts
from processing.traveling_wave import generate_wave_funscripts
from processing.axis_markers import strip_axis_suffix


def _apply_release_envelope(y, dt, tau_s):
    """Asymmetric leaky integrator: instant attack, exponential decay.

    y_out[n] = max(y_in[n], y_out[n-1] * exp(-dt[n]/τ))

    Signal rises instantly to match any increase in input. When input
    falls below the held level, the held level decays exponentially
    toward it with time constant τ. Passing τ ≤ 0 returns the input
    unchanged.
    """
    import numpy as np
    if tau_s <= 0.0 or len(y) == 0:
        return y
    y_in = np.asarray(y, dtype=float)
    dt_arr = np.asarray(dt, dtype=float)
    out = np.empty_like(y_in)
    out[0] = y_in[0]
    # alpha[n] = exp(-dt[n]/τ); closer to 1 = slower decay.
    alpha = np.exp(-np.clip(dt_arr, 0.0, None) / tau_s)
    for i in range(1, len(y_in)):
        decayed = out[i - 1] * alpha[i]
        out[i] = y_in[i] if y_in[i] >= decayed else decayed
    return out


def _reverb_feedback_delay(y, fs_hz, delay_ms, feedback, mix):
    """Single-tap IIR comb (echo with feedback).

    Internal recursion: z[n] = x[n] + fb * z[n-d]
    Output: (1 - mix) * x[n] + mix * z[n], clipped to [0, 1].

    feedback is clamped to < 0.95 for stability; mix is clamped to
    [0, 1]. Returns input unchanged when mix ≤ 0 or delay_ms ≤ 0.
    """
    import numpy as np
    if mix <= 0.0 or delay_ms <= 0.0 or fs_hz <= 0.0:
        return y
    y_in = np.asarray(y, dtype=float)
    n = len(y_in)
    if n == 0:
        return y_in
    d = int(round(delay_ms * 1e-3 * fs_hz))
    if d <= 0 or d >= n:
        return y_in
    fb = float(np.clip(feedback, 0.0, 0.95))
    m = float(np.clip(mix, 0.0, 1.0))
    z = np.zeros(n, dtype=float)
    for i in range(n):
        prev = z[i - d] if i >= d else 0.0
        z[i] = y_in[i] + fb * prev
    return np.clip((1.0 - m) * y_in + m * z, 0.0, 1.0)


def _reverb_multitap(y, fs_hz, delays_ms, gains, mix):
    """FIR sum of delayed attenuated copies (Schroeder-ish comb).

    z[n] = x[n] + Σ_k (g_k * x[n - d_k])
    Output: (1 - mix) * x[n] + mix * z[n], clipped to [0, 1].

    Unconditionally stable (no feedback). delays_ms and gains must
    have the same length. Taps beyond the signal length are skipped.
    """
    import numpy as np
    if mix <= 0.0 or not delays_ms or fs_hz <= 0.0:
        return y
    y_in = np.asarray(y, dtype=float)
    n = len(y_in)
    if n == 0:
        return y_in
    m = float(np.clip(mix, 0.0, 1.0))
    wet = y_in.copy()
    for delay_ms, g in zip(delays_ms, gains):
        d = int(round(float(delay_ms) * 1e-3 * fs_hz))
        if d <= 0 or d >= n:
            continue
        shifted = np.concatenate([np.zeros(d), y_in[:-d]])
        wet = wet + float(g) * shifted
    return np.clip((1.0 - m) * y_in + m * wet, 0.0, 1.0)


def _reverb_cross_electrode(intensities, fs_hz, delay_ms,
                            feedback, mix):
    """Add delayed neighbor contributions to each electrode.

    For each electrode i: its output receives
        mix * mean(delayed(E_{i-1}), delayed(E_{i+1}))
    (neighbors that don't exist are skipped; endpoints have one
    neighbor each.)

    If feedback > 0, also adds a per-electrode IIR feedback tap
    for self-sustain.

    Operates on and returns a new intensities dict (input unchanged).
    Output is clipped to [0, 1] per electrode.
    """
    import numpy as np
    keys = sorted(intensities.keys())
    if (mix <= 0.0 or delay_ms <= 0.0 or fs_hz <= 0.0
            or len(keys) < 2):
        return {k: np.asarray(intensities[k], dtype=float).copy()
                for k in keys}
    d = int(round(delay_ms * 1e-3 * fs_hz))
    n = len(intensities[keys[0]])
    if d <= 0 or d >= n:
        return {k: np.asarray(intensities[k], dtype=float).copy()
                for k in keys}
    m = float(np.clip(mix, 0.0, 1.0))
    fb = float(np.clip(feedback, 0.0, 0.95))

    delayed = {}
    for k in keys:
        arr = np.asarray(intensities[k], dtype=float)
        delayed[k] = np.concatenate([np.zeros(d), arr[:-d]])

    out = {}
    for i, k in enumerate(keys):
        x = np.asarray(intensities[k], dtype=float)
        neighbors = []
        if i > 0:
            neighbors.append(delayed[keys[i - 1]])
        if i < len(keys) - 1:
            neighbors.append(delayed[keys[i + 1]])
        bleed = (sum(neighbors) / len(neighbors)
                 if neighbors else np.zeros_like(x))
        wet = x + m * bleed
        if fb > 0.0:
            z = np.zeros(n, dtype=float)
            for j in range(n):
                prev = z[j - d] if j >= d else 0.0
                z[j] = wet[j] + fb * prev
            wet = z
        out[k] = np.clip(wet, 0.0, 1.0)
    return out


def _apply_ema(y, dt, tau_s):
    """Symmetric one-pole EMA with time constant τ.

    y_out[n] = α[n] * y_out[n-1] + (1 - α[n]) * y_in[n]
    where α[n] = exp(-dt[n]/τ). Causal (adds phase lag equal to τ).
    Passing τ ≤ 0 returns the input unchanged.
    """
    import numpy as np
    if tau_s <= 0.0 or len(y) == 0:
        return y
    y_in = np.asarray(y, dtype=float)
    dt_arr = np.asarray(dt, dtype=float)
    out = np.empty_like(y_in)
    out[0] = y_in[0]
    alpha = np.exp(-np.clip(dt_arr, 0.0, None) / tau_s)
    for i in range(1, len(y_in)):
        out[i] = alpha[i] * out[i - 1] + (1.0 - alpha[i]) * y_in[i]
    return out


class RestimProcessor:
    def __init__(self, parameters: Dict[str, Any]):
        self.params = parameters
        self.temp_dir: Optional[Path] = None
        self.input_path: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.filename_only: str = ""

    def _add_metadata(self, funscript: Funscript, file_type: str, description: str, additional_params: dict = None):
        """Add metadata to a generated funscript."""
        from version import __version__, __app_name__, __url__

        funscript.metadata = {
            "creator": __app_name__,
            "description": f"Generated by {__app_name__} v{__version__} - {description}",
            "url": __url__,
            "title": file_type.replace('_', ' ').title(),
            "metadata": {
                "generator": __app_name__,
                "generator_version": __version__,
                "file_type": file_type
            }
        }

        # Add additional parameters to metadata
        if additional_params:
            funscript.metadata["metadata"].update(additional_params)

    def process(self, input_file_path: str, progress_callback: Optional[Callable[[int, str], None]] = None) -> bool:
        """
        Process the input funscript file and generate all output files.

        Args:
            input_file_path: Path to the input .funscript file
            progress_callback: Optional callback function for progress updates (progress_percent, status_message)

        Returns:
            bool: True if processing completed successfully, False otherwise
        """
        try:
            self._update_progress(progress_callback, 0, "Initializing...")

            # Setup
            self.input_path = Path(input_file_path)
            self.filename_only = self.input_path.stem
            self._setup_directories()

            # Create backup if in central mode with backups enabled
            file_mgmt = self.params.get('file_management', {})
            if file_mgmt.get('mode') == 'central':
                if file_mgmt.get('create_backups', False):
                    self._create_backup(progress_callback)
                else:
                    # Delete existing output files if backups are disabled
                    # This ensures fresh generation instead of reusing old files
                    self._delete_existing_output_files(progress_callback)

            # Load main funscript
            self._update_progress(progress_callback, 5, "Loading input file...")
            main_funscript = Funscript.from_file(self.input_path)

            # Execute processing pipeline
            self._execute_pipeline(main_funscript, progress_callback)

            # Cleanup if requested
            if self.params['options']['delete_intermediary_files']:
                self._update_progress(progress_callback, 95, "Cleaning up intermediary files...")
                self._cleanup_intermediary_files()

            # Zip output files if enabled in central mode
            file_mgmt = self.params.get('file_management', {})
            if file_mgmt.get('mode') == 'central' and file_mgmt.get('zip_output', False):
                self._zip_output_files(progress_callback)

            self._update_progress(progress_callback, 100, "Processing complete!")
            return True

        except Exception as e:
            self._update_progress(progress_callback, -1, f"Error: {str(e)}")
            return False

    def process_triplet(
        self,
        paths,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> bool:
        """
        Process three input funscripts (X, Y, Z) as a single 3D signal.

        Emits one set of e1..eN funscript outputs produced by projecting
        the 3D signal onto a straight-line electrode array. Config knobs
        come from self.params['spatial_3d_linear']. The first path is
        treated as X and its stem is used for output filenames; Y and Z
        contribute axes but not naming.

        Unlike process(), this mode produces ONLY electrode outputs —
        no main/speed/alpha-beta/frequency/volume/pulse files. Run an
        independent 1D pass on whichever axis if those are wanted too.
        """
        try:
            if len(paths) < 3:
                self._update_progress(
                    progress_callback, -1,
                    f"triplet mode needs 3 paths, got {len(paths)}")
                return False

            self._update_progress(progress_callback, 0,
                                  "Initializing triplet...")

            # Use X (first path) as the anchor for filenames and directory
            # setup. Y and Z are loaded for axis data only.
            #
            # Strip the trailing axis marker (.x / .sway / etc.) from
            # the stem so the output files are keyed to the clean
            # project basename — ``capture_123.alpha.funscript`` rather
            # than ``capture_123.sway.alpha.funscript`` — and the
            # variant folder mirrors the video filename. The viewer
            # uses the same strip to find the video + variant dir.
            self.input_path = Path(paths[0])
            self.filename_only = strip_axis_suffix(self.input_path.stem)
            self._setup_directories()

            s3d = self.params.get('spatial_3d_linear', {}) or {}
            n_elec = int(s3d.get('n_electrodes', 4))
            sharpness = float(s3d.get('sharpness', 1.0))
            normalize = str(s3d.get('normalize', 'clamped'))
            center_yz = s3d.get('center_yz', [0.5, 0.5])
            # Output smoothing — One-Euro adaptive low-pass per electrode,
            # applied after the cross-electrode normalize. Off by default
            # so existing runs are byte-identical; turn on when the raw
            # electrode outputs feel clicky at sharp sharpness / tight
            # tracker input.
            _osm = s3d.get('output_smoothing', {}) or {}
            osm_enabled = bool(_osm.get('enabled', False))
            osm_min_cutoff = float(_osm.get('min_cutoff_hz', 1.0))
            osm_beta = float(_osm.get('beta', 0.05))
            # Per-electrode gain/trim. List form: positional for e1..eN.
            # Silently coerced — bad entries → 1.0 — so a half-edited
            # config can't crash the processor.
            _eg = s3d.get('electrode_gain') or []
            electrode_gain = []
            for i in range(n_elec):
                try:
                    electrode_gain.append(
                        float(_eg[i]) if i < len(_eg) else 1.0)
                except (TypeError, ValueError):
                    electrode_gain.append(1.0)
            # Soft-knee output limiter settings.
            _ol = s3d.get('output_limiter', {}) or {}
            ol_enabled = bool(_ol.get('enabled', False))
            ol_threshold = float(_ol.get('threshold', 0.85))
            # Velocity-weighted intensity settings.
            _vw = s3d.get('velocity_weight', {}) or {}
            vw_enabled = bool(_vw.get('enabled', False))
            vw_floor = float(_vw.get('floor', 0.0))
            vw_response = float(_vw.get('response', 1.0))
            vw_smoothing = float(_vw.get('smoothing_hz', 3.0))
            vw_norm_pct = float(_vw.get('normalization_percentile', 0.99))
            vw_gate = float(_vw.get('gate_threshold', 0.05))
            # Per-axis weights inside the distance calc. 1.0 each =
            # rotation-symmetric Y/Z (the historic behavior).
            y_w = float(s3d.get('y_weight', 1.0))
            z_w = float(s3d.get('z_weight', 1.0))
            # Distance falloff shape — dispatched inside the kernel.
            # `width` scales the effective cube diagonal to produce
            # the characteristic distance each shape interprets.
            falloff_shape = str(s3d.get('falloff_shape', 'linear'))
            falloff_width = float(s3d.get('falloff_width', 1.0))
            # Per-electrode sharpness override. When enabled, the
            # kernel receives a list; otherwise the scalar is used.
            if bool(s3d.get('per_electrode_sharpness_enabled', False)):
                _pes_raw = s3d.get('per_electrode_sharpness') or []
                try:
                    sharpness_arg = [
                        float(_pes_raw[i]) if i < len(_pes_raw) else 1.0
                        for i in range(n_elec)]
                except (TypeError, ValueError):
                    sharpness_arg = sharpness
            else:
                sharpness_arg = sharpness
            # Per-electrode X positions. Read the first n_elec entries;
            # fall back to linspace if any entry is malformed.
            _xp_raw = s3d.get('electrode_x_positions') or []
            electrode_x_arr = None
            try:
                electrode_x_arr = np.array(
                    [float(_xp_raw[i]) for i in range(n_elec)],
                    dtype=float)
            except (IndexError, TypeError, ValueError):
                electrode_x_arr = None  # kernel default linspace
            try:
                center_yz = (float(center_yz[0]), float(center_yz[1]))
            except (TypeError, ValueError, IndexError):
                center_yz = (0.5, 0.5)

            # Optional 4th axis: roll around the shaft (.rz). If the
            # caller passed 4+ paths, we'll take the 4th as rz. With
            # 3 paths, rza defaults to a flat 0.5 (neutral) and the
            # omega modulator produces zeros.
            self._update_progress(progress_callback, 15,
                                  "Loading X/Y/Z(/rz) triplet...")
            from processing.multi_script_loader import load_dof_scripts
            from processing.spatial_3d_linear import (
                compute_linear_intensities_3d)
            path_rz = paths[3] if len(paths) >= 4 else None

            # One-Euro input smoothing config (opt-in, off by
            # default). When tracker output is visibly jumpy, this
            # filters per-axis jitter before the spatial projection
            # without the lag-ringing artifacts a heavy fixed EMA
            # produces.
            ism_cfg = s3d.get('input_smoothing', {}) or {}
            ism_enabled = bool(ism_cfg.get('enabled', False))
            ism_min_cutoff = float(ism_cfg.get('min_cutoff_hz', 1.0))
            ism_beta = float(ism_cfg.get('beta', 0.05))
            ism_d_cutoff = float(ism_cfg.get('d_cutoff_hz', 1.0))

            # Input sharpener config (opt-in, off by default). The
            # complement of input_smoothing: adds back transient
            # energy + pushes signal toward [0, 1] extremes to
            # make smooth-tracker sources (Mask-Moments) behave
            # more like sharp-tracker sources (Quad) when the
            # downstream projection reads them.
            ish_cfg = s3d.get('input_sharpen', {}) or {}
            ish_enabled = bool(ish_cfg.get('enabled', False))
            ish_pre = float(ish_cfg.get('pre_emphasis', 1.0))
            ish_sat = float(ish_cfg.get('saturation', 1.0))
            ish_cutoff = float(
                ish_cfg.get('pre_emphasis_cutoff_hz', 3.0))

            # Noise gate config (opt-in, off by default). Applied
            # per-axis-combined inside load_dof_scripts BEFORE
            # smoothing/sharpening so the downstream stages operate
            # on a pre-gated signal. Uses a single envelope across
            # all axes (see gate_uniform_signals_combined).
            ng_cfg = s3d.get('noise_gate', {}) or {}
            ng_enabled = bool(ng_cfg.get('enabled', False))
            ng_threshold = float(ng_cfg.get('threshold', 0.05))
            ng_window_s = float(ng_cfg.get('window_s', 0.5))
            ng_attack_s = float(ng_cfg.get('attack_s', 0.02))
            ng_release_s = float(ng_cfg.get('release_s', 0.3))
            ng_rest_level = float(ng_cfg.get('rest_level', 0.5))

            t, xa, ya, za, rza = load_dof_scripts(
                paths[0], paths[1], paths[2], path_rz, hz=50.0,
                input_smoothing_enabled=ism_enabled,
                input_smoothing_min_cutoff_hz=ism_min_cutoff,
                input_smoothing_beta=ism_beta,
                input_smoothing_d_cutoff_hz=ism_d_cutoff,
                input_sharpen_enabled=ish_enabled,
                input_sharpen_pre_emphasis=ish_pre,
                input_sharpen_saturation=ish_sat,
                input_sharpen_pre_emphasis_cutoff_hz=ish_cutoff,
                noise_gate_enabled=ng_enabled,
                noise_gate_threshold=ng_threshold,
                noise_gate_window_s=ng_window_s,
                noise_gate_attack_s=ng_attack_s,
                noise_gate_release_s=ng_release_s,
                noise_gate_rest_level=ng_rest_level,
            )

            self._update_progress(progress_callback, 50,
                                  "Projecting onto electrode array...")
            # Always compute clamped intensities first. They feed the
            # volume envelope (per-frame max reflects absolute signal-
            # to-electrode proximity) regardless of whichever normalize
            # mode the user picks for the electrode outputs.
            clamped = compute_linear_intensities_3d(
                xa, ya, za,
                n_electrodes=n_elec,
                electrode_x=electrode_x_arr,
                center_yz=center_yz,
                sharpness=sharpness_arg,
                normalize='clamped',
                y_weight=y_w,
                z_weight=z_w,
                falloff_shape=falloff_shape,
                falloff_width=falloff_width,
            )
            # Second pass computes the user's selected normalize mode
            # and applies output smoothing + per-electrode gain (if
            # non-unity). The clamped pass above always stays
            # unshaped so the volume envelope reflects raw proximity.
            # When the user hasn't opted into anything, reuse clamped.
            # Velocity weight derived from the XYZ triplet — computed
            # here so the kernel stays geometry-agnostic.
            vw_array = None
            if vw_enabled:
                from processing.output_shaping import compute_velocity_weight
                vw_array = compute_velocity_weight(
                    [xa, ya, za], t,
                    floor=vw_floor,
                    response=vw_response,
                    smoothing_hz=vw_smoothing,
                    normalization_percentile=vw_norm_pct,
                    gate_threshold=vw_gate,
                )

            _gain_is_unity = all(abs(g - 1.0) < 1e-9 for g in electrode_gain)
            if (normalize == 'clamped' and not osm_enabled
                    and _gain_is_unity and not ol_enabled and not vw_enabled):
                # Reuse the clamped pass — it already saw y_w/z_w, so
                # no geometry drift.
                intensities = clamped
            else:
                intensities = compute_linear_intensities_3d(
                    xa, ya, za,
                    n_electrodes=n_elec,
                    electrode_x=electrode_x_arr,
                    center_yz=center_yz,
                    sharpness=sharpness_arg,
                    normalize=normalize,
                    t_sec=t,
                    output_smoothing_enabled=osm_enabled,
                    output_smoothing_min_cutoff_hz=osm_min_cutoff,
                    output_smoothing_beta=osm_beta,
                    electrode_gain=electrode_gain,
                    output_limiter_enabled=ol_enabled,
                    output_limiter_threshold=ol_threshold,
                    velocity_weight=vw_array,
                    y_weight=y_w,
                    z_weight=z_w,
                    falloff_shape=falloff_shape,
                    falloff_width=falloff_width,
                )

            # Volume envelope = per-frame max across clamped electrodes.
            # When the 3D signal is close to the electrode line, at
            # least one electrode is hot → volume high. Drifts toward
            # the cube corners collapse all intensities → volume dips.
            import numpy as np
            vol_stack = np.stack(
                [clamped[f'e{i + 1}'] for i in range(n_elec)], axis=0)
            volume_y = vol_stack.max(axis=0)
            volume_y = np.clip(volume_y, 0.0, 1.0)

            # EXPERIMENTAL — reverb-analog effects on volume_y.
            # Run before the ramp so the end fade-out still bounds
            # the reverb tail. _reverb_* helpers no-op when their
            # mix <= 0 so this is free when disabled.
            _rv_cfg = s3d.get('reverb', {}) or {}
            if _rv_cfg.get('enabled', False):
                _vt = _rv_cfg.get('volume_tail', {}) or {}
                volume_y = _reverb_feedback_delay(
                    volume_y, 50.0,
                    float(_vt.get('delay_ms', 200.0)),
                    float(_vt.get('feedback', 0.4)),
                    float(_vt.get('mix', 0.0)))
                _vm = _rv_cfg.get('volume_multitap', {}) or {}
                volume_y = _reverb_multitap(
                    volume_y, 50.0,
                    list(_vm.get('delays_ms', [])),
                    list(_vm.get('gains', [])),
                    float(_vm.get('mix', 0.0)))

            # Ramp envelope: linear rise across the clip, multiplied
            # into the max-E envelope. Uses `ramp_percent_total` (the
            # total % rise from clip start to end) instead of the 1D
            # pipeline's `%/hour` rate, since preview clips are
            # seconds-to-minutes long and the rate-based math made
            # the slider appear dead on short captures.
            if len(t) >= 4 and float(t[-1]) > float(t[0]):
                ramp_pct = float(s3d.get('ramp_percent_total', 40.0))
                if ramp_pct > 0.0:
                    ramp_src = Funscript(t.copy(),
                                         np.zeros_like(t, dtype=float))
                    ramp_fs = make_volume_ramp_per_clip(ramp_src, ramp_pct)
                    ramp_y = np.clip(
                        np.interp(t, np.asarray(ramp_fs.x, dtype=float),
                                  np.asarray(ramp_fs.y, dtype=float)),
                        0.0, 1.0)
                    volume_y = volume_y * ramp_y

            # 3D speed = |v| = sqrt(ẋ² + ẏ² + ż²), normalized by a
            # high-percentile of its own distribution so a single
            # artifact spike doesn't flatten the rest of the signal.
            # Falls back to peak-normalize if the percentile is 1.0
            # or if the percentile value is degenerate.
            dt = np.diff(t, prepend=t[0])
            dt = np.where(dt > 0, dt, 1.0 / 50.0)  # safety for dt=0
            vx = np.diff(xa, prepend=xa[0]) / dt
            vy = np.diff(ya, prepend=ya[0]) / dt
            vz = np.diff(za, prepend=za[0]) / dt
            speed_raw = np.sqrt(vx * vx + vy * vy + vz * vz)
            speed_pct = float(s3d.get('speed_normalization_percentile', 0.99))
            speed_pct = min(1.0, max(0.5, speed_pct))
            if speed_pct >= 1.0:
                speed_norm = float(np.nanmax(speed_raw)) or 1.0
            else:
                speed_norm = float(np.quantile(speed_raw, speed_pct))
                if not np.isfinite(speed_norm) or speed_norm <= 0:
                    speed_norm = float(np.nanmax(speed_raw)) or 1.0
            speed_y = np.clip(speed_raw / speed_norm, 0.0, 1.0)

            # Release envelope on speed_y: instant attack, exponential
            # decay toward 0 when motion slows. Gives intensity a
            # natural tail instead of snapping dead at every pause.
            # τ=0 is a no-op. At sample rate 50 Hz, dt ≈ 0.02 s.
            _release_tau = float(s3d.get('release_tau_s', 0.0))
            if _release_tau > 0.0:
                speed_y = _apply_release_envelope(speed_y, dt, _release_tau)

            # Speed floor: rest-level style minimum on the motion-
            # derived signal so the carrier doesn't drop fully quiet
            # during pauses. Applied AFTER the release envelope so
            # decay floors at this value rather than continuing to 0.
            _speed_floor = float(np.clip(
                float(s3d.get('speed_floor', 0.0)), 0.0, 1.0))
            if _speed_floor > 0.0:
                speed_y = np.maximum(speed_y, _speed_floor)

            # Geometric mapping signals for optionally driving the
            # pulse_* channels. Always computed; only used when the
            # corresponding mix knob is > 0. Cheap relative to the
            # rest of the pipeline so we don't gate behind a flag.
            cy, cz = center_yz
            _dy_pos = ya - cy
            _dz_pos = za - cz
            _r = np.sqrt(_dy_pos * _dy_pos + _dz_pos * _dz_pos)
            # Corner of the unit square from center = max reachable r.
            _r_max = float(np.sqrt(
                max(cy, 1.0 - cy) ** 2 + max(cz, 1.0 - cz) ** 2)) or 1.0
            radial_norm = np.clip(_r / _r_max, 0.0, 1.0)
            # Azimuth via cos() — smooth, no ±π wrap. Undefined at r=0,
            # so substitute 0.5 (neutral midpoint) there.
            _phi = np.arctan2(_dz_pos, _dy_pos)
            azimuth_norm = np.where(
                _r > 1e-9, (np.cos(_phi) + 1.0) / 2.0, 0.5)
            # dr/dt — outward (>0) vs inward (<0). Percentile-normalized
            # and centered at 0.5 so sign survives the mapping.
            _dr = np.diff(_r, prepend=_r[0]) / dt
            _gmap = s3d.get('geometric_mapping', {}) or {}
            _vr_pct = float(_gmap.get(
                'vradial_normalization_percentile', 0.99))
            _vr_pct = min(1.0, max(0.5, _vr_pct))
            _vr_scale = float(np.quantile(np.abs(_dr), _vr_pct))
            if not np.isfinite(_vr_scale) or _vr_scale <= 0:
                _vr_scale = float(np.nanmax(np.abs(_dr))) or 1.0
            vradial_norm = np.clip(
                0.5 + 0.5 * (_dr / _vr_scale), 0.0, 1.0)

            # Roll angular velocity dω/dt (only non-trivial when a .rz
            # funscript was dropped; otherwise rza is a flat 0.5 and
            # _drz is all zeros). Percentile-normalized and centered
            # at 0.5 like vradial, so sign survives: CW and CCW feel
            # distinct downstream.
            _drz = np.diff(rza, prepend=rza[0]) / dt
            _om_pct = float(_gmap.get(
                'omega_normalization_percentile', 0.99))
            _om_pct = min(1.0, max(0.5, _om_pct))
            _om_abs = np.abs(_drz)
            _om_scale = float(np.quantile(_om_abs, _om_pct))
            if not np.isfinite(_om_scale) or _om_scale <= 0:
                _om_scale = float(np.nanmax(_om_abs)) or 1.0
            omega_roll_norm = np.clip(
                0.5 + 0.5 * (_drz / _om_scale), 0.0, 1.0)

            # τ-hold on the geometric signals before they drive the
            # pulse channels. Symmetric EMA so rapid wobbles don't
            # chatter the pulse shape. τ=0 is a no-op. Applied to
            # the roll-velocity signal as well so it smooths the
            # same way as the others.
            _hold_tau = float(_gmap.get('hold_tau_s', 0.0))
            if _hold_tau > 0.0:
                radial_norm = _apply_ema(radial_norm, dt, _hold_tau)
                azimuth_norm = _apply_ema(azimuth_norm, dt, _hold_tau)
                vradial_norm = _apply_ema(vradial_norm, dt, _hold_tau)
                omega_roll_norm = _apply_ema(
                    omega_roll_norm, dt, _hold_tau)

            # Optional dynamic-range compressor. Applied BEFORE the
            # Butterworth smoothing so the smoother can clean up any
            # fast gain-reduction edges the compressor introduced.
            # Global-envelope compression (max across electrodes
            # drives the gain, applied uniformly to all channels) so
            # the per-frame spatial balance is preserved — the cycle
            # we're flattening is the cross-frames loudness
            # ("mild ↔ grabbing"), not the per-frame which-electrode
            # variation that defines the spatial character.
            comp_cfg = s3d.get('compression', {}) or {}
            if comp_cfg.get('enabled', False):
                from processing.dynamic_range_compressor import (
                    compress_intensities)
                intensities = compress_intensities(
                    intensities,
                    threshold=float(comp_cfg.get('threshold', 0.4)),
                    ratio=float(comp_cfg.get('ratio', 3.0)),
                    attack_ms=float(comp_cfg.get('attack_ms', 10.0)),
                    release_ms=float(comp_cfg.get('release_ms', 150.0)),
                    makeup=float(comp_cfg.get('makeup', 1.0)),
                    sample_rate_hz=50.0,
                )

            # Optional low-pass smoothing on the electrode intensities
            # to tame high-frequency flicker. Uses zero-phase Butterworth
            # (filtfilt) so the envelope stays time-aligned. Applied to
            # the final `intensities` dict — the volume envelope was
            # already derived from the raw clamped values above.
            sm_cfg = s3d.get('smoothing', {}) or {}
            if sm_cfg.get('enabled', False):
                cutoff_hz = float(sm_cfg.get('cutoff_hz', 8.0))
                order = int(sm_cfg.get('order', 2))
                fs_hz = 50.0  # load_xyz_triplet resamples at this rate
                nyq = fs_hz / 2.0
                if 0.0 < cutoff_hz < nyq:
                    from scipy import signal as _sig
                    b_coef, a_coef = _sig.butter(
                        order, cutoff_hz / nyq, btype='low')
                    min_len = 3 * max(len(a_coef), len(b_coef))
                    for k in list(intensities.keys()):
                        arr = np.asarray(intensities[k], dtype=float)
                        if len(arr) > min_len:
                            smoothed = _sig.filtfilt(b_coef, a_coef, arr)
                            intensities[k] = np.clip(smoothed, 0.0, 1.0)

            # EXPERIMENTAL — cross-electrode bleed reverb. Each E gets
            # a delayed copy of its neighbors' envelopes summed in.
            # Runs after smoothing (which would otherwise remove the
            # introduced edges) and before dedup (which would collapse
            # the reverb tails). No-op when mix = 0.
            if _rv_cfg.get('enabled', False):
                _xe = _rv_cfg.get('cross_electrode', {}) or {}
                intensities = _reverb_cross_electrode(
                    intensities, 50.0,
                    float(_xe.get('delay_ms', 100.0)),
                    float(_xe.get('feedback', 0.0)),
                    float(_xe.get('mix', 0.0)))

            # Dedup-holds: drop interior samples of constant-within-tolerance
            # runs on each electrode so the device's linear interp doesn't
            # slope across held windows. Applied per-electrode so each can
            # have its own time axis (funscripts are independent files).
            dd_cfg = s3d.get('deduplicate_holds', {}) or {}
            dd_enabled = bool(dd_cfg.get('enabled', False))
            dd_tol = float(dd_cfg.get('tolerance', 0.005))

            self._update_progress(progress_callback, 70,
                                  "Writing electrode funscripts...")
            x_name = os.path.basename(paths[0])
            y_name = os.path.basename(paths[1]) if paths[1] else 'flat'
            z_name = os.path.basename(paths[2]) if paths[2] else 'flat'
            for key in sorted(intensities.keys()):
                arr = intensities[key]
                if dd_enabled:
                    tmp = Funscript(t.copy(), arr.copy())
                    tmp = deduplicate_holds(tmp, atol=dd_tol)
                    t_e = np.asarray(tmp.x, dtype=float)
                    y_e = np.asarray(tmp.y, dtype=float)
                else:
                    t_e = t
                    y_e = arr
                fs = Funscript(t_e.copy(), y_e.copy())
                meta_params = {
                    "mode": "spatial_3d_linear",
                    "n_electrodes": n_elec,
                    "sharpness": sharpness,
                    "normalize": normalize,
                    "center_yz": list(center_yz),
                    "x_source": x_name,
                    "y_source": y_name,
                    "z_source": z_name,
                }
                if dd_enabled:
                    meta_params["deduplicated"] = True
                    meta_params["dedup_tolerance"] = dd_tol
                    meta_params["samples_before_dedup"] = int(len(t))
                    meta_params["samples_after_dedup"] = int(len(t_e))
                self._add_metadata(
                    fs, f"motion_axis_{key}",
                    f"Spatial 3D Linear ({key.upper()}) — "
                    f"X={x_name}, Y={y_name}, Z={z_name}",
                    meta_params)
                fs.save_to_path(self._get_temp_path(key))
                shutil.copy2(
                    self._get_temp_path(key),
                    self._get_output_path(key))

            # Volume envelope.
            self._update_progress(progress_callback, 85,
                                  "Writing volume envelope...")
            volume_fs = Funscript(t.copy(), volume_y.copy())
            self._add_metadata(
                volume_fs, "volume",
                f"Spatial 3D Linear volume envelope "
                f"(max of clamped E1..E{n_elec})",
                {
                    "mode": "spatial_3d_linear",
                    "volume_source": "max_clamped_electrodes",
                    "n_electrodes": n_elec,
                    "sharpness": sharpness,
                })
            volume_fs.save_to_path(self._get_temp_path("volume"))
            shutil.copy2(
                self._get_temp_path("volume"),
                self._get_output_path("volume"))

            # 3D speed (|velocity| normalized to [0, 1]).
            speed_fs = Funscript(t.copy(), speed_y.copy())
            self._add_metadata(
                speed_fs, "speed",
                "3D speed |v| = sqrt(dx² + dy² + dz²), "
                f"normalized to {speed_pct:.2f}-percentile",
                {
                    "mode": "spatial_3d_linear",
                    "speed_source": "3d_velocity_magnitude",
                    "normalization_percentile": speed_pct,
                    "normalization_value": float(speed_norm),
                })
            speed_fs.save_to_path(self._get_temp_path("speed"))
            shutil.copy2(
                self._get_temp_path("speed"),
                self._get_output_path("speed"))

            # Device-critical parameter channels: carrier frequency and
            # pulse shape. Pulse channels are emitted as flat 2-point
            # funscripts (restim just needs a valid file). Frequency can
            # optionally blend the flat default with per-frame |v| via
            # `frequency_speed_mix` — 0 = flat, 1 = fully |v|-driven.
            self._update_progress(progress_callback, 92,
                                  "Writing parameter defaults...")
            t_bounds = np.array([float(t[0]), float(t[-1])], dtype=float)

            # Carrier frequency: optionally driven by |v|.
            freq_default = float(
                np.clip(float(s3d.get('default_frequency', 0.5)), 0.0, 1.0))
            freq_mix = float(
                np.clip(float(s3d.get('frequency_speed_mix', 0.0)), 0.0, 1.0))
            if freq_mix > 0.0:
                freq_y = np.clip(
                    (1.0 - freq_mix) * freq_default + freq_mix * speed_y,
                    0.0, 1.0)
                freq_fs = Funscript(t.copy(), freq_y.copy())
                self._add_metadata(
                    freq_fs, "frequency",
                    "Carrier frequency blended from |v| "
                    f"(mix={freq_mix:.2f}, default={freq_default:.2f})",
                    {
                        "mode": "spatial_3d_linear",
                        "source": "speed_blend",
                        "default_frequency": freq_default,
                        "frequency_speed_mix": freq_mix,
                    })
            else:
                freq_fs = Funscript(
                    t_bounds.copy(),
                    np.array([freq_default, freq_default], dtype=float))
                self._add_metadata(
                    freq_fs, "frequency",
                    "Flat default carrier frequency (spatial 3D mode)",
                    {
                        "mode": "spatial_3d_linear",
                        "source": "flat_default",
                        "value": freq_default,
                    })
            freq_fs.save_to_path(self._get_temp_path("frequency"))
            shutil.copy2(
                self._get_temp_path("frequency"),
                self._get_output_path("frequency"))

            # Pulse shape channels. Each stays flat unless its
            # geometric_mapping mix knob is > 0, in which case it
            # blends the flat default with a per-frame signal:
            #   pulse_frequency ← dr/dt (outward vs inward)
            #   pulse_width     ← radial distance from shaft axis
            #   pulse_rise_time ← azimuth around shaft axis
            gmap = s3d.get('geometric_mapping', {}) or {}
            pulse_channels = [
                {
                    "name": "pulse_frequency",
                    "default": float(s3d.get('default_pulse_frequency', 0.5)),
                    "mix": float(gmap.get('pulse_frequency_vradial_mix', 0.0)),
                    "geom_y": vradial_norm,
                    "source_tag": "vradial_blend",
                    "desc_flat": "Flat default pulse frequency (spatial 3D mode)",
                    "desc_geom": "Pulse frequency blended from dr/dt",
                    "mix_key": "pulse_frequency_vradial_mix",
                    # Optional second modulator (sum-and-clip): roll-ω.
                    "mix2": float(gmap.get('pulse_frequency_omega_mix', 0.0)),
                    "geom_y2": omega_roll_norm,
                    "mix_key2": "pulse_frequency_omega_mix",
                    "desc_geom2": "blended from dω/dt (roll)",
                },
                {
                    "name": "pulse_width",
                    "default": float(s3d.get('default_pulse_width', 0.5)),
                    "mix": float(gmap.get('pulse_width_radial_mix', 0.0)),
                    "geom_y": radial_norm,
                    "source_tag": "radial_blend",
                    "desc_flat": "Flat default pulse width (spatial 3D mode)",
                    "desc_geom": "Pulse width blended from radial distance",
                    "mix_key": "pulse_width_radial_mix",
                },
                {
                    "name": "pulse_rise_time",
                    "default": float(s3d.get('default_pulse_rise_time', 0.5)),
                    "mix": float(gmap.get('pulse_rise_azimuth_mix', 0.0)),
                    "geom_y": azimuth_norm,
                    "source_tag": "azimuth_blend",
                    "desc_flat": "Flat default pulse rise time (spatial 3D mode)",
                    "desc_geom": "Pulse rise time blended from azimuth",
                    "mix_key": "pulse_rise_azimuth_mix",
                },
            ]
            for ch in pulse_channels:
                v = float(np.clip(ch["default"], 0.0, 1.0))
                mix = float(np.clip(ch["mix"], 0.0, 1.0))
                # Optional second modulator (only pulse_frequency
                # carries one today). Sum-and-clip semantics: each
                # modulator contributes an offset from the default,
                # and both offsets add into the final y.
                mix2 = float(np.clip(ch.get("mix2", 0.0), 0.0, 1.0))
                if mix > 0.0 or mix2 > 0.0:
                    y = np.full_like(t, v, dtype=float)
                    if mix > 0.0:
                        y = y + mix * (ch["geom_y"] - v)
                    if mix2 > 0.0:
                        y = y + mix2 * (ch["geom_y2"] - v)
                    y = np.clip(y, 0.0, 1.0)
                    # EXPERIMENTAL — pulse_width-tail reverb. Only
                    # applies to pulse_width (others have no analog
                    # knob) and only when reverb is enabled. No-op
                    # when mix = 0.
                    if (ch["name"] == "pulse_width"
                            and _rv_cfg.get('enabled', False)):
                        _pwt = _rv_cfg.get('pulse_width_tail', {}) or {}
                        y = _reverb_feedback_delay(
                            y, 50.0,
                            float(_pwt.get('delay_ms', 150.0)),
                            float(_pwt.get('feedback', 0.3)),
                            float(_pwt.get('mix', 0.0)))
                    pulse_fs = Funscript(t.copy(), y.copy())
                    meta = {
                        "mode": "spatial_3d_linear",
                        "source": ch["source_tag"],
                        "default_value": v,
                        ch["mix_key"]: mix,
                    }
                    if mix2 > 0.0:
                        meta[ch["mix_key2"]] = mix2
                        meta["source"] = ch["source_tag"] + "+omega"
                        desc = (f"{ch['desc_geom']} "
                                f"+ {ch['desc_geom2']} "
                                f"(mix={mix:.2f}+{mix2:.2f}, "
                                f"default={v:.2f})")
                    else:
                        desc = (f"{ch['desc_geom']} "
                                f"(mix={mix:.2f}, default={v:.2f})")
                else:
                    pulse_fs = Funscript(
                        t_bounds.copy(),
                        np.array([v, v], dtype=float))
                    meta = {
                        "mode": "spatial_3d_linear",
                        "source": "flat_default",
                        "value": v,
                    }
                    desc = ch["desc_flat"]
                self._add_metadata(pulse_fs, ch["name"], desc, meta)
                pulse_fs.save_to_path(self._get_temp_path(ch["name"]))
                shutil.copy2(
                    self._get_temp_path(ch["name"]),
                    self._get_output_path(ch["name"]))

            # Central-mode zip + cleanup follow the same rules as process().
            if (self.params.get('file_management', {}).get('mode')
                    == 'central'
                    and self.params.get('file_management', {})
                    .get('zip_output', False)):
                self._zip_output_files(progress_callback)

            if (self.params.get('options', {})
                    .get('delete_intermediary_files', True)):
                self._cleanup_intermediary_files()

            self._update_progress(progress_callback, 100,
                                  "Triplet complete.")
            return True
        except Exception as e:
            self._update_progress(
                progress_callback, -1,
                f"Error in triplet processing: {e}")
            return False

    def _setup_directories(self):
        """Create the temporary directory for intermediary files and set output directory."""
        # Set output directory based on file management mode
        file_mgmt = self.params.get('file_management', {})
        mode = file_mgmt.get('mode', 'local')

        if mode == 'central':
            # Use central folder if specified
            central_path = file_mgmt.get('central_folder_path', '').strip()
            if central_path:
                self.output_dir = Path(central_path)
                # Ensure the output directory exists
                self.output_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Fallback to local mode if central path not set
                self.output_dir = self.input_path.parent
        else:
            # Local mode: use input file directory
            self.output_dir = self.input_path.parent

        # Create temporary directory
        self.temp_dir = self.input_path.parent / "funscript-temp"
        self.temp_dir.mkdir(exist_ok=True)

    def _cleanup_intermediary_files(self):
        """Remove the temporary directory and all its contents."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _delete_existing_output_files(self, progress_callback: Optional[Callable]):
        """Delete existing restim files in central mode (when backups are disabled)."""
        try:
            # Collect all existing restim files for this input
            existing_files = list(self.output_dir.glob(f"{self.filename_only}.*.funscript"))

            # Also delete any existing output zip from a previous run
            existing_zip = self.output_dir / f"{self.filename_only}.zip"
            if existing_zip.exists():
                existing_files.append(existing_zip)

            if not existing_files:
                return  # No existing files to delete

            self._update_progress(progress_callback, 3, f"Cleaning {len(existing_files)} existing output files...")

            for file_path in existing_files:
                file_path.unlink()

            self._update_progress(progress_callback, 4, f"Deleted {len(existing_files)} old files")

        except Exception as e:
            # Log the error but don't fail the entire process
            self._update_progress(progress_callback, -1, f"Warning: Failed to delete existing files: {str(e)}")

    def _zip_output_files(self, progress_callback: Optional[Callable]):
        """Zip all output funscript files in central folder into a single zip, then delete the originals."""
        try:
            output_files = list(self.output_dir.glob(f"{self.filename_only}.*.funscript"))
            if not output_files:
                return

            zip_path = self.output_dir / f"{self.filename_only}.zip"
            self._update_progress(progress_callback, 97, f"Creating output zip: {zip_path.name}")

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in output_files:
                    zipf.write(file_path, file_path.name)

            for file_path in output_files:
                file_path.unlink()

            self._update_progress(progress_callback, 98, f"Zipped {len(output_files)} files into {zip_path.name}")

        except Exception as e:
            self._update_progress(progress_callback, -1, f"Warning: Failed to create output zip: {str(e)}")

    def _create_backup(self, progress_callback: Optional[Callable]):
        """Create backup zip of existing restim files in central mode before overwriting."""
        try:
            from datetime import datetime

            # Collect all existing restim files for this input
            backup_files = []
            for file_path in self.output_dir.glob(f"{self.filename_only}.*.funscript"):
                backup_files.append(file_path)

            if not backup_files:
                return  # No existing files to backup

            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_zip_path = self.output_dir / f"{self.filename_only}.backup-{timestamp}.zip"

            self._update_progress(progress_callback, 3, f"Creating backup: {backup_zip_path.name}")

            # Create the backup zip file
            with zipfile.ZipFile(backup_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in backup_files:
                    # Add file to zip with just the filename (no path)
                    zipf.write(file_path, file_path.name)

            # Delete the original files after successful backup
            for file_path in backup_files:
                file_path.unlink()

            self._update_progress(progress_callback, 4, f"Backup created with {len(backup_files)} files")

        except Exception as e:
            # Log the error but don't fail the entire process
            self._update_progress(progress_callback, -1, f"Warning: Failed to create backup: {str(e)}")

    def _update_progress(self, progress_callback: Optional[Callable], percent: int, message: str):
        """Update progress if callback is provided."""
        if progress_callback:
            progress_callback(percent, message)

    def _get_temp_path(self, suffix: str) -> Path:
        """Get path for a temporary file."""
        return self.temp_dir / f"{self.filename_only}.{suffix}.funscript"

    def _get_output_path(self, suffix: str) -> Path:
        """Get path for a final output file."""
        return self.output_dir / f"{self.filename_only}.{suffix}.funscript"

    def _copy_if_exists(self, source_suffix: str, dest_suffix: str) -> bool:
        """Copy auxiliary file if it exists."""
        source_path = self.input_path.parent / f"{self.filename_only}.{source_suffix}.funscript"
        if source_path.exists():
            dest_path = self._get_temp_path(dest_suffix)
            shutil.copy2(source_path, dest_path)
            return True
        return False

    def _output_file_exists(self, suffix: str) -> bool:
        """Check if an output file already exists in the output directory."""
        output_path = self._get_output_path(suffix)
        return output_path.exists()

    def _copy_output_to_temp_if_exists(self, suffix: str) -> bool:
        """Copy existing output file to temp directory if it exists."""
        output_path = self._get_output_path(suffix)
        if output_path.exists():
            dest_path = self._get_temp_path(suffix)
            shutil.copy2(output_path, dest_path)
            return True
        return False

    def _create_events_template(self, events_file_path: Path):
        """Create an empty events.yml template file with helpful comments."""
        template_content = """# Custom Events File
# Add timed events to modify the generated funscript files.
# All time values are in MILLISECONDS.
# Event names must match definitions in config.event_definitions.yml
#
# Example event structure:
# events:
#   - time: 60000        # Event at 1 minute (60,000 ms)
#     name: edge         # Event name from event_definitions.yml
#     params:            # Optional parameter overrides
#       duration_ms: 15000

events:
  # Add your custom events here
"""
        try:
            with open(events_file_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
        except Exception as e:
            # Don't fail the entire process if template creation fails
            print(f"Warning: Failed to create events template: {str(e)}")

    def _execute_pipeline(self, main_funscript: Funscript, progress_callback: Optional[Callable]):
        """Execute the complete processing pipeline."""

        # Noise gate (pre-pipeline). Pulls low-activity regions toward
        # a rest level so quantization, alpha/beta generation, and all
        # downstream stages see a gated signal rather than tracker
        # jitter / DC-offset idle sections.
        ng_cfg = self.params.get('noise_gate', {})
        if ng_cfg.get('enabled', False):
            try:
                from processing.noise_gate import apply_noise_gate
                self._update_progress(
                    progress_callback, 5,
                    "Applying noise gate to main funscript...")
                main_funscript = apply_noise_gate(
                    main_funscript,
                    threshold=float(ng_cfg.get('threshold', 0.05)),
                    window_s=float(ng_cfg.get('window_s', 0.5)),
                    attack_s=float(ng_cfg.get('attack_s', 0.02)),
                    release_s=float(ng_cfg.get('release_s', 0.3)),
                    rest_level=float(ng_cfg.get('rest_level', 0.5)),
                )
                print(
                    f"Noise gate: threshold={ng_cfg.get('threshold')} "
                    f"window_s={ng_cfg.get('window_s')} "
                    f"attack_s={ng_cfg.get('attack_s')} "
                    f"release_s={ng_cfg.get('release_s')} "
                    f"rest_level={ng_cfg.get('rest_level')}")
            except (ValueError, TypeError) as e:
                print(f"Warning: noise gate skipped: {e}")

        # Trochoid quantization (pre-pipeline). Snap input positions to N
        # discrete levels derived from a hypotrochoid sample. Applied to the
        # main funscript so every downstream file inherits the quantized
        # signal.
        tq_cfg = self.params.get('trochoid_quantization', {})
        if tq_cfg.get('enabled', False):
            try:
                n_pts = int(tq_cfg.get('n_points', 23))
                projection = str(tq_cfg.get('projection', 'radius'))
                # Family + params (with backward-compat for old flat config).
                family = str(tq_cfg.get('family',
                                        tq_cfg.get('curve_type', 'hypo')))
                params_by_family = tq_cfg.get('params_by_family') or {}
                family_params = dict(params_by_family.get(family) or {})
                if not family_params and family in ('hypo', 'epi'):
                    # Old-style flat keys
                    family_params = {
                        'R': float(tq_cfg.get('R', 5.0)),
                        'r': float(tq_cfg.get('r', 3.0)),
                        'd': float(tq_cfg.get('d', 2.0)),
                    }
                if not family_params:
                    family_params = dict(
                        _CURVE_FAMILY_DEFAULTS.get(family, {}).get('params', {}))
                self._update_progress(
                    progress_callback, 7,
                    f"Quantizing input to {n_pts} {family} levels...")
                main_funscript = quantize_to_curve(
                    main_funscript, n_pts, family, family_params, projection)
                if tq_cfg.get('deduplicate_holds', False):
                    before = len(main_funscript.y)
                    main_funscript = deduplicate_holds(main_funscript)
                    print(f"  dedup_holds: {before} -> {len(main_funscript.y)} samples")
                print(f"Trochoid quantization: family={family} n={n_pts} "
                      f"params={family_params} projection={projection}")
            except (ValueError, TypeError) as e:
                print(f"Warning: trochoid quantization skipped: {e}")

        # Phase 1: Auxiliary File Preparation (10-20%)
        self._update_progress(progress_callback, 10, "Preparing auxiliary files...")

        # Check if we should overwrite existing output files
        overwrite_existing = self.params.get('options', {}).get('overwrite_existing_files', False)

        # Always reuse ramp and speed if they exist (they are intentionally provided)
        # These are only generated in temp folder if not present
        ramp_exists = self._copy_if_exists("ramp", "ramp")
        speed_exists = self._copy_if_exists("speed", "speed")

        # Copy alpha/beta files based on overwrite setting
        # When overwrite=False: reuse existing alpha/beta from output directory if available
        # When overwrite=True: regenerate alpha/beta even if they exist
        alpha_exists = False if overwrite_existing else self._copy_output_to_temp_if_exists("alpha")
        beta_exists = False if overwrite_existing else self._copy_output_to_temp_if_exists("beta")

        # Generate speed early if needed for alpha/beta generation
        speed_funscript = None
        if not alpha_exists or not beta_exists:
            # Need speed funscript for alpha/beta generation
            if not speed_exists:
                self._update_progress(progress_callback, 12, "Generating speed file for alpha/beta...")
                speed_method = self.params['speed'].get('method', 'rolling_average')
                savgol_opts = self.params['speed'].get('savgol_options', {})
                speed_funscript = convert_to_speed(
                    main_funscript,
                    self.params['general']['speed_window_size'],
                    self.params['speed']['interpolation_interval'],
                    method=speed_method,
                    savgol_options=savgol_opts
                )
                speed_funscript.save_to_path(self._get_temp_path("speed"))
                speed_exists = True
            else:
                speed_funscript = Funscript.from_file(self._get_temp_path("speed"))

        # Always generate alpha and beta files (they are mandatory)
        if not alpha_exists or not beta_exists:
            self._update_progress(progress_callback, 15, "Generating alpha and beta files from main funscript...")
            alpha_beta_config = self.params.get('alpha_beta_generation', {})
            # Derive points_per_second from interpolation_interval so the alpha/beta grid
            # matches the speed funscript's grid exactly (used as fallback when speed_funscript is None).
            interpolation_interval = self.params['speed']['interpolation_interval']
            points_per_second = round(1.0 / interpolation_interval)
            algorithm = alpha_beta_config.get('algorithm', 'circular')
            min_distance_from_center = alpha_beta_config.get('min_distance_from_center', 0.1)
            speed_threshold_percent = alpha_beta_config.get('speed_threshold_percent', 50)
            direction_change_probability = alpha_beta_config.get('direction_change_probability', 0.1)
            min_stroke_amplitude = alpha_beta_config.get('min_stroke_amplitude', 0.0)
            point_density_scale = alpha_beta_config.get('point_density_scale', 1.0)
            alpha_funscript, beta_funscript = generate_alpha_beta_from_main(
                main_funscript, speed_funscript, points_per_second, algorithm, min_distance_from_center, speed_threshold_percent, direction_change_probability,
                min_stroke_amplitude=min_stroke_amplitude, point_density_scale=point_density_scale
            )

            if not alpha_exists:
                alpha_funscript.save_to_path(self._get_temp_path("alpha"))
                alpha_exists = True

            if not beta_exists:
                beta_funscript.save_to_path(self._get_temp_path("beta"))
                beta_exists = True

        # Generate prostate alpha and beta files if prostate generation is enabled
        if self.params.get('prostate_generation', {}).get('generate_prostate_files', True):
            # Copy prostate files based on overwrite setting from output directory
            # When overwrite=False: reuse existing prostate files if available
            # When overwrite=True: regenerate prostate files even if they exist
            alpha_prostate_exists = False if overwrite_existing else self._copy_output_to_temp_if_exists("alpha-prostate")
            beta_prostate_exists = False if overwrite_existing else self._copy_output_to_temp_if_exists("beta-prostate")

            if not alpha_prostate_exists or not beta_prostate_exists:
                self._update_progress(progress_callback, 17, "Generating prostate alpha and beta files from main funscript...")
                prostate_config = self.params.get('prostate_generation', {})
                prostate_points_per_second = prostate_config.get('points_per_second', 25)
                prostate_algorithm = prostate_config.get('algorithm', 'tear-shaped')
                prostate_min_distance = prostate_config.get('min_distance_from_center', 0.5)
                prostate_generate_from_inverted = prostate_config.get('generate_from_inverted', True)

                alpha_prostate_funscript, beta_prostate_funscript = generate_alpha_beta_prostate_from_main(
                    main_funscript, prostate_points_per_second, prostate_algorithm,
                    prostate_min_distance, prostate_generate_from_inverted
                )

                if not alpha_prostate_exists:
                    alpha_prostate_funscript.save_to_path(self._get_temp_path("alpha-prostate"))
                if not beta_prostate_exists:
                    beta_prostate_funscript.save_to_path(self._get_temp_path("beta-prostate"))

        # Traveling Wave mapping — time-driven crest along the shaft
        # that fires each electrode as the crest passes its position.
        # Highest-priority E1-E4 override: when enabled, both trochoid-
        # spatial and response-curve generation below are skipped.
        tw_cfg = self.params.get('traveling_wave', {}) or {}
        wave_active = bool(tw_cfg.get('enabled', False))
        if wave_active:
            self._update_progress(progress_callback, 17,
                                  "Generating traveling-wave E1-E4...")
            try:
                positions = tuple(
                    float(p) for p in tw_cfg.get(
                        'electrode_positions', [0.85, 0.65, 0.45, 0.25]))
                direction = str(tw_cfg.get('direction', 'bounce'))
                envelope = str(tw_cfg.get('envelope_mode', 'input'))
                wave_speed = float(tw_cfg.get('wave_speed_hz', 1.0))
                wave_width = float(tw_cfg.get('wave_width', 0.18))
                speed_mod = float(tw_cfg.get('speed_mod', 0.0))
                sharpness = float(tw_cfg.get('sharpness', 1.0))
                vel_window = float(tw_cfg.get('velocity_window_s', 0.10))
                noise_gate = float(tw_cfg.get('noise_gate', 0.10))
                exclusive = bool(tw_cfg.get('exclusive', False))
                wave_fs = generate_wave_funscripts(
                    main_funscript,
                    electrode_positions=positions,
                    wave_speed_hz=wave_speed,
                    wave_width=wave_width,
                    direction=direction,
                    envelope_mode=envelope,
                    speed_mod=speed_mod,
                    sharpness=sharpness,
                    velocity_window_s=vel_window,
                    noise_gate=noise_gate,
                    exclusive=exclusive,
                )
                # Reuse the per-axis modulation block already used by the
                # response-curve path. Each axis's modulation config lives
                # under positional_axes.eN.modulation; if `enabled`, multiply
                # the wave's intensity envelope by the configured LFO.
                axes_cfg = self.params.get('positional_axes', {}) or {}
                for key, fs in wave_fs.items():
                    mod_cfg = ((axes_cfg.get(key) or {})
                               .get('modulation') or {})
                    mod_meta = None
                    if mod_cfg.get('enabled', False):
                        try:
                            fs = apply_modulation(
                                fs,
                                frequency_hz=float(mod_cfg.get(
                                    'frequency_hz', 0.5)),
                                depth=float(mod_cfg.get('depth', 0.15)),
                                phase_deg=float(mod_cfg.get(
                                    'phase_deg', 0.0))
                                    if mod_cfg.get('phase_enabled', True)
                                    else 0.0,
                            )
                            mod_meta = {
                                "frequency_hz": float(mod_cfg.get(
                                    'frequency_hz', 0.5)),
                                "depth": float(mod_cfg.get('depth', 0.15)),
                                "phase_deg": float(mod_cfg.get(
                                    'phase_deg', 0.0)),
                            }
                        except Exception as e:
                            print(f"[traveling_wave] modulation "
                                  f"on {key} skipped: {e}")
                    extra = {
                        "direction": direction,
                        "envelope_mode": envelope,
                        "wave_speed_hz": wave_speed,
                        "wave_width": wave_width,
                        "speed_mod": speed_mod,
                        "sharpness": sharpness,
                        "electrode_position": float(
                            positions[int(key[1:]) - 1]),
                    }
                    if mod_meta:
                        extra["modulation"] = mod_meta
                    self._add_metadata(
                        fs, f"motion_axis_{key}",
                        f"Traveling-wave {direction} ({key.upper()})",
                        extra)
                    fs.save_to_path(self._get_temp_path(key))
                print(f"Traveling wave: dir={direction} "
                      f"env={envelope} speed={wave_speed} "
                      f"width={wave_width} speed_mod={speed_mod}")
            except (ValueError, TypeError) as e:
                print(f"Warning: traveling wave skipped: {e}")
                wave_active = False

        # Trochoid Spatial mapping — alternative E1-E4 generator that
        # parameterizes a 2D curve by the input position and projects each
        # (x, y) onto N electrode directions. When enabled, OVERRIDES the
        # response-curve motion-axis generation below for the e1-e4 files.
        # Skipped if traveling-wave already produced e1-e4.
        ts_cfg = self.params.get('trochoid_spatial', {}) or {}
        spatial_active = (not wave_active) and bool(ts_cfg.get('enabled', False))
        if spatial_active:
            self._update_progress(progress_callback, 18,
                                  "Generating trochoid-spatial E1-E4...")
            try:
                family = str(ts_cfg.get('family', 'hypo'))
                params_by_family = ts_cfg.get('params_by_family') or {}
                family_params = dict(params_by_family.get(family) or {})
                if not family_params:
                    family_params = dict(
                        _CURVE_FAMILY_DEFAULTS.get(family, {})
                        .get('params', {}))
                angles = tuple(
                    float(a) for a in ts_cfg.get(
                        'electrode_angles_deg', [0, 90, 180, 270]))
                spatial_fs = generate_spatial_funscripts(
                    main_funscript, family, family_params,
                    electrode_angles_deg=angles,
                    mapping=str(ts_cfg.get('mapping', 'directional')),
                    sharpness=float(ts_cfg.get('sharpness', 1.0)),
                    cycles_per_unit=float(ts_cfg.get('cycles_per_unit', 1.0)),
                    normalize=str(ts_cfg.get('normalize', 'clamped')),
                    theta_offset=float(ts_cfg.get('theta_offset', 0.0)),
                    close_on_loop=bool(ts_cfg.get('close_on_loop', False)),
                    smoothing_enabled=bool(
                        ts_cfg.get('smoothing_enabled', False)),
                    smoothing_min_cutoff_hz=float(
                        ts_cfg.get('smoothing_min_cutoff_hz', 1.0)),
                    smoothing_beta=float(
                        ts_cfg.get('smoothing_beta', 0.05)),
                    blend_directional=float(
                        ts_cfg.get('blend_directional', 0.0)),
                    blend_tangent_directional=float(
                        ts_cfg.get('blend_tangent_directional', 0.0)),
                    blend_distance=float(
                        ts_cfg.get('blend_distance', 0.0)),
                    blend_amplitude=float(
                        ts_cfg.get('blend_amplitude', 0.0)),
                    electrode_gain=ts_cfg.get('electrode_gain'),
                    output_limiter_enabled=bool(
                        ts_cfg.get('output_limiter_enabled', False)),
                    output_limiter_threshold=float(
                        ts_cfg.get('output_limiter_threshold', 0.85)),
                    velocity_weight_enabled=bool(
                        ts_cfg.get('velocity_weight_enabled', False)),
                    velocity_weight_floor=float(
                        ts_cfg.get('velocity_weight_floor', 0.0)),
                    velocity_weight_response=float(
                        ts_cfg.get('velocity_weight_response', 1.0)),
                    velocity_weight_smoothing_hz=float(
                        ts_cfg.get('velocity_weight_smoothing_hz', 3.0)),
                    velocity_weight_normalization_percentile=float(
                        ts_cfg.get('velocity_weight_normalization_percentile', 0.99)),
                    velocity_weight_gate_threshold=float(
                        ts_cfg.get('velocity_weight_gate_threshold', 0.05)),
                )
                for key, fs in spatial_fs.items():
                    self._add_metadata(
                        fs, f"motion_axis_{key}",
                        f"Trochoid-spatial {family} mapping ({key.upper()})",
                        {
                            "family": family,
                            "params": family_params,
                            "mapping": str(ts_cfg.get('mapping', 'directional')),
                            "sharpness": float(ts_cfg.get('sharpness', 1.0)),
                            "cycles_per_unit": float(
                                ts_cfg.get('cycles_per_unit', 1.0)),
                            "normalize": str(
                                ts_cfg.get('normalize', 'clamped')),
                            "electrode_angle_deg": float(
                                angles[int(key[1:]) - 1]),
                        })
                    fs.save_to_path(self._get_temp_path(key))
                print(f"Trochoid spatial: family={family} "
                      f"mapping={ts_cfg.get('mapping')} "
                      f"sharpness={ts_cfg.get('sharpness')} "
                      f"cycles_per_unit={ts_cfg.get('cycles_per_unit')}")
            except (ValueError, TypeError) as e:
                print(f"Warning: trochoid spatial skipped: {e}")
                spatial_active = False

        # Motion Axis Generation (18-19%) — skipped if traveling-wave or
        # trochoid-spatial already produced e1-e4 above.
        if (not wave_active and not spatial_active
                and self.params.get('positional_axes', {})
                .get('generate_motion_axis', False)):
            self._update_progress(progress_callback, 18, "Generating motion axis files...")
            motion_config = self.params.get('positional_axes', {})

            # Validate configuration
            config_errors = validate_motion_axis_config(motion_config)
            if config_errors:
                print(f"Motion axis configuration errors: {config_errors}")
            else:
                # Copy existing axis files first
                enabled_axes = [axis for axis in ['e1', 'e2', 'e3', 'e4']
                              if motion_config.get(axis, {}).get('enabled', False)]

                copied_files = copy_existing_axis_files(
                    self.input_path.parent,
                    self.temp_dir,
                    self.filename_only,
                    enabled_axes
                )

                # Generate any missing axis files
                axes_to_generate = [axis for axis in enabled_axes if axis not in copied_files]
                if axes_to_generate:
                    generate_config = {axis: motion_config[axis] for axis in axes_to_generate}
                    # Pass the physical_model block through so the
                    # per-axis cascade shift can be applied inside
                    # generate_motion_axes.
                    if 'physical_model' in motion_config:
                        generate_config['physical_model'] = motion_config['physical_model']
                    generated_files = generate_motion_axes(
                        main_funscript,
                        generate_config,
                        self.temp_dir,
                        self.filename_only
                    )

        # Phase-Shifted Output Generation (19%) — handled independently per mode
        axes_config = self.params.get('positional_axes', {})

        # 3P (legacy) phase shift: alpha/beta
        if axes_config.get('generate_legacy', False):
            phase_shift_config = axes_config.get('phase_shift', {})
            if phase_shift_config.get('enabled', False):
                self._update_progress(progress_callback, 19, "Generating phase-shifted versions (3P)...")
                delay_pct = phase_shift_config.get('delay_percentage', 10.0)
                print(f"3P phase shift enabled: delay={delay_pct}%")
                funscripts_to_shift = {}
                for key in ['alpha', 'beta']:
                    path = self._get_temp_path(key)
                    if path.exists():
                        funscripts_to_shift[key] = Funscript.from_file(path)
                if funscripts_to_shift:
                    shifted = generate_all_phase_shifted_funscripts(
                        funscripts_to_shift, main_funscript,
                        delay_pct,
                        phase_shift_config.get('min_segment_duration', 0.25)
                    )
                    for key, funscript in shifted.items():
                        funscript.save_to_path(self._get_temp_path(key))
                        print(f"  Saved phase-shifted file: {self._get_temp_path(key).name}")
                else:
                    print("  Warning: No alpha/beta funscripts found to phase-shift (3P)")

        # 4P (motion axis) phase shift: e1-e4
        if axes_config.get('generate_motion_axis', False):
            ma_phase_config = axes_config.get('motion_axis_phase_shift', axes_config.get('phase_shift', {}))
            if ma_phase_config.get('enabled', False):
                self._update_progress(progress_callback, 19, "Generating phase-shifted versions (4P)...")
                ma_delay_pct = ma_phase_config.get('delay_percentage', 10.0)
                print(f"4P phase shift enabled: delay={ma_delay_pct}%")
                funscripts_to_shift = {}
                for axis in ['e1', 'e2', 'e3', 'e4']:
                    path = self._get_temp_path(axis)
                    if path.exists():
                        funscripts_to_shift[axis] = Funscript.from_file(path)
                if funscripts_to_shift:
                    shifted = generate_all_phase_shifted_funscripts(
                        funscripts_to_shift, main_funscript,
                        ma_delay_pct,
                        ma_phase_config.get('min_segment_duration', 0.25)
                    )
                    for key, funscript in shifted.items():
                        funscript.save_to_path(self._get_temp_path(key))
                        print(f"  Saved phase-shifted file: {self._get_temp_path(key).name}")
                else:
                    print("  Warning: No E1-E4 funscripts found to phase-shift (4P)")

        # Phase 2: Core File Generation (20-40%)
        self._update_progress(progress_callback, 20, "Generating speed file...")

        # Generate speed if not already generated earlier
        speed_method = self.params['speed'].get('method', 'rolling_average')
        savgol_opts = self.params['speed'].get('savgol_options', {})
        if not speed_exists and speed_funscript is None:
            speed_funscript = convert_to_speed(
                main_funscript,
                self.params['general']['speed_window_size'],
                self.params['speed']['interpolation_interval'],
                method=speed_method,
                savgol_options=savgol_opts
            )
            speed_funscript.save_to_path(self._get_temp_path("speed"))
        elif speed_funscript is None:
            speed_funscript = Funscript.from_file(self._get_temp_path("speed"))

        # Invert speed
        speed_inverted = invert_funscript(speed_funscript)
        speed_inverted.save_to_path(self._get_temp_path("speed_inverted"))

        self._update_progress(progress_callback, 25, "Generating acceleration file...")

        # Generate acceleration from speed (same method used for speed-of-speed)
        accel_funscript = convert_to_speed(
            speed_funscript,
            self.params['general']['accel_window_size'],
            self.params['speed']['interpolation_interval'],
            method=speed_method,
            savgol_options=savgol_opts
        )
        accel_funscript.save_to_path(self._get_temp_path("accel"))

        self._update_progress(progress_callback, 30, "Generating volume ramp...")

        # Generate volume ramp if not provided
        if not ramp_exists:
            ramp_percent_per_hour = self.params.get('volume', {}).get('ramp_percent_per_hour', 15)
            ramp_funscript = make_volume_ramp(main_funscript, ramp_percent_per_hour)
            ramp_funscript.save_to_path(self._get_temp_path("ramp"))
        else:
            ramp_funscript = Funscript.from_file(self._get_temp_path("ramp"))

        # Invert ramp
        ramp_inverted = invert_funscript(ramp_funscript)
        ramp_inverted.save_to_path(self._get_temp_path("ramp_inverted"))

        # Phase 3: Frequency Processing (40-50%)
        self._update_progress(progress_callback, 40, "Processing frequency data...")

        # Check if pulse_frequency already exists
        if not overwrite_existing and self._output_file_exists("pulse_frequency"):
            self._update_progress(progress_callback, 42, "Reusing existing pulse_frequency...")
            pulse_frequency = Funscript.from_file(self._get_output_path("pulse_frequency"))
        elif self.params['frequency'].get('map_pulse_freq_to_position', False):
            # Map pulse frequency directly from input position
            pulse_frequency = map_funscript(
                main_funscript,
                self.params['frequency']['pulse_freq_min'],
                self.params['frequency']['pulse_freq_max']
            )
            self._add_metadata(pulse_frequency, "pulse_frequency", "Pulse frequency mapped from position", {
                "pulse_freq_min": self.params['frequency']['pulse_freq_min'],
                "pulse_freq_max": self.params['frequency']['pulse_freq_max'],
                "mode": "position_mapped"
            })
            pulse_frequency.save_to_path(self._get_output_path("pulse_frequency"))
        else:
            # Load alpha funscript for pulse frequency generation
            alpha_funscript = Funscript.from_file(self._get_temp_path("alpha"))

            # Combine alpha with speed (no pre-mapping)
            pulse_frequency_combined = combine_funscripts(
                speed_funscript,
                alpha_funscript,
                self.params['frequency']['pulse_frequency_combine_ratio']
            )

            # Map the combined result to the pulse frequency range (min/max)
            pulse_frequency = map_funscript(
                pulse_frequency_combined,
                self.params['frequency']['pulse_freq_min'],
                self.params['frequency']['pulse_freq_max']
            )

            self._add_metadata(pulse_frequency, "pulse_frequency", "Pulse frequency modulation", {
                "pulse_freq_min": self.params['frequency']['pulse_freq_min'],
                "pulse_freq_max": self.params['frequency']['pulse_freq_max'],
                "pulse_frequency_combine_ratio": self.params['frequency']['pulse_frequency_combine_ratio']
            })
            pulse_frequency.save_to_path(self._get_output_path("pulse_frequency"))

        # Generate alpha-prostate output using inverted main funscript (only if enabled)
        if self.params.get('prostate_generation', {}).get('generate_prostate_files', True):
            main_inverted = invert_funscript(main_funscript)
            self._add_metadata(main_inverted, "alpha-prostate", "Inverted main funscript for prostate stimulation")
            main_inverted.save_to_path(self._get_output_path("alpha-prostate"))
        else:
            main_inverted = invert_funscript(main_funscript)

        # Check if frequency already exists
        if not overwrite_existing and self._output_file_exists("frequency"):
            self._update_progress(progress_callback, 45, "Reusing existing frequency...")
            frequency = Funscript.from_file(self._get_output_path("frequency"))
        else:
            # Primary frequency generation
            frequency = combine_funscripts(
                ramp_funscript,
                speed_funscript,
                self.params['frequency']['frequency_ramp_combine_ratio']
            )
            freq_meta = {
                "frequency_ramp_combine_ratio": self.params['frequency']['frequency_ramp_combine_ratio']
            }
            direction_bias = float(self.params['frequency'].get('direction_bias', 0.0))
            if direction_bias > 0.0:
                polarity = str(self.params['frequency'].get('direction_polarity', 'up_higher'))
                smoothing_s = float(self.params['frequency'].get('direction_smoothing_s', 0.3))
                frequency = apply_direction_bias(
                    frequency, main_funscript,
                    bias=direction_bias,
                    polarity=polarity,
                    smoothing_s=smoothing_s,
                )
                freq_meta["direction_bias"] = direction_bias
                freq_meta["direction_polarity"] = polarity
                freq_meta["direction_smoothing_s"] = smoothing_s
            self._add_metadata(frequency, "frequency", "Primary frequency modulation", freq_meta)
            frequency.save_to_path(self._get_output_path("frequency"))

        # Phase 4: Volume Processing (50-70%)
        self._update_progress(progress_callback, 50, "Processing volume data...")

        # Check if volume already exists
        if not overwrite_existing and self._output_file_exists("volume"):
            self._update_progress(progress_callback, 52, "Reusing existing volume...")
            volume = Funscript.from_file(self._get_output_path("volume"))
        else:
            # Standard volume
            volume = combine_funscripts(
                ramp_funscript,
                speed_funscript,
                self.params['volume']['volume_ramp_combine_ratio'],
                self.params['general']['rest_level'],
                self.params['general']['ramp_up_duration_after_rest']
            )

            # Volume normalization
            if self.params['options']['normalize_volume']:
                volume_not_normalized = volume.copy()
                volume_not_normalized.save_to_path(self._get_temp_path("volume_not_normalized"))
                volume = normalize_funscript(volume)

            self._add_metadata(volume, "volume", "Standard volume control", {
                "volume_ramp_combine_ratio": self.params['volume']['volume_ramp_combine_ratio'],
                "rest_level": self.params['general']['rest_level'],
                "ramp_up_duration_after_rest": self.params['general']['ramp_up_duration_after_rest'],
                "normalized": self.params['options']['normalize_volume']
            })
            volume.save_to_path(self._get_output_path("volume"))

        # Prostate volume (only if enabled)
        if self.params.get('prostate_generation', {}).get('generate_prostate_files', True):
            # Check if volume-prostate already exists
            if not overwrite_existing and self._output_file_exists("volume-prostate"):
                self._update_progress(progress_callback, 55, "Reusing existing volume-prostate...")
            else:
                prostate_volume = combine_funscripts(
                    ramp_funscript,
                    speed_funscript,
                    self.params['volume']['volume_ramp_combine_ratio'] * self.params['volume']['prostate_volume_multiplier'],
                    self.params['volume']['prostate_rest_level'],
                    self.params['general']['ramp_up_duration_after_rest']
                )
                self._add_metadata(prostate_volume, "volume-prostate", "Prostate volume control", {
                    "volume_ramp_combine_ratio": self.params['volume']['volume_ramp_combine_ratio'],
                    "prostate_volume_multiplier": self.params['volume']['prostate_volume_multiplier'],
                    "prostate_rest_level": self.params['volume']['prostate_rest_level'],
                    "ramp_up_duration_after_rest": self.params['general']['ramp_up_duration_after_rest']
                })
                prostate_volume.save_to_path(self._get_output_path("volume-prostate"))

        # Phase 5: Pulse Parameters (70-90%)
        self._update_progress(progress_callback, 70, "Processing pulse parameters...")

        # Check if pulse_rise_time already exists
        if not overwrite_existing and self._output_file_exists("pulse_rise_time"):
            self._update_progress(progress_callback, 72, "Reusing existing pulse_rise_time...")
        else:
            # Generate pulse rise time using ramp_inverted and speed_inverted directly
            # Simplified approach without beta dependency
            pulse_rise_time = combine_funscripts(
                ramp_inverted,
                speed_inverted,
                self.params['pulse']['pulse_rise_combine_ratio']
            )
            pulse_rise_time = map_funscript(
                pulse_rise_time,
                self.params['pulse']['pulse_rise_min'],
                self.params['pulse']['pulse_rise_max']
            )
            self._add_metadata(pulse_rise_time, "pulse_rise_time", "Pulse rise time modulation", {
                "pulse_rise_combine_ratio": self.params['pulse']['pulse_rise_combine_ratio'],
                "pulse_rise_min": self.params['pulse']['pulse_rise_min'],
                "pulse_rise_max": self.params['pulse']['pulse_rise_max']
            })
            pulse_rise_time.save_to_path(self._get_output_path("pulse_rise_time"))

        # Check if pulse_width already exists
        if not overwrite_existing and self._output_file_exists("pulse_width"):
            self._update_progress(progress_callback, 75, "Reusing existing pulse_width...")
        else:
            # Generate pulse width using inverted original funscript
            # Reuse main_inverted from alpha-prostate generation above
            pulse_width_main = limit_funscript(
                main_inverted,
                self.params['pulse']['pulse_width_min'],
                self.params['pulse']['pulse_width_max']
            )
            pulse_width_main.save_to_path(self._get_temp_path("pulse_width-main"))

            # Combine with speed for final pulse width
            pulse_width = combine_funscripts(
                speed_funscript,
                pulse_width_main,
                self.params['pulse']['pulse_width_combine_ratio']
            )
            self._add_metadata(pulse_width, "pulse_width", "Pulse width modulation", {
                "pulse_width_min": self.params['pulse']['pulse_width_min'],
                "pulse_width_max": self.params['pulse']['pulse_width_max'],
                "pulse_width_combine_ratio": self.params['pulse']['pulse_width_combine_ratio']
            })
            pulse_width.save_to_path(self._get_output_path("pulse_width"))

        # Phase 6: Copy remaining outputs (90-95%)
        self._update_progress(progress_callback, 90, "Finalizing outputs...")

        # Copy alpha and beta to outputs if they exist
        if alpha_exists:
            shutil.copy2(self._get_temp_path("alpha"), self._get_output_path("alpha"))
        if beta_exists:
            shutil.copy2(self._get_temp_path("beta"), self._get_output_path("beta"))

        # Copy prostate alpha and beta to outputs if they exist and prostate generation is enabled
        if self.params.get('prostate_generation', {}).get('generate_prostate_files', True):
            alpha_temp_path = self._get_temp_path("alpha-prostate")
            if alpha_temp_path.exists():
                shutil.copy2(alpha_temp_path, self._get_output_path("alpha-prostate"))

            beta_temp_path = self._get_temp_path("beta-prostate")
            if beta_temp_path.exists():
                shutil.copy2(beta_temp_path, self._get_output_path("beta-prostate"))

        axes_config = self.params.get('positional_axes', {})

        # Copy motion axis files to outputs if 4P mode is enabled
        if axes_config.get('generate_motion_axis', False):
            for axis_name in ['e1', 'e2', 'e3', 'e4']:
                if axes_config.get(axis_name, {}).get('enabled', False):
                    temp_path = self._get_temp_path(axis_name)
                    if temp_path.exists():
                        shutil.copy2(temp_path, self._get_output_path(axis_name))

        # Copy phase-shifted outputs — 3P (legacy)
        if axes_config.get('generate_legacy', False):
            ps_config = axes_config.get('phase_shift', {})
            if ps_config.get('enabled', False):
                for suffix in ['alpha-2', 'beta-2']:
                    temp_path = self._get_temp_path(suffix)
                    if temp_path.exists():
                        shutil.copy2(temp_path, self._get_output_path(suffix))

        # Copy phase-shifted outputs — 4P (motion axis)
        if axes_config.get('generate_motion_axis', False):
            ma_ps_config = axes_config.get('motion_axis_phase_shift', axes_config.get('phase_shift', {}))
            if ma_ps_config.get('enabled', False):
                for axis_name in ['e1', 'e2', 'e3', 'e4']:
                    if axes_config.get(axis_name, {}).get('enabled', False):
                        suffix = f"{axis_name}-2"
                        temp_path = self._get_temp_path(suffix)
                        if temp_path.exists():
                            shutil.copy2(temp_path, self._get_output_path(suffix))

        # Create empty events.yml template if it doesn't exist
        # Events file is always created in local directory (next to source .funscript)
        events_file_path = self.input_path.parent / f"{self.filename_only}.events.yml"
        if not events_file_path.exists():
            self._create_events_template(events_file_path)

        # Generate optional inverted files if enabled
        if self.params['advanced']['enable_pulse_frequency_inversion']:
            if not overwrite_existing and self._output_file_exists("pulse_frequency_inverted"):
                self._update_progress(progress_callback, 92, "Reusing existing pulse_frequency_inverted...")
            else:
                pulse_freq_inverted = invert_funscript(pulse_frequency)
                self._add_metadata(pulse_freq_inverted, "pulse_frequency_inverted", "Inverted pulse frequency modulation")
                pulse_freq_inverted.save_to_path(self._get_output_path("pulse_frequency_inverted"))

        if self.params['advanced']['enable_volume_inversion']:
            if not overwrite_existing and self._output_file_exists("volume_inverted"):
                self._update_progress(progress_callback, 93, "Reusing existing volume_inverted...")
            else:
                volume_inverted = invert_funscript(volume)
                self._add_metadata(volume_inverted, "volume_inverted", "Inverted volume control")
                volume_inverted.save_to_path(self._get_output_path("volume_inverted"))

        if self.params['advanced']['enable_frequency_inversion']:
            if not overwrite_existing and self._output_file_exists("frequency_inverted"):
                self._update_progress(progress_callback, 94, "Reusing existing frequency_inverted...")
            else:
                freq_inverted = invert_funscript(frequency)
                self._add_metadata(freq_inverted, "frequency_inverted", "Inverted frequency modulation")
                freq_inverted.save_to_path(self._get_output_path("frequency_inverted"))