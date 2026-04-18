import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import sys
from pathlib import Path
from typing import Optional

try:
    from tkinterdnd2 import TkinterDnD, DND_ALL
    HAS_DND = True
except ImportError:
    HAS_DND = False

sys.path.append(str(Path(__file__).parent.parent))
from config import ConfigManager
from processor import RestimProcessor
from ui.parameter_tabs import ParameterTabs
from ui.conversion_tabs import ConversionTabs
from ui.custom_events_builder import CustomEventsBuilderDialog
from ui.tooltip_helper import create_tooltip
import ui.theme as _theme


class MainWindow:
    def __init__(self):
        # Use TkinterDnD for drag-and-drop support if available
        if HAS_DND:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()
        self.root.title("Restim Funscript Processor")
        self.root.geometry("850x735")
        self.root.resizable(True, True)

        # Configuration
        self.config_manager = ConfigManager()
        self.current_config = self.config_manager.get_config()

        # Variables
        self.input_file_var = tk.StringVar()
        self.input_files = []  # Store list of selected files for batch processing
        self.last_processed_filename = None  # Track last processed filename for auto-loading events
        self.last_processed_directory = None  # Track directory of last processed file

        # Progress tracking
        self.progress_var = tk.IntVar()
        self.status_var = tk.StringVar(value="Ready to process...")

        self.setup_ui()
        self.update_config_display()
        dark = self.current_config.get('ui', {}).get('dark_mode', False)
        _theme.apply(dark)
        if dark:
            self._dark_btn.config(text='\u2600 Light')
            self.drop_zone.config(bg='#2d2d3f')

    def setup_ui(self):
        """Setup the main user interface."""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Help Contents", command=self._open_help)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._open_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        row = 0

        # Input file selection with drop zone
        input_frame = ttk.LabelFrame(main_frame, text="Input File (drop .funscript files here)", padding="5")
        input_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        input_frame.columnconfigure(1, weight=1)

        # Create a visible drop zone using tk.Frame (not ttk) for better DnD support
        self.drop_zone = tk.Frame(input_frame, bg='#f0f0f0', relief='sunken', bd=1)
        self.drop_zone.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=2, pady=2)
        self.drop_zone.columnconfigure(1, weight=1)

        ttk.Label(self.drop_zone, text="File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        self.input_entry = ttk.Entry(self.drop_zone, textvariable=self.input_file_var, width=50)
        self.input_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)

        ttk.Button(self.drop_zone, text="Browse...", command=self.browse_input_file).grid(row=0, column=2, padx=5, pady=5)

        row += 1

        # Spatial 3D Linear tuning panel. Sits between the drop zone and
        # the variants bar so tuning knobs live close to the file input
        # they affect. Visible only when the Spatial 3D checkbox (built
        # further down in the buttons row) is on.
        self._build_s3d_tuning_panel(main_frame, row)
        row += 1

        # Variants bar: A/B/C/D snapshots of the whole config. Switching
        # slots auto-saves the current UI state into the slot you're
        # leaving, then loads the new slot and refreshes all tabs.
        self._build_variants_bar(main_frame, row)
        row += 1

        # Parameters frame (1D to 2D conversion is now in Motion Axis tab).
        # Held as an instance ref so _s3d_update_visibility can hide the
        # whole block when Spatial 3D Linear is active — none of those
        # tabs feed the 3D pipeline.
        self._params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        params_frame = self._params_frame
        params_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        params_frame.columnconfigure(0, weight=1)
        params_frame.rowconfigure(0, weight=1)

        # Parameter tabs
        self.parameter_tabs = ParameterTabs(params_frame, self.current_config)
        self.parameter_tabs.main_window = self
        self.parameter_tabs.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Set callback for mode changes (for future extensibility)
        self.parameter_tabs.set_mode_change_callback(self.on_mode_change)

        # Set conversion callbacks for embedded conversion tabs
        self.parameter_tabs.set_conversion_callbacks(self.convert_basic_2d, self.convert_prostate_2d)

        row += 1

        # Progress and status frame
        status_frame = ttk.LabelFrame(main_frame, text="Output Status", padding="10")
        status_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)

        # Progress bar
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Status label
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=1, column=0, sticky=tk.W, pady=5)

        # Scrollable buttons frame
        buttons_outer = ttk.Frame(status_frame)
        buttons_outer.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        buttons_outer.columnconfigure(0, weight=1)

        buttons_canvas = tk.Canvas(buttons_outer, height=40, highlightthickness=0)
        buttons_scrollbar = ttk.Scrollbar(buttons_outer, orient=tk.HORIZONTAL, command=buttons_canvas.xview)
        buttons_frame = ttk.Frame(buttons_canvas)

        buttons_frame.bind("<Configure>", lambda e: buttons_canvas.configure(scrollregion=buttons_canvas.bbox("all")))
        buttons_canvas.create_window((0, 0), window=buttons_frame, anchor="nw")
        buttons_canvas.configure(xscrollcommand=buttons_scrollbar.set)

        buttons_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))
        buttons_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.process_button = ttk.Button(buttons_frame, text="Process All Files", command=self.start_processing)
        self.process_button.pack(side=tk.LEFT, padx=(0, 10))

        # Spatial 3D Linear toggle. When on, "Process All Files" treats
        # the first three dropped scripts as X/Y/Z of a single 3D signal
        # and emits one set of E1..EN funscripts; otherwise the drop
        # zone behaves as today (independent batch items).
        self._s3d_var = tk.BooleanVar(
            value=bool(self.current_config.get('spatial_3d_linear', {})
                       .get('enabled', False)))
        ttk.Checkbutton(
            buttons_frame,
            text="Spatial 3D (X,Y,Z triplet)",
            variable=self._s3d_var,
            command=self._on_s3d_toggle,
        ).pack(side=tk.LEFT, padx=(0, 10))
        # Now that _s3d_var exists, re-sync the tuning panel so it
        # matches the persisted "enabled" flag (panel was built earlier
        # in setup_ui and fell back to the config value then — this
        # tightens that to the Var the checkbox is bound to).
        self._s3d_update_visibility()

        self.process_motion_button = ttk.Button(buttons_frame, text="Process Motion Files", command=self.start_motion_processing)
        self.process_motion_button.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(buttons_frame, text="Custom Event Builder", command=self.open_custom_events_builder).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(buttons_frame, text="Animation Viewer", command=self._open_animation_viewer).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(buttons_frame, text="Signal Analyzer", command=self._open_signal_analyzer).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(buttons_frame, text="Compare Funscripts", command=self._open_compare_viewer).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(buttons_frame, text="Shaft Viewer", command=self._open_shaft_viewer).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(buttons_frame, text="Multi-Script 3D", command=self._open_multi_script_viewer).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(buttons_frame, text="Trochoid Viewer", command=self._open_trochoid_viewer).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(buttons_frame, text="T-Code Preview", command=self._open_tcode_preview).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(buttons_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(buttons_frame, text="Save Preset", command=self.save_config_preset).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(buttons_frame, text="Load Preset", command=self.load_config_preset).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(buttons_frame, text="Reset to Defaults", command=self.reset_config).pack(side=tk.LEFT, padx=(0, 10))

        self._dark_btn = ttk.Button(buttons_frame, text='\u263d Dark', width=8, command=self._toggle_dark_mode)
        self._dark_btn.pack(side=tk.LEFT)

        # Configure main_frame row weights
        main_frame.rowconfigure(row-1, weight=1)  # Parameters frame gets extra space

        # Enable drag-and-drop if available
        if HAS_DND:
            try:
                # Register drop target on the drop zone frame and entry widget
                for widget in (self.drop_zone, self.input_entry):
                    widget.drop_target_register(DND_ALL)
                    widget.dnd_bind('<<Drop>>', self.handle_drop)
                    widget.dnd_bind('<<DragEnter>>', self.on_drag_enter)
                    widget.dnd_bind('<<DragLeave>>', self.on_drag_leave)
            except Exception as e:
                pass  # Silently fail if drag-and-drop setup fails



    def open_custom_events_builder(self):
        """Open the new visual custom events builder."""
        dialog = CustomEventsBuilderDialog(
            self.root,
            self.current_config,
            self.last_processed_filename,
            self.last_processed_directory
        )
        self.root.wait_window(dialog)

    def _open_animation_viewer(self):
        """Open the animated 2D trajectory + electrode intensity viewer."""
        # Update config from current UI state before opening
        self.parameter_tabs.update_config(self.current_config)
        from ui.animation_viewer import AnimationViewer
        AnimationViewer(self.root, self)

    def _open_multi_script_viewer(self):
        """Open the multi-script 3D viewer (X/Y/Z from three funscripts)."""
        from ui.multi_script_viewer import MultiScriptViewer
        MultiScriptViewer(self.root, self)

    def _on_s3d_toggle(self):
        """Persist the Spatial 3D Linear enable flag into current_config."""
        self.current_config.setdefault(
            'spatial_3d_linear', {})['enabled'] = bool(self._s3d_var.get())
        self._s3d_update_visibility()

    def _build_s3d_tuning_panel(self, parent, grid_row):
        """LabelFrame with inline sliders/spinner/dropdown for the
        Spatial 3D Linear config. Slider commits write directly into
        self.current_config['spatial_3d_linear'] so the next Process run
        picks up the new values.
        """
        s3d = self.current_config.setdefault('spatial_3d_linear', {})
        # Registry for _s3d_refresh_from_config. Populated by
        # _s3d_make_slider; keyed by config path (tuple).
        self._s3d_slider_refs = {}
        self._s3d_panel = ttk.LabelFrame(
            parent, text="Spatial 3D Linear — tuning", padding="6")
        self._s3d_panel.grid(
            row=grid_row, column=0, columnspan=3,
            sticky=(tk.W, tk.E), pady=(0, 4))
        self._s3d_panel.columnconfigure(0, weight=1)

        # Row 1: electrode math (sharpness + n_electrodes + normalize)
        r1 = ttk.Frame(self._s3d_panel)
        r1.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self._s3d_make_slider(
            r1, "Sharpness", 'sharpness', 0.1, 8.0,
            float(s3d.get('sharpness', 1.0)), col=0, fmt="{:.1f}",
            tooltip=(
                "Exponent on the (1 − d/√3) intensity falloff, where d "
                "is distance from the signal point to each electrode. "
                "1.0 = smooth overlap between adjacent electrodes. "
                "4+ = highly selective (one electrode active at a time). "
                "Higher values make transitions feel more discrete."))

        n_elec_label = ttk.Label(r1, text="  Electrodes:")
        n_elec_label.grid(row=0, column=3, padx=(10, 2))
        self._s3d_n_elec_var = tk.IntVar(
            value=int(s3d.get('n_electrodes', 4)))
        n_elec_spin = ttk.Spinbox(
            r1, from_=2, to=4, width=4,
            textvariable=self._s3d_n_elec_var,
            command=lambda: self._s3d_write(
                'n_electrodes', int(self._s3d_n_elec_var.get())))
        n_elec_spin.grid(row=0, column=4)
        _elec_tt = ("Number of electrodes arranged along the shaft "
                    "axis (2–4). Must match your physical setup.")
        create_tooltip(n_elec_label, _elec_tt)
        create_tooltip(n_elec_spin, _elec_tt)

        norm_label = ttk.Label(r1, text="  Normalize:")
        norm_label.grid(row=0, column=5, padx=(10, 2))
        self._s3d_norm_var = tk.StringVar(
            value=str(s3d.get('normalize', 'clamped')))
        norm_combo = ttk.Combobox(
            r1, textvariable=self._s3d_norm_var,
            values=['clamped', 'per_frame'], width=10, state='readonly')
        norm_combo.grid(row=0, column=6)
        norm_combo.bind('<<ComboboxSelected>>', lambda e: self._s3d_write(
            'normalize', self._s3d_norm_var.get()))
        _norm_tt = (
            "clamped = raw per-electrode intensity clipped to [0, 1]. "
            "per_frame = renormalize each frame so the hottest electrode "
            "always hits 1.0 regardless of overall proximity. Use "
            "per_frame when you want the 'most active electrode' feel "
            "to stay consistent even as the signal drifts.")
        create_tooltip(norm_label, _norm_tt)
        create_tooltip(norm_combo, _norm_tt)

        # Row 2: envelope shaping. The volume ramp now mirrors the 1D
        # pipeline's make_volume_ramp (driven by volume.ramp_percent_per_hour)
        # so there's no 3D-specific knob here.
        r2 = ttk.Frame(self._s3d_panel)
        r2.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(4, 0))
        self._s3d_make_slider(
            r2, "Speed norm pct", 'speed_normalization_percentile',
            0.5, 1.0, float(s3d.get('speed_normalization_percentile', 0.99)),
            col=0, fmt="{:.2f}",
            tooltip=(
                "Percentile used to normalize |v| before clipping to "
                "[0, 1]. 0.99 ignores single-sample spikes so the "
                "envelope isn't flattened by one outlier. 1.0 uses the "
                "true peak (and is more spike-sensitive)."))
        self._s3d_make_slider(
            r2, "Freq×|v| mix", 'frequency_speed_mix', 0.0, 1.0,
            float(s3d.get('frequency_speed_mix', 0.0)), col=3, fmt="{:.2f}",
            tooltip=(
                "Blend the flat 'Freq default' with per-frame speed "
                "magnitude |v|. 0.0 = flat carrier (prior behavior). "
                "1.0 = fully |v|-driven (faster motion → higher "
                "frequency). 0.3 is a good starting point."))
        # Ramp % / hour drives the make_volume_ramp envelope. Shared
        # with the 1D pipeline (config key volume.ramp_percent_per_hour)
        # so tuning it here is the same as tuning it in the Volume tab.
        # Exposed in this panel so S3D mode is self-contained.
        self._s3d_make_slider(
            r2, "Ramp %/hr", None, 0.0, 40.0,
            float(self.current_config.get('volume', {})
                  .get('ramp_percent_per_hour', 15.0)),
            col=6, fmt="{:.1f}",
            external_path=('volume', 'ramp_percent_per_hour'),
            tooltip=(
                "Volume ramp rate from the 1D pipeline's make_volume_ramp, "
                "shared between S3D and 1D. Baseline volume rises this "
                "many % per hour of runtime; the 4-point envelope also "
                "adds an end fade-out. 15 is a reasonable default. "
                "Raise for more dramatic build-up, lower for a flatter "
                "envelope."))

        # Row 3a: frequency defaults (Freq default + Pulse freq)
        r3a = ttk.Frame(self._s3d_panel)
        r3a.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(4, 0))
        self._s3d_make_slider(
            r3a, "Freq default", 'default_frequency', 0.0, 1.0,
            float(s3d.get('default_frequency', 0.5)), col=0, fmt="{:.2f}",
            tooltip=(
                "Flat baseline for carrier frequency (0–1 normalized; "
                "restim maps to actual Hz on playback — roughly 500–"
                "1000 Hz). Active when Freq×|v| mix = 0; blended when "
                "it's > 0."))
        self._s3d_make_slider(
            r3a, "Pulse freq", 'default_pulse_frequency', 0.0, 1.0,
            float(s3d.get('default_pulse_frequency', 0.5)), col=3,
            fmt="{:.2f}",
            tooltip=(
                "Flat baseline for pulse rate (0–1 normalized). The 1D "
                "pipeline clips its pulse_frequency output to [0.5, "
                "0.99], so 0.5 here sits right at the 1D floor. Try "
                "0.75 if the pulse feels weak."))

        # Row 3b: pulse shape defaults (Pulse width + Pulse rise)
        # Split off from 3a so all four don't overflow the window.
        r3b = ttk.Frame(self._s3d_panel)
        r3b.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(4, 0))
        self._s3d_make_slider(
            r3b, "Pulse width", 'default_pulse_width', 0.0, 1.0,
            float(s3d.get('default_pulse_width', 0.5)), col=0, fmt="{:.2f}",
            tooltip=(
                "Flat baseline for pulse duration (0–1 normalized; "
                "fullness vs sharpness). Blended with radial distance "
                "when PW × radial > 0."))
        self._s3d_make_slider(
            r3b, "Pulse rise", 'default_pulse_rise_time', 0.0, 1.0,
            float(s3d.get('default_pulse_rise_time', 0.5)), col=3,
            fmt="{:.2f}",
            tooltip=(
                "Flat baseline for pulse attack shape (0–1 normalized; "
                "0 = sharp edge, 1 = soft onset). Blended with azimuth "
                "when PR × azimuth > 0."))

        # Row 4: electrode smoothing (Butterworth low-pass on E1..En)
        sm_cfg = s3d.setdefault('smoothing', {})
        r4 = ttk.Frame(self._s3d_panel)
        r4.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(4, 0))
        self._s3d_smooth_var = tk.BooleanVar(
            value=bool(sm_cfg.get('enabled', False)))
        smooth_chk = ttk.Checkbutton(
            r4, text="Smooth E1..En", variable=self._s3d_smooth_var,
            command=lambda: self._s3d_write(
                ('smoothing', 'enabled'),
                bool(self._s3d_smooth_var.get())))
        smooth_chk.grid(row=0, column=0, padx=(6, 10), sticky=tk.W)
        create_tooltip(
            smooth_chk,
            "Zero-phase Butterworth low-pass filter on E1..En. Reduces "
            "fast transitions that manifest as device flicker without "
            "time-shifting the signal. OFF by default.")
        self._s3d_make_slider(
            r4, "Cutoff Hz", ('smoothing', 'cutoff_hz'), 1.0, 24.0,
            float(sm_cfg.get('cutoff_hz', 8.0)), col=1, fmt="{:.1f}",
            tooltip=(
                "Low-pass cutoff frequency. Lower = more smoothing. "
                "8 Hz is a reasonable start if you hear flicker; drop "
                "to 5 Hz if it persists, raise to 12+ Hz if output "
                "feels smeared or dulled."))
        order_label = ttk.Label(r4, text="  Order:")
        order_label.grid(row=0, column=4, padx=(10, 2), sticky=tk.E)
        self._s3d_smooth_order_var = tk.IntVar(
            value=int(sm_cfg.get('order', 2)))
        order_spin = ttk.Spinbox(
            r4, from_=1, to=6, width=4,
            textvariable=self._s3d_smooth_order_var,
            command=lambda: self._s3d_write(
                ('smoothing', 'order'),
                int(self._s3d_smooth_order_var.get())))
        order_spin.grid(row=0, column=5)
        _order_tt = (
            "Butterworth filter order. Higher = steeper rolloff at the "
            "cost of more risk of ringing at fast transients. 2 is a "
            "safe default; bump to 3–4 only if you need a sharper "
            "cutoff.")
        create_tooltip(order_label, _order_tt)
        create_tooltip(order_spin, _order_tt)

        # Row 5: dedup-holds on E1..En after smoothing
        dd_cfg = s3d.setdefault('deduplicate_holds', {})
        r5 = ttk.Frame(self._s3d_panel)
        r5.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(4, 0))
        self._s3d_dedup_var = tk.BooleanVar(
            value=bool(dd_cfg.get('enabled', False)))
        dedup_chk = ttk.Checkbutton(
            r5, text="Dedup holds", variable=self._s3d_dedup_var,
            command=lambda: self._s3d_write(
                ('deduplicate_holds', 'enabled'),
                bool(self._s3d_dedup_var.get())))
        dedup_chk.grid(row=0, column=0, padx=(6, 10), sticky=tk.W)
        create_tooltip(
            dedup_chk,
            "After smoothing, drop interior samples of constant-within-"
            "tolerance runs on each electrode. Shrinks output files "
            "and prevents the device's linear interpolation from "
            "sloping across held windows. OFF by default.")
        self._s3d_make_slider(
            r5, "Tolerance", ('deduplicate_holds', 'tolerance'),
            0.0, 0.05, float(dd_cfg.get('tolerance', 0.005)),
            col=1, fmt="{:.3f}",
            tooltip=(
                "Absolute tolerance for 'constant'. 0.005 = 0.5% of "
                "full scale (conservative). Raise to 0.02 for "
                "aggressive compression; you'll start to feel the "
                "quantization steps as it gets higher."))

        # Row 6: geometric mapping for pulse channels. Each mix blends
        # the flat default with a per-frame geometry signal.
        gm_cfg = s3d.setdefault('geometric_mapping', {})
        r6 = ttk.Frame(self._s3d_panel)
        r6.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(4, 0))
        self._s3d_make_slider(
            r6, "PW × radial",
            ('geometric_mapping', 'pulse_width_radial_mix'), 0.0, 1.0,
            float(gm_cfg.get('pulse_width_radial_mix', 0.0)),
            col=0, fmt="{:.2f}",
            tooltip=(
                "Blend pulse_width with radial distance from the shaft "
                "axis. 0 = flat default, 1 = fully radial-driven. "
                "Further off-axis (more wobble) = fuller/wider pulse. "
                "Try 0.3 for subtle width modulation."))
        self._s3d_make_slider(
            r6, "PR × azimuth",
            ('geometric_mapping', 'pulse_rise_azimuth_mix'), 0.0, 1.0,
            float(gm_cfg.get('pulse_rise_azimuth_mix', 0.0)),
            col=3, fmt="{:.2f}",
            tooltip=(
                "Blend pulse_rise_time with azimuth around the shaft "
                "axis via (cos(φ)+1)/2. Wrap-free but sign-collapsing "
                "(rise-time is symmetric anyway). Creates a rotational "
                "'texture' feel as the signal orbits around the axis."))
        self._s3d_make_slider(
            r6, "PF × dr/dt",
            ('geometric_mapping', 'pulse_frequency_vradial_mix'),
            0.0, 1.0,
            float(gm_cfg.get('pulse_frequency_vradial_mix', 0.0)),
            col=6, fmt="{:.2f}",
            tooltip=(
                "Blend pulse_frequency with radial velocity dr/dt. "
                "Outward motion pushes it above 0.5, inward pulls it "
                "below. Sign-preserving — push and pull feel distinct. "
                "Percentile-normalized so spikes don't saturate."))

        # Row 7: temporal dynamics (τ knobs). Release acts on speed_y
        # (→ carrier when Freq×|v| mix > 0). Hold smooths the three
        # geometric source signals before they drive the pulse channels.
        r7 = ttk.Frame(self._s3d_panel)
        r7.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(4, 0))
        self._s3d_make_slider(
            r7, "Release τ (s)", 'release_tau_s', 0.0, 2.0,
            float(s3d.get('release_tau_s', 0.0)), col=0, fmt="{:.2f}",
            tooltip=(
                "Exponential release envelope on speed_y (the signal "
                "fueling Freq×|v| mix). Instant attack, decays toward "
                "0 when motion slows. Intensity hangs briefly after a "
                "pause instead of snapping dead. 0.0 = off; 0.3 ≈ 37% "
                "remaining 300 ms after motion stops; 1.0 = long hold. "
                "Only audible when Freq×|v| mix > 0."))
        self._s3d_make_slider(
            r7, "Hold τ (s)", ('geometric_mapping', 'hold_tau_s'),
            0.0, 1.0,
            float(gm_cfg.get('hold_tau_s', 0.0)), col=3, fmt="{:.2f}",
            tooltip=(
                "Symmetric EMA smoothing on the three geometric source "
                "signals (radial, azimuth, dr/dt) before they blend "
                "into the pulse channels. Kills chatter from small "
                "wobbles. 0.0 = off; 0.1 = ~100 ms settling; 0.3 = "
                "calm. Only audible when at least one PW/PR/PF "
                "geometric mix is > 0."))
        self._s3d_make_slider(
            r7, "Speed floor", 'speed_floor', 0.0, 1.0,
            float(s3d.get('speed_floor', 0.0)), col=6, fmt="{:.2f}",
            tooltip=(
                "Rest-level floor on speed_y AFTER the release "
                "envelope. Prevents the motion-derived carrier from "
                "going fully quiet during pauses. 0.0 = off "
                "(signal can decay to 0); 0.3 = always at least "
                "30% intensity. Only audible when Freq×|v| mix > 0."))

        # Row 8: EXPERIMENTAL reverb block. Enable + 4 wet/dry mixes.
        # Advanced params (delays, feedback) live in config.json —
        # the mix knobs are what you A/B with on-device.
        rv_cfg = s3d.setdefault('reverb', {})
        r8 = ttk.Frame(self._s3d_panel)
        r8.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=(4, 0))
        self._s3d_reverb_var = tk.BooleanVar(
            value=bool(rv_cfg.get('enabled', False)))
        reverb_chk = ttk.Checkbutton(
            r8, text="Reverb (experimental)",
            variable=self._s3d_reverb_var,
            command=lambda: self._s3d_write(
                ('reverb', 'enabled'),
                bool(self._s3d_reverb_var.get())))
        reverb_chk.grid(row=0, column=0, padx=(6, 10), sticky=tk.W)
        create_tooltip(
            reverb_chk,
            "EXPERIMENTAL. Reverb-analog effects at envelope rate — "
            "delayed + attenuated copies summed back into signals. "
            "Won't add energy to a dead baseline; tune the baseline "
            "first, then A/B these on top. Delay/feedback params are "
            "in config.json — these sliders are wet/dry mixes.")
        self._s3d_make_slider(
            r8, "Vol tail", ('reverb', 'volume_tail', 'mix'), 0.0, 1.0,
            float(rv_cfg.get('volume_tail', {}).get('mix', 0.0)),
            col=1, fmt="{:.2f}",
            tooltip=(
                "Single-tap feedback delay on volume_y. Discrete "
                "echoes of intensity; default 200 ms delay + 0.4 "
                "feedback. Set mix = 0.3 to hear an obvious echo on "
                "each stroke."))
        self._s3d_make_slider(
            r8, "Vol multi", ('reverb', 'volume_multitap', 'mix'),
            0.0, 1.0,
            float(rv_cfg.get('volume_multitap', {}).get('mix', 0.0)),
            col=4, fmt="{:.2f}",
            tooltip=(
                "FIR multi-tap comb on volume_y (Schroeder-style). "
                "Four incommensurate delays (83/127/191/307 ms) "
                "summed back for a dense, no-single-echo tail. Most "
                "traditionally reverb-like. Set mix = 0.3 for subtle "
                "spaciousness."))
        self._s3d_make_slider(
            r8, "Cross-E", ('reverb', 'cross_electrode', 'mix'),
            0.0, 1.0,
            float(rv_cfg.get('cross_electrode', {}).get('mix', 0.0)),
            col=7, fmt="{:.2f}",
            tooltip=(
                "Cross-electrode bleed: each E receives a delayed "
                "copy of its neighbors' envelopes. Creates sensation "
                "movement through the electrode array even when "
                "position is stationary. No audio equivalent. Set "
                "mix = 0.3 and feel it travel."))
        self._s3d_make_slider(
            r8, "PW tail",
            ('reverb', 'pulse_width_tail', 'mix'), 0.0, 1.0,
            float(rv_cfg.get('pulse_width_tail', {}).get('mix', 0.0)),
            col=10, fmt="{:.2f}",
            tooltip=(
                "Single-tap feedback delay on pulse_width. Only "
                "audible when PW × radial mix > 0 (otherwise "
                "pulse_width is flat and nothing to echo). Gives "
                "pulse character a 'breathing' quality."))

        self._s3d_update_visibility()

    def _s3d_make_slider(self, parent, label, config_key,
                         from_, to, initial, col, fmt="{:.2f}",
                         tooltip=None, external_path=None):
        """Create a labeled Scale widget wired to config.

        Default behavior writes under self.current_config['spatial_3d_linear']
        [config_key]. When external_path is given (tuple like
        ('volume', 'ramp_percent_per_hour')), the slider writes to
        that absolute config path instead — used for cross-section
        settings that belong in the S3D panel for self-containment.

        Uses a Tk variable so the Scale and its readout stay in sync;
        writes on ButtonRelease-1 to avoid thrashing the config dict.

        Registers the (var, readout, fmt) tuple in self._s3d_slider_refs
        so variant switches can refresh all sliders from config. The
        registry key encodes whether the path is s3d-relative or
        absolute (prefixed with a sentinel).
        """
        text_label = ttk.Label(parent, text=label + ":")
        text_label.grid(row=0, column=col, padx=(6, 2), sticky=tk.E)
        var = tk.DoubleVar(value=float(initial))
        readout = ttk.Label(parent, text=fmt.format(float(initial)),
                            width=6, anchor=tk.W)
        readout.grid(row=0, column=col + 2, padx=(2, 6), sticky=tk.W)

        def on_drag(val):
            try:
                readout.config(text=fmt.format(float(val)))
            except (TypeError, ValueError):
                pass

        if external_path is not None:
            abs_path = tuple(external_path)

            def on_release(_event=None):
                self._write_config_path(abs_path, float(var.get()))
            ref_key = ('__abs__',) + abs_path
        else:
            def on_release(_event=None):
                self._s3d_write(config_key, float(var.get()))
            ref_key = tuple(config_key) if isinstance(
                config_key, (tuple, list)) else (config_key,)

        scale = ttk.Scale(parent, from_=from_, to=to, orient=tk.HORIZONTAL,
                          length=120, variable=var, command=on_drag)
        scale.grid(row=0, column=col + 1, padx=(0, 2))
        scale.bind("<ButtonRelease-1>", on_release)
        if tooltip:
            for w in (text_label, scale, readout):
                create_tooltip(w, tooltip)
        self._s3d_slider_refs[ref_key] = (var, readout, fmt)
        return scale

    def _write_config_path(self, path, value):
        """Write `value` into self.current_config at absolute `path`
        (tuple of keys), creating intermediate dicts as needed."""
        node = self.current_config
        for k in path[:-1]:
            node = node.setdefault(k, {})
        node[path[-1]] = value

    def _s3d_refresh_from_config(self):
        """Re-sync the Spatial 3D Linear panel widgets with whatever
        is currently in self.current_config['spatial_3d_linear'].
        Called after variant switches so slider positions, checkboxes,
        etc. reflect the now-active slot's saved values.

        Silently no-ops if the panel hasn't been built yet (early
        startup) — keeps variant init safe.
        """
        if not hasattr(self, '_s3d_slider_refs'):
            return
        s3d = self.current_config.get('spatial_3d_linear', {}) or {}

        def _get_path(root, path):
            node = root
            for key in path:
                if not isinstance(node, dict) or key not in node:
                    return None
                node = node[key]
            return node

        # Sliders (any depth). Paths starting with '__abs__' are
        # absolute under current_config; otherwise they're relative
        # to spatial_3d_linear.
        for path, (var, readout, fmt) in self._s3d_slider_refs.items():
            if path and path[0] == '__abs__':
                val = _get_path(self.current_config, path[1:])
            else:
                val = _get_path(s3d, path)
            if val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            var.set(v)
            try:
                readout.config(text=fmt.format(v))
            except (tk.TclError, TypeError, ValueError):
                pass

        # Non-slider widgets.
        if hasattr(self, '_s3d_n_elec_var'):
            self._s3d_n_elec_var.set(int(s3d.get('n_electrodes', 4)))
        if hasattr(self, '_s3d_norm_var'):
            self._s3d_norm_var.set(str(s3d.get('normalize', 'clamped')))
        if hasattr(self, '_s3d_smooth_var'):
            self._s3d_smooth_var.set(
                bool(s3d.get('smoothing', {}).get('enabled', False)))
        if hasattr(self, '_s3d_smooth_order_var'):
            self._s3d_smooth_order_var.set(
                int(s3d.get('smoothing', {}).get('order', 2)))
        if hasattr(self, '_s3d_dedup_var'):
            self._s3d_dedup_var.set(
                bool(s3d.get('deduplicate_holds', {}).get('enabled', False)))
        if hasattr(self, '_s3d_var'):
            self._s3d_var.set(bool(s3d.get('enabled', False)))
        self._s3d_update_visibility()

    def _s3d_write(self, key, value):
        """Persist a Spatial 3D Linear setting into current_config.

        `key` is either a top-level string or a tuple/list of keys for
        nested dicts (e.g. ('smoothing', 'cutoff_hz')).
        """
        s3d = self.current_config.setdefault('spatial_3d_linear', {})
        if isinstance(key, (tuple, list)):
            sub = s3d
            for k in key[:-1]:
                sub = sub.setdefault(k, {})
            sub[key[-1]] = value
        else:
            s3d[key] = value

    @staticmethod
    def _order_xyz_triplet(paths):
        """Reorder paths so the first three become X, Y, Z in a
        predictable way for the Spatial 3D Linear pipeline.

        If at least one path has an explicit axis marker in its basename
        (`.x.`, `.y.`, `.z.` with any case), those paths are assigned
        to the matching axis slot; the remaining slots are filled
        alphabetically from the unmarked paths. If no markers are
        present, the full list is simply sorted alphabetically by
        basename so the order is reproducible regardless of drop order.
        """
        if len(paths) < 3:
            return list(paths)

        import re
        marker_re = re.compile(r'\.([xyz])\.', re.IGNORECASE)
        slots = {'x': None, 'y': None, 'z': None}
        unmarked = []
        for p in paths:
            m = marker_re.search(Path(p).name)
            if m and slots[m.group(1).lower()] is None:
                slots[m.group(1).lower()] = p
            else:
                unmarked.append(p)
        unmarked.sort(key=lambda q: Path(q).name.lower())

        # If nothing was marked, just alphabetize everything.
        if all(v is None for v in slots.values()):
            return sorted(paths, key=lambda q: Path(q).name.lower())

        # Fill empty slots in X, Y, Z order from unmarked pool.
        ordered = []
        for axis in ('x', 'y', 'z'):
            if slots[axis] is not None:
                ordered.append(slots[axis])
            elif unmarked:
                ordered.append(unmarked.pop(0))
        # Any extras tacked on the end (won't be used by the triplet
        # processor but still visible in input_files).
        ordered.extend(unmarked)
        return ordered

    def _s3d_update_visibility(self):
        """Show/hide the Spatial 3D tuning panel based on the toggle.

        The panel is built during setup_ui BEFORE the checkbox (which
        lives in the buttons row below), so `_s3d_var` may not exist
        yet the first time we're called. Fall back to the config flag
        in that case so the panel's initial state still matches the
        persisted setting.
        """
        if not hasattr(self, '_s3d_panel'):
            return
        if hasattr(self, '_s3d_var'):
            enabled = bool(self._s3d_var.get())
        else:
            enabled = bool(self.current_config.get('spatial_3d_linear', {})
                           .get('enabled', False))
        if enabled:
            self._s3d_panel.grid()
        else:
            self._s3d_panel.grid_remove()
        # Hide the 1D Parameters tab-bar when S3D is active — none of
        # those tabs feed process_triplet (only volume.ramp_percent_per_hour
        # is shared, and it's mirrored into the S3D panel).
        if hasattr(self, '_params_frame'):
            if enabled:
                self._params_frame.grid_remove()
            else:
                self._params_frame.grid()

    def _open_signal_analyzer(self):
        """Open the signal analyzer to examine the loaded funscript."""
        self.parameter_tabs.update_config(self.current_config)
        from ui.signal_analyzer import SignalAnalyzer
        SignalAnalyzer(self.root, self)

    def _open_compare_viewer(self):
        """Open the dual-timeline comparison viewer.

        Pre-fills slot A with the currently selected input file (if any)
        so the common case (compare original vs processed output) is one
        click away. Slot B is left empty for the user to browse.
        """
        from ui.compare_viewer import CompareViewer
        first_file = None
        if hasattr(self, 'input_files') and self.input_files:
            first_file = self.input_files[0]
        CompareViewer(self.root, file_a=first_file)

    def _open_tcode_preview(self):
        """Open the T-Code Live Preview — streams to restim over UDP."""
        from ui.tcode_preview import TCodePreviewViewer
        TCodePreviewViewer(self.root, self)

    def _open_shaft_viewer(self):
        """Open the Shaft Viewer (cylinder with E1-E4 along the length).

        Syncs current UI settings into current_config first so the viewer's
        'auto' mode and 'Apply trochoid quantization' default reflect
        whatever is actively selected in the tabs.
        """
        self.parameter_tabs.update_config(self.current_config)
        from ui.shaft_viewer import ShaftViewer
        ShaftViewer(self.root, self)

    def _open_trochoid_viewer(self):
        """Open the Trochoid Viewer — 2D trochoid curve on the left,
        its shadow projected onto the shaft on the right. The pen on
        the curve is driven by the funscript y value; the shadow is
        the normalized x/y/radius projection of that point.
        """
        self.parameter_tabs.update_config(self.current_config)
        from ui.trochoid_viewer import TrochoidViewer
        TrochoidViewer(self.root, self)

    def _open_help(self):
        """Open the built-in help viewer."""
        from ui.help_viewer import HelpViewer
        HelpViewer(self.root)

    def _open_about(self):
        """Show about dialog."""
        from version import __version__, __app_name__, __url__
        from tkinter import messagebox
        messagebox.showinfo(
            "About",
            f"{__app_name__} v{__version__}\n\n"
            f"Converts .funscript files into e-stim control signals\n"
            f"for use with restim.\n\n"
            f"{__url__}",
            parent=self.root)

    def _toggle_dark_mode(self):
        _theme.toggle()
        dark = _theme.is_dark()
        self._dark_btn.config(text='\u2600 Light' if dark else '\u263d Dark')
        self.drop_zone.config(bg='#2d2d3f' if dark else '#f0f0f0')
        # Persist preference
        self.current_config.setdefault('ui', {})['dark_mode'] = dark
        self.save_config()

    def on_mode_change(self, mode):
        """Called when positional axis mode changes."""
        # Mode changes are now handled within the Motion Axis tab
        pass




    def browse_input_file(self):
        """Open file dialog to select input funscript file(s)."""
        file_paths = filedialog.askopenfilenames(
            title="Select Funscript File(s)",
            filetypes=[("Funscript files", "*.funscript"), ("All files", "*.*")]
        )
        if file_paths:
            s3d_active = bool(self.current_config.get(
                'spatial_3d_linear', {}).get('enabled', False))
            paths = list(file_paths)
            if s3d_active and len(paths) >= 3:
                paths = self._order_xyz_triplet(paths)
            self.input_files = paths
            # Update display.
            if s3d_active and len(self.input_files) >= 3:
                labels = ['X', 'Y', 'Z']
                summary = " / ".join(
                    f"{lab}: {Path(p).name}"
                    for lab, p in zip(labels, self.input_files[:3]))
                if len(self.input_files) > 3:
                    summary += f" (+{len(self.input_files) - 3} ignored)"
                self.input_file_var.set(summary)
            elif len(self.input_files) == 1:
                self.input_file_var.set(self.input_files[0])
            else:
                self.input_file_var.set(f"{len(self.input_files)} files selected")

    def on_drag_enter(self, event):
        """Visual feedback when dragging over drop zone."""
        self.drop_zone.config(bg='#d4edda')  # Light green
        return event.action

    def on_drag_leave(self, event):
        """Reset visual feedback when leaving drop zone."""
        self.drop_zone.config(bg='#f0f0f0')  # Original color
        return event.action

    def handle_drop(self, event):
        """Handle files dropped onto the window. Only accepts .funscript files."""
        # Reset drop zone color
        self.drop_zone.config(bg='#f0f0f0')
        # Parse dropped file paths - tkinterdnd2 returns space-separated paths
        # with curly braces around paths containing spaces
        dropped_data = event.data

        # Parse the dropped data - handles paths with spaces (wrapped in {})
        file_paths = []
        current_path = ""
        in_braces = False

        for char in dropped_data:
            if char == '{':
                in_braces = True
            elif char == '}':
                in_braces = False
                if current_path:
                    file_paths.append(current_path)
                    current_path = ""
            elif char == ' ' and not in_braces:
                if current_path:
                    file_paths.append(current_path)
                    current_path = ""
            else:
                current_path += char

        # Don't forget the last path if not in braces
        if current_path:
            file_paths.append(current_path)

        # Filter to only .funscript files
        funscript_files = [
            path for path in file_paths
            if path.lower().endswith('.funscript') and Path(path).exists()
        ]

        if funscript_files:
            # If Spatial 3D Linear is active and 3+ files were dropped,
            # reorder deterministically so the UI can show X/Y/Z and
            # the processor gets a predictable triplet regardless of
            # drop order. Prefer explicit .x. / .y. / .z. markers in
            # the basename; fall back to alphabetical.
            s3d_active = bool(self.current_config.get(
                'spatial_3d_linear', {}).get('enabled', False))
            if s3d_active and len(funscript_files) >= 3:
                funscript_files = self._order_xyz_triplet(funscript_files)
            self.input_files = funscript_files
            # Update display.
            if s3d_active and len(self.input_files) >= 3:
                labels = ['X', 'Y', 'Z']
                summary = " / ".join(
                    f"{lab}: {Path(p).name}"
                    for lab, p in zip(labels, self.input_files[:3]))
                if len(self.input_files) > 3:
                    summary += f" (+{len(self.input_files) - 3} ignored)"
                self.input_file_var.set(summary)
            elif len(self.input_files) == 1:
                self.input_file_var.set(self.input_files[0])
            else:
                self.input_file_var.set(f"{len(self.input_files)} files selected")
        elif file_paths:
            # Files were dropped but none were .funscript
            messagebox.showwarning(
                "Invalid Files",
                "Only .funscript files are accepted. Please drop .funscript files."
            )

    def convert_basic_2d(self):
        """Convert 1D funscript to 2D alpha/beta files using basic algorithms."""
        self._convert_2d('basic')

    def convert_prostate_2d(self):
        """Convert 1D funscript to 2D alpha-prostate/beta-prostate files."""
        self._convert_2d('prostate')

    def _convert_2d(self, conversion_type):
        """Common 2D conversion logic."""
        input_file = self.input_file_var.get().strip()

        if not input_file:
            messagebox.showerror("Error", "Please select an input file first.")
            return

        if not Path(input_file).exists():
            messagebox.showerror("Error", "Input file does not exist.")
            return

        if not input_file.lower().endswith('.funscript'):
            messagebox.showerror("Error", "Input file must be a .funscript file.")
            return

        # Disable the convert buttons during processing
        if hasattr(self.parameter_tabs, 'embedded_conversion_tabs'):
            self.parameter_tabs.embedded_conversion_tabs.set_button_state('disabled')

        # Start conversion in background thread
        conversion_thread = threading.Thread(target=self._perform_2d_conversion, args=(conversion_type,), daemon=True)
        conversion_thread.start()


    def _perform_2d_conversion(self, conversion_type):
        """Perform 2D conversion in background thread."""
        try:
            input_file = self.input_file_var.get().strip()
            input_path = Path(input_file)

            self.update_progress(10, "Loading input file...")

            # Import necessary modules
            from funscript import Funscript

            # Load main funscript
            main_funscript = Funscript.from_file(input_path)

            self.update_progress(30, "Converting to 2D...")

            # Determine which conversion_tabs to use (always use embedded 3P tab)
            if hasattr(self.parameter_tabs, 'embedded_conversion_tabs'):
                conversion_tabs = self.parameter_tabs.embedded_conversion_tabs
            else:
                conversion_tabs = self.conversion_tabs

            # Determine output directory - respect file_management mode (central vs local)
            file_mgmt = self.current_config.get('file_management', {})
            if file_mgmt.get('mode') == 'central':
                central_path = file_mgmt.get('central_folder_path', '').strip()
                if central_path:
                    output_dir = Path(central_path)
                    output_dir.mkdir(parents=True, exist_ok=True)
                else:
                    output_dir = input_path.parent  # fallback if central path not set
            else:
                output_dir = input_path.parent

            if conversion_type == 'basic':
                from processing.funscript_1d_to_2d import generate_alpha_beta_from_main

                # Get basic conversion parameters
                config = conversion_tabs.get_basic_config()

                # Generate speed funscript (required for radius scaling)
                from processing.speed_processing import convert_to_speed
                speed_funscript = convert_to_speed(
                    main_funscript,
                    self.current_config['general']['speed_window_size'],
                    self.current_config['speed']['interpolation_interval']
                )

                # Generate alpha and beta files
                alpha_funscript, beta_funscript = generate_alpha_beta_from_main(
                    main_funscript, speed_funscript, config['points_per_second'], config['algorithm'],
                    config['min_distance_from_center'], config['speed_threshold_percent'],
                    config['direction_change_probability'],
                    min_stroke_amplitude=config.get('min_stroke_amplitude', 0.0),
                    point_density_scale=config.get('point_density_scale', 1.0),
                )

                # Save files
                filename_only = input_path.stem
                alpha_path = output_dir / f"{filename_only}.alpha.funscript"
                beta_path = output_dir / f"{filename_only}.beta.funscript"

                alpha_funscript.save_to_path(alpha_path)
                beta_funscript.save_to_path(beta_path)

                success_message = f"Basic conversion complete! Created {alpha_path.name} and {beta_path.name}"
                files_created = [alpha_path.name, beta_path.name]

            elif conversion_type == 'prostate':
                from processing.funscript_prostate_2d import generate_alpha_beta_prostate_from_main

                # Get prostate conversion parameters
                config = conversion_tabs.get_prostate_config()

                # Generate alpha-prostate and beta-prostate files
                alpha_prostate_funscript, beta_prostate_funscript = generate_alpha_beta_prostate_from_main(
                    main_funscript, config['points_per_second'], config['algorithm'],
                    config['min_distance_from_center'], config['generate_from_inverted']
                )

                # Save files
                filename_only = input_path.stem
                alpha_prostate_path = output_dir / f"{filename_only}.alpha-prostate.funscript"
                beta_prostate_path = output_dir / f"{filename_only}.beta-prostate.funscript"

                alpha_prostate_funscript.save_to_path(alpha_prostate_path)
                beta_prostate_funscript.save_to_path(beta_prostate_path)

                success_message = f"Prostate conversion complete! Created {alpha_prostate_path.name} and {beta_prostate_path.name}"
                files_created = [alpha_prostate_path.name, beta_prostate_path.name]

            self.update_progress(70, "Saving output files...")
            self.update_progress(100, success_message)

            # Show success message
            files_list = "\n".join([f"• {filename}" for filename in files_created])
            self.root.after(100, lambda: messagebox.showinfo("Success",
                f"2D conversion completed successfully!\n\nCreated files:\n{files_list}"))

        except Exception as e:
            error_msg = f"2D conversion failed: {str(e)}"
            self.update_progress(-1, error_msg)
            self.root.after(100, lambda: messagebox.showerror("Error", error_msg))

        finally:
            # Re-enable the convert buttons
            if hasattr(self.parameter_tabs, 'embedded_conversion_tabs'):
                self.root.after(100, lambda: self.parameter_tabs.embedded_conversion_tabs.set_button_state('normal'))

    def _generate_motion_axis_files(self, input_path: Path):
        """Generate motion axis files (E1-E4) based on current configuration."""
        try:
            self.update_progress(30, "Loading input file...")

            # Import necessary modules
            from funscript import Funscript
            from processing.motion_axis_generation import generate_motion_axes

            # Load main funscript
            main_funscript = Funscript.from_file(input_path)

            self.update_progress(50, "Generating motion axis files...")

            # Get motion axis configuration
            motion_config = self.current_config['positional_axes']

            # Determine output directory - respect file_management mode (central vs local)
            file_mgmt = self.current_config.get('file_management', {})
            if file_mgmt.get('mode') == 'central':
                central_path = file_mgmt.get('central_folder_path', '').strip()
                if central_path:
                    output_dir = Path(central_path)
                    output_dir.mkdir(parents=True, exist_ok=True)
                else:
                    output_dir = input_path.parent  # fallback if central path not set
            else:
                output_dir = input_path.parent

            # Generate motion axis files
            generated_files = generate_motion_axes(
                main_funscript,
                motion_config,
                output_dir,
                input_path.stem  # Use input filename without extension
            )

            self.update_progress(80, "Saving motion axis files...")

            if generated_files:
                # Create success message with list of generated files
                files_list = "\n".join([f"• {path.name}" for path in generated_files.values()])
                success_message = f"Motion axis generation complete! Created {len(generated_files)} files."

                self.update_progress(100, success_message)

                # Show success message
                self.root.after(100, lambda: messagebox.showinfo("Success",
                    f"Motion axis files generated successfully!\n\nCreated files:\n{files_list}"))

            else:
                # No files were generated (all axes disabled)
                warning_message = "No motion axis files generated - all axes are disabled."
                self.update_progress(100, warning_message)
                self.root.after(100, lambda: messagebox.showwarning("No Files Generated",
                    "No motion axis files were generated because all axes (E1-E4) are disabled.\n\n"
                    "Enable at least one axis in the Motion Axis tab to generate files."))

        except Exception as e:
            error_msg = f"Motion axis generation failed: {str(e)}"
            self.update_progress(-1, error_msg)
            self.root.after(100, lambda: messagebox.showerror("Error", error_msg))
            raise  # Re-raise to be caught by the calling method

    def update_config_from_ui(self):
        """Update configuration with current UI values."""
        # Update all parameters from parameter tabs (which now includes embedded conversion tabs)
        self.parameter_tabs.update_config(self.current_config)

    def update_config_display(self):
        """Update UI display with current configuration values."""
        # The conversion tabs will handle their own display updates
        # since they manage their own variables internally

        # Parameter tabs now handle all parameters including processing options
        self.parameter_tabs.update_display(self.current_config)

    def save_config(self):
        """Save current configuration to file."""
        self.update_config_from_ui()
        if self.config_manager.update_config(self.current_config):
            if self.config_manager.save_config():
                messagebox.showinfo("Configuration", "Configuration saved successfully!")
            else:
                messagebox.showerror("Error", "Failed to save configuration file.")
        else:
            messagebox.showerror("Error", "Invalid configuration values.")

    def save_config_preset(self):
        """Save current configuration to a named preset file."""
        import json
        self.update_config_from_ui()
        path = filedialog.asksaveasfilename(
            title="Save Configuration Preset",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(Path.home() / "Documents"),
            initialfile="restim_preset.json"
        )
        if not path:
            return
        try:
            with open(path, 'w') as f:
                json.dump(self.current_config, f, indent=2)
            messagebox.showinfo("Preset Saved", f"Configuration preset saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preset:\n{e}")

    def load_config_preset(self):
        """Load configuration from a preset file."""
        import json
        path = filedialog.askopenfilename(
            title="Load Configuration Preset",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(Path.home() / "Documents")
        )
        if not path:
            return
        try:
            with open(path, 'r') as f:
                loaded = json.load(f)
            if self.config_manager.update_config(loaded):
                self.current_config = self.config_manager.get_config()
                self.update_config_display()
                messagebox.showinfo("Preset Loaded", f"Configuration loaded from:\n{Path(path).name}")
            else:
                messagebox.showerror("Error", "Preset file contains invalid configuration values.")
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Invalid JSON file.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preset:\n{e}")

    def reset_config(self):
        """Reset configuration to defaults."""
        if messagebox.askyesno("Reset Configuration", "Reset all parameters to default values?"):
            self.config_manager.reset_to_defaults()
            self.current_config = self.config_manager.get_config()
            self.update_config_display()
            if hasattr(self, '_variant_active_var'):
                self._refresh_variants_bar()

    # ─────────────────────────────────────────────────────────────
    # Variants: 4-slot (A/B/C/D) whole-config snapshots for A/B test.
    # ─────────────────────────────────────────────────────────────

    _VARIANT_SLOTS = ('A', 'B', 'C', 'D')

    def _build_variants_bar(self, parent, row):
        """Compact top-bar widget: active-slot radio group, enabled
        checkboxes, save-to-slot button, and process-all action."""
        bar = ttk.LabelFrame(
            parent,
            text="Variants — A/B test whole-config snapshots",
            padding=(8, 4))
        bar.grid(row=row, column=0, columnspan=3,
                 sticky=(tk.W, tk.E), pady=(0, 4))

        self._variants_ensure_slots()
        v_cfg = self.current_config['variants']
        self._variant_active_var = tk.StringVar(value=str(v_cfg.get('active', 'A')))
        self._variant_enabled_vars = {}

        # Two stacked rows so the action buttons on row 2 stay visible
        # even when the window isn't wide enough for everything inline.
        # Row 1: slot selectors. Row 2: actions.
        row_top = ttk.Frame(bar)
        row_top.pack(side=tk.TOP, fill=tk.X, expand=False, anchor=tk.W)
        row_bot = ttk.Frame(bar)
        row_bot.pack(side=tk.TOP, fill=tk.X, expand=False, anchor=tk.W,
                     pady=(4, 0))

        ttk.Label(row_top, text="Active:").pack(side=tk.LEFT, padx=(0, 6))

        self._variant_radios = {}
        for slot in self._VARIANT_SLOTS:
            rb = ttk.Radiobutton(
                row_top, text=slot,
                variable=self._variant_active_var, value=slot,
                command=lambda s=slot: self._variant_switch_to(s))
            rb.pack(side=tk.LEFT, padx=(0, 4))
            self._variant_radios[slot] = rb

        ttk.Separator(row_top, orient='vertical').pack(
            side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(row_top, text="Enabled:").pack(side=tk.LEFT, padx=(0, 4))
        for slot in self._VARIANT_SLOTS:
            slot_cfg = v_cfg['slots'][slot]
            var = tk.BooleanVar(value=bool(slot_cfg.get('enabled', False)))
            self._variant_enabled_vars[slot] = var
            ttk.Checkbutton(
                row_top, text=slot, variable=var,
                command=lambda s=slot: self._variant_set_enabled(s)
            ).pack(side=tk.LEFT, padx=(0, 4))

        # Row 2: actions
        ttk.Button(row_bot, text="Save current to active slot",
                   command=self._variant_save_current).pack(
            side=tk.LEFT, padx=(0, 6))

        # Copy-from-slot: pick a source variant and clone its saved
        # settings into the currently active slot. Lets the user iterate
        # B/C/D off a tuned A baseline without re-entering every value.
        ttk.Label(row_bot, text="Copy from:").pack(side=tk.LEFT, padx=(0, 4))
        self._variant_copy_src_var = tk.StringVar(value='A')
        copy_combo = ttk.Combobox(
            row_bot, textvariable=self._variant_copy_src_var,
            values=list(self._VARIANT_SLOTS), state='readonly', width=3)
        copy_combo.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(row_bot, text="→ active",
                   command=self._variant_copy_from_to_active).pack(
            side=tk.LEFT, padx=(0, 6))

        ttk.Separator(row_bot, orient='vertical').pack(
            side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(row_bot, text="Process active variant",
                   command=self.start_processing_active_variant).pack(
            side=tk.LEFT, padx=(0, 6))
        ttk.Button(row_bot, text="Process all enabled variants",
                   command=self.start_processing_all_variants).pack(
            side=tk.LEFT, padx=(0, 4))

    def _variants_ensure_slots(self):
        """Guarantee the variants block exists with all 4 slots."""
        v = self.current_config.setdefault('variants', {})
        v.setdefault('active', 'A')
        slots = v.setdefault('slots', {})
        for s in self._VARIANT_SLOTS:
            slot = slots.setdefault(s, {})
            slot.setdefault('label', s)
            slot.setdefault('enabled', s == 'A')
            slot.setdefault('config', {})

    def _refresh_variants_bar(self):
        """Resync the bar's widgets with self.current_config after a
        config swap or reset."""
        self._variants_ensure_slots()
        v = self.current_config['variants']
        try:
            self._variant_active_var.set(str(v.get('active', 'A')))
        except tk.TclError:
            pass
        for s in self._VARIANT_SLOTS:
            try:
                self._variant_enabled_vars[s].set(
                    bool(v['slots'][s].get('enabled', False)))
            except (tk.TclError, KeyError):
                pass

    def _variant_snapshot_current(self) -> dict:
        """Take a copy of current_config minus the variants block
        itself, so slot-configs don't recursively nest variants.

        Also strips spatial_3d_linear.enabled so the S3D mode toggle
        stays global — switching variants shouldn't flip the pipeline
        on or off. Variants still carry all S3D tuning values (mixes,
        smoothing, dedup, defaults, etc.), just not the mode bit.
        """
        import copy
        self.update_config_from_ui()  # capture any pending UI edits
        snap = copy.deepcopy(self.current_config)
        snap.pop('variants', None)
        if 'spatial_3d_linear' in snap:
            snap['spatial_3d_linear'].pop('enabled', None)
        return snap

    def _variant_save_current(self):
        """Manual save: push current UI state into the active slot."""
        self._variants_ensure_slots()
        active = str(self._variant_active_var.get())
        snap = self._variant_snapshot_current()
        self.current_config['variants']['slots'][active]['config'] = snap
        self.current_config['variants']['active'] = active
        self.status_var.set(f"Variant {active} saved.")

    def _variant_copy_from_to_active(self):
        """Clone the source slot's saved config onto the currently
        active slot AND into the live UI. If the source slot is empty
        (never populated), fall back to the source's defaults so the
        action is never silently a no-op."""
        import copy
        self._variants_ensure_slots()
        v = self.current_config['variants']
        src = str(self._variant_copy_src_var.get())
        active = str(v.get('active', 'A'))
        if src not in v['slots']:
            self.status_var.set(f"Variant {src} not found.")
            return
        if src == active:
            self.status_var.set(
                f"Source and active are both {src} — nothing to copy.")
            return
        src_cfg = v['slots'][src].get('config') or {}
        if not src_cfg:
            self.status_var.set(
                f"Variant {src} has no saved settings yet — switch to it "
                f"and 'Save current to active slot' first.")
            return

        # Write into the active slot's stored config…
        cloned = copy.deepcopy(src_cfg)
        v['slots'][active]['config'] = cloned

        # …and load it into the live UI so the user sees the paste land.
        # S3D mode toggle is global — preserve it across the paste.
        current_mode = bool(
            self.current_config.get('spatial_3d_linear', {})
            .get('enabled', False))
        new_config = copy.deepcopy(cloned)
        new_config['variants'] = v
        new_config.setdefault(
            'spatial_3d_linear', {})['enabled'] = current_mode
        self.current_config = new_config
        self.config_manager.config = new_config
        self.update_config_display()
        self._s3d_refresh_from_config()
        self.status_var.set(f"Copied {src} → {active}.")

    def _variant_set_enabled(self, slot: str):
        self._variants_ensure_slots()
        try:
            val = bool(self._variant_enabled_vars[slot].get())
        except tk.TclError:
            val = False
        self.current_config['variants']['slots'][slot]['enabled'] = val

    def _variant_switch_to(self, new_slot: str):
        """Auto-save the leaving slot, then load `new_slot`'s config.

        If the target slot has an empty config (never populated), we
        treat the current UI as its starting point — the first time
        you switch to an empty slot it inherits whatever you had.
        """
        import copy
        self._variants_ensure_slots()
        v = self.current_config['variants']
        leaving = str(v.get('active', 'A'))

        # 1. Snapshot current UI into the leaving slot.
        leaving_snap = self._variant_snapshot_current()
        v['slots'][leaving]['config'] = leaving_snap

        # 2. Load target slot. Empty slot -> inherit current.
        target = v['slots'].get(new_slot, {})
        target_cfg = target.get('config') or {}
        if not target_cfg:
            target_cfg = copy.deepcopy(leaving_snap)
            target['config'] = target_cfg
            target['enabled'] = True
            try:
                self._variant_enabled_vars[new_slot].set(True)
            except tk.TclError:
                pass

        # 3. Swap current_config. Preserve the variants block AND
        #    the current global S3D mode toggle (the toggle isn't
        #    part of the variant snapshot — see
        #    _variant_snapshot_current).
        current_mode = bool(
            self.current_config.get('spatial_3d_linear', {})
            .get('enabled', False))
        new_config = copy.deepcopy(target_cfg)
        new_config['variants'] = v
        v['active'] = new_slot
        new_config.setdefault(
            'spatial_3d_linear', {})['enabled'] = current_mode
        self.current_config = new_config
        self.config_manager.config = new_config

        # 4. Push into tabs + conversion tabs + Spatial 3D panel.
        self.update_config_display()
        self._s3d_refresh_from_config()
        self.status_var.set(f"Active variant: {new_slot}")

    # ─── Processing all enabled variants ─────────────────────────

    def start_processing_active_variant(self):
        """Re-render only the currently active slot into its
        _variants/<slot>/ subfolder. Use this when you've tuned one
        variant and want to refresh just its outputs without touching
        the others."""
        if not self.validate_inputs():
            return
        self._variants_ensure_slots()
        active = str(self.current_config['variants'].get('active', 'A'))
        # Make sure the live UI state is captured into the slot first.
        self._variant_save_current()
        self.process_button.config(state='disabled')
        self.process_motion_button.config(state='disabled')
        self.progress_var.set(0)
        t = threading.Thread(
            target=self._process_all_variants_worker,
            args=([active],), daemon=True)
        t.start()

    def start_processing_all_variants(self):
        """Run the full pipeline for every enabled variant, writing
        each one's outputs into its own subfolder next to the input
        file (<input_dir>/<basename>_variants/<slot>/). The currently
        active variant is restored at the end."""
        if not self.validate_inputs():
            return
        self._variants_ensure_slots()
        enabled = [s for s in self._VARIANT_SLOTS
                   if self.current_config['variants']['slots'][s].get(
                       'enabled', False)]
        if not enabled:
            messagebox.showinfo(
                "Variants",
                "No variants are enabled. Check at least one slot "
                "under 'Enabled' first.")
            return
        # Make sure the current UI state is saved into the ACTIVE slot
        # first so it's included if that slot is enabled.
        self._variant_save_current()

        self.process_button.config(state='disabled')
        self.process_motion_button.config(state='disabled')
        self.progress_var.set(0)

        t = threading.Thread(
            target=self._process_all_variants_worker,
            args=(enabled,), daemon=True)
        t.start()

    def _process_all_variants_worker(self, enabled_slots):
        """Thread body: iterate variants, run processor, collect
        successes and failures."""
        import copy
        saved_active = str(self.current_config['variants'].get('active'))
        total_variants = len(enabled_slots)
        total_files = len(self.input_files)
        all_successes = 0
        all_failures = 0
        try:
            for v_idx, slot in enumerate(enabled_slots, 1):
                slot_cfg = copy.deepcopy(
                    self.current_config['variants']['slots'][slot]
                    .get('config') or {})
                if not slot_cfg:
                    continue
                # Force per-variant subfolder as the central output path.
                fm = slot_cfg.setdefault('file_management', {})
                fm['mode'] = 'central'
                for file_idx, input_file in enumerate(self.input_files, 1):
                    base = Path(input_file).stem
                    parent = Path(input_file).parent
                    out_dir = parent / f"{base}_variants" / slot
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fm['central_folder_path'] = str(out_dir)
                    processor = RestimProcessor(slot_cfg)

                    def prog(percent, message, s=slot, fi=file_idx,
                             vi=v_idx):
                        self.update_progress(
                            percent,
                            f"Variant {s} [{vi}/{total_variants}] — "
                            f"file {fi}/{total_files}: {message}")

                    ok = processor.process(input_file, prog)
                    if ok:
                        all_successes += 1
                        self.last_processed_filename = base
                        self.last_processed_directory = out_dir
                    else:
                        all_failures += 1
            self.update_progress(
                100,
                f"Processed {total_variants} variant(s): "
                f"{all_successes} ok, {all_failures} failed.")
            if all_failures == 0:
                self.root.after(
                    100, lambda: messagebox.showinfo(
                        "Variants",
                        f"Processed all {total_variants} enabled "
                        f"variants.\nOutputs under:\n"
                        f"{Path(self.input_files[0]).parent}/"
                        f"{Path(self.input_files[0]).stem}_variants/"))
            else:
                self.root.after(
                    100, lambda: messagebox.showwarning(
                        "Variants",
                        f"{all_failures} variant runs failed. "
                        f"See console for details."))
        except Exception as e:
            self.root.after(
                100, lambda msg=str(e): messagebox.showerror(
                    "Variants", f"Unexpected error:\n{msg}"))
        finally:
            # Restore the active slot's state (we may have nudged
            # file_management inside slot_cfg copies above; the
            # authoritative slot config in current_config is unchanged
            # because we deep-copied).
            self.current_config['variants']['active'] = saved_active
            self.root.after(0, lambda: (
                self.process_button.config(state='normal'),
                self.process_motion_button.config(state='normal')))

    def validate_inputs(self) -> bool:
        """Validate user inputs before processing."""
        # Check if files are selected
        if not self.input_files:
            messagebox.showerror("Error", "Please select at least one input file.")
            return False

        # Validate all selected files
        for input_file in self.input_files:
            if not Path(input_file).exists():
                messagebox.showerror("Error", f"Input file does not exist:\n{input_file}")
                return False

            if not input_file.lower().endswith('.funscript'):
                messagebox.showerror("Error", f"File must be a .funscript file:\n{input_file}")
                return False

        # Update and validate configuration
        self.update_config_from_ui()
        try:
            self.config_manager.validate_config()
        except ValueError as e:
            messagebox.showerror("Configuration Error", str(e))
            return False

        return True

    def start_processing(self):
        """Start the processing in a separate thread."""
        if not self.validate_inputs():
            return

        # Disable both process buttons during processing
        self.process_button.config(state='disabled')
        self.process_motion_button.config(state='disabled')
        self.progress_var.set(0)

        # Start processing thread
        processing_thread = threading.Thread(target=self.process_files, daemon=True)
        processing_thread.start()

    def start_motion_processing(self):
        """Start motion file processing in a separate thread."""
        if not self.validate_inputs():
            return

        # Disable both process buttons during processing
        self.process_button.config(state='disabled')
        self.process_motion_button.config(state='disabled')
        self.progress_var.set(0)

        # Start motion processing thread
        processing_thread = threading.Thread(target=self.process_motion_files, daemon=True)
        processing_thread.start()

    def process_files(self):
        """Process files in background thread."""
        try:
            total_files = len(self.input_files)
            successful = 0
            failed = 0

            # Spatial 3D Linear mode reinterprets the drop zone: the
            # first three scripts become X/Y/Z of a single 3D signal
            # instead of three independent batch items. Short-circuit
            # the batch loop here when the mode is enabled.
            s3d_cfg = self.current_config.get('spatial_3d_linear', {}) or {}
            if s3d_cfg.get('enabled', False):
                if total_files < 3:
                    msg = ("Spatial 3D Linear is enabled but fewer than "
                           "three scripts were dropped. Drop three "
                           ".funscript files — the first is X, then Y, "
                           "then Z — or disable the mode for batch "
                           "processing.")
                    self.update_progress(-1, "Spatial 3D Linear: need 3 files")
                    self.root.after(100,
                                    lambda: messagebox.showerror(
                                        "Spatial 3D Linear", msg))
                    return

                triplet = self.input_files[:3]
                names = " / ".join(Path(p).name for p in triplet)
                self.update_progress(0, f"Spatial 3D: {names}")

                def triplet_cb(percent, message):
                    self.update_progress(percent, f"Spatial 3D: {message}")

                processor = RestimProcessor(self.current_config)
                ok = processor.process_triplet(triplet, triplet_cb)
                if ok:
                    self.update_progress(100, "Spatial 3D complete.")
                    anchor = Path(triplet[0])
                    self.last_processed_filename = anchor.stem
                    self.last_processed_directory = anchor.parent
                    self.root.after(
                        100, lambda: messagebox.showinfo(
                            "Spatial 3D Linear",
                            f"Produced E1..E{s3d_cfg.get('n_electrodes', 4)}"
                            f" funscripts from:\n  X={triplet[0]}\n"
                            f"  Y={triplet[1]}\n  Z={triplet[2]}"))
                else:
                    self.root.after(
                        100, lambda: messagebox.showerror(
                            "Spatial 3D Linear",
                            "Processing failed — see console for details."))
                return

            for index, input_file in enumerate(self.input_files, 1):
                # Update status for current file
                file_name = Path(input_file).name
                self.update_progress(0, f"Processing file {index}/{total_files}: {file_name}")
                
                # Create processor with current configuration
                processor = RestimProcessor(self.current_config)

                # Process with progress callback that includes file index
                def file_progress_callback(percent, message):
                    status_msg = f"[{index}/{total_files}] {file_name}: {message}"
                    self.update_progress(percent, status_msg)

                success = processor.process(input_file, file_progress_callback)

                if success:
                    successful += 1
                    # Track the last successfully processed file
                    input_path = Path(input_file)
                    self.last_processed_filename = input_path.stem
                    self.last_processed_directory = input_path.parent
                else:
                    failed += 1

            # Show final summary
            if total_files == 1:
                if successful:
                    self.update_progress(100, "Processing completed successfully!")
                    self.root.after(100, lambda: messagebox.showinfo("Success", "Processing completed successfully!"))
            else:
                # Batch processing summary
                summary = f"Batch processing complete!\n\nSuccessful: {successful}\nFailed: {failed}\nTotal: {total_files}"
                self.update_progress(100, f"Batch complete: {successful}/{total_files} successful")
                self.root.after(100, lambda: messagebox.showinfo("Batch Complete", summary))

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.update_progress(-1, error_msg)
            self.root.after(100, lambda: messagebox.showerror("Error", error_msg))

        finally:
            # Re-enable both process buttons
            self.root.after(100, lambda: self.process_button.config(state='normal'))
            self.root.after(100, lambda: self.process_motion_button.config(state='normal'))

    def process_motion_files(self):
        """Process motion files in background thread based on current mode."""
        try:
            total_files = len(self.input_files)
            successful = 0
            failed = 0
            
            axes_config = self.current_config['positional_axes']
            generate_legacy = axes_config.get('generate_legacy', False)
            generate_motion_axis = axes_config.get('generate_motion_axis', False)
            modes = ([" 3P"] if generate_legacy else []) + (["4P"] if generate_motion_axis else [])
            mode_str = "+".join(modes) if modes else "none"

            for index, input_file in enumerate(self.input_files, 1):
                file_name = Path(input_file).name
                self.update_progress(0, f"[{index}/{total_files}] Processing {file_name} ({mode_str})...")

                try:
                    input_path = Path(input_file)

                    if generate_legacy:
                        # Use existing 2D conversion logic
                        self.update_progress(20, f"[{index}/{total_files}] Converting to 2D (3P)...")
                        original_value = self.input_file_var.get()
                        self.input_file_var.set(input_file)
                        self._perform_2d_conversion('basic')
                        self.input_file_var.set(original_value)

                    if generate_motion_axis:
                        # Generate motion axis files
                        self.update_progress(20, f"[{index}/{total_files}] Generating motion axis files (4P)...")
                        self._generate_motion_axis_files(input_path)

                    if not generate_legacy and not generate_motion_axis:
                        raise ValueError("No motion scripts enabled — enable 'Generate motion scripts' in the Motion Axis (3P) or (4P) tab")

                    successful += 1
                    # Track the last successfully processed file
                    self.last_processed_filename = input_path.stem
                    self.last_processed_directory = input_path.parent

                except Exception as file_error:
                    failed += 1
                    error_msg = f"Failed to process {file_name}: {str(file_error)}"
                    self.update_progress(-1, error_msg)

            # Show final summary
            if total_files == 1:
                if successful:
                    self.update_progress(100, "Motion processing completed successfully!")
                    self.root.after(100, lambda: messagebox.showinfo("Success", "Motion processing completed successfully!"))
            else:
                # Batch processing summary
                summary = f"Batch motion processing complete!\n\nSuccessful: {successful}\nFailed: {failed}\nTotal: {total_files}"
                self.update_progress(100, f"Batch complete: {successful}/{total_files} successful")
                self.root.after(100, lambda: messagebox.showinfo("Batch Complete", summary))

        except Exception as e:
            error_msg = f"Motion processing failed: {str(e)}"
            self.update_progress(-1, error_msg)
            self.root.after(100, lambda: messagebox.showerror("Error", error_msg))

        finally:
            # Re-enable both process buttons
            self.root.after(100, lambda: self.process_button.config(state='normal'))
            self.root.after(100, lambda: self.process_motion_button.config(state='normal'))

    def update_progress(self, percent: int, message: str):
        """Update progress bar and status message. Thread-safe."""
        def update_ui():
            if percent >= 0:
                self.progress_var.set(percent)
            else:
                # Error indicated by negative percent
                self.progress_var.set(0)
                messagebox.showerror("Processing Error", message)

            self.status_var.set(message)

        # Schedule UI update in main thread
        self.root.after(0, update_ui)

    def run(self):
        """Start the main application loop."""
        self.root.mainloop()


def main():
    """Entry point for the application."""
    import traceback
    from datetime import datetime

    def log_exception(exc_type, exc_value, exc_traceback):
        """Log uncaught exceptions to a file."""
        with open("restimfunscriptprocessor.log", "a") as f:
            f.write(f"--- {datetime.now()} ---\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
            f.write("\n")
        
        # Also show a user-friendly error message
        # Make sure this runs in the main thread if called from a background thread
        def show_error():
            messagebox.showerror("Unhandled Exception",
                                 "An unexpected error occurred. Please check restimfunscriptprocessor.log for details.")
        
        # This check is crude. A better way would involve a cross-thread communication queue.
        # But for this application, it's a reasonable starting point.
        if isinstance(threading.current_thread(), threading._MainThread):
            show_error()
        else:
            # If we are not in the main thread, we can't directly show a messagebox.
            # The logging is the most important part.
            print("ERROR: Unhandled exception in background thread. See log file.")


    app = MainWindow()
    
    # Set the global exception handlers
    app.root.report_callback_exception = log_exception
    threading.excepthook = log_exception

    app.run()


if __name__ == "__main__":
    main()