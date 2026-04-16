"""
Per-axis smoothing preview: overlays the pre-smoothing axis signal
(post curve / modulation / cascade) against the post-smoothing version
for each enabled E1-E4 axis. Lets the user visually tune the
Butterworth cutoff/order without rendering full output files.
"""

import os
import math
import tkinter as tk
from tkinter import ttk
import warnings


# Per-axis "smoothed" color (vivid). The matching "raw axis" trace uses
# a paler tint so the two read as a family but the smoothed line pops.
_AXIS_COLORS = {
    'e1': '#1f77b4',  # blue
    'e2': '#ff7f0e',  # orange
    'e3': '#2ca02c',  # green
    'e4': '#d62728',  # red
}
_AXIS_RAW_COLORS = {
    'e1': '#9ec5e8',  # pale blue
    'e2': '#ffc99b',  # pale orange
    'e3': '#9fd9a0',  # pale green
    'e4': '#f4a3a4',  # pale red
}
# Source overlay — single distinct color used on every subplot since
# it's the same input on each. Purple stays out of the axis palette.
_SOURCE_COLOR = '#7e57c2'


class SmoothingPreview(tk.Toplevel):
    """Popup: 4 stacked subplots (E1-E4) with raw vs smoothed overlay."""

    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.title("Smoothing Preview")
        self.geometry("950x720")
        self.minsize(700, 500)
        self.main_window = main_window
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # State
        self._main_fs = None
        self._source_label = ""
        self._window_start_var = tk.DoubleVar(value=0.0)
        self._window_len_var = tk.DoubleVar(value=10.0)

        # Try matplotlib up front so failures show as a clear message.
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            self._Figure = Figure
            self._FigureCanvasTkAgg = FigureCanvasTkAgg
            self._matplotlib_ok = True
        except ImportError:
            self._matplotlib_ok = False

        self._build_ui()
        if self._matplotlib_ok:
            self._reload_input()
            self._render()

    # ── UI ─────────────────────────────────────────────────────────

    def _build_ui(self):
        top = ttk.Frame(self, padding=(8, 6))
        top.pack(side=tk.TOP, fill=tk.X)

        self._source_var = tk.StringVar(value="(no input)")
        ttk.Label(top, text="Source:").pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self._source_var,
                  foreground="#1976D2").pack(side=tk.LEFT, padx=(4, 12))

        ttk.Button(top, text="Reload input",
                   command=self._on_reload).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(top, text="Refresh plot",
                   command=self._render).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(top, text="Window start (s):").pack(side=tk.LEFT)
        start_entry = ttk.Entry(top, textvariable=self._window_start_var, width=7)
        start_entry.pack(side=tk.LEFT, padx=(2, 8))
        start_entry.bind('<Return>', lambda _e: self._render())
        start_entry.bind('<FocusOut>', lambda _e: self._render())

        ttk.Label(top, text="Length (s):").pack(side=tk.LEFT)
        len_entry = ttk.Entry(top, textvariable=self._window_len_var, width=6)
        len_entry.pack(side=tk.LEFT, padx=(2, 4))
        len_entry.bind('<Return>', lambda _e: self._render())
        len_entry.bind('<FocusOut>', lambda _e: self._render())

        if not self._matplotlib_ok:
            msg = ttk.Label(self,
                            text="matplotlib not available — install it to use this preview.",
                            foreground="red", padding=20)
            msg.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            return

        # Plot area
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._fig = self._Figure(figsize=(9, 6.5), dpi=90)
        self._fig.patch.set_facecolor('white')
        self._axes = self._fig.subplots(4, 1, sharex=True)
        for ax, name in zip(self._axes, ['E1', 'E2', 'E3', 'E4']):
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel(name, rotation=0, labelpad=18, va='center')
            ax.grid(True, alpha=0.3)
        self._axes[-1].set_xlabel("Time (s)")
        self._fig.subplots_adjust(left=0.08, right=0.97, top=0.96,
                                  bottom=0.08, hspace=0.25)

        self._canvas = self._FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Status bar
        self._status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._status_var,
                  foreground="#555", padding=(8, 2)).pack(side=tk.BOTTOM, fill=tk.X)

    # ── Data prep ──────────────────────────────────────────────────

    def _reload_input(self):
        """Pull the currently-selected input file from main_window, or
        fall back to a synthetic stroke pattern. We use the SAME source
        the processor would, so the preview matches actual output."""
        from funscript import Funscript
        import numpy as np

        path = None
        if (hasattr(self.main_window, 'input_files')
                and self.main_window.input_files):
            candidate = self.main_window.input_files[0]
            if os.path.isfile(candidate) and candidate.endswith('.funscript'):
                path = candidate

        if path is not None:
            try:
                self._main_fs = Funscript.from_file(path)
                self._source_label = os.path.basename(path)
            except Exception as e:
                self._main_fs = None
                self._source_label = f"failed to load: {e}"
        if self._main_fs is None:
            # Demo: 8s of mixed-amplitude strokes so the LPF effect is visible.
            t = np.linspace(0, 8.0, 200)
            y = (0.5
                 + 0.4 * np.sin(2 * np.pi * 0.6 * t)
                 + 0.05 * np.sin(2 * np.pi * 4.0 * t))
            y = np.clip(y, 0.0, 1.0)
            self._main_fs = Funscript(t, y)
            self._source_label = "(demo — load a .funscript for real preview)"
        self._source_var.set(self._source_label)

    def _on_reload(self):
        self._reload_input()
        self._render()

    # ── Render ─────────────────────────────────────────────────────

    def _render(self):
        if not self._matplotlib_ok or self._main_fs is None:
            return
        try:
            self._render_inner()
        except Exception as e:
            self._status_var.set(f"render error: {e}")

    def _render_inner(self):
        import numpy as np
        from funscript import Funscript
        from processing.linear_mapping import (
            apply_response_curve_to_funscript)
        from processing.motion_axis_generation import (
            apply_modulation, apply_lowpass, apply_cascade_shift,
            DEFAULT_MODULATION_PHASE_DEG, _AXIS_LINE_INDEX)

        config = self.main_window.current_config
        axes_config = config.get('positional_axes', {})

        # Physical-model cascade: same logic as generate_motion_axes.
        phys_model = axes_config.get('physical_model', {}) or {}
        phys_enabled = bool(phys_model.get('enabled', False))
        phys_spacing = float(phys_model.get('electrode_spacing_mm', 20.0))
        phys_speed = float(phys_model.get('propagation_speed_mm_s', 300.0))
        phys_direction = str(phys_model.get('sweep_direction', 'e1_to_e4'))
        if phys_enabled and phys_spacing > 0 and phys_speed > 0:
            step_s = phys_spacing / phys_speed
        else:
            step_s = 0.0
        if len(self._main_fs.x) >= 1:
            source_duration_s = float(self._main_fs.x[-1])
        else:
            source_duration_s = 0.0

        # Time window
        try:
            t0 = max(0.0, float(self._window_start_var.get()))
            wlen = max(0.5, float(self._window_len_var.get()))
        except (tk.TclError, ValueError):
            t0, wlen = 0.0, 10.0
        t1 = t0 + wlen

        any_enabled = False
        for ax_idx, axis_name in enumerate(['e1', 'e2', 'e3', 'e4']):
            ax = self._axes[ax_idx]
            ax.clear()
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel(axis_name.upper(), rotation=0,
                          labelpad=18, va='center')
            ax.grid(True, alpha=0.3)

            axis_cfg = axes_config.get(axis_name, {})
            if not axis_cfg.get('enabled', False):
                ax.text(0.5, 0.5, f"{axis_name.upper()} disabled",
                        transform=ax.transAxes, ha='center', va='center',
                        color='#999')
                continue
            any_enabled = True

            # Replicate the same per-axis pipeline as generate_motion_axes,
            # WITHOUT smoothing — that's our "raw" baseline.
            curve_cfg = axis_cfg.get('curve', {}) or {}
            control_points = curve_cfg.get(
                'control_points', [(0.0, 0.0), (1.0, 1.0)])

            signal_angle = axis_cfg.get('signal_angle', 0)
            if signal_angle:
                cos_a = math.cos(math.radians(signal_angle))
                rotated_y = [
                    max(0.0, min(1.0, 0.5 + (p - 0.5) * cos_a))
                    for p in self._main_fs.y]
                rotated_fs = Funscript(self._main_fs.x.copy(), rotated_y)
            else:
                rotated_fs = self._main_fs

            axis_fs = apply_response_curve_to_funscript(
                rotated_fs, control_points)

            mod_cfg = axis_cfg.get('modulation', {}) or {}
            mod_enabled = bool(mod_cfg.get('enabled', False))
            mod_freq = float(mod_cfg.get('frequency_hz', 0.5))
            mod_depth = float(mod_cfg.get('depth', 0.15))
            phase_enabled = bool(mod_cfg.get('phase_enabled', True))
            mod_phase = float(mod_cfg.get(
                'phase_deg',
                DEFAULT_MODULATION_PHASE_DEG.get(axis_name, 0.0)))
            if not phase_enabled:
                mod_phase = 0.0
            if mod_enabled and mod_depth > 0.0 and mod_freq > 0.0:
                axis_fs = apply_modulation(
                    axis_fs, mod_freq, mod_depth, mod_phase)

            # Cascade shift
            if step_s > 0:
                line_index = _AXIS_LINE_INDEX[axis_name]
                if phys_direction != 'signal_direction':
                    if phys_direction == 'e4_to_e1':
                        line_index = 3 - line_index
                    cascade_shift_s = line_index * step_s
                    if cascade_shift_s > 0:
                        axis_fs = apply_cascade_shift(
                            axis_fs, cascade_shift_s, source_duration_s)
                # signal_direction cascade is too expensive to preview
                # (per-stroke); skip — preview shows the simpler case.

            raw_fs = axis_fs

            # Smoothing
            sm_cfg = axis_cfg.get('smoothing', {}) or {}
            sm_enabled = bool(sm_cfg.get('enabled', False))
            sm_cutoff = float(sm_cfg.get('cutoff_hz', 8.0))
            sm_order = int(sm_cfg.get('order', 2))
            if sm_enabled and sm_cutoff > 0.0:
                smoothed_fs = apply_lowpass(raw_fs, sm_cutoff, sm_order)
            else:
                smoothed_fs = None

            # Original 1D source — pre-curve, pre-modulation, pre-everything.
            # Drawn first so it sits behind the axis-pipeline traces.
            xo = np.asarray(self._main_fs.x, dtype=float)
            yo = np.asarray(self._main_fs.y, dtype=float)
            mask_o = (xo >= t0) & (xo <= t1)
            if mask_o.any():
                ax.plot(xo[mask_o], yo[mask_o], color=_SOURCE_COLOR,
                        lw=1.0, ls='--', alpha=0.7, label='source')

            # Raw axis output (post curve/mod/cascade, pre-smoothing) — the
            # paler tint of the axis color so it visually pairs with the
            # smoothed line of the same hue.
            xr = np.asarray(raw_fs.x, dtype=float)
            yr = np.asarray(raw_fs.y, dtype=float)
            mask = (xr >= t0) & (xr <= t1)
            if mask.any():
                ax.plot(xr[mask], yr[mask],
                        color=_AXIS_RAW_COLORS[axis_name],
                        lw=1.2, alpha=0.9, label='raw axis')

            if smoothed_fs is not None:
                xs = np.asarray(smoothed_fs.x, dtype=float)
                ys = np.asarray(smoothed_fs.y, dtype=float)
                mask_s = (xs >= t0) & (xs <= t1)
                if mask_s.any():
                    ax.plot(xs[mask_s], ys[mask_s],
                            color=_AXIS_COLORS[axis_name],
                            lw=1.6,
                            label=f'smoothed (cutoff {sm_cutoff:.1f} Hz, order {sm_order})')
            else:
                ax.text(0.99, 0.95, "smoothing OFF",
                        transform=ax.transAxes, ha='right', va='top',
                        color='#999', fontsize=8)

            # Always show the legend so the source/raw/smoothed colors
            # stay self-documenting even when smoothing is off.
            ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
            ax.set_xlim(t0, t1)

        self._axes[-1].set_xlabel("Time (s)")

        if not any_enabled:
            self._status_var.set("All four axes are disabled.")
        else:
            self._status_var.set(
                f"Window {t0:.1f}-{t1:.1f}s of "
                f"{float(self._main_fs.x[-1]):.1f}s. "
                f"Edit per-axis Smooth/Cutoff/Order and click 'Refresh plot'.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._canvas.draw_idle()

    # ── Cleanup ────────────────────────────────────────────────────

    def _on_close(self):
        try:
            if hasattr(self, '_canvas'):
                self._canvas.get_tk_widget().destroy()
        except Exception:
            pass
        self.destroy()
