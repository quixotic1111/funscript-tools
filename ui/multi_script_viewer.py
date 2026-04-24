"""
Multi-Script 3D Viewer.

Takes three funscripts and treats them as X/Y/Z axes of a single 3D
signal. Renders the trajectory as a color-by-time trail in a 3D axes,
plus three 2D shadow projections (XY, XZ, YZ) so the path can be read
from any plane. Time appears as trail color, not as a spatial axis —
so all three spatial dimensions come from user-supplied scripts.

Playhead, scrubber, play/pause, and follow-mode plumbing are
transplanted from ui/animation_viewer.py.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import warnings


AXIS_LABELS = ('X', 'Y', 'Z')
AXIS_COLORS = ('#1f77b4', '#2ca02c', '#d62728')  # blue / green / red


class MultiScriptViewer(tk.Toplevel):
    """Popup window: 3D trail from three funscripts + three 2D shadows."""

    _FPS = 20
    _TRAIL_LENGTH = 60
    _DOT_COLOR = '#e53935'
    _TRAIL_COLOR = '#1976D2'

    def __init__(self, parent, main_window=None):
        super().__init__(parent)
        self.title("Multi-Script 3D Viewer")
        self.geometry("1100x820")
        self.minsize(800, 600)
        self.main_window = main_window
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._playing = False
        self._after_id = None
        self._frame_index = 0
        self._data = None
        self._n_frames = 0

        # Three slots — X, Y, Z. Prepopulate X with main window's current
        # input if one is loaded; user picks Y and Z manually.
        self._slot_paths = [None, None, None]
        if (main_window is not None
                and getattr(main_window, 'input_files', None)):
            first = main_window.input_files[0]
            if isinstance(first, str) and first.endswith('.funscript'):
                self._slot_paths[0] = first

        self._build_ui()
        self._refresh_slot_labels()

    # ── UI Construction ─────────────────────────────────────────────

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        # Three slot picker rows
        self._slot_labels = []
        for i, name in enumerate(AXIS_LABELS):
            row = ttk.Frame(self, padding=(8, 4, 8, 2))
            row.grid(row=i, column=0, sticky='ew')
            row.columnconfigure(2, weight=1)
            ttk.Label(row, text=f"{name}:", width=2,
                      foreground=AXIS_COLORS[i]
                      ).grid(row=0, column=0, padx=(0, 4))
            ttk.Button(row, text="Browse…",
                       command=lambda idx=i: self._browse(idx)
                       ).grid(row=0, column=1)
            lbl = ttk.Label(row, text="(no file)", foreground='#666')
            lbl.grid(row=0, column=2, sticky='w', padx=8)
            self._slot_labels.append(lbl)
            ttk.Button(row, text="Clear",
                       command=lambda idx=i: self._clear_slot(idx)
                       ).grid(row=0, column=3)

        # Canvas area
        self._canvas_frame = ttk.Frame(self)
        self._canvas_frame.grid(row=3, column=0, sticky='nsew',
                                padx=8, pady=4)
        self._canvas_frame.columnconfigure(0, weight=1)
        self._canvas_frame.rowconfigure(0, weight=1)

        # Controls row
        ctrl = ttk.Frame(self, padding=(8, 4, 8, 4))
        ctrl.grid(row=4, column=0, sticky='ew')
        ctrl.columnconfigure(1, weight=1)

        self._play_btn = ttk.Button(ctrl, text="\u25b6 Play", width=8,
                                    command=self._toggle_play,
                                    state='disabled')
        self._play_btn.grid(row=0, column=0, padx=(0, 5))

        self._scrubber_var = tk.IntVar(value=0)
        self._scrubber = ttk.Scale(ctrl, from_=0, to=1,
                                   orient=tk.HORIZONTAL,
                                   variable=self._scrubber_var,
                                   command=self._on_scrub)
        self._scrubber.grid(row=0, column=1, sticky='ew', padx=5)

        self._time_label = ttk.Label(ctrl, text="0.00 / 0.00s", width=16)
        self._time_label.grid(row=0, column=2, padx=(2, 0))

        speed_frame = ttk.Frame(ctrl)
        speed_frame.grid(row=0, column=3, padx=(10, 0))
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        self._speed_var = tk.DoubleVar(value=1.0)
        speed_combo = ttk.Combobox(speed_frame, textvariable=self._speed_var,
                                   values=["0.25", "0.5", "1.0", "2.0", "4.0"],
                                   width=5, state='readonly')
        speed_combo.pack(side=tk.LEFT, padx=(3, 0))
        speed_combo.set("1.0")

        self._render_btn = ttk.Button(ctrl, text="Render",
                                      command=self._prepare_and_draw)
        self._render_btn.grid(row=0, column=4, padx=(10, 0))

        # Status / hint
        self._status = ttk.Label(self, text="Pick up to three .funscript "
                                 "files (X, Y, Z) and click Render.",
                                 foreground='#666')
        self._status.grid(row=5, column=0, sticky='w', padx=10, pady=(0, 6))

    # ── Slot handling ───────────────────────────────────────────────

    def _browse(self, idx):
        initial = None
        for p in self._slot_paths:
            if p and os.path.isfile(p):
                initial = os.path.dirname(p)
                break
        path = filedialog.askopenfilename(
            parent=self,
            title=f"Pick {AXIS_LABELS[idx]}-axis funscript",
            initialdir=initial,
            filetypes=[("Funscript", "*.funscript"), ("All files", "*.*")])
        if path:
            self._slot_paths[idx] = path
            self._refresh_slot_labels()

    def _clear_slot(self, idx):
        self._slot_paths[idx] = None
        self._refresh_slot_labels()

    def _refresh_slot_labels(self):
        for lbl, path in zip(self._slot_labels, self._slot_paths):
            if path:
                lbl.config(text=os.path.basename(path), foreground='#000')
            else:
                lbl.config(text="(no file)", foreground='#666')

    # ── Data preparation ────────────────────────────────────────────

    def _prepare_data(self):
        """Load up to 3 funscripts and resample onto a shared time grid.

        Missing slots are filled with a flat 0.5 series — so a 1-script or
        2-script input still renders (as a line or a plane), giving useful
        feedback while the user wires up more inputs.
        """
        from funscript import Funscript

        loaded = []  # list of (label, Funscript or None)
        any_loaded = False
        for idx, path in enumerate(self._slot_paths):
            if path and os.path.isfile(path):
                try:
                    fs = Funscript.from_file(path)
                    loaded.append((os.path.basename(path), fs))
                    any_loaded = True
                except Exception as e:
                    print(f"[multi-viewer] failed to load {path}: {e}")
                    loaded.append((f"(load failed: {AXIS_LABELS[idx]})", None))
            else:
                loaded.append((f"(unset: {AXIS_LABELS[idx]})", None))

        if not any_loaded:
            return None

        # Common time grid: union of all loaded scripts' timestamps,
        # resampled uniformly at ~50 Hz between min and max t so the 3D
        # line renders smoothly without being dominated by one script's
        # sample density.
        t_min, t_max = np.inf, -np.inf
        for _, fs in loaded:
            if fs is not None and len(fs.x) > 0:
                t_min = min(t_min, float(fs.x[0]))
                t_max = max(t_max, float(fs.x[-1]))
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            return None

        dt = 1.0 / 50.0  # 50 Hz resample
        n = int(np.ceil((t_max - t_min) / dt)) + 1
        n = min(n, 50000)  # safety cap
        t = np.linspace(t_min, t_max, n)

        axes = []
        labels = []
        for (label, fs) in loaded:
            labels.append(label)
            if fs is None or len(fs.x) == 0:
                axes.append(np.full_like(t, 0.5))
            else:
                axes.append(np.interp(t, np.asarray(fs.x, dtype=float),
                                      np.asarray(fs.y, dtype=float)))

        return {
            't': t,
            'x': axes[0], 'y': axes[1], 'z': axes[2],
            'labels': labels,
        }

    def _prepare_and_draw(self):
        """Compute data and (re)build the matplotlib figure."""
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import matplotlib.gridspec as gridspec
        except ImportError:
            for w in self._canvas_frame.winfo_children():
                w.destroy()
            ttk.Label(self._canvas_frame,
                      text="matplotlib is required for this viewer.",
                      foreground='red').pack(padx=20, pady=20)
            return

        self._data = self._prepare_data()
        if self._data is None:
            for w in self._canvas_frame.winfo_children():
                w.destroy()
            ttk.Label(self._canvas_frame,
                      text="Load at least one funscript to render.",
                      foreground='#666').pack(padx=20, pady=20)
            self._play_btn.config(state='disabled')
            self._status.config(text="No data loaded.")
            return

        n = len(self._data['t'])
        self._n_frames = n
        self._frame_index = 0
        self._scrubber.configure(to=max(0, n - 1))
        self._scrubber_var.set(0)
        self._play_btn.config(state='normal')

        labels = self._data['labels']
        self._status.config(text=(
            f"X: {labels[0]}   Y: {labels[1]}   Z: {labels[2]}"
            f"   ({n} samples, {self._data['t'][-1]:.2f}s)"))

        # Clear old canvas
        for w in self._canvas_frame.winfo_children():
            w.destroy()

        fig = Figure(figsize=(10, 7), dpi=90)
        fig.patch.set_facecolor('#f5f5f5')
        gs = gridspec.GridSpec(2, 3, figure=fig,
                               width_ratios=[1, 1, 1],
                               height_ratios=[1.3, 1],
                               wspace=0.3, hspace=0.35)

        # Top row: big 3D trail spans all three columns
        self._build_3d(fig, gs[0, :])
        # Bottom row: three 2D shadow projections
        self._build_shadow(fig, gs[1, 0], 'x', 'y', 'XY (top-down)')
        self._build_shadow(fig, gs[1, 1], 'x', 'z', 'XZ (front)')
        self._build_shadow(fig, gs[1, 2], 'y', 'z', 'YZ (side)')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fig.tight_layout(pad=1.0)
            except Exception:
                pass

        self._canvas = FigureCanvasTkAgg(fig, self._canvas_frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._fig = fig
        self._update_frame()

    # ── Subplots ─────────────────────────────────────────────────────

    def _build_3d(self, fig, subplot_spec):
        """Main 3D trajectory, colored by time."""
        from matplotlib import cm
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        ax = fig.add_subplot(subplot_spec, projection='3d')
        d = self._data
        t = d['t']
        t_norm = (t - t[0]) / max(t[-1] - t[0], 1e-6)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel('X', fontsize=8, color=AXIS_COLORS[0], labelpad=2)
        ax.set_ylabel('Y', fontsize=8, color=AXIS_COLORS[1], labelpad=2)
        ax.set_zlabel('Z', fontsize=8, color=AXIS_COLORS[2], labelpad=2)
        ax.set_title('3D trajectory (color = time)', fontsize=9)
        ax.tick_params(labelsize=6)

        # Static full-path line, color-gradient by time
        points = np.array([d['x'], d['y'], d['z']]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = cm.viridis(t_norm[:-1])
        lc = Line3DCollection(segments, colors=colors, linewidth=0.7,
                              alpha=0.55)
        ax.add_collection3d(lc)

        # Animated trail (more recent history)
        self._trail_3d, = ax.plot([], [], [], color=self._TRAIL_COLOR,
                                  linewidth=2.0, alpha=0.85)
        self._dot_3d, = ax.plot([d['x'][0]], [d['y'][0]], [d['z'][0]],
                                'o', color=self._DOT_COLOR,
                                markersize=7, markeredgecolor='white',
                                markeredgewidth=0.5, zorder=10)
        self._ax_3d = ax

    def _build_shadow(self, fig, subplot_spec, axis_a, axis_b, title):
        """2D projection onto the (axis_a, axis_b) plane."""
        ax = fig.add_subplot(subplot_spec)
        ax.set_aspect('equal')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        label_a = axis_a.upper()
        label_b = axis_b.upper()
        color_a = AXIS_COLORS['xyz'.index(axis_a)]
        color_b = AXIS_COLORS['xyz'.index(axis_b)]
        ax.set_xlabel(label_a, fontsize=8, color=color_a)
        ax.set_ylabel(label_b, fontsize=8, color=color_b)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.15)

        d = self._data
        # Static faint path
        ax.plot(d[axis_a], d[axis_b], color='#aaaaaa',
                linewidth=0.4, alpha=0.5, zorder=2)

        trail, = ax.plot([], [], color=self._TRAIL_COLOR,
                         linewidth=1.5, alpha=0.7, zorder=3)
        dot = ax.scatter([d[axis_a][0]], [d[axis_b][0]], s=50,
                         c=self._DOT_COLOR, edgecolors='white',
                         linewidths=0.5, zorder=4)

        # Stash artists on self for per-frame update
        if not hasattr(self, '_shadows'):
            self._shadows = []
        self._shadows.append({
            'ax': ax, 'axis_a': axis_a, 'axis_b': axis_b,
            'trail': trail, 'dot': dot,
        })

    # ── Animation ────────────────────────────────────────────────────

    def _update_frame(self):
        if self._data is None or self._n_frames == 0:
            return
        i = self._frame_index
        d = self._data
        trail_start = max(0, i - self._TRAIL_LENGTH)

        # 3D
        if hasattr(self, '_dot_3d'):
            self._dot_3d.set_data_3d([d['x'][i]], [d['y'][i]], [d['z'][i]])
            self._trail_3d.set_data_3d(
                d['x'][trail_start:i + 1],
                d['y'][trail_start:i + 1],
                d['z'][trail_start:i + 1])

        # Shadows
        for s in getattr(self, '_shadows', []):
            a, b = s['axis_a'], s['axis_b']
            s['trail'].set_data(
                d[a][trail_start:i + 1],
                d[b][trail_start:i + 1])
            s['dot'].set_offsets([[d[a][i], d[b][i]]])

        current_t = float(d['t'][i])
        total_t = float(d['t'][-1])
        self._time_label.config(text=f"{current_t:.2f} / {total_t:.2f}s")

        if hasattr(self, '_canvas'):
            self._canvas.draw_idle()

    def _tick(self):
        if not self._playing or self._data is None:
            return
        speed = max(0.25, float(self._speed_var.get()))
        self._frame_index += max(1, int(speed))
        if self._frame_index >= self._n_frames:
            self._frame_index = 0  # loop
        self._scrubber_var.set(self._frame_index)
        self._update_frame()
        self._after_id = self.after(int(1000 / self._FPS), self._tick)

    def _on_scrub(self, value):
        try:
            idx = int(float(value))
        except (ValueError, TypeError):
            return
        idx = max(0, min(idx, self._n_frames - 1))
        self._frame_index = idx
        self._update_frame()

    def _toggle_play(self):
        if self._data is None:
            return
        self._playing = not self._playing
        if self._playing:
            self._play_btn.config(text="\u23f8 Pause")
            self._tick()
        else:
            self._play_btn.config(text="\u25b6 Play")
            if self._after_id is not None:
                self.after_cancel(self._after_id)
                self._after_id = None

    def _on_close(self):
        self._playing = False
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        self.destroy()
