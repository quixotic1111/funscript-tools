## Unreleased

### Noise Gate — new pipeline stage

1. **1D pipeline** — new "Noise Gate" tab (between Advanced and Trochoid). Activity-based gate applied to the dropped funscript before every other stage (trochoid quantization, speed, alpha/beta, prostate, traveling wave, volume ramp, etc.). Rolling peak-to-peak over a centered window; when amplitude drops below threshold the signal is pulled toward a rest level with asymmetric attack/release so transitions don't click. Off by default.
2. **Spatial 3D Linear** — same gate added to the XYZ-triplet pipeline inside the main-window tuning panel. New "Noise gate" checkbox + threshold / window / attack / release / rest sliders at rows 15-16 of the panel. Runs **before** `Smooth input (1€)` and `Sharpen input` inside `load_dof_scripts`. All axes gate synchronously off a combined activity metric (max of per-axis rolling p2p) — the 3D trajectory stays coherent; a quiet section collapses X/Y/Z/rz toward rest together instead of warping the position.
3. **Variant refresh fix** — `_s3d_refresh_from_config` now refreshes seven checkboxes on A/B/C/D variant switch: Smooth E1..En, Dedup holds, Reverb, Smooth input (1€), Sharpen input, Compress output, and Noise gate. Previously only the first two (plus all sliders) round-tripped; the newer booleans held their on-screen state regardless of which slot was active.

**Config keys added:**
- `noise_gate.{enabled, threshold, window_s, attack_s, release_s, rest_level}` — 1D pipeline
- `spatial_3d_linear.noise_gate.{enabled, threshold, window_s, attack_s, release_s, rest_level}` — 3D pipeline

**Files touched:**
- `processing/noise_gate.py` — new: `apply_noise_gate(funscript, ...)` and `gate_uniform_signals_combined(signals, dt, ...)` sharing internal helpers (`_rolling_peak_to_peak`, `_gate_envelope_from_p2p`)
- `processing/multi_script_loader.py` — `load_dof_scripts` gained 6 `noise_gate_*` kwargs; gate applied right after resample
- `processor.py` — 1D pipeline stage at the top of `_execute_pipeline`; 3D pipeline config parsed in `process_triplet` and forwarded to `load_dof_scripts`
- `config.py` — defaults + validation ranges for both new sections
- `ui/parameter_tabs.py` — new `setup_noise_gate_tab`, tab registered between Advanced and Trochoid
- `ui/main_window.py` — 2 new rows in the S3D tuning panel, variant-refresh hooks for all seven checkboxes
- `SETTINGS_GUIDE.md` — new "Noise Gate Tab" section and new "Noise Gate" subsection inside Spatial 3D Linear

---

## What's New in v2.3.2

### New Features (merged from contributor PR #10 + follow-up fixes)

**Canvas Timeline (Custom Event Builder)**
1. Replaced the basic event list with a fully interactive canvas timeline — drag blocks to reposition, drag right edge to resize event duration
2. Zoom with Ctrl+scroll, pan with scroll or drag background
3. Snap-to-grid (Off / 0.5s / 1s / 5s / 10s / 30s / 1m)
4. Undo / Redo support
5. Funscript waveform overlay — auto-loads matching `.funscript` when opening an events file
6. Playhead indicator
7. Conflict detection — overlapping events warn before save/apply
8. Category-coloured event blocks (mcb / clutch / test / general)

**Video Playback & Timeline**
9. Synchronized video playback window (ffpyplayer) with timeline scrubbing
10. Arrow key frame stepping and spacebar play/pause on timeline; keys work when video window is focused
11. Seek bar in video window syncs timeline playhead
12. "Show waveform" checkbox in Options bar to hide/show funscript track
13. Timeline ruler minor tick subdivisions and two-level grid
14. Timeline zoom extended to support long videos (>15 min)
15. Auto-load matching video file when opening events for same source

**Dark Mode**
16. Dark/light mode toggle button in main toolbar (sv_ttk theme)
17. Dark mode preference is now persisted in config and restored on next launch

### Dependencies added
- `ffpyplayer>=4.3.0`
- `Pillow>=10.0.0`
- `sv-ttk>=2.6.0`
