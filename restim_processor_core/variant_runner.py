"""
Subprocess-friendly variant runner.

Module-level entry point `run_variants_in_subprocess` that the UI spawns
via multiprocessing.Process. Runs the full variant-processing loop in
full OS-process isolation so Tk events (video playback, slider feedback,
button clicks) and the T-code scheduler's tick thread don't contend for
the GIL with heavy numpy work inside the processor.

Why this matters: even with targeted GIL-yield nudges inside individual
hot loops (one_euro_filter, velocity-weight EMA, etc.), the processor
has dozens of other per-sample Python loops across motion-axis
generation, traveling-wave, volume/speed derivation, etc. Each briefly
holds the GIL. Running the whole thing in a separate process is the
only bulletproof way to keep live preview smooth.

Communication protocol via a multiprocessing.Queue (tuples):
    ('progress', percent: int, message: str)
    ('file_result', slot: str, base: str, out_dir: str, ok: bool)
    ('done', successes: int, failures: int, total_variants: int)
    ('error', traceback_str: str)
"""

import copy
import sys
import traceback
from pathlib import Path
from typing import Any, Dict


def run_variants_in_subprocess(payload: Dict[str, Any], queue) -> None:
    """
    Subprocess entry point. Must be module-level so `spawn` start method
    can pickle it.

    Args:
        payload: dict with keys:
            config        — the full current_config (deep-copied)
            enabled_slots — list of variant slot names to run
            input_files   — list of input funscript paths
            triplet_mode  — bool; True if Spatial 3D Linear XYZ mode
        queue: multiprocessing.Queue (main process drains it).
    """
    try:
        # Imports happen inside the function so spawn can re-import the
        # module without executing anything at import time.
        import restim_processor_core  # noqa: F401 — sets up shim aliases
        from processor import RestimProcessor
        from processing.axis_markers import strip_axis_suffix

        enabled_slots = list(payload.get('enabled_slots') or [])
        input_files = list(payload.get('input_files') or [])
        triplet_mode = bool(payload.get('triplet_mode', False))
        current_config = payload.get('config') or {}

        total_variants = len(enabled_slots)
        total_files = len(input_files)
        all_successes = 0
        all_failures = 0

        for v_idx, slot in enumerate(enabled_slots, 1):
            slot_cfg = copy.deepcopy(
                (current_config.get('variants', {}) or {})
                .get('slots', {}).get(slot, {}).get('config') or {})
            if not slot_cfg:
                continue

            # Force per-variant subfolder as the central output path.
            fm = slot_cfg.setdefault('file_management', {})
            fm['mode'] = 'central'

            if triplet_mode:
                triplet = input_files[:4]
                if len(triplet) < 3:
                    all_failures += 1
                    continue
                base = strip_axis_suffix(Path(triplet[0]).stem)
                parent = Path(triplet[0]).parent
                out_dir = parent / f"{base}_variants" / slot
                out_dir.mkdir(parents=True, exist_ok=True)
                fm['central_folder_path'] = str(out_dir)
                processor = RestimProcessor(slot_cfg)

                def prog(percent, message, s=slot, vi=v_idx,
                         tv=total_variants):
                    queue.put((
                        'progress', int(percent),
                        f"Variant {s} [{vi}/{tv}] — "
                        f"Spatial 3D: {message}"))

                ok = processor.process_triplet(triplet, prog)
                if ok:
                    all_successes += 1
                    queue.put((
                        'file_result', slot, str(base),
                        str(out_dir), True))
                else:
                    all_failures += 1
                    queue.put((
                        'file_result', slot, str(base),
                        str(out_dir), False))
                continue

            for file_idx, input_file in enumerate(input_files, 1):
                base = strip_axis_suffix(Path(input_file).stem)
                parent = Path(input_file).parent
                out_dir = parent / f"{base}_variants" / slot
                out_dir.mkdir(parents=True, exist_ok=True)
                fm['central_folder_path'] = str(out_dir)
                processor = RestimProcessor(slot_cfg)

                def prog(percent, message, s=slot, fi=file_idx,
                         vi=v_idx, tf=total_files, tv=total_variants):
                    queue.put((
                        'progress', int(percent),
                        f"Variant {s} [{vi}/{tv}] — "
                        f"file {fi}/{tf}: {message}"))

                ok = processor.process(input_file, prog)
                if ok:
                    all_successes += 1
                    queue.put((
                        'file_result', slot, str(base),
                        str(out_dir), True))
                else:
                    all_failures += 1
                    queue.put((
                        'file_result', slot, str(base),
                        str(out_dir), False))

        queue.put((
            'done', all_successes, all_failures, total_variants))
    except Exception:
        # Include the full traceback so the main process can show
        # something useful. queue.put is pickled; traceback.format_exc()
        # returns a plain str, safe to ship.
        queue.put(('error', traceback.format_exc()))
