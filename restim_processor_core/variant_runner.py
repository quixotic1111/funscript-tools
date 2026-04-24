"""
Subprocess-friendly variant runner.

Two entry points, both module-level so ``spawn`` can pickle them:

- ``run_variants_in_subprocess`` — legacy one-shot runner. Kept for
  the fallback path if the persistent worker fails to start.
- ``run_persistent_worker`` — long-lived worker that imports the
  processor once, then loops on a task queue. Main process reuses
  it across clicks, eliminating the 1-2 s cold-start that the
  one-shot runner pays on every invocation.

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
    ('ready',)                            # persistent worker only

Main → persistent worker:
    ('run', payload_dict)                 # run a batch
    ('shutdown',)                         # exit cleanly
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
        _run_batch(
            payload, queue, RestimProcessor, strip_axis_suffix)
    except Exception:
        # Include the full traceback so the main process can show
        # something useful. queue.put is pickled; traceback.format_exc()
        # returns a plain str, safe to ship.
        queue.put(('error', traceback.format_exc()))


def run_persistent_worker(task_queue, result_queue) -> None:
    """
    Long-lived worker entry point. Imports the processor exactly once
    at startup, then loops on ``task_queue`` for batches to run.

    Main process sends:
        ('run', payload_dict)   — run a batch; results stream back
                                  on ``result_queue`` ending with a
                                  ('done', ...) message.
        ('shutdown',)           — exit the worker cleanly.

    Worker emits:
        ('ready',)              — initialization complete, safe to
                                  send the first 'run' task.
        ('progress', pct, msg)  — per-batch progress update.
        ('file_result', ...)    — per-file success/failure.
        ('done', s, f, tv)      — batch complete; worker is idle
                                  and awaiting the next task.
        ('error', tb)           — uncaught exception anywhere.

    Same wire format as the one-shot runner above, so the main-side
    poll loop is identical — except it must NOT join the worker on
    'done' (the worker is persistent; only 'shutdown' exits it).
    """
    try:
        # The cold-start cost lives here — paid once per app session
        # instead of once per variant click. Subsequent 'run' tasks
        # reuse the already-imported modules.
        import restim_processor_core  # noqa: F401
        from processor import RestimProcessor
        from processing.axis_markers import strip_axis_suffix
        result_queue.put(('ready',))
    except Exception:
        result_queue.put(('error', traceback.format_exc()))
        return

    while True:
        try:
            msg = task_queue.get()
        except (EOFError, KeyboardInterrupt, OSError):
            # Main process went away — exit quietly.
            return
        if not msg:
            continue
        cmd = msg[0]
        if cmd == 'shutdown':
            return
        if cmd != 'run':
            # Unknown command — ignore so a protocol typo doesn't
            # wedge the worker.
            continue
        payload = msg[1] if len(msg) > 1 else {}
        try:
            _run_batch(
                payload, result_queue,
                RestimProcessor, strip_axis_suffix)
        except Exception:
            # Report the traceback and keep looping — a single bad
            # payload shouldn't kill the whole worker.
            result_queue.put(('error', traceback.format_exc()))


def _run_batch(payload: Dict[str, Any], queue,
               RestimProcessor, strip_axis_suffix) -> None:
    """Execute one variant batch. Shared body between the one-shot
    and persistent entry points — they only differ in how long the
    process lives."""
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
