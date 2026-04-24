"""Profile a single variant slot's process_triplet (or process) call.

Reads the project's ``config.json`` and whatever input files are listed
in it or passed on the CLI, runs one slot's pipeline under cProfile,
and writes both a raw stats dump and a readable top-N report.

Usage:
    python scripts/profile_one_slot.py <slot> <x.funscript> \\
        <y.funscript> <z.funscript> [rz.funscript]

Or for a single-input (non-triplet) run:
    python scripts/profile_one_slot.py <slot> <input.funscript>

Outputs:
    profile_<slot>.prof   — raw pstats dump (snakeviz-compatible)
    profile_<slot>.txt    — top-40 by cumulative time

Open the .prof in snakeviz for an interactive flame-graph view:
    pip install snakeviz
    snakeviz profile_<slot>.prof
"""

from __future__ import annotations

import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / 'restim_processor_core'))
    import restim_processor_core  # noqa: F401  — installs shim aliases
    from processor import RestimProcessor

    slot = sys.argv[1]
    inputs = [str(Path(p).resolve()) for p in sys.argv[2:]]

    # Load the project's config.json.
    cfg_path = repo / 'config.json'
    with cfg_path.open() as f:
        full_cfg = json.load(f)

    # Pull out the slot's config.
    slot_cfg = (
        full_cfg.get('variants', {})
        .get('slots', {})
        .get(slot, {})
        .get('config')
    )
    if slot_cfg is None:
        # Fall back to the top-level config if no slots section.
        slot_cfg = full_cfg
        print(f"[profile] variant slot {slot!r} not found; "
              f"falling back to top-level config")

    # Route output to /tmp so we don't clobber the user's project.
    import copy
    slot_cfg = copy.deepcopy(slot_cfg)
    fm = slot_cfg.setdefault('file_management', {})
    fm['mode'] = 'central'
    out_dir = Path(f"/tmp/profile_{slot}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fm['central_folder_path'] = str(out_dir)

    triplet_mode = bool(
        slot_cfg.get('spatial_3d_linear', {}).get('enabled', False))

    def progress(pct, msg):
        # Silent — we don't want tk-style print flooding to skew the
        # profile numbers.
        pass

    # One warm run first so Python's bytecode cache + funscript cache
    # + any lazy imports are all warm. The measured run then reflects
    # steady-state cost, which is what matters for tuning perf.
    print(f"[profile] warm-up run…")
    t0 = time.perf_counter()
    processor = RestimProcessor(slot_cfg)
    if triplet_mode or len(inputs) >= 3:
        processor.process_triplet(inputs[:4], progress)
    else:
        processor.process(inputs[0], progress)
    warm_elapsed = time.perf_counter() - t0
    print(f"[profile] warm-up: {warm_elapsed:.2f} s")

    # Measured run under cProfile.
    print(f"[profile] measured run (under cProfile)…")
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    processor = RestimProcessor(slot_cfg)
    if triplet_mode or len(inputs) >= 3:
        processor.process_triplet(inputs[:4], progress)
    else:
        processor.process(inputs[0], progress)
    pr.disable()
    measured = time.perf_counter() - t0
    print(f"[profile] measured: {measured:.2f} s "
          f"(cProfile overhead ~5-15%, so actual is slightly less)")

    prof_path = Path.cwd() / f"profile_{slot}.prof"
    txt_path = Path.cwd() / f"profile_{slot}.txt"
    pr.dump_stats(str(prof_path))
    print(f"[profile] raw stats: {prof_path}")

    # Human-readable: top 40 by cumulative time, then by total time.
    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).sort_stats('cumulative')
    buf.write("=" * 70 + "\nTOP 40 BY CUMULATIVE TIME\n" + "=" * 70 + "\n")
    stats.print_stats(40)
    stats = pstats.Stats(pr, stream=buf).sort_stats('tottime')
    buf.write("\n" + "=" * 70 + "\nTOP 40 BY TOTAL TIME (self only)\n"
              + "=" * 70 + "\n")
    stats.print_stats(40)
    txt_path.write_text(buf.getvalue())
    print(f"[profile] top-40 report: {txt_path}")
    print(f"[profile] preview first 60 lines ↓\n")
    print('\n'.join(buf.getvalue().splitlines()[:60]))


if __name__ == '__main__':
    main()
