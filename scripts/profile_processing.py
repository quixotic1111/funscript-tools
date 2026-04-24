#!/usr/bin/env python3
"""
Profiling harness for the Restim processing pipeline.

Runs `RestimProcessor.process()` on a sample input under cProfile, writes
the raw stats to disk, and prints the top hot-spots by cumulative and
total time. Use this before touching any optimization code so you know
what's actually slow instead of guessing.

Usage:

    python scripts/profile_processing.py
        # Profile the default Spatial 3D triplet from analysis_samples/

    python scripts/profile_processing.py --input examples/sample.funscript
        # Profile a single-axis 1D run

    python scripts/profile_processing.py --input analysis_samples/target/123.x.funscript --triplet
        # Profile S3D with an explicit X anchor (Y and Z will be auto-discovered
        # alongside in the same directory)

    python scripts/profile_processing.py --open
        # After profiling, open the .prof file in snakeviz (if installed).

Output:

    /tmp/restim_profile_<timestamp>/
        profile.prof         raw cProfile stats (open with snakeviz/flameprof)
        top_cumulative.txt   top 40 functions by cumulative time
        top_total.txt        top 40 functions by total time (self-time)

Exit status 0 on success. Non-zero if the profiled run itself failed.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure the library is importable when this script is run from the repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import restim_processor_core  # noqa: F401 — sets up shim aliases
from restim_processor_core import RestimProcessor, DEFAULT_CONFIG
import copy


DEFAULT_TRIPLET_X = REPO_ROOT / "analysis_samples" / "target" / "123.x.funscript"


def _build_s3d_config(base: dict) -> dict:
    """Clone the default config and turn on Spatial 3D Linear mode.

    Exercises the 3D pipeline — projection, geometric mapping, all
    parameter channels — so the profile reflects the fuller code path
    rather than just the 1D default.
    """
    cfg = copy.deepcopy(base)
    s3d = cfg.setdefault("spatial_3d_linear", {})
    s3d["enabled"] = True
    # Enable the geometric mixers so the pulse-param channels actually
    # evaluate rather than short-circuiting to flat funscripts. Without
    # this the profile understates Stage 3 / Stage 5 cost.
    geom = s3d.setdefault("geometric_mapping", {})
    geom.setdefault("pulse_width_radial_mix", 0.3)
    geom.setdefault("pulse_rise_azimuth_mix", 0.2)
    geom.setdefault("pulse_frequency_vradial_mix", 0.2)
    # Input-stage features stay at their config defaults — flip them
    # on here if you want to profile the noise-gate / one-euro / sharpen
    # stages specifically.
    return cfg


def _run_profile(input_path: Path, use_s3d: bool, out_dir: Path) -> float:
    """Run the pipeline under cProfile. Returns wall-clock seconds."""
    cfg = _build_s3d_config(DEFAULT_CONFIG) if use_s3d else copy.deepcopy(DEFAULT_CONFIG)

    # Send outputs to a scratch dir so we don't pollute the sample area.
    scratch = out_dir / "outputs"
    scratch.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("general", {})["output_directory"] = str(scratch)

    processor = RestimProcessor(cfg)

    profiler = cProfile.Profile()
    t_start = time.monotonic()
    profiler.enable()
    try:
        ok = processor.process(str(input_path))
    finally:
        profiler.disable()
    t_wall = time.monotonic() - t_start

    if not ok:
        print(f"WARNING: processor.process returned False for {input_path}",
              file=sys.stderr)

    # Raw stats file for interactive tools (snakeviz, flameprof, etc.)
    prof_path = out_dir / "profile.prof"
    profiler.dump_stats(str(prof_path))
    return t_wall


def _write_reports(prof_path: Path, out_dir: Path, top_n: int = 40) -> None:
    """Produce human-readable top-N reports from the raw .prof file."""
    stats = pstats.Stats(str(prof_path))

    # Strip long path prefixes for readability — keep last two components
    # so package + file is still identifiable.
    stats.strip_dirs()

    for sort_key, label, filename in [
        ("cumulative", "cumulative time (includes callees)", "top_cumulative.txt"),
        ("tottime", "total time (self-time only, excludes callees)", "top_total.txt"),
    ]:
        path = out_dir / filename
        with path.open("w") as f:
            f.write(f"Top {top_n} functions by {label}\n")
            f.write("=" * 72 + "\n\n")
            # Redirect pstats output into the file
            saved_stream = stats.stream
            stats.stream = f
            stats.sort_stats(sort_key).print_stats(top_n)
            stats.stream = saved_stream


def _summarise_to_console(out_dir: Path, wall_s: float) -> None:
    """Print a tight console summary after profiling."""
    cum_path = out_dir / "top_cumulative.txt"
    print(f"\n=== Profile complete — {wall_s:.2f}s wall ===\n")
    print(f"Raw stats:  {out_dir / 'profile.prof'}")
    print(f"Reports:    {cum_path}")
    print(f"            {out_dir / 'top_total.txt'}\n")

    # Print top 15 cumulative inline so hot-spots are visible without
    # opening files.
    print("Top 15 by cumulative time:")
    print("-" * 72)
    lines = cum_path.read_text().splitlines()
    # Skip the pstats header (first ~6 lines) and print the following 15
    # data rows.
    started = False
    printed = 0
    for line in lines:
        if line.strip().startswith("ncalls"):
            started = True
            print(line)
            continue
        if started and line.strip():
            print(line)
            printed += 1
            if printed >= 15:
                break


def _maybe_open_snakeviz(prof_path: Path) -> None:
    """If snakeviz is installed, launch it on the profile file."""
    if shutil.which("snakeviz") is None:
        print("\nsnakeviz not installed. To view interactively:")
        print("    pip install snakeviz")
        print(f"    snakeviz {prof_path}")
        return
    print(f"\nOpening snakeviz on {prof_path} ...")
    subprocess.Popen(["snakeviz", str(prof_path)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_TRIPLET_X,
        help=f"Input .funscript path. Default: {DEFAULT_TRIPLET_X}")
    parser.add_argument(
        "--triplet",
        action="store_true",
        default=True,
        help="Treat the input as the X-axis of a Spatial 3D triplet "
             "(Y and Z are discovered alongside). Default: ON.")
    parser.add_argument(
        "--no-triplet", dest="triplet", action="store_false",
        help="Disable S3D mode; run the 1D pipeline on a single file.")
    parser.add_argument(
        "--open", action="store_true",
        help="After profiling, open snakeviz on the .prof file if installed.")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output directory for profile artifacts. Default: a "
             "timestamped directory under /tmp.")
    parser.add_argument(
        "--top", type=int, default=40,
        help="How many functions to include in the top-N reports.")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2

    if args.out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(tempfile.gettempdir()) / f"restim_profile_{ts}"
    else:
        out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "Spatial 3D triplet" if args.triplet else "1D pipeline"
    print(f"Profiling {mode} on: {args.input}")
    print(f"Output dir:            {out_dir}")

    wall_s = _run_profile(args.input, args.triplet, out_dir)
    _write_reports(out_dir / "profile.prof", out_dir, top_n=args.top)
    _summarise_to_console(out_dir, wall_s)

    if args.open:
        _maybe_open_snakeviz(out_dir / "profile.prof")

    return 0


if __name__ == "__main__":
    sys.exit(main())
