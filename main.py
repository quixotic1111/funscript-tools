#!/usr/bin/env python3
"""
Restim Funscript Processor
A GUI application for processing funscript files for electrostimulation devices.

Usage:
    python main.py                           # Open with no files loaded
    python main.py file.funscript            # Pre-load a single file
    python main.py x.funscript y.funscript z.funscript   # Pre-load a triplet
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Load the extracted processing package first so its back-compat shim
# aliases `processor`, `config`, `processing`, and `funscript` into
# sys.modules under the bare names the app's existing imports expect.
import restim_processor_core  # noqa: F401

from ui.main_window import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Restim Funscript Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help=".funscript files to pre-load (single file or 3-4 X/Y/Z[/rz] triplet)",
    )
    args = parser.parse_args()

    # Filter to existing .funscript files; warn but don't fail on bad paths
    # so a broken launcher doesn't prevent the GUI from opening.
    preload: list[Path] = []
    for p in args.files:
        if not p.exists():
            print(f"Warning: file not found, skipping: {p}", file=sys.stderr)
            continue
        if p.suffix.lower() != ".funscript":
            print(f"Warning: not a .funscript, skipping: {p}", file=sys.stderr)
            continue
        preload.append(p.resolve())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    main(preload_files=preload)
