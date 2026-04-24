#!/usr/bin/env python3
"""
Create a lightweight macOS .app wrapper that launches run.py with the
current Python interpreter.

Unlike build_macos.py (which bundles Python + numpy + ffmpeg into a 1GB+
self-contained .app via PyInstaller), this produces a ~2KB .app that:

  - Contains a shell script launcher only
  - Uses the Python interpreter / site-packages already on your machine
  - Points at this checkout of the repo (so `git pull` updates the app)

Usage:
    python3 create_dock_app.py              # create in ./dist
    python3 create_dock_app.py --install    # also copy to /Applications

Then drag it to the Dock once and forget about it.
"""

import argparse
import os
import platform
import plistlib
import shutil
import stat
import sys
from pathlib import Path

from version import __version__

APP_NAME = "Funscript Tools"
BUNDLE_ID = "com.funscript-tools.launcher"


def build_app(app_path: Path, repo_dir: Path, python_exe: Path) -> None:
    """Write a minimal .app bundle at app_path."""
    if app_path.exists():
        shutil.rmtree(app_path)

    contents = app_path / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir()

    # Info.plist — the minimum keys macOS needs to treat this as a real app
    info = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleVersion": __version__,
        "CFBundleShortVersionString": __version__,
        "CFBundleExecutable": "launcher",
        "CFBundlePackageType": "APPL",
        "CFBundleInfoDictionaryVersion": "6.0",
        "LSMinimumSystemVersion": "10.13",
        "NSHighResolutionCapable": True,
        # LSUIElement: False -> show Dock icon and menu bar
        "LSUIElement": False,
    }
    # Optional icon if the user drops one in assets/icon.icns
    icon_src = repo_dir / "assets" / "icon.icns"
    if icon_src.exists():
        target_icon = resources / "icon.icns"
        shutil.copy2(icon_src, target_icon)
        info["CFBundleIconFile"] = "icon"

    with open(contents / "Info.plist", "wb") as f:
        plistlib.dump(info, f)

    # Force the launcher to run under the machine's native architecture.
    # When macOS launches a .app via LaunchServices, a universal2 Python
    # (e.g. python.org's installer) defaults to x86_64 even on Apple
    # Silicon, which then fails to load arm64-only wheels like numpy.
    # `arch -arm64` / `arch -x86_64` pins it to the native arch.
    native_arch = platform.machine()  # "arm64" on M-series, "x86_64" on Intel

    # Launcher shell script. Logs stdout/stderr so future crashes are not
    # silent — check ~/Library/Logs/FunscriptTools/launcher.log if the Dock
    # icon bounces and disappears.
    log_dir = "$HOME/Library/Logs/FunscriptTools"
    launcher = macos / "launcher"
    launcher.write_text(
        f"""#!/bin/bash
set -e
mkdir -p "{log_dir}"
LOG="{log_dir}/launcher.log"
{{
  echo "---- $(date) ----"
  echo "Python:  {python_exe}"
  echo "Repo:    {repo_dir}"
  echo "Arch:    {native_arch}"
  cd "{repo_dir}"
  exec arch -{native_arch} "{python_exe}" run.py "$@"
}} >> "$LOG" 2>&1
"""
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"Created: {app_path}")
    print(f"  -> launches: {python_exe} {repo_dir}/run.py")
    print(f"  -> logs to:  ~/Library/Logs/FunscriptTools/launcher.log")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install",
        action="store_true",
        help="Also copy the .app to /Applications after creating it.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Path to the Python interpreter the launcher should use "
        "(default: the Python running this script).",
    )
    args = parser.parse_args()

    if sys.platform != "darwin":
        print("Error: .app bundles only work on macOS.")
        return 1

    repo_dir = Path(__file__).resolve().parent
    python_exe = Path(args.python).resolve()

    if not (repo_dir / "run.py").exists():
        print(f"Error: run.py not found in {repo_dir}")
        return 1
    if not python_exe.exists():
        print(f"Error: Python interpreter not found: {python_exe}")
        return 1

    # Quick preflight: confirm the target Python can import the app's deps.
    # If not, the Dock icon will bounce and the log will show the import
    # error — but better to catch it here.
    import subprocess
    check = subprocess.run(
        [str(python_exe), "-c", "import numpy, matplotlib, tkinter, yaml, PIL, tkinterdnd2, sv_ttk, ffpyplayer"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        print("Warning: the selected Python is missing required packages:")
        print(check.stderr.strip())
        print(f"\nInstall them with:")
        print(f"    {python_exe} -m pip install -r requirements.txt")
        print("\nContinuing anyway — fix this before launching the app.")

    dist_dir = repo_dir / "dist"
    dist_dir.mkdir(exist_ok=True)
    app_path = dist_dir / f"{APP_NAME}.app"
    build_app(app_path, repo_dir, python_exe)

    if args.install:
        target = Path("/Applications") / f"{APP_NAME}.app"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(app_path, target, symlinks=True)
        print(f"\nInstalled: {target}")
        print("Now open /Applications in Finder and drag the icon to your Dock.")
    else:
        print(f"\nTo install:")
        print(f"    cp -R '{app_path}' /Applications/")
        print(f"or re-run with --install.")
        print("\nThen open /Applications in Finder and drag the icon to your Dock.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
