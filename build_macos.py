#!/usr/bin/env python3
"""
Build script for creating a macOS .app bundle using PyInstaller.

Output:
    dist/macos/Restim Funscript Processor.app
    dist/RestimFunscriptProcessor-v<version>-macOS-<arch>.zip

Usage:
    python build_macos.py

Distribution notes:
    The produced .app is unsigned. On another Mac, Gatekeeper will block it on
    first launch — right-click the app -> Open, or run:
        xattr -dr com.apple.quarantine "Restim Funscript Processor.app"
    For the building user there is no prompt. Signing/notarization requires an
    Apple Developer account and is out of scope here.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from version import __version__, __app_name__

APP_NAME = "Restim Funscript Processor"
BUNDLE_ID = "com.funscript-tools.restim-processor"


def build_macos_app():
    """Build macOS .app bundle using PyInstaller."""
    print(f"Building {__app_name__} v{__version__} for macOS...")

    dist_dir = Path("dist")
    build_dir = Path("build")

    if dist_dir.exists():
        print("Cleaning previous dist folder...")
        shutil.rmtree(dist_dir)

    if build_dir.exists():
        print("Cleaning previous build folder...")
        shutil.rmtree(build_dir)

    # Invoke PyInstaller via `python -m PyInstaller` using the *current*
    # interpreter, so it picks up the same site-packages (numpy, matplotlib,
    # etc.) that are visible to this script. Using the `pyinstaller` CLI can
    # silently resolve to a different Python and produce a bundle that
    # crashes with ModuleNotFoundError on launch.
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--windowed",              # produce .app bundle, no console
        "--onedir",                # standard for .app bundles
        "--name", APP_NAME,
        "--osx-bundle-identifier", BUNDLE_ID,
        "--paths", ".",
        # Package data
        "--collect-all", "ui",
        "--collect-all", "processing",
        # Core scientific stack — --collect-all forces PyInstaller to grab
        # every submodule and binary extension, which avoids
        # ModuleNotFoundError for lazily-imported pieces.
        "--collect-all", "numpy",
        "--collect-all", "matplotlib",
        # Third-party packages that ship non-Python resources (tcl files,
        # theme assets, ffmpeg shared libs) — these need --collect-all
        # so PyInstaller grabs the whole package, not just the .py files.
        "--collect-all", "tkinterdnd2",
        "--collect-all", "sv_ttk",
        "--collect-all", "ffpyplayer",
        # Hidden imports
        "--hidden-import", "tkinter",
        "--hidden-import", "numpy",
        "--hidden-import", "matplotlib",
        "--hidden-import", "matplotlib.pyplot",
        "--hidden-import", "matplotlib.backends.backend_tkagg",
        "--hidden-import", "matplotlib.figure",
        "--hidden-import", "matplotlib.patches",
        "--hidden-import", "PIL",
        "--hidden-import", "yaml",
        "--hidden-import", "json",
        "--hidden-import", "pathlib",
        "--hidden-import", "processing.linear_mapping",
        "--hidden-import", "processing.motion_axis_generation",
        "--clean",
        "--distpath", "dist/macos",
        "main.py",
    ]

    # Data files (macOS/Linux use `:` as the --add-data separator)
    if Path("restim_config.json").exists():
        cmd.insert(-1, "--add-data")
        cmd.insert(-1, "restim_config.json:.")

    if Path("config.json").exists():
        cmd.insert(-1, "--add-data")
        cmd.insert(-1, "config.json:.")

    if Path("config.event_definitions.yml").exists():
        cmd.insert(-1, "--add-data")
        cmd.insert(-1, "config.event_definitions.yml:.")

    # Optional icon. PyInstaller expects .icns on macOS. Drop one at
    # assets/icon.icns and it will be picked up automatically.
    if Path("assets/icon.icns").exists():
        cmd.insert(-1, "--icon")
        cmd.insert(-1, "assets/icon.icns")

    print("Running PyInstaller...")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print("Build successful!")

        app_path = Path("dist/macos") / f"{APP_NAME}.app"
        if app_path.exists():
            # Bundle size reported by `du -sh` equivalent
            total = sum(f.stat().st_size for f in app_path.rglob("*") if f.is_file())
            print(f"Created: {app_path}")
            print(f"Size: {total / (1024 * 1024):.1f} MB")
            return app_path

        print("Warning: .app bundle not found in dist/macos")
        return None

    except subprocess.CalledProcessError as e:
        print(f"Build failed with exit code {e.returncode}")
        return None
    except FileNotFoundError:
        print("Error: PyInstaller not found. Install with: pip install pyinstaller")
        return None


def preflight():
    """Verify required deps are importable in *this* interpreter before building."""
    required = ["numpy", "matplotlib", "yaml", "PIL", "tkinterdnd2", "sv_ttk", "ffpyplayer", "PyInstaller"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print("Preflight check failed. The following packages are not installed")
        print(f"in this Python interpreter ({sys.executable}):")
        for m in missing:
            print(f"  - {m}")
        print("\nInstall them with:")
        print(f"    {sys.executable} -m pip install -r requirements.txt")
        return False
    print(f"Preflight OK. Building with: {sys.executable}")
    return True


def create_release_package(app_path: Path):
    """Zip up the .app bundle with docs for distribution."""
    print("Creating release package...")

    arch = platform.machine()  # "arm64" on Apple Silicon, "x86_64" on Intel
    release_name = f"RestimFunscriptProcessor-v{__version__}-macOS-{arch}"
    release_dir = Path("dist") / release_name
    release_dir.mkdir(parents=True, exist_ok=True)

    # Copy the .app bundle preserving symlinks and permissions
    target_app = release_dir / f"{APP_NAME}.app"
    if target_app.exists():
        shutil.rmtree(target_app)
    shutil.copytree(app_path, target_app, symlinks=True)
    print(f"Copied .app to: {target_app}")

    # Documentation
    for doc in ["README.md", "config.json", "config.event_definitions.yml"]:
        src = Path(doc)
        if src.exists():
            shutil.copy2(src, release_dir / doc)
            print(f"Copied: {doc}")

    install_guide = release_dir / "INSTALLATION.txt"
    install_guide.write_text(
        f"""Restim Funscript Processor v{__version__} - macOS Installation

QUICK START:
1. Drag "{APP_NAME}.app" into /Applications (or anywhere you like)
2. Double-click to launch. If macOS blocks it ("unidentified developer"),
   right-click -> Open, then click Open in the dialog. You only need to do
   this once.
3. To pin to the Dock: launch the app, then right-click its Dock icon
   -> Options -> Keep in Dock.

GATEKEEPER WORKAROUND (if right-click Open does not work):
    xattr -dr com.apple.quarantine "{APP_NAME}.app"

USAGE:
- Select your .funscript file using the Browse button or drag-and-drop
- Configure parameters in the tabs
- Click "Process Files" to generate output files
- Output files are created alongside your input file (or central folder)

ARCHITECTURE:
This build targets: {arch}
If you need the other arch (Intel/Apple Silicon), rebuild on that machine.

VERSION: {__version__}
"""
    )
    print(f"Created installation guide: {install_guide}")

    # Create zip using ditto to preserve resource forks / symlinks inside
    # the .app bundle. `shutil.make_archive` works too but ditto is the
    # Apple-recommended tool for app bundles.
    archive_path = Path("dist") / f"{release_name}.zip"
    if archive_path.exists():
        archive_path.unlink()

    try:
        subprocess.run(
            ["ditto", "-c", "-k", "--keepParent", str(release_dir), str(archive_path)],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to shutil if ditto unavailable (shouldn't happen on macOS)
        print("ditto not available, falling back to shutil.make_archive")
        shutil.make_archive(
            str(archive_path.with_suffix("")),
            "zip",
            release_dir.parent,
            release_dir.name,
        )

    if archive_path.exists():
        size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"Release package created: {archive_path}")
        print(f"Archive size: {size_mb:.1f} MB")
        return archive_path

    return None


def main():
    print("=" * 60)
    print(f"Building {__app_name__} v{__version__} for macOS")
    print("=" * 60)

    if sys.platform != "darwin":
        print("Error: macOS .app bundles can only be built on macOS.")
        print("Run this script on a Mac.")
        return False

    if not preflight():
        return False

    app_path = build_macos_app()
    if not app_path:
        print("Build failed!")
        return False

    archive_path = create_release_package(app_path)
    if archive_path:
        print("\n" + "=" * 60)
        print("BUILD SUCCESSFUL!")
        print(f"App bundle:      {app_path}")
        print(f"Release package: {archive_path}")
        print("=" * 60)
        print("\nTo install: drag the .app into /Applications, then launch it.")
        print("To pin to Dock: right-click Dock icon -> Options -> Keep in Dock.")
        return True

    print("Failed to create release package")
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
