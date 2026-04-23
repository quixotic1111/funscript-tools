"""
restim_processor_core — Signal processing pipeline for Restim funscripts.

Converts position funscripts (single-axis or X/Y/Z[/rz] triplets) into
device-ready output files: electrode intensities, frequency and pulse
parameter channels, volume envelopes, and speed/acceleration signals.

Extracted from funscript-tools' processing engine to be consumable as
a library by FunGen and other downstream tools. See NOTICE.md for
attribution to the original authors.

Typical usage:

    from restim_processor_core import RestimProcessor, DEFAULT_CONFIG

    proc = RestimProcessor(DEFAULT_CONFIG)
    proc.process("input.funscript")
"""

import sys as _sys

# ---------------------------------------------------------------------------
# Back-compat shim for the bundled processing engine
# ---------------------------------------------------------------------------
# The engine's internal imports use bare names that pre-date the package
# restructure — `from funscript import Funscript`, `from processing.X import
# Y`, `from processor import RestimProcessor`, `from config import ...`.
# Rather than rewrite every one of those imports across ~25 modules in a
# single commit (which would conflict with in-progress feature work in the
# app), alias the package's submodules into sys.modules under the bare
# names. Python's import machinery then resolves those imports through the
# aliases whether the caller runs from the repo root (Restim Funscript
# Processor app) or has installed the package via pip (e.g. FunGen).
#
# This is transitional. A later commit will rewrite the engine's imports
# to proper relative paths and remove this shim.
# ---------------------------------------------------------------------------

from . import funscript as _funscript_pkg
_sys.modules.setdefault('funscript', _funscript_pkg)

from . import processing as _processing_pkg
_sys.modules.setdefault('processing', _processing_pkg)

from . import config as _config_mod
_sys.modules.setdefault('config', _config_mod)

from . import processor as _processor_mod
_sys.modules.setdefault('processor', _processor_mod)


# Public API ----------------------------------------------------------------

from .processor import RestimProcessor
from .config import DEFAULT_CONFIG, ConfigManager
from .funscript import Funscript

__all__ = [
    "RestimProcessor",
    "DEFAULT_CONFIG",
    "ConfigManager",
    "Funscript",
]

__version__ = "0.1.0"
