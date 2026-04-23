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
