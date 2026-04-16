"""T-code encoder for streaming funscript signals to restim.

Maps in-memory Funscript signal buffers to T-code commands per the
mapping locked in for restim's T-code listener (default ports 12347
TCP/UDP, 12346 WebSocket):

    L0 ← alpha                (position, primary)
    L1 ← beta                 (position, secondary)
    V0 ← volume               (master gain)
    C0 ← frequency            (carrier; restim maps to 500-1000 Hz)
    P0 ← pulse_frequency      (→ 0-100 Hz in restim)
    P1 ← pulse_width          (→ 4-10 cycles in restim)
    P3 ← pulse_rise_time      (→ 2-20 cycles in restim)
    E1..E4 ← e1..e4           (4P intensities; only if the user has
                               assigned these 2-char T-code names in
                               restim's Funscript Kit preferences)

Wire format: ``<2-char axis><4-digit integer 0-9999>``, commands
separated by newlines. All signal values are 0.0-1.0 in our domain
and are scaled to 0-9999 before sending; restim normalizes back to
0-1 and applies its own per-axis range mapping.
"""

from typing import Dict, List, Optional

import numpy as np

from funscript import Funscript


# Default 3-phase mapping. Safe to stream without any user-side
# configuration in restim — all channels have default names.
DEFAULT_AXIS_MAP: Dict[str, str] = {
    'L0': 'alpha',
    'L1': 'beta',
    'V0': 'volume',
    'C0': 'frequency',
    'P0': 'pulse_frequency',
    'P1': 'pulse_width',
    'P3': 'pulse_rise_time',
}

# 4-phase extension. Requires the user to have assigned E1..E4 as
# T-code names for the INTENSITY_A..INTENSITY_D axes in restim's
# Funscript Kit preferences; otherwise restim silently drops them.
AXIS_MAP_4P: Dict[str, str] = {
    **DEFAULT_AXIS_MAP,
    'E1': 'e1',
    'E2': 'e2',
    'E3': 'e3',
    'E4': 'e4',
}


def sample_at(fs: Funscript, t: float) -> float:
    """Linearly interpolate a Funscript at time t (seconds).

    Clamps to the first/last sample — no extrapolation, so sending
    past the end of the script holds the final value.
    """
    x = fs.x
    y = fs.y
    n = len(x)
    if n == 0:
        return 0.0
    if n == 1 or t <= x[0]:
        return float(y[0])
    if t >= x[-1]:
        return float(y[-1])
    return float(np.interp(t, x, y))


def format_tcode_value(value: float) -> int:
    """Convert a normalized 0.0-1.0 signal to the 0-9999 wire integer."""
    if not np.isfinite(value):
        return 0
    return max(0, min(9999, int(round(value * 9999))))


def format_command(axis: str, value: float) -> str:
    """Format a single T-code command (no trailing newline)."""
    return f"{axis}{format_tcode_value(value):04d}"


def encode_frame(buffers: Dict[str, Funscript],
                 t: float,
                 axis_map: Optional[Dict[str, str]] = None,
                 enabled: Optional[Dict[str, bool]] = None) -> List[str]:
    """Build T-code commands for every mapped channel at time t.

    Args:
        buffers: Map of signal name → Funscript. Missing signals are
            skipped silently so the same axis_map can serve runs
            where, e.g., 4P outputs weren't generated.
        t: Sample time in seconds. Same domain as Funscript.x.
        axis_map: Map of T-code axis → signal name. Defaults to
            DEFAULT_AXIS_MAP.
        enabled: Optional per-axis on/off. Missing keys default to
            enabled; False explicitly mutes the channel.

    Returns:
        Commands in the order defined by axis_map. The caller joins
        them with '\\n' (see encode_frame_bytes).
    """
    if axis_map is None:
        axis_map = DEFAULT_AXIS_MAP
    commands = []
    for axis, signal_name in axis_map.items():
        if enabled is not None and not enabled.get(axis, True):
            continue
        fs = buffers.get(signal_name)
        if fs is None:
            continue
        commands.append(format_command(axis, sample_at(fs, t)))
    return commands


def encode_frame_bytes(buffers: Dict[str, Funscript],
                       t: float,
                       axis_map: Optional[Dict[str, str]] = None,
                       enabled: Optional[Dict[str, bool]] = None) -> bytes:
    """Encode a frame as a single ASCII payload ready for UDP send.

    Returns b'' when no channels would be sent (e.g. empty buffers,
    or everything muted) so the caller can skip the socket write.
    """
    commands = encode_frame(buffers, t, axis_map, enabled)
    if not commands:
        return b''
    return ('\n'.join(commands) + '\n').encode('ascii')
