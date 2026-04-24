"""Shared axis-marker helpers.

Spatial-3D inputs come from several ecosystems with different axis
naming conventions (funscript ``.x/.y/.z/.rz``, fungen / heresphere
``.sway/.stroke/.surge/.roll``, etc). This module centralizes the
recognized marker set so the triplet orderer, processor output-naming,
and viewer all agree on what suffix belongs to which axis — and on how
to recover the clean "project basename" from a suffixed stem.

Marker → axis mapping:
    .x    .sway               → X
    .y    .heave  .stroke     → Y (stroke / primary)
    .z    .surge              → Z
    .rz   .roll   .twist      → rz (roll around shaft)

The regex lists longer/more-specific alternatives first so ``.rz``
wins over ``.z`` and ``.stroke`` wins over ``.x``.
"""

import re

_AXIS_MARKER_RE = re.compile(
    r'\.(rz|roll|twist|sway|surge|heave|stroke|x|y|z)$',
    re.IGNORECASE)

MARKER_TO_AXIS = {
    'rz': 'rz', 'roll': 'rz', 'twist': 'rz',
    'x':  'x',  'sway':  'x',
    'y':  'y',  'heave': 'y', 'stroke': 'y',
    'z':  'z',  'surge': 'z',
}

# Known multi-axis markers that Spatial 3D Linear does NOT consume —
# rotations around sway / vertical (pitch, yaw), device-specific
# haptic aux channels. The triplet orderer drops these silently so
# they can't leak into the X/Y/Z/rz slots via the alphabetical
# unmarked-fallback. Extend this list as new conventions show up.
_NON_TRIPLET_MARKER_RE = re.compile(
    r'\.(pitch|yaw|vib|valve|suck)$',
    re.IGNORECASE)


def strip_axis_suffix(stem: str) -> str:
    """Strip a trailing axis marker from a funscript stem.

    ``capture_123.sway`` → ``capture_123``. Case-insensitive. Returns
    the stem unchanged when no marker is present (plain stroke files,
    non-triplet inputs).
    """
    return _AXIS_MARKER_RE.sub('', stem)


def axis_from_stem(stem: str):
    """Return the canonical axis ('x'/'y'/'z'/'rz') for a stem, or
    None if it carries no recognized marker."""
    m = _AXIS_MARKER_RE.search(stem)
    return MARKER_TO_AXIS[m.group(1).lower()] if m else None


def is_non_triplet_axis(stem: str) -> bool:
    """True if the stem ends in a known irrelevant-to-S3D marker
    (pitch, yaw, device aux channels). Used to prune drops before
    slot assignment so unknown axes can't contaminate X/Y/Z."""
    return _NON_TRIPLET_MARKER_RE.search(stem) is not None
