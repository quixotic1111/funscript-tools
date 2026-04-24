"""
External media-source adapters for funscript-tools.

Adapted from restim's net/media_source/ (MIT-licensed, Copyright 2023
diglet48 — https://github.com/diglet48/restim). The original uses
PySide6 for its I/O layer; this port strips Qt out and replaces it
with Python stdlib + `requests` so the adapters can be used from
tkinter-based UIs without dragging in a second UI framework.

See NOTICE below for attribution.

Structure:
  interface.py   — abstract base + connection-state enum
  mediasource.py — shared state-machine / timeline-mapping logic
  vlc.py         — HTTP XML-RPC adapter for VLC's web interface

Only the VLC adapter is ported in V1. Other restim adapters (MPC-HC,
Kodi, HereSphere) can be ported later following the same pattern.

Typical usage:

    from ui.media_source.vlc import VLC

    player = VLC(address='http://127.0.0.1:8080', password='1234')
    player.on_connection_changed(lambda: print('state:', player.state()))
    player.enable()
    # ... later ...
    if player.is_playing():
        t_seconds = player.map_timestamp(time.time())
    player.disable()
"""

# NOTICE -------------------------------------------------------------------
# This package is a Python-stdlib port of restim's media-source layer.
#
# Original:  https://github.com/diglet48/restim
#            MIT License, Copyright (c) 2023 diglet48
#
# The original interfaces (MediaConnectionState enum, MediaSource state
# machine, VLC HTTP protocol parsing) are preserved near-verbatim. Only
# the I/O layer has been rewritten: Qt timers/signals/sockets replaced
# with threading.Timer / callback lists / requests.
# --------------------------------------------------------------------------

from .interface import MediaConnectionState, MediaSourceInterface
from .mediasource import MediaSource, MediaState, MediaStatusReport
from .vlc import VLC

__all__ = [
    "MediaConnectionState",
    "MediaSourceInterface",
    "MediaSource",
    "MediaState",
    "MediaStatusReport",
    "VLC",
]
