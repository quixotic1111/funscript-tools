"""
Shared state machine for external media-source adapters.

Ported from restim's net/media_source/mediasource.py (MIT, Copyright
2023 diglet48). The original inherits from QObject for signal support;
this port replaces it with the callback-registry in
MediaSourceInterface, so no Qt dependency remains.

The state machine and timeline-mapping logic are preserved verbatim
— only the class hierarchy and signal emission differ.
"""

import copy
import logging
from dataclasses import dataclass

from .interface import MediaSourceInterface, MediaConnectionState


logger = logging.getLogger("funscript_tools.media")


@dataclass
class MediaState:
    """Current media state tracked by the state machine.

    Attributes:
        connectionState: current MediaConnectionState.
        filePath: the loaded file path (empty string if no file).
        cursor: when paused, the media position in seconds; when
            playing, the time at which media was started.
        media_play_timestamp: wall-clock timestamp (time.time()) when
            the currently-playing media started playing.
    """
    connectionState: MediaConnectionState
    filePath: str = ""
    cursor: float = -1
    media_play_timestamp: float = -1


@dataclass
class MediaStatusReport:
    """Single sample of player state, produced by a polling adapter.

    Adapters build one of these per poll, pass it to
    MediaSource.set_state(), which runs the state machine.
    """
    timestamp: float
    errorString: str = ""
    connectionState: MediaConnectionState = MediaConnectionState.NOT_CONNECTED
    filePath: str = ""
    playbackRate: float = 1
    claimed_media_position: float = -1


class MediaSource(MediaSourceInterface):
    """Base class for concrete adapters.

    Subclasses poll their player and call ``self.set_state(report)``
    with each sample. The state machine here normalises the report
    into the ``self.last_state`` dataclass and fires
    ``connection_changed`` when the connection state or loaded file
    path transitions.

    map_timestamp() converts a wall-clock timestamp into a media-time
    timestamp — the primary consumer-facing API.
    """

    def __init__(self):
        super().__init__()
        self.last_state = MediaState(MediaConnectionState.NOT_CONNECTED)
        self.media_sync_offset = 0.0

    def state(self) -> MediaConnectionState:
        return self.last_state.connectionState

    def pre_update(self, last_state: MediaState, state: MediaStatusReport):
        """Hook for subclasses to mutate the incoming report before
        the state machine runs. Default: identity."""
        return state

    def set_state(self, report: MediaStatusReport) -> None:
        """Advance the state machine with a new status report.

        This is a direct port of restim's logic — see the original
        project for the rationale behind each branch. The short
        version: we treat the last observed state + the new report
        as inputs, compute the resulting new state, and emit a
        change notification if the connection state or file path
        differs.
        """
        new_state = copy.copy(self.last_state)

        # Only support playback rate == 1. Anything else is treated
        # as paused so the map_timestamp() math stays linear.
        if report.playbackRate != 1:
            if report.connectionState.is_playing():
                report.connectionState = MediaConnectionState.CONNECTED_AND_PAUSED

        # Initial connect.
        if self.last_state.connectionState == MediaConnectionState.NOT_CONNECTED:
            if report.connectionState.is_connected():
                logger.info("connected")
                new_state.connectionState = report.connectionState
                new_state.filePath = report.filePath
                new_state.cursor = report.claimed_media_position
                new_state.media_play_timestamp = report.timestamp

                if report.connectionState.is_playing():
                    logger.info("play-on-connect")
                    new_state.media_play_timestamp = report.timestamp
                    new_state.cursor = report.claimed_media_position

        # Any disconnect.
        elif not report.connectionState.is_connected():
            logger.info("disconnected")
            new_state.connectionState = MediaConnectionState.NOT_CONNECTED
            new_state.filePath = ""

        elif self.last_state.connectionState == MediaConnectionState.CONNECTED_BUT_NO_FILE_LOADED:
            if report.connectionState.is_file_loaded():
                logger.info("file loaded")
                new_state.filePath = report.filePath
                new_state.connectionState = report.connectionState
                new_state.cursor = report.claimed_media_position
                new_state.media_play_timestamp = report.timestamp

                if report.connectionState.is_playing():
                    logger.info("play-on-load")

        elif self.last_state.connectionState == MediaConnectionState.CONNECTED_AND_PAUSED:
            if self.last_state.connectionState.is_file_loaded():
                if report.connectionState.is_file_loaded():
                    if self.last_state.filePath != report.filePath:
                        logger.info("loaded file changed")
                        new_state.filePath = report.filePath

            if report.connectionState.is_playing():
                logger.info("play")
                new_state.connectionState = report.connectionState
                new_state.media_play_timestamp = report.timestamp
                new_state.cursor = report.claimed_media_position
            elif not report.connectionState.is_file_loaded():
                logger.info("file unload")
                new_state.connectionState = report.connectionState
                new_state.filePath = ""
            else:
                # seek while paused
                new_state.cursor = report.claimed_media_position

        elif self.last_state.connectionState == MediaConnectionState.CONNECTED_AND_PLAYING:
            if self.last_state.connectionState.is_file_loaded():
                if report.connectionState.is_file_loaded():
                    if self.last_state.filePath != report.filePath:
                        logger.info("loaded file changed")
                        new_state.filePath = report.filePath

            # play → unloaded
            if report.connectionState == MediaConnectionState.CONNECTED_BUT_NO_FILE_LOADED:
                logger.info("file unload")
                new_state.connectionState = report.connectionState
                new_state.filePath = ""
            # play → pause
            elif report.connectionState == MediaConnectionState.CONNECTED_AND_PAUSED:
                new_state.connectionState = MediaConnectionState.CONNECTED_AND_PAUSED
                new_state.cursor = report.claimed_media_position
                logger.info("pause")
            else:
                # Still playing. Compare wall-clock elapsed against
                # reported-position delta to detect scrub/resync.
                drift = ((report.timestamp - self.last_state.media_play_timestamp)
                         - (report.claimed_media_position - self.last_state.cursor))
                if abs(drift) > 2.0:
                    new_state.media_play_timestamp = report.timestamp
                    new_state.connectionState = MediaConnectionState.CONNECTED_AND_PLAYING
                    new_state.cursor = report.claimed_media_position
                    logger.info("drift too much (%s), re-sync", drift)

        prev_state = self.last_state
        self.last_state = new_state
        if (prev_state.connectionState != new_state.connectionState
                or prev_state.filePath != new_state.filePath):
            self._emit_connection_changed()

    def map_timestamp(self, timestamp: float) -> float:
        """Convert a wall-clock timestamp to a media-time timestamp.

        Args:
            timestamp: wall-clock time (e.g. ``time.time()``).

        Returns:
            Position in seconds within the currently-loaded media. If
            playing, uses the linear extrapolation
            ``wall_clock_now - play_start_wall + cursor_at_play_start``.
            If paused, returns the fixed cursor position.

        This is the primary API consumers use to drive scheduled
        output (e.g. T-code generation) against the external player's
        timeline between polls.
        """
        if self.is_playing():
            return (timestamp
                    - self.last_state.media_play_timestamp
                    + self.last_state.cursor
                    - self.media_sync_offset)
        return self.last_state.cursor - self.media_sync_offset

    def media_path(self) -> str:
        return self.last_state.filePath

    def set_media_sync_offset(self, offset_in_seconds: float) -> None:
        self.media_sync_offset = offset_in_seconds
