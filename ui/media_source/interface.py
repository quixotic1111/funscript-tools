"""
Abstract base class for external media-source adapters.

Ported from restim's net/media_source/interface.py (MIT, Copyright 2023
diglet48). The only change vs. the original is the Qt signal replacement:
restim's `connectionStatusChanged = QtCore.Signal()` becomes a plain
callback list with subscribe/emit semantics, so no Qt dependency.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List


class MediaConnectionState(Enum):
    NOT_CONNECTED = 0
    CONNECTED_BUT_NO_FILE_LOADED = 1
    CONNECTED_AND_PAUSED = 2
    CONNECTED_AND_PLAYING = 3

    def is_connected(self) -> bool:
        return self in (
            MediaConnectionState.CONNECTED_BUT_NO_FILE_LOADED,
            MediaConnectionState.CONNECTED_AND_PAUSED,
            MediaConnectionState.CONNECTED_AND_PLAYING,
        )

    def is_file_loaded(self) -> bool:
        return self in (
            MediaConnectionState.CONNECTED_AND_PAUSED,
            MediaConnectionState.CONNECTED_AND_PLAYING,
        )

    def is_playing(self) -> bool:
        return self == MediaConnectionState.CONNECTED_AND_PLAYING


class MediaSourceInterface(ABC):
    """Abstract surface for an external media player adapter.

    Adapters poll their player, parse responses into MediaStatusReport,
    and update an internal state machine (in MediaSource, not here).
    Consumers subscribe via `on_connection_changed` to be notified when
    the connection-state or loaded-file changes; they call `media_path`
    and `map_timestamp` to read the current state.

    Callbacks fire from the polling thread — consumers that touch UI
    widgets should dispatch onto their UI thread (e.g. via tk.after()).
    """

    def __init__(self):
        # Simple callback registry — replaces restim's Qt signal.
        self._connection_status_listeners: List[Callable[[], None]] = []

    # --- lifecycle ------------------------------------------------------

    @abstractmethod
    def enable(self) -> None:
        """Start the polling / connection loop."""
        ...

    @abstractmethod
    def disable(self) -> None:
        """Stop the polling / connection loop."""
        ...

    @abstractmethod
    def is_enabled(self) -> bool:
        ...

    # --- state introspection --------------------------------------------

    @abstractmethod
    def state(self) -> MediaConnectionState:
        ...

    def is_internal(self) -> bool:
        return False

    def is_connected(self) -> bool:
        return self.state() in (
            MediaConnectionState.CONNECTED_BUT_NO_FILE_LOADED,
            MediaConnectionState.CONNECTED_AND_PAUSED,
            MediaConnectionState.CONNECTED_AND_PLAYING,
        )

    def is_media_loaded(self) -> bool:
        return self.state() in (
            MediaConnectionState.CONNECTED_AND_PAUSED,
            MediaConnectionState.CONNECTED_AND_PLAYING,
        )

    def is_playing(self) -> bool:
        return self.state() == MediaConnectionState.CONNECTED_AND_PLAYING

    @abstractmethod
    def media_path(self) -> str:
        ...

    def set_media_sync_offset(self, offset_in_seconds: float) -> None:
        """Adjust a fixed time offset applied to map_timestamp() output."""
        pass

    # --- event subscription ---------------------------------------------

    def on_connection_changed(
        self, callback: Callable[[], None]
    ) -> Callable[[], None]:
        """Subscribe to connection-state or loaded-file changes.

        Returns a callable that unsubscribes when called.
        """
        self._connection_status_listeners.append(callback)

        def _unsubscribe() -> None:
            try:
                self._connection_status_listeners.remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    def _emit_connection_changed(self) -> None:
        """Notify all subscribed listeners. Swallow per-listener
        exceptions so one bad callback doesn't break the others."""
        for cb in list(self._connection_status_listeners):
            try:
                cb()
            except Exception:
                # Mirror Qt's default signal-emission behaviour: a
                # listener failure shouldn't tear down the source.
                pass
