"""
VLC HTTP-interface media-source adapter.

Ported from restim's net/media_source/vlc.py (MIT, Copyright 2023
diglet48). The protocol — polling VLC's /requests/status.xml and
/requests/playlist.xml every 100ms, parsing XML, feeding into the
MediaSource state machine — is preserved verbatim. The I/O layer is
rewritten: Qt's QTimer / QNetworkAccessManager / QXmlStreamReader
replaced with threading.Timer / requests / stdlib xml.etree.

Set up VLC for remote control:
    Preferences -> Show All -> Interface -> Main interfaces -> tick
    "Web" -> Lua -> HTTP -> set a password. Restart VLC.

    Then verify: http://127.0.0.1:8080/ in a browser with user "" and
    the password you set. You should see VLC's web UI.
"""

import logging
import threading
import time
import xml.etree.ElementTree as ET
from typing import Optional

try:
    import requests
    from requests.auth import HTTPBasicAuth
    _HAVE_REQUESTS = True
except ImportError:
    _HAVE_REQUESTS = False

from .mediasource import MediaSource, MediaStatusReport, MediaConnectionState


logger = logging.getLogger("funscript_tools.media.VLC")


# Polling intervals (ms). Matches restim exactly.
_POLL_INTERVAL_MS = 100        # connected, nominal rate (10 Hz)
_RETRY_INTERVAL_MS = 2000      # between failed connection attempts
_INVALID_ADDR_INTERVAL_MS = 5000  # bad URL format; back off
_REQUEST_TIMEOUT_S = 2.0


class VLC(MediaSource):
    """Poll VLC's HTTP web interface and feed the MediaSource state
    machine. See module docstring for VLC-side setup.

    Args:
        address: Base URL of the VLC HTTP interface, e.g.
            "http://127.0.0.1:8080". Trailing slash optional.
        password: HTTP password (VLC sends basic auth challenge on
            first request). Leave empty if VLC is unconfigured.
        username: VLC accepts any username by default; leave empty
            unless your setup requires one.
    """

    def __init__(self, address: str = "http://127.0.0.1:8080",
                 password: str = "", username: str = ""):
        super().__init__()
        if not _HAVE_REQUESTS:
            raise ImportError(
                "The VLC adapter requires the `requests` library. "
                "Install it with: pip install requests")

        self._enabled = False
        self._address = address.rstrip("/")
        self._auth = HTTPBasicAuth(username, password) if (password or username) else None

        self._playlist_id: Optional[str] = None
        self._filename: Optional[str] = None

        # Exposed so the UI can show a human-readable reason when
        # we're NOT_CONNECTED — the state machine only emits on
        # transitions, but first-poll-fails-repeatedly is a single
        # NOT_CONNECTED → NOT_CONNECTED non-transition that would
        # otherwise be invisible to the user.
        self.last_error: str = ""

        # Polling thread state. A single threading.Timer drives the
        # loop; enable() primes it, disable() cancels it. All network
        # work happens off the main thread.
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.RLock()

        # Short-lived HTTP session so we get connection pooling between
        # polls instead of a fresh socket every 100ms.
        self._session = requests.Session()

    # --- public controls ------------------------------------------------

    def enable(self) -> None:
        with self._lock:
            self._enabled = True
            self._cancel_timer_locked()
            self._timer = threading.Timer(0, self._poll_once)
            self._timer.daemon = True
            self._timer.start()

    def disable(self) -> None:
        with self._lock:
            self._enabled = False
            self._cancel_timer_locked()
        # Tell the state machine we've disconnected. Safe to call
        # outside the lock — set_state only touches its own fields.
        self.set_state(MediaStatusReport(timestamp=time.time()))

    def is_enabled(self) -> bool:
        return self._enabled

    def update_address(self, address: str, password: str = "",
                       username: str = "") -> None:
        """Update connection settings live. Takes effect on next poll."""
        with self._lock:
            self._address = address.rstrip("/")
            self._auth = HTTPBasicAuth(username, password) if (password or username) else None
            # Force playlist lookup on next successful status poll.
            self._playlist_id = None
            self._filename = None

    # --- polling loop ---------------------------------------------------

    def _schedule_next(self, delay_ms: int) -> None:
        """Schedule the next poll after delay_ms. Caller holds the lock
        or guarantees no concurrent enable/disable."""
        if not self._enabled:
            return
        self._timer = threading.Timer(delay_ms / 1000.0, self._poll_once)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer_locked(self) -> None:
        """Cancel any pending timer. Caller must hold the lock."""
        if self._timer is not None:
            try:
                self._timer.cancel()
            except Exception:
                pass
            self._timer = None

    def _poll_once(self) -> None:
        """One iteration of the polling loop — query VLC's status and
        (if needed) playlist, parse, feed the state machine, reschedule.
        """
        if not self._enabled:
            return

        address = self._address
        if not address or not address.startswith(("http://", "https://")):
            logger.warning("Invalid VLC address: %r", address)
            with self._lock:
                self._schedule_next(_INVALID_ADDR_INTERVAL_MS)
            return

        status_url = f"{address}/requests/status.xml"

        try:
            resp = self._session.get(
                status_url, auth=self._auth,
                timeout=_REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            # Typical during initial attempts / VLC not running.
            err = str(exc)
            self.last_error = err
            logger.debug("VLC status request failed: %s", err)
            report = MediaStatusReport(
                timestamp=time.time(),
                connectionState=MediaConnectionState.NOT_CONNECTED,
                errorString=err)
            self.set_state(report)
            self._reset_media_cache()
            with self._lock:
                self._schedule_next(_RETRY_INTERVAL_MS)
            return

        if resp.status_code != 200:
            err = f"HTTP {resp.status_code}"
            self.last_error = err
            logger.debug("VLC status HTTP %d", resp.status_code)
            self.set_state(MediaStatusReport(
                timestamp=time.time(),
                connectionState=MediaConnectionState.NOT_CONNECTED,
                errorString=err))
            self._reset_media_cache()
            with self._lock:
                self._schedule_next(_RETRY_INTERVAL_MS)
            return

        # Successful request → clear any lingering error.
        self.last_error = ""

        try:
            report = self._parse_status_xml(resp.content)
        except ET.ParseError as exc:
            logger.debug("VLC status XML parse error: %s", exc)
            with self._lock:
                self._schedule_next(_RETRY_INTERVAL_MS)
            return

        self.set_state(report)

        # If we saw a new playlist id, fetch the playlist to learn the
        # currently-loaded file path. Decoupled from status polling so
        # we don't spam playlist.xml every 100ms.
        if self._playlist_id is not None and self._filename is None:
            self._fetch_playlist(address)

        with self._lock:
            self._schedule_next(_POLL_INTERVAL_MS)

    def _reset_media_cache(self) -> None:
        self._playlist_id = None
        self._filename = None

    # --- XML parsing ----------------------------------------------------

    def _parse_status_xml(self, payload: bytes) -> MediaStatusReport:
        """Parse VLC's /requests/status.xml response."""
        root = ET.fromstring(payload)
        state = _first_text(root, "state")
        rate = _first_text(root, "rate")
        position = _first_text(root, "position")
        currentplid = _first_text(root, "currentplid")
        length = _first_text(root, "length")

        try:
            playback_rate = float(rate) if rate is not None else 1.0
        except ValueError:
            playback_rate = 1.0

        if currentplid == "-1":
            currentplid = None

        # Track playlist id to decide whether we need to fetch
        # playlist.xml for the filename.
        if currentplid is None:
            self._reset_media_cache()
        elif self._playlist_id != currentplid:
            self._playlist_id = currentplid
            # Invalidate filename — will refetch.
            self._filename = None

        file_path = self._filename or ""

        # VLC reports position as a fraction of length; combine to
        # get seconds. If length is unknown, fall back to zero.
        try:
            length_s = float(length) if length else 0.0
            position_frac = float(position) if position else 0.0
        except ValueError:
            length_s = 0.0
            position_frac = 0.0
        media_position = length_s * position_frac

        if file_path and state == "playing":
            conn = MediaConnectionState.CONNECTED_AND_PLAYING
        elif file_path:
            conn = MediaConnectionState.CONNECTED_AND_PAUSED
        else:
            conn = MediaConnectionState.CONNECTED_BUT_NO_FILE_LOADED

        return MediaStatusReport(
            timestamp=time.time(),
            connectionState=conn,
            filePath=file_path,
            playbackRate=playback_rate,
            claimed_media_position=media_position,
        )

    def _fetch_playlist(self, address: str) -> None:
        """Fetch /requests/playlist.xml and extract the filename for
        the currently-playing playlist entry."""
        url = f"{address}/requests/playlist.xml"
        try:
            resp = self._session.get(
                url, auth=self._auth, timeout=_REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            logger.debug("VLC playlist request failed: %s", exc)
            return
        if resp.status_code != 200:
            return

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as exc:
            logger.debug("VLC playlist XML parse error: %s", exc)
            return

        # Walk <node> → <leaf> looking for the entry whose id matches
        # the current playlist id. VLC encodes file paths as file://
        # URIs on the uri attribute.
        for leaf in root.iter("leaf"):
            if leaf.get("id") == self._playlist_id:
                uri = leaf.get("uri", "")
                self._filename = _local_path_from_uri(uri)
                logger.info("VLC loaded file: %s", self._filename)
                return


def _first_text(root: ET.Element, tag: str) -> Optional[str]:
    """Return the text of the first child matching `tag`, or None."""
    node = root.find(tag)
    if node is None:
        return None
    return node.text


def _local_path_from_uri(uri: str) -> str:
    """Convert a file:// URI to a local filesystem path.

    VLC's playlist.xml encodes paths as file-URIs with percent-encoded
    special characters (spaces → %20, etc.). Returns an empty string
    for non-file schemes (e.g. network streams).
    """
    if not uri:
        return ""
    from urllib.parse import urlparse, unquote
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return ""
    # urlparse leaves the path with a leading slash on POSIX and a
    # /C:/... on Windows; unquote handles %20.
    path = unquote(parsed.path or "")
    # On Windows file URIs look like file:///C:/foo, leaving path as
    # /C:/foo — strip the leading slash before a drive letter.
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return path
