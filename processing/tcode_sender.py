"""UDP sender for T-code frames to restim.

Restim listens on UDP 12347 for T-code commands by default. Sends are
fire-and-forget: UDP to localhost on a closed port usually succeeds
silently (the kernel drops it), so "restim isn't running" is not a
send-time error and the stream keeps flowing. When restim comes up
later, subsequent packets land.

``probe()`` does a TCP connect to the same host:port (restim listens
on TCP there too) so the UI can report a real connection status before
starting the scheduler.
"""

import logging
import socket
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 12347


class TCodeUDPSender:
    """Thin UDP client. Open once, send many, close on shutdown.

    Send failures are swallowed and logged once per error kind to
    avoid spamming the log when restim is down. Use probe() for a
    pre-flight check that something is actually listening.
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None
        self._last_error_kind: Optional[str] = None
        self._open()

    def _open(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.1)
            self._sock = sock
        except OSError as e:
            logger.warning("Failed to create UDP socket: %s", e)
            self._sock = None

    def send(self, payload: bytes) -> bool:
        """Send one pre-formatted T-code payload. Returns True on success.

        Empty payloads are a no-op (returns False). If the socket
        previously errored, a successful send logs a one-line recovery.
        """
        if not payload:
            return False
        if self._sock is None:
            self._open()
            if self._sock is None:
                return False
        try:
            self._sock.sendto(payload, (self.host, self.port))
        except OSError as e:
            kind = type(e).__name__
            if kind != self._last_error_kind:
                logger.warning("UDP send to %s:%s failed: %s",
                               self.host, self.port, e)
                self._last_error_kind = kind
            return False
        if self._last_error_kind is not None:
            logger.info("UDP send to %s:%s recovered", self.host, self.port)
            self._last_error_kind = None
        return True

    def probe(self, timeout: float = 0.5) -> bool:
        """Best-effort check that restim is up on host:port.

        Opens a short-lived TCP connection to the same address. Restim
        accepts T-code over both TCP and UDP on 12347, so a successful
        TCP connect is a strong signal the UDP path will work too.
        Returns True on connect, False on refused/timeout/unreachable.
        """
        try:
            with socket.create_connection((self.host, self.port),
                                          timeout=timeout):
                return True
        except (OSError, socket.timeout):
            return False

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
