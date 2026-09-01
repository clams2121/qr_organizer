"""systemd integration: readiness notification and the hang-detection watchdog.

Implemented directly against the `NOTIFY_SOCKET` protocol so the service does
not grow a dependency for eleven lines of datagram. Every function is a no-op
outside systemd, which is what makes the plain-venv deployment work unchanged.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from typing import Callable

log = logging.getLogger(__name__)


def _notify(message: str) -> bool:
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return False
    if address.startswith("@"):  # abstract namespace
        address = "\0" + address[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(address)
            sock.sendall(message.encode("utf-8"))
        return True
    except OSError as exc:
        log.debug("sd_notify(%r) failed: %s", message, exc)
        return False


def ready(status: str = "") -> None:
    _notify(f"READY=1\nSTATUS={status}" if status else "READY=1")


def set_status(status: str) -> None:
    _notify(f"STATUS={status}")


def stopping() -> None:
    _notify("STOPPING=1")


def watchdog_interval_seconds() -> float | None:
    """Half of `WatchdogSec`, which is the interval systemd expects pings at."""
    raw = os.environ.get("WATCHDOG_USEC")
    if not raw:
        return None
    pid = os.environ.get("WATCHDOG_PID")
    if pid and pid != str(os.getpid()):
        return None
    try:
        return int(raw) / 1_000_000 / 2
    except ValueError:
        return None


class Watchdog:
    """Pings systemd only while a liveness probe still says the service works.

    Pinging unconditionally from a background thread would defeat the point:
    systemd would keep a wedged service alive forever. If the probe says the
    service is broken, the pings stop and systemd restarts it.
    """

    def __init__(self, probe: Callable[[], bool], interval: float) -> None:
        self.probe = probe
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="sd-watchdog", daemon=True)
        self._thread.start()
        log.info("systemd watchdog active, pinging every %.1fs", self.interval)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                healthy = self.probe()
            except Exception:
                log.exception("watchdog probe raised; withholding the keepalive ping")
                continue
            if healthy:
                _notify("WATCHDOG=1")
            else:
                log.error("watchdog probe reports the service cannot work; withholding ping")

    def stop(self) -> None:
        self._stop.set()
