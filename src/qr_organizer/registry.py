"""Service-registry breadcrumbs for the status aggregator.

The aggregator is a separate, read-only app. It does not maintain a list of
services; it reads this directory. Writing here at startup and every few minutes
means a late-installed aggregator picks up an already-running service on its
next periodic write, not only at the next restart.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from . import APP_NAME
from .util import now_iso

log = logging.getLogger(__name__)


class RegistryWriter:
    """Writes and refreshes this service's entry in the shared registry directory."""

    def __init__(
        self,
        *,
        directory: Path,
        name: str = APP_NAME,
        host: str = "",
        port: int = 0,
        base_url: str = "",
        log_path: str = "",
        refresh_minutes: int = 7,
    ) -> None:
        self.directory = directory
        self.name = name
        self.host = host
        self.port = port
        self.base_url = base_url.rstrip("/")
        self.log_path = log_path
        self.refresh_seconds = max(60, refresh_minutes * 60)
        self._timer: threading.Timer | None = None
        self._stopped = threading.Event()
        self.last_write: str | None = None
        self.last_error: str | None = None

    def payload(self, *, status: str = "running", extra: dict[str, Any] | None = None) -> dict:
        entry = {
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "health_url": f"{self.base_url}/health",
            "log_path": self.log_path,
            "config_url": f"{self.base_url}/config",
            "status_url": f"{self.base_url}/status",
            "status": status,
            "pid": os.getpid(),
            "updated_at": now_iso(),
        }
        if extra:
            entry.update(extra)
        return entry

    def write(self, *, status: str = "running", extra: dict[str, Any] | None = None) -> bool:
        """Write the breadcrumb. Returns False (having logged) if it can't."""
        if not self.directory.is_dir():
            self.last_error = f"{self.directory} does not exist"
            return False
        target = self.directory / f"{self.name}.json"
        try:
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.payload(status=status, extra=extra), indent=2))
            os.replace(tmp, target)
        except OSError as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("could not write the service registry entry %s: %s", target, exc)
            return False
        self.last_write = now_iso()
        self.last_error = None
        log.debug("service registry entry refreshed at %s", target)
        return True

    def start(self) -> None:
        self.write()
        self._schedule()

    def _schedule(self) -> None:
        if self._stopped.is_set():
            return
        self._timer = threading.Timer(self.refresh_seconds, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self) -> None:
        self.write()
        self._schedule()

    def stop(self) -> None:
        self._stopped.set()
        if self._timer is not None:
            self._timer.cancel()


def write_failure_breadcrumb(directory: Path, message: str, *, name: str = APP_NAME) -> None:
    """Last-ditch trace for a service that is too broken to start.

    Called from the top-level exception handler. If even this fails there is
    genuinely nothing left to try, so it stays quiet about its own failure.
    """
    try:
        if not directory.is_dir():
            return
        (directory / f"{name}.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "status": "error",
                    "error": message[:2000],
                    "pid": os.getpid(),
                    "updated_at": now_iso(),
                },
                indent=2,
            )
        )
    except Exception:
        pass
