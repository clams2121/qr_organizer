"""Readiness-focused health reporting.

`/health` answers "can this service currently do its job?", not "is the process
alive". Its job is: accept a scan, identify a photo, and answer a search. A
missing API key or an unreachable Ollama means `degraded` even though the web
UI is perfectly responsive, because photos submitted right now would fail.
"""

from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .util import now_iso

log = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_ERROR = "error"

#: Below this much free space, thumbnails and photos are at risk.
MIN_FREE_BYTES = 200 * 1024 * 1024


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    critical: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"status": STATUS_OK if self.ok else
                (STATUS_ERROR if self.critical else STATUS_DEGRADED),
                "detail": self.detail}


@dataclass
class HealthReport:
    status: str
    checks: list[Check] = field(default_factory=list)
    last_success: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checks": {check.name: check.to_dict() for check in self.checks},
            "last_success": self.last_success,
            "checked_at": now_iso(),
        }

    @property
    def http_status(self) -> int:
        # A degraded service is still worth talking to, so only a hard error is
        # a 503. The aggregator keys off the JSON body, not the status code.
        return 503 if self.status == STATUS_ERROR else 200


class HealthMonitor:
    """Collects checks and remembers the last time real work succeeded."""

    def __init__(self) -> None:
        self._checks: list[tuple[str, Callable[[], tuple[bool, str]], bool]] = []
        self._last_success: str | None = None
        self._lock = threading.Lock()

    def register(
        self, name: str, probe: Callable[[], tuple[bool, str]], *, critical: bool = False
    ) -> None:
        self._checks.append((name, probe, critical))

    def record_success(self) -> None:
        with self._lock:
            self._last_success = now_iso()

    @property
    def last_success(self) -> str | None:
        with self._lock:
            return self._last_success

    def report(self) -> HealthReport:
        checks: list[Check] = []
        for name, probe, critical in self._checks:
            try:
                ok, detail = probe()
            except Exception as exc:  # a probe that throws is itself a failure
                ok, detail = False, f"probe raised {type(exc).__name__}: {exc}"
            checks.append(Check(name=name, ok=ok, detail=detail, critical=critical))

        if any(not check.ok and check.critical for check in checks):
            status = STATUS_ERROR
        elif any(not check.ok for check in checks):
            status = STATUS_DEGRADED
        else:
            status = STATUS_OK
        return HealthReport(status=status, checks=checks, last_success=self.last_success)


def disk_probe(path) -> tuple[bool, str]:
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return False, f"cannot stat {path}: {exc}"
    free_gb = usage.free / 1024**3
    if usage.free < MIN_FREE_BYTES:
        return False, f"only {free_gb:.1f} GiB free on {path}"
    return True, f"{free_gb:.1f} GiB free on {path}"
