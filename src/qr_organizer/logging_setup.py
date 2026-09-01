"""Logging: rotated plain-text file, stderr, and an in-memory ring for the status page."""

from __future__ import annotations

import logging
import logging.handlers
import os
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class LogRecordView:
    """A log line as the status page renders it."""

    when: str
    level: str
    logger: str
    message: str


class RingBufferHandler(logging.Handler):
    """Keeps the most recent records in memory so the status page can show them.

    The file on disk is the real log; this exists so that a user looking at the
    web UI sees failures without needing shell access.
    """

    def __init__(self, capacity: int = 400) -> None:
        super().__init__(level=logging.INFO)
        self._records: deque[LogRecordView] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if record.exc_info:
                message = f"{message}\n{self.format(record)}"
            view = LogRecordView(
                when=datetime.fromtimestamp(record.created, tz=timezone.utc)
                .astimezone()
                .strftime(DATE_FORMAT),
                level=record.levelname,
                logger=record.name,
                message=message,
            )
        except Exception:  # pragma: no cover -- logging must never raise
            return
        with self._lock:
            self._records.append(view)

    def snapshot(self, min_level: str = "INFO", limit: int = 200) -> list[LogRecordView]:
        threshold = logging.getLevelName(min_level)
        if not isinstance(threshold, int):
            threshold = logging.INFO
        with self._lock:
            records = list(self._records)
        keep = [
            record
            for record in records
            if _level_value(record.level) >= threshold
        ]
        return keep[-limit:][::-1]

    def count_at_least(self, min_level: str) -> int:
        threshold = _level_value(min_level)
        with self._lock:
            return sum(1 for record in self._records if _level_value(record.level) >= threshold)


def _level_value(name: str) -> int:
    value = logging.getLevelName(name)
    return value if isinstance(value, int) else 0


RING = RingBufferHandler()


def resolve_log_file(log_dir: Path, data_dir: Path) -> tuple[Path, str | None]:
    """Return the log file to use, plus a note if we had to fall back.

    The standard path is `/var/log/qr-organizer/app.log` so the status
    aggregator can find it predictably. When that is not writable (a plain
    venv run as an unprivileged user), fall back inside the data dir rather
    than dying -- but say so, loudly, once.
    """
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        probe = log_dir / ".write-probe"
        probe.touch()
        probe.unlink()
        return log_dir / "app.log", None
    except OSError as exc:
        fallback = data_dir / "logs"
        fallback.mkdir(parents=True, exist_ok=True)
        note = (
            f"log directory {log_dir} is not writable ({exc}); "
            f"logging to {fallback / 'app.log'} instead"
        )
        return fallback / "app.log", note


def configure(
    *,
    log_dir: Path,
    data_dir: Path,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> Path:
    """Install handlers on the root logger. Returns the active log file path."""
    log_file, note = resolve_log_file(log_dir, data_dir)

    root = logging.getLogger()
    root.setLevel(_level_value(level) or logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    RING.setFormatter(formatter)
    root.addHandler(RING)

    # Flask/werkzeug request logs are noise at INFO for a personal app.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if note:
        logging.getLogger(__name__).warning(note)
    logging.getLogger(__name__).info("logging to %s (pid %d)", log_file, os.getpid())
    return log_file


def bootstrap_stderr_logging(level: str = "INFO") -> None:
    """Minimal logging for the window before config is loaded."""
    logging.basicConfig(level=_level_value(level) or logging.INFO, format=LOG_FORMAT,
                        datefmt=DATE_FORMAT)
