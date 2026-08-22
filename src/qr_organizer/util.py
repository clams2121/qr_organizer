"""Small shared helpers: timestamps, code formatting, slugs."""

from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone

CODE_RE = re.compile(r"^([A-Z][A-Z0-9]{1,7})-([0-9]{2,8})$")


def now() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return now().isoformat(timespec="seconds")


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def is_expired(timestamp: str | None, minutes: int) -> bool:
    moment = parse_iso(timestamp)
    if moment is None:
        return True
    return now() - moment > timedelta(minutes=minutes)


def humanise_age(timestamp: str | None) -> str:
    moment = parse_iso(timestamp)
    if moment is None:
        return "never"
    delta = now() - moment
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def format_code(prefix: str, number: int, digits: int = 4) -> str:
    return f"{prefix.upper()}-{number:0{digits}d}"


def parse_code(raw: str) -> tuple[str, int] | None:
    """Split `BIN-0042` into ('BIN', 42). Returns None if it isn't a code."""
    match = CODE_RE.match(raw.strip().upper())
    if not match:
        return None
    return match.group(1), int(match.group(2))


def extract_code(raw: str) -> str | None:
    """Pull a bin/location code out of a raw scan payload.

    Labels encode a full URL (`https://host/s/BIN-0042`) so a phone's own
    camera app can open them, but the in-app scanner may also see a bare code.
    Both are accepted; anything else is rejected rather than guessed at.
    """
    if not raw:
        return None
    text = raw.strip()
    if "/" in text:
        text = text.rstrip("/").rsplit("/", 1)[-1]
    if "?" in text:
        text = text.split("?", 1)[0]
    parsed = parse_code(text)
    if parsed is None:
        return None
    return format_code(parsed[0], parsed[1], digits=len(text.split("-", 1)[1]))


def new_session_key() -> str:
    return secrets.token_urlsafe(16)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify(value: str, fallback: str = "item") -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned[:48] or fallback
