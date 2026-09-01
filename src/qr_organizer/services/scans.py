"""The scan log, and the informal device identity attached to each entry.

There is no auth here by design. Identity is best-effort attribution so that
"who pulled that?" has an answer: the Tailscale peer name when the Tailscale
CLI can tell us, otherwise the source IP, otherwise nothing.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from typing import Any

from ..db import Database
from ..util import now_iso
from . import rows_to_dicts

log = logging.getLogger(__name__)

_WHOIS_CACHE: dict[str, tuple[float, str]] = {}
_WHOIS_TTL_SECONDS = 600


def device_name_for_ip(ip: str) -> str:
    """Ask Tailscale who owns this address. Cached; failure is simply 'unknown'."""
    if not ip:
        return ""
    cached = _WHOIS_CACHE.get(ip)
    if cached and time.monotonic() - cached[0] < _WHOIS_TTL_SECONDS:
        return cached[1]

    name = ""
    binary = shutil.which("tailscale")
    if binary:
        try:
            result = subprocess.run(
                [binary, "whois", "--json", ip], capture_output=True, timeout=4, check=False
            )
            if result.returncode == 0:
                payload = json.loads(result.stdout or "{}")
                node = (payload.get("Node") or {}).get("Name", "")
                user = (payload.get("UserProfile") or {}).get("DisplayName", "")
                name = " / ".join(part for part in (node.split(".")[0], user) if part)
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
            log.debug("tailscale whois failed for %s: %s", ip, exc)

    _WHOIS_CACHE[ip] = (time.monotonic(), name)
    return name


def record(
    db: Database,
    *,
    kind: str,
    code: str = "",
    bin_id: int | None = None,
    location_id: int | None = None,
    item_id: int | None = None,
    device_key: str = "",
    device_ip: str = "",
    device_name: str = "",
    user_agent: str = "",
    detail: str = "",
) -> None:
    with db.write() as conn:
        conn.execute(
            "INSERT INTO scan_events(kind, code, bin_id, location_id, item_id, device_key, "
            "device_ip, device_name, user_agent, detail, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (kind, code, bin_id, location_id, item_id, device_key, device_ip, device_name,
             user_agent[:200], detail, now_iso()),
        )


def recent(db: Database, limit: int = 50) -> list[dict[str, Any]]:
    return rows_to_dicts(
        db.query(
            "SELECT scan_events.*, bins.code AS bin_code, locations.name AS location_name "
            "FROM scan_events "
            "LEFT JOIN bins ON bins.id = scan_events.bin_id "
            "LEFT JOIN locations ON locations.id = scan_events.location_id "
            "ORDER BY scan_events.id DESC LIMIT ?",
            (limit,),
        )
    )
