"""Locations and the active-location context.

Two rules from the spec drive this module, and both are about *not* being
clever:

* A bin scan inherits the active location only while that context is live. Once
  it has timed out the app asks for a fresh location scan rather than reusing a
  stale one.
* A re-inventory scan never moves a bin. Location changes happen only through
  `move_bin`, which is reached from a deliberate "change location" action.
"""

from __future__ import annotations

import logging
from typing import Any

from ..db import Database
from ..errors import ConflictError, NotFoundError
from ..util import format_code, is_expired, now_iso
from . import row_to_dict, rows_to_dicts

log = logging.getLogger(__name__)


def list_locations(db: Database) -> list[dict[str, Any]]:
    return rows_to_dicts(
        db.query(
            "SELECT locations.*, "
            "(SELECT COUNT(*) FROM bins WHERE bins.location_id = locations.id) AS bin_count "
            "FROM locations ORDER BY locations.name COLLATE NOCASE"
        )
    )


def get_location(db: Database, code: str) -> dict[str, Any] | None:
    return row_to_dict(db.query_one("SELECT * FROM locations WHERE code = ?", (code.upper(),)))


def get_location_by_id(db: Database, location_id: int) -> dict[str, Any] | None:
    return row_to_dict(db.query_one("SELECT * FROM locations WHERE id = ?", (location_id,)))


def next_location_code(db: Database, prefix: str, digits: int) -> str:
    row = db.query_one(
        "SELECT code FROM locations WHERE code LIKE ? ORDER BY code DESC LIMIT 1",
        (f"{prefix}-%",),
    )
    highest = 0
    if row:
        try:
            highest = int(row["code"].split("-", 1)[1])
        except (IndexError, ValueError):
            highest = 0
    return format_code(prefix, highest + 1, digits)


def create_location(db: Database, *, name: str, code: str, notes: str = "") -> dict[str, Any]:
    name = name.strip()
    if not name:
        raise ConflictError("a location needs a name")
    code = code.strip().upper()
    timestamp = now_iso()
    with db.write() as conn:
        existing = conn.execute("SELECT id FROM locations WHERE code = ?", (code,)).fetchone()
        if existing:
            raise ConflictError(f"location code {code} is already in use")
        conn.execute(
            "INSERT INTO locations(code, name, notes, created_at, updated_at) "
            "VALUES(?, ?, ?, ?, ?)",
            (code, name, notes.strip(), timestamp, timestamp),
        )
    log.info("created location %s (%s)", code, name)
    return get_location(db, code)  # type: ignore[return-value]


def rename_location(db: Database, code: str, *, name: str, notes: str) -> None:
    with db.write() as conn:
        cursor = conn.execute(
            "UPDATE locations SET name = ?, notes = ?, updated_at = ? WHERE code = ?",
            (name.strip(), notes.strip(), now_iso(), code.upper()),
        )
        if cursor.rowcount == 0:
            raise NotFoundError(f"no location {code}")


# -- active location context ---------------------------------------------


def set_active_location(db: Database, device_key: str, location_id: int) -> None:
    timestamp = now_iso()
    with db.write() as conn:
        conn.execute(
            "INSERT INTO location_context(device_key, location_id, set_at, last_used_at) "
            "VALUES(?, ?, ?, ?) "
            "ON CONFLICT(device_key) DO UPDATE SET "
            "location_id = excluded.location_id, set_at = excluded.set_at, "
            "last_used_at = excluded.last_used_at",
            (device_key, location_id, timestamp, timestamp),
        )
    log.info("device %s: active location set to id=%s", device_key, location_id)


def clear_active_location(db: Database, device_key: str) -> None:
    with db.write() as conn:
        conn.execute("DELETE FROM location_context WHERE device_key = ?", (device_key,))


def active_location(db: Database, device_key: str, timeout_minutes: int) -> dict[str, Any] | None:
    """The live location context for this device, or None if absent or expired.

    An expired context is deleted on read: the next bin scan must be preceded
    by a fresh location scan, which is the whole point of the timeout.
    """
    row = db.query_one(
        "SELECT location_context.*, locations.code AS location_code, "
        "locations.name AS location_name "
        "FROM location_context JOIN locations ON locations.id = location_context.location_id "
        "WHERE device_key = ?",
        (device_key,),
    )
    if row is None:
        return None
    if is_expired(row["last_used_at"], timeout_minutes):
        log.info(
            "device %s: location context on %s expired after %d min of inactivity",
            device_key, row["location_code"], timeout_minutes,
        )
        clear_active_location(db, device_key)
        return None
    return dict(row)


def touch_active_location(db: Database, device_key: str) -> None:
    """Refresh the inactivity clock after a scan that used the context."""
    with db.write() as conn:
        conn.execute(
            "UPDATE location_context SET last_used_at = ? WHERE device_key = ?",
            (now_iso(), device_key),
        )
