"""Bins and the pre-printed label stock they are claimed from.

Codes are not minted on demand. Sheets of sequential codes are generated and
printed ahead of time into `label_stock`; scanning an unassigned code is what
turns it into a real bin. That keeps the physical labels and the database in
agreement -- a code that exists on a sheet in the workshop always resolves,
even before anything has been put in the tote.
"""

from __future__ import annotations

import logging
from typing import Any

from ..db import Database
from ..errors import ConflictError, NotFoundError
from ..util import format_code, now_iso
from . import row_to_dict, rows_to_dicts

log = logging.getLogger(__name__)


def get_bin(db: Database, code: str) -> dict[str, Any] | None:
    return row_to_dict(
        db.query_one(
            "SELECT bins.*, locations.name AS location_name, locations.code AS location_code "
            "FROM bins LEFT JOIN locations ON locations.id = bins.location_id "
            "WHERE bins.code = ?",
            (code.upper(),),
        )
    )


def get_bin_by_id(db: Database, bin_id: int) -> dict[str, Any] | None:
    return row_to_dict(
        db.query_one(
            "SELECT bins.*, locations.name AS location_name, locations.code AS location_code "
            "FROM bins LEFT JOIN locations ON locations.id = bins.location_id "
            "WHERE bins.id = ?",
            (bin_id,),
        )
    )


def list_bins(db: Database) -> list[dict[str, Any]]:
    return rows_to_dicts(
        db.query(
            "SELECT bins.*, locations.name AS location_name, "
            "(SELECT COUNT(*) FROM items WHERE items.bin_id = bins.id "
            " AND items.status != 'missing') AS item_count, "
            "(SELECT COUNT(*) FROM items WHERE items.bin_id = bins.id "
            " AND items.status = 'missing') AS missing_count "
            "FROM bins LEFT JOIN locations ON locations.id = bins.location_id "
            "ORDER BY bins.code"
        )
    )


def create_bin(
    db: Database, *, code: str, label: str = "", location_id: int | None = None
) -> dict[str, Any]:
    code = code.strip().upper()
    timestamp = now_iso()
    with db.write() as conn:
        if conn.execute("SELECT id FROM bins WHERE code = ?", (code,)).fetchone():
            raise ConflictError(f"bin {code} already exists")
        conn.execute(
            "INSERT INTO bins(code, label, location_id, location_set_at, created_at, updated_at) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (code, label.strip(), location_id, timestamp if location_id else None,
             timestamp, timestamp),
        )
        conn.execute(
            "UPDATE label_stock SET assigned_at = ? WHERE code = ? AND assigned_at IS NULL",
            (timestamp, code),
        )
    log.info("created bin %s (location_id=%s)", code, location_id)
    return get_bin(db, code)  # type: ignore[return-value]


def update_bin(db: Database, code: str, *, label: str, notes: str) -> None:
    with db.write() as conn:
        cursor = conn.execute(
            "UPDATE bins SET label = ?, notes = ?, updated_at = ? WHERE code = ?",
            (label.strip(), notes.strip(), now_iso(), code.upper()),
        )
        if cursor.rowcount == 0:
            raise NotFoundError(f"no bin {code}")


def move_bin(db: Database, code: str, location_id: int | None) -> None:
    """Deliberately change a bin's location.

    This is the ONLY way a bin's location changes. Re-inventory scans never
    call it -- moving a bin has to be something you meant to do.
    """
    timestamp = now_iso()
    with db.write() as conn:
        cursor = conn.execute(
            "UPDATE bins SET location_id = ?, location_set_at = ?, updated_at = ? WHERE code = ?",
            (location_id, timestamp if location_id else None, timestamp, code.upper()),
        )
        if cursor.rowcount == 0:
            raise NotFoundError(f"no bin {code}")
    log.info("bin %s moved to location_id=%s", code, location_id)


# -- label stock ----------------------------------------------------------


def reserve_codes(
    db: Database, *, kind: str, prefix: str, count: int, digits: int, sheet_id: str
) -> list[str]:
    """Mint `count` sequential codes into the label stock for printing."""
    if count <= 0:
        raise ConflictError("count must be positive")
    timestamp = now_iso()
    codes: list[str] = []
    with db.write() as conn:
        row = conn.execute(
            "SELECT code FROM label_stock WHERE kind = ? ORDER BY id DESC LIMIT 1", (kind,)
        ).fetchone()
        highest = 0
        if row:
            try:
                highest = int(row["code"].split("-", 1)[1])
            except (IndexError, ValueError):
                highest = 0
        # Codes already used by hand-made bins must not be re-issued.
        table = "bins" if kind == "bin" else "locations"
        used = conn.execute(
            f"SELECT code FROM {table} WHERE code LIKE ? ORDER BY code DESC LIMIT 1",
            (f"{prefix}-%",),
        ).fetchone()
        if used:
            try:
                highest = max(highest, int(used["code"].split("-", 1)[1]))
            except (IndexError, ValueError):
                pass
        for offset in range(1, count + 1):
            code = format_code(prefix, highest + offset, digits)
            conn.execute(
                "INSERT INTO label_stock(code, kind, sheet_id, printed_at, created_at) "
                "VALUES(?, ?, ?, ?, ?)",
                (code, kind, sheet_id, timestamp, timestamp),
            )
            codes.append(code)
    log.info("reserved %d %s label(s) on sheet %s: %s..%s", count, kind, sheet_id,
             codes[0], codes[-1])
    return codes


def list_sheets(db: Database) -> list[dict[str, Any]]:
    return rows_to_dicts(
        db.query(
            "SELECT sheet_id, kind, COUNT(*) AS total, "
            "SUM(CASE WHEN assigned_at IS NULL THEN 0 ELSE 1 END) AS used, "
            "MIN(code) AS first_code, MAX(code) AS last_code, MIN(printed_at) AS printed_at "
            "FROM label_stock GROUP BY sheet_id, kind ORDER BY printed_at DESC"
        )
    )


def sheet_codes(db: Database, sheet_id: str) -> list[dict[str, Any]]:
    return rows_to_dicts(
        db.query("SELECT * FROM label_stock WHERE sheet_id = ? ORDER BY code", (sheet_id,))
    )


def is_known_stock(db: Database, code: str) -> bool:
    return db.query_one("SELECT 1 FROM label_stock WHERE code = ?", (code.upper(),)) is not None
