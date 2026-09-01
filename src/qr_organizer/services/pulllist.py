"""The pull list: a shopping-cart-style working checklist.

Adding an item to the list changes nothing about the item. Checking it off is
the moment it becomes "in use" -- that is the physical act of picking it up off
the shelf, and it stays in use until it is scanned back into some bin.
"""

from __future__ import annotations

import logging
from typing import Any

from ..db import Database
from ..util import now_iso
from . import items as items_service
from . import rows_to_dicts

log = logging.getLogger(__name__)


def entries(db: Database, session_key: str) -> list[dict[str, Any]]:
    return rows_to_dicts(
        db.query(
            "SELECT pull_list.id AS entry_id, pull_list.added_at, pull_list.checked_off_at, "
            "items.id AS item_id, items.label, items.description, items.status, "
            "items.thumbnail_path, bins.code AS bin_code, bins.label AS bin_label, "
            "locations.name AS location_name, loans.person_name AS loan_person "
            "FROM pull_list "
            "JOIN items ON items.id = pull_list.item_id "
            "LEFT JOIN bins ON bins.id = COALESCE(items.bin_id, items.home_bin_id) "
            "LEFT JOIN locations ON locations.id = bins.location_id "
            "LEFT JOIN loans ON loans.id = items.loan_id "
            "WHERE pull_list.session_key = ? "
            "ORDER BY pull_list.checked_off_at IS NOT NULL, "
            "locations.name COLLATE NOCASE, bins.code, items.label COLLATE NOCASE",
            (session_key,),
        )
    )


def counts(db: Database, session_key: str) -> tuple[int, int]:
    row = db.query_one(
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN checked_off_at IS NULL THEN 0 ELSE 1 END) AS done "
        "FROM pull_list WHERE session_key = ?",
        (session_key,),
    )
    if not row:
        return 0, 0
    return int(row["total"] or 0), int(row["done"] or 0)


def add(db: Database, session_key: str, item_id: int) -> bool:
    with db.write() as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO pull_list(session_key, item_id, added_at) VALUES(?, ?, ?)",
            (session_key, item_id, now_iso()),
        )
    return cursor.rowcount > 0


def remove(db: Database, session_key: str, item_id: int) -> None:
    with db.write() as conn:
        conn.execute(
            "DELETE FROM pull_list WHERE session_key = ? AND item_id = ?", (session_key, item_id)
        )


def clear(db: Database, session_key: str, *, only_checked: bool = False) -> None:
    clause = " AND checked_off_at IS NOT NULL" if only_checked else ""
    with db.write() as conn:
        conn.execute(f"DELETE FROM pull_list WHERE session_key = ?{clause}", (session_key,))


def check_off(db: Database, session_key: str, item_id: int) -> None:
    """Mark an item physically retrieved -- and therefore in use."""
    with db.write() as conn:
        conn.execute(
            "UPDATE pull_list SET checked_off_at = ? WHERE session_key = ? AND item_id = ?",
            (now_iso(), session_key, item_id),
        )
    items_service.mark_in_use(db, item_id, detail="checked off a pull list")


def uncheck(db: Database, session_key: str, item_id: int) -> None:
    """Undo a check-off. The item goes back to in-bin only if nothing else claims it."""
    with db.write() as conn:
        conn.execute(
            "UPDATE pull_list SET checked_off_at = NULL WHERE session_key = ? AND item_id = ?",
            (session_key, item_id),
        )
    item = items_service.get_item(db, item_id)
    if item and item["status"] == items_service.STATUS_IN_USE and item["bin_id"]:
        items_service.check_into_bin(
            db, item_id, item["bin_id"], detail="pull-list check-off undone"
        )
