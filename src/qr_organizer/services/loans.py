"""Loans -- a virtual bin tied to a person instead of a location.

A loan behaves like a bin everywhere it can: it has a code, it holds items, and
it has a photo of what went out the door. What it doesn't have is a location,
because the whole point is that the stuff isn't here.
"""

from __future__ import annotations

import logging
from typing import Any

from ..db import Database
from ..errors import ConflictError, NotFoundError
from ..util import format_code, now_iso
from . import items as items_service
from . import row_to_dict, rows_to_dicts

log = logging.getLogger(__name__)


def list_loans(db: Database, *, include_closed: bool = False) -> list[dict[str, Any]]:
    clause = "" if include_closed else " WHERE loans.status = 'open'"
    return rows_to_dicts(
        db.query(
            "SELECT loans.*, "
            "(SELECT COUNT(*) FROM items WHERE items.loan_id = loans.id) AS item_count "
            "FROM loans" + clause + " ORDER BY loans.created_at DESC"
        )
    )


def get_loan(db: Database, code: str) -> dict[str, Any] | None:
    return row_to_dict(db.query_one("SELECT * FROM loans WHERE code = ?", (code.upper(),)))


def get_loan_by_id(db: Database, loan_id: int) -> dict[str, Any] | None:
    return row_to_dict(db.query_one("SELECT * FROM loans WHERE id = ?", (loan_id,)))


def loan_items(db: Database, loan_id: int) -> list[dict[str, Any]]:
    return rows_to_dicts(
        db.query(
            "SELECT items.*, bins.code AS bin_code FROM items "
            "LEFT JOIN bins ON bins.id = items.home_bin_id "
            "WHERE items.loan_id = ? ORDER BY items.label COLLATE NOCASE",
            (loan_id,),
        )
    )


def next_loan_code(db: Database, prefix: str, digits: int) -> str:
    row = db.query_one(
        "SELECT code FROM loans WHERE code LIKE ? ORDER BY code DESC LIMIT 1", (f"{prefix}-%",)
    )
    highest = 0
    if row:
        try:
            highest = int(row["code"].split("-", 1)[1])
        except (IndexError, ValueError):
            highest = 0
    return format_code(prefix, highest + 1, digits)


def create_loan(
    db: Database, *, person_name: str, code: str, notes: str = ""
) -> dict[str, Any]:
    person_name = person_name.strip()
    if not person_name:
        raise ConflictError("a loan needs a person's name")
    timestamp = now_iso()
    with db.write() as conn:
        if conn.execute("SELECT id FROM loans WHERE code = ?", (code.upper(),)).fetchone():
            raise ConflictError(f"loan code {code} already exists")
        conn.execute(
            "INSERT INTO loans(code, person_name, notes, status, created_at) "
            "VALUES(?, ?, ?, 'open', ?)",
            (code.upper(), person_name, notes.strip(), timestamp),
        )
    log.info("opened loan %s for %s", code, person_name)
    return get_loan(db, code)  # type: ignore[return-value]


def assign_items(db: Database, loan_id: int, item_ids: list[int]) -> int:
    loan = get_loan_by_id(db, loan_id)
    if loan is None:
        raise NotFoundError(f"no loan with id {loan_id}")
    for item_id in item_ids:
        items_service.mark_loaned(db, item_id, loan_id, loan["person_name"])
    log.info(
        "assigned %d item(s) to loan %s (%s)",
        len(item_ids), loan["code"], loan["person_name"],
    )
    return len(item_ids)


def attach_photo(db: Database, loan_id: int, photo_id: int) -> None:
    with db.write() as conn:
        conn.execute("UPDATE loans SET photo_id = ? WHERE id = ?", (photo_id, loan_id))


def close_loan(db: Database, loan_id: int) -> None:
    """Close a loan. Refuses while items are still out -- check them into a bin first."""
    outstanding = db.query_one(
        "SELECT COUNT(*) AS n FROM items WHERE loan_id = ? AND status = 'loaned'", (loan_id,)
    )
    if outstanding and outstanding["n"]:
        raise ConflictError(
            f"{outstanding['n']} item(s) are still out on this loan; "
            "scan them back into a bin first"
        )
    with db.write() as conn:
        conn.execute(
            "UPDATE loans SET status = 'closed', closed_at = ? WHERE id = ?",
            (now_iso(), loan_id),
        )
    log.info("closed loan id=%s", loan_id)
