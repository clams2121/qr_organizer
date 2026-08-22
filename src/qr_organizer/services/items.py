"""Items: creation, labelling, status, and the live embedding library.

The RAG library is not a separate store. An item's label lives on the item and
its vector lives in `item_embeddings` keyed by the same id, so correcting a
label instantly changes what future lookups will suggest -- no retraining step,
no batch job, exactly as specified.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Iterable

import numpy as np

from ..db import Database
from ..errors import NotFoundError, StorageError
from ..util import now_iso
from . import row_to_dict, rows_to_dicts

log = logging.getLogger(__name__)

STATUS_IN_BIN = "in_bin"
STATUS_IN_USE = "in_use"
STATUS_LOANED = "loaned"
STATUS_MISSING = "missing"
ACTIVE_STATUSES = (STATUS_IN_BIN, STATUS_IN_USE, STATUS_LOANED)


def get_item(db: Database, item_id: int) -> dict[str, Any] | None:
    return row_to_dict(
        db.query_one(
            "SELECT items.*, bins.code AS bin_code, bins.label AS bin_label, "
            "locations.name AS location_name, loans.person_name AS loan_person, "
            "loans.code AS loan_code "
            "FROM items "
            "LEFT JOIN bins ON bins.id = items.bin_id "
            "LEFT JOIN locations ON locations.id = bins.location_id "
            "LEFT JOIN loans ON loans.id = items.loan_id "
            "WHERE items.id = ?",
            (item_id,),
        )
    )


def list_bin_items(
    db: Database, bin_id: int, *, include_missing: bool = True
) -> list[dict[str, Any]]:
    clause = "" if include_missing else " AND items.status != 'missing'"
    return rows_to_dicts(
        db.query(
            "SELECT items.*, loans.person_name AS loan_person, loans.code AS loan_code "
            "FROM items LEFT JOIN loans ON loans.id = items.loan_id "
            "WHERE (items.bin_id = ? OR (items.home_bin_id = ? AND items.bin_id IS NULL))"
            + clause
            + " ORDER BY items.needs_review DESC, items.label COLLATE NOCASE",
            (bin_id, bin_id),
        )
    )


def list_needing_review(db: Database, limit: int = 200) -> list[dict[str, Any]]:
    return rows_to_dicts(
        db.query(
            "SELECT items.*, bins.code AS bin_code FROM items "
            "LEFT JOIN bins ON bins.id = items.bin_id "
            "WHERE items.needs_review = 1 ORDER BY items.created_at DESC LIMIT ?",
            (limit,),
        )
    )


def create_item(
    db: Database,
    *,
    label: str,
    bin_id: int | None,
    thumbnail_path: str,
    source_photo_id: int | None,
    bbox: tuple[float, float, float, float] | None,
    description: str = "",
    label_source: str = "ai",
    label_confidence: float = 0.0,
    needs_review: bool = False,
    status: str = STATUS_IN_BIN,
) -> int:
    timestamp = now_iso()
    with db.write() as conn:
        cursor = conn.execute(
            "INSERT INTO items("
            "label, description, status, bin_id, home_bin_id, thumbnail_path, source_photo_id, "
            "bbox_json, label_source, label_confidence, needs_review, "
            "first_seen_at, last_seen_at, status_changed_at, created_at, updated_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                label.strip(),
                description.strip(),
                status,
                bin_id,
                bin_id,
                thumbnail_path,
                source_photo_id,
                json.dumps(list(bbox)) if bbox else "",
                label_source,
                float(label_confidence),
                1 if needs_review else 0,
                timestamp, timestamp, timestamp, timestamp, timestamp,
            ),
        )
        item_id = int(cursor.lastrowid)
        conn.execute(
            "INSERT INTO item_events(item_id, event, detail, created_at) VALUES(?, ?, ?, ?)",
            (item_id, "created", f"{label_source} label {label!r}", timestamp),
        )
    return item_id


def set_label(
    db: Database,
    item_id: int,
    *,
    label: str,
    description: str | None = None,
    source: str = "user",
    confidence: float = 1.0,
) -> None:
    """Correct or confirm a label. This is what updates the RAG library."""
    label = label.strip()
    if not label:
        raise StorageError("an item label cannot be empty")
    timestamp = now_iso()
    with db.write() as conn:
        previous = conn.execute("SELECT label FROM items WHERE id = ?", (item_id,)).fetchone()
        if previous is None:
            raise NotFoundError(f"no item {item_id}")
        if description is None:
            conn.execute(
                "UPDATE items SET label = ?, label_source = ?, label_confidence = ?, "
                "needs_review = 0, updated_at = ? WHERE id = ?",
                (label, source, confidence, timestamp, item_id),
            )
        else:
            conn.execute(
                "UPDATE items SET label = ?, description = ?, label_source = ?, "
                "label_confidence = ?, needs_review = 0, updated_at = ? WHERE id = ?",
                (label, description.strip(), source, confidence, timestamp, item_id),
            )
        conn.execute(
            "INSERT INTO item_events(item_id, event, detail, created_at) VALUES(?, ?, ?, ?)",
            (item_id, "relabelled", f"{previous['label']!r} -> {label!r} ({source})", timestamp),
        )
    log.info("item %d relabelled %r -> %r by %s", item_id, previous["label"], label, source)


def set_notes(db: Database, item_id: int, notes: str) -> None:
    with db.write() as conn:
        conn.execute(
            "UPDATE items SET notes = ?, updated_at = ? WHERE id = ?",
            (notes.strip(), now_iso(), item_id),
        )


def delete_item(db: Database, item_id: int) -> None:
    with db.write() as conn:
        if _has_vec(conn):
            conn.execute("DELETE FROM item_vectors WHERE item_id = ?", (item_id,))
        cursor = conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        if cursor.rowcount == 0:
            raise NotFoundError(f"no item {item_id}")
    log.info("item %d deleted", item_id)


def _has_vec(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'item_vectors'"
    ).fetchone()
    return row is not None


# -- status transitions ---------------------------------------------------


def _set_status(
    db: Database,
    item_id: int,
    status: str,
    *,
    bin_id: int | None = None,
    loan_id: int | None = None,
    detail: str = "",
    keep_bin: bool = False,
) -> None:
    timestamp = now_iso()
    with db.write() as conn:
        row = conn.execute("SELECT status FROM items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"no item {item_id}")
        if keep_bin:
            conn.execute(
                "UPDATE items SET status = ?, loan_id = ?, status_changed_at = ?, updated_at = ? "
                "WHERE id = ?",
                (status, loan_id, timestamp, timestamp, item_id),
            )
        else:
            conn.execute(
                "UPDATE items SET status = ?, bin_id = ?, loan_id = ?, status_changed_at = ?, "
                "updated_at = ? WHERE id = ?",
                (status, bin_id, loan_id, timestamp, timestamp, item_id),
            )
        conn.execute(
            "INSERT INTO item_events(item_id, event, detail, created_at) VALUES(?, ?, ?, ?)",
            (item_id, status, detail or f"{row['status']} -> {status}", timestamp),
        )


def mark_in_use(db: Database, item_id: int, detail: str = "pulled from search") -> None:
    """Pulled for use. The bin link is kept so the item still lists as 'from BIN-x'."""
    _set_status(db, item_id, STATUS_IN_USE, keep_bin=True, detail=detail)


def mark_loaned(db: Database, item_id: int, loan_id: int, person: str) -> None:
    _set_status(db, item_id, STATUS_LOANED, keep_bin=True, loan_id=loan_id,
                detail=f"loaned to {person}")


def mark_missing(db: Database, item_id: int, detail: str) -> None:
    """Not detected in a re-inventory. Flagged, never deleted."""
    _set_status(db, item_id, STATUS_MISSING, keep_bin=True, detail=detail)


def check_into_bin(db: Database, item_id: int, bin_id: int, detail: str = "") -> None:
    """Return an item to stock, in any bin -- not necessarily its original one."""
    _set_status(db, item_id, STATUS_IN_BIN, bin_id=bin_id, loan_id=None,
                detail=detail or "checked back in")
    with db.write() as conn:
        conn.execute(
            "UPDATE items SET home_bin_id = ?, last_seen_at = ? WHERE id = ?",
            (bin_id, now_iso(), item_id),
        )
    # The item is demonstrably back, so any queued "did this come back?" question
    # is moot -- don't leave it sitting in the review queue.
    resolve_pending_returns(db, item_id, resolution="superseded")


def touch_seen(db: Database, item_ids: Iterable[int]) -> None:
    """Record that these items were seen, without changing their status.

    Being visible in a photo is evidence, not a decision. An item the user put
    somewhere -- out on loan, in use, or flagged missing -- keeps that status
    until the user confirms the return; see `record_pending_return`.
    """
    ids = list(item_ids)
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    with db.write() as conn:
        conn.execute(
            f"UPDATE items SET last_seen_at = ? WHERE id IN ({placeholders})",
            (now_iso(), *ids),
        )


# -- pending returns ------------------------------------------------------
#
# When a re-inventory photo turns up something the user had put elsewhere -- out
# on loan, in use, or flagged missing -- the app does NOT decide that it came
# back. A visual match is evidence; the user makes the call. Until they do, the
# item keeps the status they gave it.


def record_pending_return(
    db: Database,
    *,
    item_id: int,
    bin_id: int,
    prior_status: str,
    session_id: int | None = None,
    confidence: float = 0.0,
) -> int | None:
    """Queue a return for the user to confirm. Returns None if already queued."""
    with db.write() as conn:
        existing = conn.execute(
            "SELECT id FROM pending_returns WHERE item_id = ? AND resolved_at IS NULL",
            (item_id,),
        ).fetchone()
        if existing is not None:
            # Photographing the same bin twice must not ask the same question twice.
            return None
        cursor = conn.execute(
            "INSERT INTO pending_returns("
            "item_id, bin_id, session_id, prior_status, confidence, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (item_id, bin_id, session_id, prior_status, float(confidence), now_iso()),
        )
        entry_id = int(cursor.lastrowid)
        conn.execute(
            "INSERT INTO item_events(item_id, event, detail, created_at) VALUES(?, ?, ?, ?)",
            (item_id, "return_seen",
             f"seen in a bin photo while {prior_status}; awaiting confirmation", now_iso()),
        )
    log.info("item %d looks like it came back (was %s); waiting on the user",
             item_id, prior_status)
    return entry_id


def pending_returns(db: Database, bin_id: int | None = None) -> list[dict[str, Any]]:
    clause = " AND pending_returns.bin_id = ?" if bin_id is not None else ""
    params: tuple = (bin_id,) if bin_id is not None else ()
    return rows_to_dicts(
        db.query(
            "SELECT pending_returns.id AS entry_id, pending_returns.prior_status, "
            "pending_returns.confidence, pending_returns.created_at AS seen_at, "
            "pending_returns.bin_id, items.id AS item_id, items.label, items.status, "
            "items.thumbnail_path, bins.code AS bin_code, bins.label AS bin_label, "
            "loans.person_name AS loan_person, loans.code AS loan_code "
            "FROM pending_returns "
            "JOIN items ON items.id = pending_returns.item_id "
            "JOIN bins ON bins.id = pending_returns.bin_id "
            "LEFT JOIN loans ON loans.id = items.loan_id "
            "WHERE pending_returns.resolved_at IS NULL" + clause +
            " ORDER BY pending_returns.created_at DESC",
            params,
        )
    )


def pending_return_count(db: Database) -> int:
    row = db.query_one("SELECT COUNT(*) AS n FROM pending_returns WHERE resolved_at IS NULL")
    return int(row["n"]) if row else 0


def get_pending_return(db: Database, entry_id: int) -> dict[str, Any] | None:
    return row_to_dict(
        db.query_one(
            "SELECT * FROM pending_returns WHERE id = ? AND resolved_at IS NULL", (entry_id,)
        )
    )


def resolve_pending_returns(db: Database, item_id: int, *, resolution: str) -> int:
    with db.write() as conn:
        cursor = conn.execute(
            "UPDATE pending_returns SET resolution = ?, resolved_at = ? "
            "WHERE item_id = ? AND resolved_at IS NULL",
            (resolution, now_iso(), item_id),
        )
    return cursor.rowcount


def confirm_return(db: Database, entry_id: int) -> int:
    """The user says yes, it's back. Now -- and only now -- the status changes."""
    entry = get_pending_return(db, entry_id)
    if entry is None:
        raise NotFoundError(f"no open return to confirm with id {entry_id}")
    item_id = int(entry["item_id"])
    check_into_bin(
        db, item_id, int(entry["bin_id"]),
        detail=f"return confirmed by the user (was {entry['prior_status']})",
    )
    # check_into_bin already resolves the entry as superseded; say what it was.
    with db.write() as conn:
        conn.execute(
            "UPDATE pending_returns SET resolution = 'confirmed' WHERE id = ?", (entry_id,)
        )
    return item_id


def dismiss_return(db: Database, entry_id: int) -> int:
    """The user says no. The item keeps the status they gave it."""
    entry = get_pending_return(db, entry_id)
    if entry is None:
        raise NotFoundError(f"no open return to dismiss with id {entry_id}")
    item_id = int(entry["item_id"])
    with db.write() as conn:
        conn.execute(
            "UPDATE pending_returns SET resolution = 'dismissed', resolved_at = ? WHERE id = ?",
            (now_iso(), entry_id),
        )
        conn.execute(
            "INSERT INTO item_events(item_id, event, detail, created_at) VALUES(?, ?, ?, ?)",
            (item_id, "return_dismissed",
             f"user confirmed it is still {entry['prior_status']}", now_iso()),
        )
    log.info("item %d is not back after all; left as %s", item_id, entry["prior_status"])
    return item_id


def item_history(db: Database, item_id: int, limit: int = 50) -> list[dict[str, Any]]:
    return rows_to_dicts(
        db.query(
            "SELECT * FROM item_events WHERE item_id = ? ORDER BY id DESC LIMIT ?",
            (item_id, limit),
        )
    )


# -- embeddings -----------------------------------------------------------


def store_embedding(db: Database, item_id: int, *, model: str, vector: np.ndarray) -> None:
    """Write (or replace) an item's vector in both the BLOB table and the index."""
    flat = np.asarray(vector, dtype=np.float32).ravel()
    if flat.size == 0:
        raise StorageError(f"refusing to store an empty embedding for item {item_id}")
    blob = flat.tobytes()
    timestamp = now_iso()
    with db.write() as conn:
        conn.execute(
            "INSERT INTO item_embeddings(item_id, model, dim, vector, created_at) "
            "VALUES(?, ?, ?, ?, ?) "
            "ON CONFLICT(item_id) DO UPDATE SET model = excluded.model, dim = excluded.dim, "
            "vector = excluded.vector, created_at = excluded.created_at",
            (item_id, model, int(flat.size), blob, timestamp),
        )
        if db.vec_available and _has_vec(conn):
            conn.execute("DELETE FROM item_vectors WHERE item_id = ?", (item_id,))
            conn.execute(
                "INSERT INTO item_vectors(item_id, embedding) VALUES(?, ?)", (item_id, blob)
            )


def labels_for_items(db: Database, item_ids: Iterable[int]) -> dict[int, str]:
    ids = list(item_ids)
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = db.query(f"SELECT id, label FROM items WHERE id IN ({placeholders})", tuple(ids))
    return {int(row["id"]): row["label"] for row in rows}


def embedding_stats(db: Database) -> dict[str, int]:
    total = db.query_one("SELECT COUNT(*) AS n FROM items")
    embedded = db.query_one("SELECT COUNT(*) AS n FROM item_embeddings")
    return {
        "items": int(total["n"]) if total else 0,
        "embedded": int(embedded["n"]) if embedded else 0,
    }
