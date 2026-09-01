"""Photo intake: normalise, store on disk, record in the database."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..db import Database
from ..imaging import store_photo
from ..util import now_iso, sha256_file
from . import row_to_dict

log = logging.getLogger(__name__)


def ingest(
    db: Database,
    *,
    source: Path,
    photos_root: Path,
    kind: str,
    bin_id: int | None = None,
    loan_id: int | None = None,
    max_dimension: int = 2048,
    jpeg_quality: int = 88,
) -> dict[str, Any]:
    """Store an uploaded photo and return its database row."""
    timestamp = now_iso()
    stamp = timestamp.replace(":", "").replace("-", "")
    subdir = f"{kind}/{stamp[:6]}"
    filename = f"{stamp}-{source.stem[:24] or 'photo'}.jpg"
    relative = f"{subdir}/{filename}"
    stored = store_photo(
        source,
        photos_root / relative,
        max_dimension=max_dimension,
        jpeg_quality=jpeg_quality,
    )
    with db.write() as conn:
        cursor = conn.execute(
            "INSERT INTO photos(path, kind, bin_id, loan_id, width, height, sha256, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (relative, kind, bin_id, loan_id, stored.width, stored.height,
             sha256_file(stored.path), timestamp),
        )
        photo_id = int(cursor.lastrowid)
    log.info("stored %s photo %s (%dx%d) as id=%d", kind, relative, stored.width, stored.height,
             photo_id)
    return get_photo(db, photo_id)  # type: ignore[return-value]


def get_photo(db: Database, photo_id: int) -> dict[str, Any] | None:
    return row_to_dict(db.query_one("SELECT * FROM photos WHERE id = ?", (photo_id,)))


def bin_photos(db: Database, bin_id: int, limit: int = 20) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in db.query(
            "SELECT * FROM photos WHERE bin_id = ? ORDER BY id DESC LIMIT ?", (bin_id, limit)
        )
    ]
