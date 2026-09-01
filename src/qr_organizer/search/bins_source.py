"""The bin inventory as a `SearchSource`.

Vector search runs through sqlite-vec when the extension loaded, and falls back
to an exact NumPy scan over the same BLOBs when it didn't. Both read
`item_embeddings`, so the two paths cannot drift apart.
"""

from __future__ import annotations

import logging
import re
import sqlite3

import numpy as np

from ..db import Database
from . import SearchHit

log = logging.getLogger(__name__)

SOURCE_ID = "bins"

_BASE_SELECT = """
SELECT
    items.id                AS item_id,
    items.label             AS label,
    items.description       AS description,
    items.status            AS status,
    items.thumbnail_path    AS thumbnail_path,
    items.needs_review      AS needs_review,
    items.label_source      AS label_source,
    bins.code               AS bin_code,
    bins.label              AS bin_label,
    locations.name          AS location_name,
    loans.person_name       AS loan_person,
    loans.code              AS loan_code
FROM items
LEFT JOIN bins      ON bins.id = items.bin_id
LEFT JOIN locations ON locations.id = bins.location_id
LEFT JOIN loans     ON loans.id = items.loan_id
"""

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def fts_query(raw: str) -> str:
    """Turn a human query into a safe FTS5 MATCH expression.

    Every token is quoted (so punctuation can't become FTS syntax) and the last
    one gets a prefix wildcard, which is what makes as-you-type search feel
    right.
    """
    tokens = _TOKEN_RE.findall(raw)
    if not tokens:
        return ""
    quoted = [f'"{token}"' for token in tokens[:-1]]
    quoted.append(f'"{tokens[-1]}"*')
    return " AND ".join(quoted)


class BinInventorySource:
    """Keyword + visual search over everything stored in bins and loans."""

    id = SOURCE_ID
    name = "Bins"

    def __init__(self, database: Database) -> None:
        self.db = database

    # -- hydration --------------------------------------------------------
    def _hit(self, row: sqlite3.Row, score: float, match_kind: str) -> SearchHit:
        status = row["status"]
        if status == "loaned" and row["loan_person"]:
            status_detail = f"with {row['loan_person']}"
        elif status == "in_use":
            status_detail = "pulled for use"
        elif status == "missing":
            status_detail = "not seen in the last re-inventory"
        else:
            status_detail = ""

        bin_code = row["bin_code"] or ""
        container_label = bin_code
        if row["bin_label"]:
            container_label = f"{bin_code} - {row['bin_label']}"
        container_url = f"/bins/{bin_code}" if bin_code else ""
        if status == "loaned" and row["loan_code"]:
            container_label = f"Loan to {row['loan_person']}"
            container_url = f"/loans/{row['loan_code']}"

        return SearchHit(
            source_id=self.id,
            source_name=self.name,
            external_id=str(row["item_id"]),
            label=row["label"],
            description=row["description"] or "",
            score=score,
            thumbnail_url=f"/media/thumbnails/{row['thumbnail_path']}"
            if row["thumbnail_path"]
            else "",
            detail_url=f"/items/{row['item_id']}",
            status=status,
            status_detail=status_detail,
            container_label=container_label,
            container_url=container_url,
            location_name=row["location_name"] or "",
            needs_review=bool(row["needs_review"]),
            match_kind=match_kind,
            extra={"label_source": row["label_source"]},
        )

    # -- SearchSource -----------------------------------------------------
    def keyword(self, query: str, *, limit: int, include_in_use: bool) -> list[SearchHit]:
        match = fts_query(query)
        if not match:
            return self.recent(limit=limit)

        status_clause = "" if include_in_use else " AND items.status = 'in_bin'"
        sql = (
            _BASE_SELECT
            + " JOIN items_fts ON items_fts.rowid = items.id"
            + " WHERE items_fts MATCH ?"
            + status_clause
            + " ORDER BY bm25(items_fts, 4.0, 1.0, 0.5) LIMIT ?"
        )
        try:
            rows = self.db.query(sql, (match, limit))
        except sqlite3.OperationalError as exc:
            # A malformed MATCH is the only realistic cause and it means the
            # sanitiser above has a hole -- say so rather than silently
            # returning nothing.
            log.error("FTS query failed for %r (match=%r): %s", query, match, exc)
            return self._like_fallback(query, limit=limit, include_in_use=include_in_use)

        hits = []
        for position, row in enumerate(rows):
            # bm25 ordering is already correct; map rank to a descending score
            # in the same 0..1 band as vector similarity so the registry can
            # merge sources sensibly.
            hits.append(self._hit(row, score=1.0 - (position / max(len(rows), 1)) * 0.5,
                                  match_kind="keyword"))
        return hits

    def _like_fallback(self, query: str, *, limit: int, include_in_use: bool) -> list[SearchHit]:
        status_clause = "" if include_in_use else " AND items.status = 'in_bin'"
        sql = (
            _BASE_SELECT
            + " WHERE (items.label LIKE ? OR items.description LIKE ?)"
            + status_clause
            + " ORDER BY items.last_seen_at DESC LIMIT ?"
        )
        pattern = f"%{query.strip()}%"
        rows = self.db.query(sql, (pattern, pattern, limit))
        return [self._hit(row, score=0.4, match_kind="keyword") for row in rows]

    def vector(self, vector: np.ndarray, *, limit: int, min_score: float) -> list[SearchHit]:
        pairs = self.nearest(vector, limit=limit, min_score=min_score)
        if not pairs:
            return []
        placeholders = ",".join("?" for _ in pairs)
        rows = self.db.query(
            _BASE_SELECT + f" WHERE items.id IN ({placeholders})",
            tuple(item_id for item_id, _ in pairs),
        )
        scores = dict(pairs)
        hits = [self._hit(row, score=scores[row["item_id"]], match_kind="visual") for row in rows]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)

    def recent(self, *, limit: int) -> list[SearchHit]:
        rows = self.db.query(
            _BASE_SELECT + " ORDER BY items.last_seen_at DESC LIMIT ?", (limit,)
        )
        return [self._hit(row, score=0.0, match_kind="recent") for row in rows]

    def health(self) -> tuple[bool, str]:
        ok, detail = self.db.health_check()
        if not ok:
            return False, detail
        row = self.db.query_one("SELECT COUNT(*) AS n FROM items")
        embedded = self.db.query_one("SELECT COUNT(*) AS n FROM item_embeddings")
        return True, (
            f"{row['n'] if row else 0} item(s), {embedded['n'] if embedded else 0} embedded, "
            f"index: {'sqlite-vec' if self.db.vec_available else 'numpy scan'}"
        )

    # -- nearest-neighbour ------------------------------------------------
    def nearest(
        self, vector: np.ndarray, *, limit: int, min_score: float = 0.0,
        exclude_item_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """(item_id, cosine similarity) pairs, best first.

        Vectors are stored L2-normalised, so cosine similarity is a dot product
        and sqlite-vec's L2 distance `d` maps to `1 - d^2 / 2`.
        """
        query = np.asarray(vector, dtype=np.float32).ravel()
        if query.size == 0:
            return []
        excluded = exclude_item_ids or set()
        fetch = limit + len(excluded)

        if self.db.vec_available and self.db.vec_dim == query.size:
            try:
                rows = self.db.query(
                    "SELECT item_id, distance FROM item_vectors "
                    "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                    (query.tobytes(), fetch),
                )
                scored = [
                    (int(row["item_id"]), 1.0 - (float(row["distance"]) ** 2) / 2.0)
                    for row in rows
                ]
            except sqlite3.Error as exc:
                log.warning("sqlite-vec lookup failed (%s); falling back to a NumPy scan", exc)
                scored = self._numpy_nearest(query, fetch)
        else:
            scored = self._numpy_nearest(query, fetch)

        return [
            (item_id, score)
            for item_id, score in scored
            if item_id not in excluded and score >= min_score
        ][:limit]

    def _numpy_nearest(self, query: np.ndarray, limit: int) -> list[tuple[int, float]]:
        rows = self.db.query("SELECT item_id, dim, vector FROM item_embeddings")
        if not rows:
            return []
        ids: list[int] = []
        vectors: list[np.ndarray] = []
        for row in rows:
            if row["dim"] != query.size:
                continue
            ids.append(int(row["item_id"]))
            vectors.append(np.frombuffer(row["vector"], dtype=np.float32))
        if not vectors:
            return []
        matrix = np.vstack(vectors)
        scores = matrix @ query
        order = np.argsort(-scores)[:limit]
        return [(ids[index], float(scores[index])) for index in order]
