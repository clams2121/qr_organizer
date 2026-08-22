"""The search layer.

This is deliberately not "search over bins". It is "search over registered
sources of items that have labels, thumbnails and embeddings", with the bin
inventory registered as one such source. A future item database -- a whole-home
inventory, say -- becomes a second class implementing `SearchSource` plus one
`register()` call, with no change to the query path or the UI.

That is as far as the abstraction goes on purpose: no plugin loader, no entry
points, no per-source config machinery invented for a consumer that doesn't
exist yet. What it buys is that nothing above this line knows the word "bin".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """One searchable thing, in the only shape the UI knows about."""

    source_id: str
    source_name: str
    external_id: str
    label: str
    score: float
    description: str = ""
    thumbnail_url: str = ""
    detail_url: str = ""
    status: str = "in_bin"
    status_detail: str = ""
    container_label: str = ""
    container_url: str = ""
    location_name: str = ""
    needs_review: bool = False
    match_kind: str = "keyword"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def in_use(self) -> bool:
        return self.status in {"in_use", "loaned"}

    @property
    def key(self) -> str:
        return f"{self.source_id}:{self.external_id}"


@runtime_checkable
class SearchSource(Protocol):
    """A source of items with labels, thumbnails and embeddings."""

    id: str
    name: str

    def keyword(self, query: str, *, limit: int, include_in_use: bool) -> list[SearchHit]: ...

    def vector(self, vector: np.ndarray, *, limit: int, min_score: float) -> list[SearchHit]: ...

    def recent(self, *, limit: int) -> list[SearchHit]: ...

    def health(self) -> tuple[bool, str]: ...


class SearchRegistry:
    """Every registered source, queried as one."""

    def __init__(self) -> None:
        self._sources: dict[str, SearchSource] = {}

    def register(self, source: SearchSource) -> None:
        if source.id in self._sources:
            raise ValueError(f"a search source with id {source.id!r} is already registered")
        self._sources[source.id] = source
        log.info("registered search source %r (%s)", source.id, source.name)

    @property
    def sources(self) -> list[SearchSource]:
        return list(self._sources.values())

    def get(self, source_id: str) -> SearchSource | None:
        return self._sources.get(source_id)

    def keyword(
        self, query: str, *, limit: int = 50, include_in_use: bool = True,
        source_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for source in self._selected(source_ids):
            hits.extend(source.keyword(query, limit=limit, include_in_use=include_in_use))
        return _rank(hits, limit)

    def vector(
        self, vector: np.ndarray, *, limit: int = 20, min_score: float = 0.0,
        source_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for source in self._selected(source_ids):
            hits.extend(source.vector(vector, limit=limit, min_score=min_score))
        return _rank(hits, limit)

    def recent(self, *, limit: int = 20) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for source in self.sources:
            hits.extend(source.recent(limit=limit))
        return hits[:limit]

    def health(self) -> dict[str, tuple[bool, str]]:
        return {source.id: source.health() for source in self.sources}

    def _selected(self, source_ids: list[str] | None) -> list[SearchSource]:
        if not source_ids:
            return self.sources
        return [self._sources[sid] for sid in source_ids if sid in self._sources]


def _rank(hits: list[SearchHit], limit: int) -> list[SearchHit]:
    """Highest score first, one hit per item, capped."""
    best: dict[str, SearchHit] = {}
    for hit in hits:
        existing = best.get(hit.key)
        if existing is None or hit.score > existing.score:
            best[hit.key] = hit
    ordered = sorted(best.values(), key=lambda hit: hit.score, reverse=True)
    return ordered[:limit]
