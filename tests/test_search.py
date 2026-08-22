"""Keyword search, vector search, and the pluggable source registry."""

from __future__ import annotations

import numpy as np
import pytest

from qr_organizer.search import SearchHit, SearchRegistry
from qr_organizer.search.bins_source import BinInventorySource, fts_query
from qr_organizer.services import bins as bins_service
from qr_organizer.services import items as items_service
from qr_organizer.services import loans as loans_service
from qr_organizer.services import locations as locations_service


@pytest.fixture()
def stocked(db):
    place = locations_service.create_location(db, name="Shed", code="LOC-0001")
    record = bins_service.create_bin(db, code="BIN-0001", label="Plumbing",
                                     location_id=int(place["id"]))
    bin_id = int(record["id"])
    ids = {
        label: items_service.create_item(
            db, label=label, bin_id=bin_id, thumbnail_path=f"{label}.jpg",
            source_photo_id=None, bbox=None,
        )
        for label in ("adjustable wrench", "roll of PTFE tape", "pipe cutter")
    }
    return db, bin_id, ids


def test_keyword_search_finds_items_without_knowing_where_they_are(stocked):
    db, _, _ = stocked
    hits = BinInventorySource(db).keyword("wrench", limit=10, include_in_use=True)
    assert [hit.label for hit in hits] == ["adjustable wrench"]
    # Location comes along for context, but was never part of the query.
    assert hits[0].location_name == "Shed"
    assert hits[0].container_label == "BIN-0001 - Plumbing"


def test_prefix_matching_makes_as_you_type_search_work(stocked):
    db, _, _ = stocked
    assert BinInventorySource(db).keyword("wren", limit=10, include_in_use=True)


def test_in_use_items_stay_in_results_with_a_status(stocked):
    db, _, ids = stocked
    items_service.mark_in_use(db, ids["pipe cutter"])
    source = BinInventorySource(db)

    shown = source.keyword("pipe", limit=10, include_in_use=True)
    assert shown and shown[0].status == "in_use" and shown[0].in_use is True

    hidden = source.keyword("pipe", limit=10, include_in_use=False)
    assert hidden == []


def test_a_loaned_item_says_who_has_it(stocked):
    db, _, ids = stocked
    loan = loans_service.create_loan(db, person_name="Dave", code="LOAN-0001")
    loans_service.assign_items(db, int(loan["id"]), [ids["pipe cutter"]])

    hit = BinInventorySource(db).keyword("pipe", limit=10, include_in_use=True)[0]
    assert hit.status == "loaned"
    assert hit.status_detail == "with Dave"
    assert hit.container_url == "/loans/LOAN-0001"


@pytest.mark.parametrize("raw", ["", "   ", "!!!"])
def test_empty_queries_fall_back_to_recent_items(stocked, raw):
    db, _, _ = stocked
    hits = BinInventorySource(db).keyword(raw, limit=10, include_in_use=True)
    assert len(hits) == 3


@pytest.mark.parametrize(
    "raw", ["wrench OR", 'tape"', "NEAR(", "*", "a AND (b", "10mm socket"]
)
def test_punctuation_cannot_break_the_fts_query(stocked, raw):
    db, _, _ = stocked
    BinInventorySource(db).keyword(raw, limit=10, include_in_use=True)


def test_fts_query_quotes_every_token_and_prefixes_the_last():
    assert fts_query("pipe cutter") == '"pipe" AND "cutter"*'
    assert fts_query("") == ""


def test_vector_search_matches_the_nearest_embedding(stocked):
    db, _, ids = stocked
    source = BinInventorySource(db)
    vectors = {
        ids["adjustable wrench"]: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        ids["roll of PTFE tape"]: np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ids["pipe cutter"]: np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    db.ensure_vector_index(3)
    for item_id, vector in vectors.items():
        items_service.store_embedding(db, item_id, model="test", vector=vector)

    hits = source.vector(np.array([0.95, 0.1, 0.0], dtype=np.float32), limit=2, min_score=0.5)
    assert hits[0].label == "adjustable wrench"
    assert hits[0].match_kind == "visual"


def test_the_numpy_fallback_agrees_with_the_extension(stocked, monkeypatch):
    """The two nearest-neighbour paths must not drift apart."""
    db, _, ids = stocked
    source = BinInventorySource(db)
    db.ensure_vector_index(3)
    for offset, item_id in enumerate(ids.values()):
        vector = np.zeros(3, dtype=np.float32)
        vector[offset] = 1.0
        items_service.store_embedding(db, item_id, model="test", vector=vector)

    query = np.array([0.9, 0.4, 0.0], dtype=np.float32)
    query /= np.linalg.norm(query)

    with_extension = source.nearest(query, limit=3)
    monkeypatch.setattr(db, "vec_available", False)
    without = source.nearest(query, limit=3)

    assert [item for item, _ in with_extension] == [item for item, _ in without]
    for (_, a), (_, b) in zip(with_extension, without):
        assert a == pytest.approx(b, abs=1e-5)


def test_a_second_source_is_searched_alongside_the_bins(stocked):
    """The whole point of the source abstraction: adding one changes nothing else."""
    db, _, _ = stocked

    class RoomSource:
        id = "rooms"
        name = "Rooms"

        def keyword(self, query, *, limit, include_in_use):
            if "wrench" not in query:
                return []
            return [
                SearchHit(
                    source_id=self.id, source_name=self.name, external_id="r1",
                    label="wrench on the pegboard", score=0.99, location_name="Garage",
                )
            ]

        def vector(self, vector, *, limit, min_score):
            return []

        def recent(self, *, limit):
            return []

        def health(self):
            return True, "fake"

    registry = SearchRegistry()
    registry.register(BinInventorySource(db))
    registry.register(RoomSource())

    hits = registry.keyword("wrench", limit=10, include_in_use=True)
    assert {hit.source_id for hit in hits} == {"bins", "rooms"}
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)
    assert {hit.label for hit in hits} == {"adjustable wrench", "wrench on the pegboard"}
    assert set(registry.health()) == {"bins", "rooms"}


def test_registering_a_duplicate_source_id_is_refused(db):
    registry = SearchRegistry()
    registry.register(BinInventorySource(db))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(BinInventorySource(db))
