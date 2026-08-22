"""The identification pipeline end to end, offline.

These cover the behaviours the spec is most specific about: a new bin gets its
contents, a re-inventory diffs rather than replaces, missing items are flagged
and never deleted, and a corrected label immediately changes what the RAG layer
suggests next time.
"""

from __future__ import annotations

import pytest
from conftest import LAYOUT, FakeEmbedder, make_photo

from qr_organizer import paths
from qr_organizer.errors import VisionError
from qr_organizer.services import bins as bins_service
from qr_organizer.services import items as items_service
from qr_organizer.services import photos as photos_service


def _ingest(db, cfg, source):
    return photos_service.ingest(
        db, source=source, photos_root=paths.photos_dir(cfg.data_dir), kind="bin_layout"
    )


def _run(pipeline, db, cfg, bin_id, photo_path):
    photo = _ingest(db, cfg, photo_path)
    session_id = pipeline.start_session(bin_id=bin_id, photo_id=int(photo["id"]), device_key="test")
    return pipeline.run(
        session_id=session_id,
        bin_id=bin_id,
        photo_id=int(photo["id"]),
        photo_path=paths.photos_dir(cfg.data_dir) / photo["path"],
    )


def test_new_bin_gets_one_item_per_detected_thing(db, cfg, photo, pipeline_factory):
    record = bins_service.create_bin(db, code="BIN-0001")
    pipeline, backend, _ = pipeline_factory(list(LAYOUT))

    result = _run(pipeline, db, cfg, int(record["id"]), photo)

    assert len(result.detected) == len(LAYOUT)
    assert len(result.added_item_ids) == len(LAYOUT)
    assert result.missing_item_ids == []
    labels = {item["label"] for item in items_service.list_bin_items(db, int(record["id"]))}
    assert labels == set(LAYOUT)
    # Two whole-image calls, and no verify pass at high confidence.
    assert backend.calls == ["enumerate pass 1", "locate pass"]


def test_every_item_gets_a_thumbnail_and_an_embedding(db, cfg, photo, pipeline_factory):
    record = bins_service.create_bin(db, code="BIN-0001")
    pipeline, _, _ = pipeline_factory(list(LAYOUT))
    result = _run(pipeline, db, cfg, int(record["id"]), photo)

    for item_id in result.added_item_ids:
        item = items_service.get_item(db, item_id)
        assert (paths.thumbnails_dir(cfg.data_dir) / item["thumbnail_path"]).is_file()
        row = db.query_one("SELECT dim FROM item_embeddings WHERE item_id = ?", (item_id,))
        assert row is not None and row["dim"] == FakeEmbedder().dim


def test_reinventory_diffs_instead_of_replacing(db, cfg, photo, pipeline_factory, tmp_path):
    record = bins_service.create_bin(db, code="BIN-0001")
    bin_id = int(record["id"])

    pipeline, _, _ = pipeline_factory(list(LAYOUT))
    first = _run(pipeline, db, cfg, bin_id, photo)
    assert len(first.added_item_ids) == 4

    # Second photo: the tape is gone, everything else is still there.
    remaining = [name for name in LAYOUT if name != "roll of tape"]
    second_photo = make_photo(tmp_path, remaining)
    pipeline2, _, _ = pipeline_factory(remaining)
    second = _run(pipeline2, db, cfg, bin_id, second_photo)

    assert second.added_item_ids == []
    assert len(second.matched_item_ids) == 3
    assert len(second.missing_item_ids) == 1

    missing = items_service.get_item(db, second.missing_item_ids[0])
    assert missing["label"] == "roll of tape"
    assert missing["status"] == "missing"


def test_missing_items_are_flagged_never_deleted(db, cfg, photo, pipeline_factory, tmp_path):
    record = bins_service.create_bin(db, code="BIN-0001")
    bin_id = int(record["id"])
    pipeline, _, _ = pipeline_factory(list(LAYOUT))
    _run(pipeline, db, cfg, bin_id, photo)

    empty_photo = make_photo(tmp_path, ["wrench"], name="just-one.jpg")
    pipeline2, _, _ = pipeline_factory(["wrench"])
    _run(pipeline2, db, cfg, bin_id, empty_photo)

    assert db.query_one("SELECT COUNT(*) AS n FROM items")["n"] == 4
    still_there = db.query_one("SELECT COUNT(*) AS n FROM items WHERE status = 'missing'")
    assert still_there["n"] == 3


def test_a_returning_item_stops_being_missing(db, cfg, photo, pipeline_factory, tmp_path):
    record = bins_service.create_bin(db, code="BIN-0001")
    bin_id = int(record["id"])
    pipeline, _, _ = pipeline_factory(list(LAYOUT))
    _run(pipeline, db, cfg, bin_id, photo)

    one = make_photo(tmp_path, ["wrench"], name="one.jpg")
    pipeline2, _, _ = pipeline_factory(["wrench"])
    _run(pipeline2, db, cfg, bin_id, one)
    assert db.query_one("SELECT COUNT(*) AS n FROM items WHERE status='missing'")["n"] == 3

    pipeline3, _, _ = pipeline_factory(list(LAYOUT))
    everything = make_photo(tmp_path, list(LAYOUT), name="all-again.jpg")
    result = _run(pipeline3, db, cfg, bin_id, everything)

    assert result.added_item_ids == []
    assert db.query_one("SELECT COUNT(*) AS n FROM items WHERE status='missing'")["n"] == 0


def test_rag_reuses_a_corrected_label_on_a_later_photo(db, cfg, photo, pipeline_factory, tmp_path):
    """The library updates live: correct a label, and the next match uses it."""
    first_bin = bins_service.create_bin(db, code="BIN-0001")
    pipeline, _, _ = pipeline_factory(["wrench"])
    result = _run(pipeline, db, cfg, int(first_bin["id"]), make_photo(tmp_path, ["wrench"]))
    item_id = result.added_item_ids[0]

    # The user renames it to a generic catch-all, as the spec describes.
    items_service.set_label(db, item_id, label="robot kit parts", source="user")

    # A different bin, a visually identical object, and the model calls it
    # something else entirely.
    second_bin = bins_service.create_bin(db, code="BIN-0002")
    pipeline2, _, _ = pipeline_factory(["wrench"])
    second = _run(
        pipeline2, db, cfg, int(second_bin["id"]),
        make_photo(tmp_path, ["wrench"], name="again.jpg"),
    )

    detected = second.detected[0]
    assert detected.label == "robot kit parts"
    assert detected.label_source == "rag"
    # An assisted label is surfaced for confirmation, not silently applied.
    assert detected.needs_review is True


def test_low_confidence_detections_get_a_verify_pass(db, cfg, photo, pipeline_factory, tmp_path):
    record = bins_service.create_bin(db, code="BIN-0001")
    pipeline, backend, _ = pipeline_factory(["wrench"], confidence=0.2)
    result = _run(pipeline, db, cfg, int(record["id"]), make_photo(tmp_path, ["wrench"]))

    assert "verify pass" in backend.calls
    # Nothing in the library yet and no candidates, so the fake declines to name
    # it -- which must reach the user rather than being guessed at.
    assert result.detected[0].label_source == "unidentified"
    assert result.detected[0].needs_review is True


def test_an_empty_enumeration_is_a_loud_failure(db, cfg, photo, pipeline_factory):
    record = bins_service.create_bin(db, code="BIN-0001")
    pipeline, _, _ = pipeline_factory([])
    with pytest.raises(VisionError, match="found no items"):
        _run(pipeline, db, cfg, int(record["id"]), photo)


def test_a_failed_run_is_recorded_on_the_session(db, cfg, photo, pipeline_factory):
    from qr_organizer.pipeline import session_status

    record = bins_service.create_bin(db, code="BIN-0001")
    pipeline, _, _ = pipeline_factory([])
    stored = photos_service.ingest(
        db, source=photo, photos_root=paths.photos_dir(cfg.data_dir), kind="bin_layout"
    )
    session_id = pipeline.start_session(
        bin_id=int(record["id"]), photo_id=int(stored["id"]), device_key="test"
    )
    pipeline._run_guarded(
        session_id, int(record["id"]), int(stored["id"]),
        paths.photos_dir(cfg.data_dir) / stored["path"],
    )

    status = session_status(db, session_id)
    assert status["status"] == "error"
    assert "found no items" in status["error"]


def test_identification_never_moves_the_bin(db, cfg, photo, pipeline_factory):
    from qr_organizer.services import locations as locations_service

    place = locations_service.create_location(db, name="Shed", code="LOC-0001")
    record = bins_service.create_bin(db, code="BIN-0001", location_id=int(place["id"]))
    pipeline, _, _ = pipeline_factory(list(LAYOUT))
    _run(pipeline, db, cfg, int(record["id"]), photo)

    assert bins_service.get_bin(db, "BIN-0001")["location_id"] == place["id"]
