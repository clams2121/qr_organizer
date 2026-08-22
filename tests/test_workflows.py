"""Location context, loans, the pull list, and label stock."""

from __future__ import annotations

from datetime import timedelta

import pytest

from qr_organizer.errors import ConflictError
from qr_organizer.services import bins as bins_service
from qr_organizer.services import items as items_service
from qr_organizer.services import loans as loans_service
from qr_organizer.services import locations as locations_service
from qr_organizer.services import pulllist
from qr_organizer.util import now


def _item(db, bin_id, label="wrench"):
    return items_service.create_item(
        db, label=label, bin_id=bin_id, thumbnail_path="", source_photo_id=None, bbox=None
    )


# -- location context -----------------------------------------------------


def test_active_location_is_returned_while_it_is_live(db):
    place = locations_service.create_location(db, name="Shed", code="LOC-0001")
    locations_service.set_active_location(db, "phone", int(place["id"]))

    active = locations_service.active_location(db, "phone", 30)
    assert active is not None
    assert active["location_code"] == "LOC-0001"


def test_stale_location_context_expires_and_is_discarded(db):
    place = locations_service.create_location(db, name="Shed", code="LOC-0001")
    locations_service.set_active_location(db, "phone", int(place["id"]))

    stale = (now() - timedelta(minutes=31)).isoformat()
    with db.write() as conn:
        conn.execute("UPDATE location_context SET last_used_at = ?", (stale,))

    assert locations_service.active_location(db, "phone", 30) is None
    # ...and it is not left lying around to be picked up later.
    assert db.query_one("SELECT COUNT(*) AS n FROM location_context")["n"] == 0


def test_each_device_has_its_own_context(db):
    shed = locations_service.create_location(db, name="Shed", code="LOC-0001")
    attic = locations_service.create_location(db, name="Attic", code="LOC-0002")
    locations_service.set_active_location(db, "phone", int(shed["id"]))
    locations_service.set_active_location(db, "tablet", int(attic["id"]))

    assert locations_service.active_location(db, "phone", 30)["location_code"] == "LOC-0001"
    assert locations_service.active_location(db, "tablet", 30)["location_code"] == "LOC-0002"


def test_moving_a_bin_is_explicit(db):
    shed = locations_service.create_location(db, name="Shed", code="LOC-0001")
    attic = locations_service.create_location(db, name="Attic", code="LOC-0002")
    bins_service.create_bin(db, code="BIN-0001", location_id=int(shed["id"]))

    bins_service.move_bin(db, "BIN-0001", int(attic["id"]))
    assert bins_service.get_bin(db, "BIN-0001")["location_name"] == "Attic"


# -- pull list ------------------------------------------------------------


def test_checking_off_a_pull_list_entry_marks_it_in_use(db):
    record = bins_service.create_bin(db, code="BIN-0001")
    item_id = _item(db, int(record["id"]))

    pulllist.add(db, "phone", item_id)
    assert pulllist.counts(db, "phone") == (1, 0)
    assert items_service.get_item(db, item_id)["status"] == "in_bin"

    pulllist.check_off(db, "phone", item_id)
    assert pulllist.counts(db, "phone") == (1, 1)
    assert items_service.get_item(db, item_id)["status"] == "in_use"


def test_an_in_use_item_returns_to_any_bin(db):
    first = bins_service.create_bin(db, code="BIN-0001")
    second = bins_service.create_bin(db, code="BIN-0002")
    item_id = _item(db, int(first["id"]))

    pulllist.add(db, "phone", item_id)
    pulllist.check_off(db, "phone", item_id)
    items_service.check_into_bin(db, item_id, int(second["id"]))

    item = items_service.get_item(db, item_id)
    assert item["status"] == "in_bin"
    assert item["bin_code"] == "BIN-0002"


def test_adding_the_same_item_twice_is_idempotent(db):
    record = bins_service.create_bin(db, code="BIN-0001")
    item_id = _item(db, int(record["id"]))
    assert pulllist.add(db, "phone", item_id) is True
    assert pulllist.add(db, "phone", item_id) is False
    assert pulllist.counts(db, "phone") == (1, 0)


def test_clearing_the_list_does_not_change_item_status(db):
    record = bins_service.create_bin(db, code="BIN-0001")
    item_id = _item(db, int(record["id"]))
    pulllist.add(db, "phone", item_id)
    pulllist.check_off(db, "phone", item_id)
    pulllist.clear(db, "phone")

    assert pulllist.counts(db, "phone") == (0, 0)
    assert items_service.get_item(db, item_id)["status"] == "in_use"


# -- loans ----------------------------------------------------------------


def test_a_loan_records_who_has_the_item(db):
    record = bins_service.create_bin(db, code="BIN-0001")
    item_id = _item(db, int(record["id"]))
    loan = loans_service.create_loan(db, person_name="Dave", code="LOAN-0001")

    loans_service.assign_items(db, int(loan["id"]), [item_id])

    item = items_service.get_item(db, item_id)
    assert item["status"] == "loaned"
    assert item["loan_person"] == "Dave"
    # The item still remembers where it lives, so it can go home later.
    assert item["bin_code"] == "BIN-0001"


def test_a_loan_cannot_be_closed_while_items_are_out(db):
    record = bins_service.create_bin(db, code="BIN-0001")
    item_id = _item(db, int(record["id"]))
    loan = loans_service.create_loan(db, person_name="Dave", code="LOAN-0001")
    loans_service.assign_items(db, int(loan["id"]), [item_id])

    with pytest.raises(ConflictError, match="still out"):
        loans_service.close_loan(db, int(loan["id"]))

    items_service.check_into_bin(db, item_id, int(record["id"]))
    loans_service.close_loan(db, int(loan["id"]))
    assert loans_service.get_loan(db, "LOAN-0001")["status"] == "closed"


# -- label stock ----------------------------------------------------------


def test_sheets_never_reissue_a_code(db):
    first = bins_service.reserve_codes(
        db, kind="bin", prefix="BIN", count=5, digits=4, sheet_id="a"
    )
    second = bins_service.reserve_codes(
        db, kind="bin", prefix="BIN", count=5, digits=4, sheet_id="b"
    )
    assert first == ["BIN-0001", "BIN-0002", "BIN-0003", "BIN-0004", "BIN-0005"]
    assert second[0] == "BIN-0006"
    assert not set(first) & set(second)


def test_a_hand_made_bin_does_not_get_its_code_reissued(db):
    bins_service.create_bin(db, code="BIN-0009")
    codes = bins_service.reserve_codes(
        db, kind="bin", prefix="BIN", count=2, digits=4, sheet_id="a"
    )
    assert codes == ["BIN-0010", "BIN-0011"]


def test_claiming_a_printed_code_marks_the_stock_used(db):
    bins_service.reserve_codes(db, kind="bin", prefix="BIN", count=2, digits=4, sheet_id="a")
    bins_service.create_bin(db, code="BIN-0001")

    sheet = {entry["code"]: entry for entry in bins_service.sheet_codes(db, "a")}
    assert sheet["BIN-0001"]["assigned_at"] is not None
    assert sheet["BIN-0002"]["assigned_at"] is None
