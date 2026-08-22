"""HTTP-level tests over the real Flask app with a fake vision backend."""

from __future__ import annotations

import io

import pytest
from conftest import LAYOUT, FakeEmbedder, FakeVisionBackend
from PIL import Image

from qr_organizer import paths
from qr_organizer.health import HealthMonitor
from qr_organizer.pipeline import IdentificationPipeline
from qr_organizer.registry import RegistryWriter
from qr_organizer.runtime import AppContext
from qr_organizer.search import SearchRegistry
from qr_organizer.search.bins_source import BinInventorySource
from qr_organizer.services import bins as bins_service
from qr_organizer.services import items as items_service
from qr_organizer.web.app import create_app


@pytest.fixture()
def client(cfg, db, tmp_path):
    paths.ensure_dir(paths.photos_dir(cfg.data_dir))
    paths.ensure_dir(paths.thumbnails_dir(cfg.data_dir))
    source = BinInventorySource(db)
    registry = SearchRegistry()
    registry.register(source)
    backend = FakeVisionBackend(list(LAYOUT))
    context = AppContext(
        cfg=cfg,
        db=db,
        embedder=FakeEmbedder(),
        vision=backend,
        search=registry,
        bins_source=source,
        pipeline=IdentificationPipeline(
            db=db, backend=backend, embedder=FakeEmbedder(), source=source,
            thumbnails_root=paths.thumbnails_dir(cfg.data_dir),
            photos_root=paths.photos_dir(cfg.data_dir), config=cfg,
        ),
        health=HealthMonitor(),
        registry_writer=RegistryWriter(directory=tmp_path / "registry"),
        log_path=tmp_path / "app.log",
        bind_host="127.0.0.1",
        bind_explanation="test",
        base_url="http://testserver",
    )
    context._register_health_checks()
    app = create_app(context)
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _photo_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (200, 150), (120, 120, 120)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "path",
    ["/", "/scan", "/search", "/bins", "/locations", "/loans", "/pull", "/review",
     "/labels", "/config", "/status", "/health"],
)
def test_every_page_renders(client, path):
    assert client.get(path).status_code == 200


def test_health_reports_readiness_not_just_liveness(client):
    payload = client.get("/health").get_json()
    assert payload["status"] in {"ok", "degraded", "error"}
    assert set(payload["checks"]) >= {"database", "vision", "embeddings", "search", "storage"}
    assert "last_success" in payload


def test_scanning_an_unknown_location_offers_to_name_it(client):
    response = client.get("/s/LOC-0001")
    assert response.status_code == 200
    assert b"Create place" in response.data
    assert b"LOC-0001" in response.data


def test_scanning_a_place_then_a_bin_tags_the_bin(client):
    client.post("/locations/create", data={"code": "LOC-0001", "name": "Shed"})
    response = client.get("/s/BIN-0001")
    assert b"the place you scanned" in response.data

    client.post("/bins/create", data={"code": "BIN-0001", "location_id": "active"})
    page = client.get("/bins/BIN-0001")
    assert b"Shed" in page.data


def test_a_bin_scan_with_no_live_location_will_not_guess(client, db):
    response = client.get("/s/BIN-0001")
    assert b"No live location context" in response.data

    # Creating it anyway is allowed, but only as a deliberate choice.
    client.post("/bins/create", data={"code": "BIN-0001", "location_id": ""})
    assert bins_service.get_bin(db, "BIN-0001")["location_id"] is None


def test_an_expired_context_is_refused_at_submit_time(client, db):
    client.post("/locations/create", data={"code": "LOC-0001", "name": "Shed"})
    with db.write() as conn:
        conn.execute("UPDATE location_context SET last_used_at = '2000-01-01T00:00:00+00:00'")

    response = client.post("/bins/create", data={"code": "BIN-0001", "location_id": "active"})
    assert response.status_code == 409
    assert b"expired while this form was open" in response.data


def test_rescanning_a_bin_never_moves_it(client, db):
    client.post("/locations/create", data={"code": "LOC-0001", "name": "Shed"})
    client.post("/bins/create", data={"code": "BIN-0001", "location_id": "active"})
    client.post("/locations/create", data={"code": "LOC-0002", "name": "Attic"})

    client.get("/s/BIN-0001")  # scanned while "Attic" is the active location

    assert bins_service.get_bin(db, "BIN-0001")["location_name"] == "Shed"


def test_moving_a_bin_from_the_form_does_move_it(client, db):
    client.post("/locations/create", data={"code": "LOC-0001", "name": "Shed"})
    client.post("/bins/create", data={"code": "BIN-0001", "location_id": "active"})
    attic = client.post("/locations/create", data={"code": "LOC-0002", "name": "Attic"})
    assert attic.status_code == 302

    from qr_organizer.services import locations as locations_service

    target = locations_service.get_location(db, "LOC-0002")
    client.post("/bins/BIN-0001/move", data={"location_id": str(target["id"])})
    assert bins_service.get_bin(db, "BIN-0001")["location_name"] == "Attic"


def test_scan_resolution_accepts_urls_and_rejects_junk(client):
    ok = client.post("/api/scan/resolve", json={"payload": "http://x/s/BIN-0042"})
    assert ok.get_json() == {"ok": True, "code": "BIN-0042", "url": "/s/BIN-0042"}

    bad = client.post("/api/scan/resolve", json={"payload": "https://example.com"})
    assert bad.status_code == 400
    assert bad.get_json()["ok"] is False


def test_uploading_a_photo_queues_a_session(client, db):
    client.post("/bins/create", data={"code": "BIN-0001", "location_id": ""})
    response = client.post(
        "/bins/BIN-0001/photo",
        data={"photo": (io.BytesIO(_photo_bytes()), "layout.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 302
    assert "session=" in response.headers["Location"]
    assert db.query_one("SELECT COUNT(*) AS n FROM inventory_sessions")["n"] == 1


def test_an_unknown_code_prefix_is_a_clear_404(client):
    response = client.get("/s/XYZ-0001")
    assert response.status_code == 404
    assert b"not a recognised code" in response.data


def test_search_page_shows_a_status_badge_for_in_use_items(client, db):
    record = bins_service.create_bin(db, code="BIN-0001")
    item_id = items_service.create_item(
        db, label="torque wrench", bin_id=int(record["id"]), thumbnail_path="",
        source_photo_id=None, bbox=None,
    )
    items_service.mark_in_use(db, item_id)

    response = client.get("/search?q=torque")
    assert b"torque wrench" in response.data
    assert b"in use" in response.data


def test_pull_list_round_trip(client, db):
    record = bins_service.create_bin(db, code="BIN-0001")
    item_id = items_service.create_item(
        db, label="hammer", bin_id=int(record["id"]), thumbnail_path="",
        source_photo_id=None, bbox=None,
    )

    client.post("/pull/add", data={"item_id": item_id})
    assert b"hammer" in client.get("/pull").data

    client.post("/pull/check", data={"item_id": item_id})
    assert items_service.get_item(db, item_id)["status"] == "in_use"

    client.post("/bins/BIN-0001/checkin", data={"item_id": item_id})
    assert items_service.get_item(db, item_id)["status"] == "in_bin"


def test_correcting_a_label_from_the_review_queue(client, db):
    record = bins_service.create_bin(db, code="BIN-0001")
    item_id = items_service.create_item(
        db, label="mystery object", bin_id=int(record["id"]), thumbnail_path="",
        source_photo_id=None, bbox=None, needs_review=True, label_source="unidentified",
    )
    assert b"mystery object" in client.get("/review").data

    client.post(f"/items/{item_id}/label", data={"label": "torx bit set", "next": "/review"})
    item = items_service.get_item(db, item_id)
    assert item["label"] == "torx bit set"
    assert item["needs_review"] == 0
    assert item["label_source"] == "user"


def test_config_form_saves_and_validates(client, db, cfg):
    response = client.post("/config", data={"field:server.port": "9100"})
    assert response.status_code == 302
    assert cfg.int_("server.port") == 9100

    rejected = client.post("/config", data={"field:server.port": "not-a-number"})
    assert rejected.status_code == 400
    assert b"not a whole number" in rejected.data
    assert cfg.int_("server.port") == 9100


def test_config_form_refuses_a_wildcard_bind(client, cfg):
    response = client.post("/config", data={"field:server.host": "0.0.0.0"})
    assert response.status_code == 400
    assert b"refused" in response.data
    assert cfg.str_("server.host") != "0.0.0.0"


def test_label_sheet_produces_a_pdf_and_reserves_codes(client, db):
    response = client.post("/labels/sheet", data={"kind": "bin", "count": "4"})
    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF-")
    assert db.query_one("SELECT COUNT(*) AS n FROM label_stock")["n"] == 4


def test_the_device_cookie_is_set_once_and_reused(client):
    first = client.get("/")
    assert any("qro_device" in header for header in first.headers.getlist("Set-Cookie"))
    second = client.get("/")
    assert not any("qro_device" in header for header in second.headers.getlist("Set-Cookie"))


def test_background_identification_completes_through_the_http_path(client, db, cfg, tmp_path):
    """The queued worker path, not just a direct pipeline.run()."""
    import time

    from conftest import make_photo

    client.post("/bins/create", data={"code": "BIN-0001", "location_id": ""})
    photo_path = make_photo(tmp_path, list(LAYOUT), name="web-layout.jpg")
    response = client.post(
        "/bins/BIN-0001/photo",
        data={"photo": (io.BytesIO(photo_path.read_bytes()), "layout.jpg")},
        content_type="multipart/form-data",
    )
    session_id = int(response.headers["Location"].rsplit("=", 1)[1])

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        payload = client.get(f"/api/sessions/{session_id}").get_json()
        if payload["status"] in {"complete", "error"}:
            break
        time.sleep(0.1)

    assert payload["status"] == "complete", payload.get("error")
    assert payload["detected_count"] == len(LAYOUT)
    assert payload["added_count"] == len(LAYOUT)
    assert b"wrench" in client.get("/bins/BIN-0001").data


def test_populated_pages_render(client, db):
    """Templates in their non-empty branches, which the empty-state pass misses."""
    from qr_organizer.services import locations as locations_service

    client.post("/locations/create", data={"code": "LOC-0001", "name": "Shed"})
    client.post("/bins/create", data={"code": "BIN-0001", "location_id": "active"})
    record = bins_service.get_bin(db, "BIN-0001")
    item_id = items_service.create_item(
        db, label="claw hammer", bin_id=int(record["id"]), thumbnail_path="a/b.jpg",
        source_photo_id=None, bbox=(0.1, 0.1, 0.2, 0.2), needs_review=True,
    )
    client.post("/loans/create", data={"code": "LOAN-0001", "person_name": "Dave",
                                       "item_id": str(item_id)})
    client.post("/pull/add", data={"item_id": item_id})

    for path in [
        f"/items/{item_id}",
        "/bins/BIN-0001",
        "/locations/LOC-0001",
        "/loans/LOAN-0001",
        "/pull",
        "/search?q=hammer",
        "/search?q=hammer&sort=location",
        "/review",
        "/status",
        "/labels",
    ]:
        response = client.get(path)
        assert response.status_code == 200, f"{path} -> {response.status_code}"
        assert b"Traceback" not in response.data

    assert locations_service.get_location(db, "LOC-0001") is not None


def _queue_a_return(client, db, tmp_path, *, prior="in_use"):
    """Get one item into the 'looks like it came back' state over HTTP."""

    from conftest import make_photo

    from qr_organizer.services import loans as loans_service

    client.post("/bins/create", data={"code": "BIN-0001", "location_id": ""})
    photo = make_photo(tmp_path, ["wrench"], name="ret-a.jpg")
    client.post(
        "/bins/BIN-0001/photo",
        data={"photo": (io.BytesIO(photo.read_bytes()), "a.jpg")},
        content_type="multipart/form-data",
    )
    _await_sessions(client, db)
    item_id = db.query_one("SELECT id FROM items")["id"]

    if prior == "loaned":
        loan = loans_service.create_loan(db, person_name="Dave", code="LOAN-0001")
        loans_service.assign_items(db, int(loan["id"]), [int(item_id)])
    else:
        items_service.mark_in_use(db, int(item_id))

    again = make_photo(tmp_path, ["wrench"], name="ret-b.jpg")
    client.post(
        "/bins/BIN-0001/photo",
        data={"photo": (io.BytesIO(again.read_bytes()), "b.jpg")},
        content_type="multipart/form-data",
    )
    _await_sessions(client, db)
    entry = items_service.pending_returns(db)[0]
    return int(item_id), int(entry["entry_id"])


def _await_sessions(client, db, timeout=20):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = db.query_one(
            "SELECT COUNT(*) AS n FROM inventory_sessions WHERE status IN ('pending','running')"
        )
        if not row["n"]:
            return
        time.sleep(0.05)
    raise AssertionError("identification did not finish in time")


def test_a_reappearing_item_asks_rather_than_assumes(client, db, tmp_path):
    item_id, _ = _queue_a_return(client, db, tmp_path)

    page = client.get("/bins/BIN-0001")
    assert b"Did these come back?" in page.data
    assert items_service.get_item(db, item_id)["status"] == "in_use"
    assert b"to confirm as returned" in client.get("/").data


def test_confirming_from_the_bin_page_returns_the_item(client, db, tmp_path):
    item_id, entry_id = _queue_a_return(client, db, tmp_path, prior="loaned")

    response = client.post(f"/returns/{entry_id}/confirm", data={"next": "/bins/BIN-0001"})
    assert response.status_code == 302
    item = items_service.get_item(db, item_id)
    assert item["status"] == "in_bin"
    assert item["loan_id"] is None
    assert b"Did these come back?" not in client.get("/bins/BIN-0001").data


def test_dismissing_from_the_bin_page_keeps_the_loan(client, db, tmp_path):
    item_id, entry_id = _queue_a_return(client, db, tmp_path, prior="loaned")

    client.post(f"/returns/{entry_id}/dismiss", data={"next": "/bins/BIN-0001"})
    item = items_service.get_item(db, item_id)
    assert item["status"] == "loaned"
    assert item["loan_person"] == "Dave"
    assert items_service.pending_returns(db) == []


def test_bulk_confirm_resolves_every_queued_return_for_a_bin(client, db, tmp_path):
    from conftest import make_photo

    client.post("/bins/create", data={"code": "BIN-0001", "location_id": ""})
    photo = make_photo(tmp_path, list(LAYOUT), name="bulk-a.jpg")
    client.post(
        "/bins/BIN-0001/photo",
        data={"photo": (io.BytesIO(photo.read_bytes()), "a.jpg")},
        content_type="multipart/form-data",
    )
    _await_sessions(client, db)
    for row in db.query("SELECT id FROM items"):
        items_service.mark_in_use(db, int(row["id"]))

    again = make_photo(tmp_path, list(LAYOUT), name="bulk-b.jpg")
    client.post(
        "/bins/BIN-0001/photo",
        data={"photo": (io.BytesIO(again.read_bytes()), "b.jpg")},
        content_type="multipart/form-data",
    )
    _await_sessions(client, db)
    assert len(items_service.pending_returns(db)) == len(LAYOUT)

    client.post("/bins/BIN-0001/returns", data={"action": "confirm"})
    assert items_service.pending_returns(db) == []
    assert db.query_one("SELECT COUNT(*) AS n FROM items WHERE status='in_bin'")["n"] == len(LAYOUT)


def test_the_review_queue_lists_returns_awaiting_a_decision(client, db, tmp_path):
    _queue_a_return(client, db, tmp_path)
    entry_id = items_service.pending_returns(db)[0]["entry_id"]
    page = client.get("/review")
    assert b"Did these come back?" in page.data
    assert f"/returns/{entry_id}/confirm".encode() in page.data
    assert f"/returns/{entry_id}/dismiss".encode() in page.data


def test_confirming_an_already_resolved_return_is_a_clean_404(client, db, tmp_path):
    _item_id, entry_id = _queue_a_return(client, db, tmp_path)
    client.post(f"/returns/{entry_id}/confirm")
    assert client.post(f"/returns/{entry_id}/confirm").status_code == 404
