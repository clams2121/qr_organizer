"""Bins, items, locations and loans."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from .. import paths
from ..errors import ConflictError, NotFoundError
from ..pipeline import session_status
from ..services import bins as bins_service
from ..services import items as items_service
from ..services import loans as loans_service
from ..services import locations as locations_service
from ..services import photos as photos_service
from .helpers import ctx, device_key, log_scan

log = logging.getLogger(__name__)

bp = Blueprint("inventory", __name__)


# -- bins -----------------------------------------------------------------


@bp.get("/bins")
def bin_list():
    context = ctx()
    return render_template("bins.html", bins=bins_service.list_bins(context.db))


@bp.get("/bins/<code>")
def bin_detail(code: str):
    context = ctx()
    record = bins_service.get_bin(context.db, code)
    if record is None:
        raise NotFoundError(f"no bin {code}")
    bin_id = int(record["id"])
    session_id = request.args.get("session", type=int)
    return render_template(
        "bin_detail.html",
        bin=record,
        items=items_service.list_bin_items(context.db, bin_id),
        photos=photos_service.bin_photos(context.db, bin_id, limit=6),
        locations=locations_service.list_locations(context.db),
        session_id=session_id,
        session=session_status(context.db, session_id) if session_id else None,
        checked_out=_checked_out_items(),
        sessions=_recent_sessions(bin_id),
        returns=items_service.pending_returns(context.db, bin_id=bin_id),
    )


def _recent_sessions(bin_id: int) -> list[dict]:
    return [
        dict(row)
        for row in ctx().db.query(
            "SELECT * FROM inventory_sessions WHERE bin_id = ? ORDER BY id DESC LIMIT 5",
            (bin_id,),
        )
    ]


def _checked_out_items() -> list[dict]:
    """Everything currently in use or loaned, for the check-in picker."""
    return [
        dict(row)
        for row in ctx().db.query(
            "SELECT items.*, loans.person_name AS loan_person, bins.code AS home_code "
            "FROM items LEFT JOIN loans ON loans.id = items.loan_id "
            "LEFT JOIN bins ON bins.id = items.home_bin_id "
            "WHERE items.status IN ('in_use', 'loaned') "
            "ORDER BY items.status_changed_at DESC LIMIT 100"
        )
    ]


@bp.post("/bins/<code>/update")
def bin_update(code: str):
    bins_service.update_bin(
        ctx().db, code, label=request.form.get("label", ""), notes=request.form.get("notes", "")
    )
    return redirect(url_for("inventory.bin_detail", code=code))


@bp.post("/bins/<code>/move")
def bin_move(code: str):
    """The one and only path by which a bin changes location."""
    context = ctx()
    raw = request.form.get("location_id", "").strip()
    location_id = int(raw) if raw.isdigit() else None
    bins_service.move_bin(context.db, code, location_id)
    record = bins_service.get_bin(context.db, code)
    log_scan(kind="bin_moved", code=code, bin_id=int(record["id"]) if record else None,
             location_id=location_id, detail="location changed deliberately")
    return redirect(url_for("inventory.bin_detail", code=code))


@bp.post("/bins/<code>/photo")
def bin_photo(code: str):
    """Upload a layout photo and queue identification."""
    context = ctx()
    record = bins_service.get_bin(context.db, code)
    if record is None:
        raise NotFoundError(f"no bin {code}")
    upload = request.files.get("photo")
    if upload is None or not upload.filename:
        raise ConflictError("no photo was attached")

    bin_id = int(record["id"])
    suffix = Path(upload.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        upload.save(handle.name)
        staged = Path(handle.name)
    try:
        photo = photos_service.ingest(
            context.db,
            source=staged,
            photos_root=paths.photos_dir(context.cfg.data_dir),
            kind="bin_layout",
            bin_id=bin_id,
            max_dimension=context.cfg.int_("images.max_dimension", 2048),
            jpeg_quality=context.cfg.int_("images.jpeg_quality", 88),
        )
    finally:
        staged.unlink(missing_ok=True)

    session_id = context.pipeline.start_session(
        bin_id=bin_id, photo_id=int(photo["id"]), device_key=device_key()
    )
    context.pipeline.submit(
        session_id=session_id,
        bin_id=bin_id,
        photo_id=int(photo["id"]),
        photo_path=paths.photos_dir(context.cfg.data_dir) / photo["path"],
    )
    log_scan(kind="photo", code=code, bin_id=bin_id, detail=f"session {session_id}")
    return redirect(url_for("inventory.bin_detail", code=code, session=session_id))


@bp.get("/api/sessions/<int:session_id>")
def session_poll(session_id: int):
    """Polled by the bin page while identification runs in the background."""
    status = session_status(ctx().db, session_id)
    if status is None:
        return jsonify({"error": "no such session"}), 404
    return jsonify(status)


@bp.post("/bins/<code>/checkin")
def bin_checkin(code: str):
    """Return selected in-use or loaned items to this bin."""
    context = ctx()
    record = bins_service.get_bin(context.db, code)
    if record is None:
        raise NotFoundError(f"no bin {code}")
    item_ids = [int(value) for value in request.form.getlist("item_id") if value.isdigit()]
    for item_id in item_ids:
        items_service.check_into_bin(
            context.db, item_id, int(record["id"]), detail=f"scanned back into {code}"
        )
        log_scan(kind="checkin", code=code, bin_id=int(record["id"]), item_id=item_id)
    log.info("checked %d item(s) into %s", len(item_ids), code)
    return redirect(url_for("inventory.bin_detail", code=code))


# -- items ----------------------------------------------------------------


@bp.get("/items/<int:item_id>")
def item_detail(item_id: int):
    context = ctx()
    item = items_service.get_item(context.db, item_id)
    if item is None:
        raise NotFoundError(f"no item {item_id}")
    return render_template(
        "item_detail.html",
        item=item,
        history=items_service.item_history(context.db, item_id),
        similar=_similar_items(item_id),
        loans=loans_service.list_loans(context.db),
        bins=bins_service.list_bins(context.db),
        pending_return=next(
            iter(
                entry
                for entry in items_service.pending_returns(context.db)
                if entry["item_id"] == item_id
            ),
            None,
        ),
    )


def _similar_items(item_id: int) -> list[dict]:
    """Visually similar items, which is also a quick way to spot duplicates."""
    context = ctx()
    row = context.db.query_one(
        "SELECT vector, dim FROM item_embeddings WHERE item_id = ?", (item_id,)
    )
    if row is None:
        return []
    import numpy as np

    vector = np.frombuffer(row["vector"], dtype=np.float32)
    hits = context.bins_source.vector(vector, limit=6, min_score=0.5)
    return [hit for hit in hits if hit.external_id != str(item_id)]


@bp.post("/items/<int:item_id>/label")
def item_label(item_id: int):
    """Correct or confirm a label. This immediately updates the RAG library."""
    context = ctx()
    label = request.form.get("label", "").strip()
    description = request.form.get("description")
    items_service.set_label(
        context.db, item_id, label=label, description=description, source="user", confidence=1.0
    )
    if request.form.get("reembed") == "1":
        context.pipeline.reembed_item(item_id)
    target = request.form.get("next") or url_for("inventory.item_detail", item_id=item_id)
    return redirect(target)


@bp.post("/items/<int:item_id>/notes")
def item_notes(item_id: int):
    items_service.set_notes(ctx().db, item_id, request.form.get("notes", ""))
    return redirect(url_for("inventory.item_detail", item_id=item_id))


@bp.post("/items/<int:item_id>/delete")
def item_delete(item_id: int):
    items_service.delete_item(ctx().db, item_id)
    return redirect(request.form.get("next") or url_for("scan.home"))


@bp.post("/items/<int:item_id>/status")
def item_status(item_id: int):
    context = ctx()
    action = request.form.get("action", "")
    if action == "in_use":
        items_service.mark_in_use(context.db, item_id, detail="marked in use by hand")
    elif action == "check_in":
        raw = request.form.get("bin_id", "")
        if not raw.isdigit():
            raise ConflictError("pick a bin to check this item into")
        items_service.check_into_bin(context.db, item_id, int(raw))
        log_scan(
            kind="checkin", bin_id=int(raw), item_id=item_id,
            detail="checked in from item page",
        )
    elif action == "missing":
        items_service.mark_missing(context.db, item_id, detail="flagged missing by hand")
    else:
        raise ConflictError(f"unknown status action {action!r}")
    return redirect(url_for("inventory.item_detail", item_id=item_id))


@bp.get("/review")
def review_queue():
    """Everything waiting on a human decision, in one place."""
    context = ctx()
    return render_template(
        "review.html",
        items=items_service.list_needing_review(context.db),
        returns=items_service.pending_returns(context.db),
    )


# -- returns awaiting confirmation ----------------------------------------


@bp.post("/returns/<int:entry_id>/confirm")
def return_confirm(entry_id: int):
    """The user says the item really is back. Only now does its status change."""
    context = ctx()
    item_id = items_service.confirm_return(context.db, entry_id)
    log_scan(kind="return_confirmed", item_id=item_id, detail="confirmed from the web UI")
    return redirect(request.form.get("next") or url_for("inventory.review_queue"))


@bp.post("/returns/<int:entry_id>/dismiss")
def return_dismiss(entry_id: int):
    """The user says it is not back. The item keeps the status they gave it."""
    context = ctx()
    item_id = items_service.dismiss_return(context.db, entry_id)
    log_scan(kind="return_dismissed", item_id=item_id, detail="dismissed from the web UI")
    return redirect(request.form.get("next") or url_for("inventory.review_queue"))


@bp.post("/bins/<code>/returns")
def bin_returns_bulk(code: str):
    """Confirm or dismiss every queued return for one bin in a single decision."""
    context = ctx()
    record = bins_service.get_bin(context.db, code)
    if record is None:
        raise NotFoundError(f"no bin {code}")
    action = request.form.get("action", "")
    if action not in {"confirm", "dismiss"}:
        raise ConflictError(f"unknown bulk return action {action!r}")

    resolve = (
        items_service.confirm_return if action == "confirm" else items_service.dismiss_return
    )
    entries = items_service.pending_returns(context.db, bin_id=int(record["id"]))
    for entry in entries:
        resolve(context.db, int(entry["entry_id"]))
    log.info("%sed %d queued return(s) for %s", action, len(entries), code)
    return redirect(url_for("inventory.bin_detail", code=code))


# -- locations ------------------------------------------------------------


@bp.get("/locations")
def location_list():
    context = ctx()
    return render_template(
        "locations.html",
        locations=locations_service.list_locations(context.db),
        next_code=locations_service.next_location_code(
            context.db,
            context.cfg.str_("labels.location_prefix", "LOC"),
            context.cfg.int_("labels.code_digits", 4),
        ),
    )


@bp.get("/locations/<code>")
def location_detail(code: str):
    context = ctx()
    location = locations_service.get_location(context.db, code)
    if location is None:
        raise NotFoundError(f"no location {code}")
    contained = [
        record
        for record in bins_service.list_bins(context.db)
        if record["location_id"] == location["id"]
    ]
    return render_template("location_detail.html", location=location, bins=contained)


@bp.post("/locations/<code>/update")
def location_update(code: str):
    locations_service.rename_location(
        ctx().db, code, name=request.form.get("name", ""), notes=request.form.get("notes", "")
    )
    return redirect(url_for("inventory.location_detail", code=code))


# -- loans ----------------------------------------------------------------


@bp.get("/loans")
def loan_list():
    context = ctx()
    return render_template(
        "loans.html",
        loans=loans_service.list_loans(context.db, include_closed=True),
        next_code=loans_service.next_loan_code(
            context.db,
            context.cfg.str_("labels.loan_prefix", "LOAN"),
            context.cfg.int_("labels.code_digits", 4),
        ),
    )


@bp.get("/loans/<code>")
def loan_detail(code: str):
    context = ctx()
    loan = loans_service.get_loan(context.db, code)
    if loan is None:
        raise NotFoundError(f"no loan {code}")
    photo = None
    if loan["photo_id"]:
        photo = photos_service.get_photo(context.db, int(loan["photo_id"]))
    return render_template(
        "loan_detail.html",
        loan=loan,
        items=loans_service.loan_items(context.db, int(loan["id"])),
        photo=photo,
    )


@bp.post("/loans/create")
def loan_create():
    context = ctx()
    code = request.form.get("code", "").strip().upper() or loans_service.next_loan_code(
        context.db,
        context.cfg.str_("labels.loan_prefix", "LOAN"),
        context.cfg.int_("labels.code_digits", 4),
    )
    loan = loans_service.create_loan(
        context.db,
        person_name=request.form.get("person_name", ""),
        code=code,
        notes=request.form.get("notes", ""),
    )
    item_ids = [int(value) for value in request.form.getlist("item_id") if value.isdigit()]
    if item_ids:
        loans_service.assign_items(context.db, int(loan["id"]), item_ids)
    return redirect(url_for("inventory.loan_detail", code=loan["code"]))


@bp.post("/loans/<code>/assign")
def loan_assign(code: str):
    context = ctx()
    loan = loans_service.get_loan(context.db, code)
    if loan is None:
        raise NotFoundError(f"no loan {code}")
    item_ids = [int(value) for value in request.form.getlist("item_id") if value.isdigit()]
    loans_service.assign_items(context.db, int(loan["id"]), item_ids)
    return redirect(url_for("inventory.loan_detail", code=code))


@bp.post("/loans/<code>/photo")
def loan_photo(code: str):
    context = ctx()
    loan = loans_service.get_loan(context.db, code)
    if loan is None:
        raise NotFoundError(f"no loan {code}")
    upload = request.files.get("photo")
    if upload is None or not upload.filename:
        raise ConflictError("no photo was attached")
    suffix = Path(upload.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        upload.save(handle.name)
        staged = Path(handle.name)
    try:
        photo = photos_service.ingest(
            context.db,
            source=staged,
            photos_root=paths.photos_dir(context.cfg.data_dir),
            kind="loan_handover",
            loan_id=int(loan["id"]),
            max_dimension=context.cfg.int_("images.max_dimension", 2048),
            jpeg_quality=context.cfg.int_("images.jpeg_quality", 88),
        )
    finally:
        staged.unlink(missing_ok=True)
    loans_service.attach_photo(context.db, int(loan["id"]), int(photo["id"]))
    return redirect(url_for("inventory.loan_detail", code=code))


@bp.post("/loans/<code>/close")
def loan_close(code: str):
    context = ctx()
    loan = loans_service.get_loan(context.db, code)
    if loan is None:
        raise NotFoundError(f"no loan {code}")
    loans_service.close_loan(context.db, int(loan["id"]))
    return redirect(url_for("inventory.loan_list"))
