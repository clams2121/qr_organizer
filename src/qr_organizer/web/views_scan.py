"""Scanning: the camera page, code resolution, and the active location context."""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from ..errors import NotFoundError
from ..services import bins as bins_service
from ..services import items as items_service
from ..services import loans as loans_service
from ..services import locations as locations_service
from ..services import pulllist, scans
from ..util import extract_code, parse_code
from .helpers import ctx, device_key, log_scan

log = logging.getLogger(__name__)

bp = Blueprint("scan", __name__)


def _prefixes() -> dict[str, str]:
    cfg = ctx().cfg
    return {
        "bin": cfg.str_("labels.bin_prefix", "BIN").upper(),
        "location": cfg.str_("labels.location_prefix", "LOC").upper(),
        "loan": cfg.str_("labels.loan_prefix", "LOAN").upper(),
    }


def code_kind(code: str) -> str | None:
    parsed = parse_code(code)
    if parsed is None:
        return None
    prefix = parsed[0]
    for kind, expected in _prefixes().items():
        if prefix == expected:
            return kind
    return None


def current_location_banner() -> dict | None:
    """The live location context for this device, for the nav bar."""
    context = ctx()
    return locations_service.active_location(
        context.db,
        device_key(),
        context.cfg.int_("scanning.location_context_timeout_minutes", 30),
    )


@bp.get("/")
def home():
    context = ctx()
    total, done = pulllist.counts(context.db, device_key())
    return render_template(
        "home.html",
        bins=bins_service.list_bins(context.db)[:12],
        locations=locations_service.list_locations(context.db),
        review=items_service.list_needing_review(context.db, limit=8),
        pending_returns=items_service.pending_return_count(context.db),
        recent_scans=scans.recent(context.db, limit=8),
        pull_total=total,
        pull_done=done,
        stats=items_service.embedding_stats(context.db),
    )


@bp.get("/scan")
def scanner():
    """In-browser continuous scanner.

    Camera access needs a secure context. Over Tailscale that means running
    `tailscale serve`; the page detects an insecure context and tells the user
    exactly that instead of silently showing a dead viewfinder.
    """
    return render_template("scan.html", prefixes=_prefixes())


@bp.post("/api/scan/resolve")
def resolve():
    """Turn a raw QR payload into the page the scanner should jump to."""
    payload = (request.get_json(silent=True) or {}).get("payload", "")
    code = extract_code(payload)
    if not code:
        return jsonify({"ok": False, "error": f"not a QR Organizer code: {payload[:80]!r}"}), 400
    return jsonify({"ok": True, "code": code, "url": url_for("scan.resolve_code", code=code)})


@bp.get("/s/<code>")
def resolve_code(code: str):
    """The URL every printed label points at."""
    context = ctx()
    code = code.strip().upper()
    kind = code_kind(code)
    if kind is None:
        raise NotFoundError(
            f"{code} is not a recognised code. Expected one of the configured prefixes: "
            + ", ".join(sorted(_prefixes().values()))
        )

    if kind == "location":
        return _scan_location(code)
    if kind == "loan":
        loan = loans_service.get_loan(context.db, code)
        if loan is None:
            raise NotFoundError(f"no loan {code}")
        log_scan(kind="loan", code=code, detail="scanned")
        return redirect(url_for("inventory.loan_detail", code=code))
    return _scan_bin(code)


def _scan_location(code: str):
    context = ctx()
    location = locations_service.get_location(context.db, code)
    if location is None:
        # An unused location placard: offer to name it rather than inventing one.
        return render_template("location_new.html", code=code)
    locations_service.set_active_location(context.db, device_key(), int(location["id"]))
    log_scan(kind="location", code=code, location_id=int(location["id"]),
             detail="active location set")
    return redirect(url_for("inventory.location_detail", code=code))


def _scan_bin(code: str):
    context = ctx()
    existing = bins_service.get_bin(context.db, code)
    if existing is not None:
        # An update scan NEVER touches the bin's location -- §3 of the spec.
        log_scan(kind="bin", code=code, bin_id=int(existing["id"]), detail="scanned")
        active = current_location_banner()
        if active:
            locations_service.touch_active_location(context.db, device_key())
        return redirect(url_for("inventory.bin_detail", code=code))

    active = current_location_banner()
    return render_template(
        "bin_new.html",
        code=code,
        active=active,
        timeout=context.cfg.int_("scanning.location_context_timeout_minutes", 30),
        known_stock=bins_service.is_known_stock(context.db, code),
        locations=locations_service.list_locations(context.db),
    )


@bp.post("/bins/create")
def create_bin():
    """Claim a scanned code as a real bin."""
    context = ctx()
    code = request.form.get("code", "").strip().upper()
    label = request.form.get("label", "").strip()
    raw_location = request.form.get("location_id", "").strip()

    location_id: int | None = None
    if raw_location == "active":
        active = current_location_banner()
        if active is None:
            # The context expired between rendering the form and submitting it.
            return render_template(
                "bin_new.html",
                code=code,
                active=None,
                expired=True,
                timeout=context.cfg.int_("scanning.location_context_timeout_minutes", 30),
                known_stock=bins_service.is_known_stock(context.db, code),
                locations=locations_service.list_locations(context.db),
            ), 409
        location_id = int(active["location_id"])
        locations_service.touch_active_location(context.db, device_key())
    elif raw_location.isdigit():
        location_id = int(raw_location)

    created = bins_service.create_bin(context.db, code=code, label=label, location_id=location_id)
    log_scan(kind="bin_created", code=code, bin_id=int(created["id"]), location_id=location_id,
             detail="new bin")
    return redirect(url_for("inventory.bin_detail", code=code))


@bp.post("/locations/create")
def create_location():
    context = ctx()
    code = request.form.get("code", "").strip().upper()
    name = request.form.get("name", "").strip()
    notes = request.form.get("notes", "").strip()
    if not code:
        code = locations_service.next_location_code(
            context.db,
            context.cfg.str_("labels.location_prefix", "LOC"),
            context.cfg.int_("labels.code_digits", 4),
        )
    location = locations_service.create_location(context.db, name=name, code=code, notes=notes)
    locations_service.set_active_location(context.db, device_key(), int(location["id"]))
    log_scan(kind="location_created", code=code, location_id=int(location["id"]), detail=name)
    return redirect(url_for("inventory.location_detail", code=code))


@bp.post("/location/clear")
def clear_location():
    locations_service.clear_active_location(ctx().db, device_key())
    return redirect(request.referrer or url_for("scan.home"))
