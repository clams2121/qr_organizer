"""Config form, status/error page, health endpoint, and label printing.

The config form and the status page share this blueprint and the app's
navigation deliberately: when a config change breaks something, the error it
caused is one click away.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from flask import Blueprint, Response, jsonify, redirect, render_template, request, url_for

from .. import labels as labels_module
from .. import net, secrets
from ..config import validate
from ..errors import ConfigError, ConflictError
from ..logging_setup import RING
from ..services import bins as bins_service
from ..services import items as items_service
from ..services import scans
from ..util import now_iso, slugify
from .helpers import ctx

log = logging.getLogger(__name__)

bp = Blueprint("admin", __name__)

#: Config values the form must never round-trip as editable text.
READ_ONLY_KEYS = {"app.data_dir"}


@bp.get("/health")
def health():
    """Readiness: can this service do its job right now?"""
    report = ctx().health.report()
    return jsonify(report.to_dict()), report.http_status


@bp.get("/status")
def status():
    context = ctx()
    report = context.health.report()
    level = request.args.get("level", "INFO").upper()
    tailscale_ip = net.detect_tailscale_ip()
    return render_template(
        "status.html",
        report=report,
        records=RING.snapshot(min_level=level, limit=250),
        level=level,
        error_count=RING.count_at_least("ERROR"),
        warning_count=RING.count_at_least("WARNING"),
        log_path=context.log_path,
        bind_host=context.bind_host,
        bind_explanation=context.bind_explanation,
        base_url=context.base_url,
        tailscale_ip=tailscale_ip,
        tailscale_host=net.tailscale_hostname(),
        registry=context.registry_writer,
        recent_scans=scans.recent(context.db, limit=25),
        sessions=[
            dict(row)
            for row in context.db.query(
                "SELECT inventory_sessions.*, bins.code AS bin_code FROM inventory_sessions "
                "LEFT JOIN bins ON bins.id = inventory_sessions.bin_id "
                "ORDER BY inventory_sessions.id DESC LIMIT 15"
            )
        ],
        stats=items_service.embedding_stats(context.db),
        secret_sources=secrets.describe_sources(
            context.cfg.str_("vision.anthropic.credential_name", "anthropic_api_key"),
            context.cfg.str_("vision.anthropic.api_key_env", "ANTHROPIC_API_KEY"),
        ),
    )


# -- config form ----------------------------------------------------------


def _flatten(tree: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    fields: list[tuple[str, Any]] = []
    for key, value in tree.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            fields.extend(_flatten(value, f"{dotted}."))
        else:
            fields.append((dotted, value))
    return fields


def _sections(cfg) -> list[dict[str, Any]]:
    """Group config into sections, marking migration state per field."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for dotted, value in _flatten(cfg.data):
        section = dotted.rsplit(".", 1)[0] if "." in dotted else "general"
        grouped.setdefault(section, []).append(
            {
                "key": dotted,
                "name": dotted.rsplit(".", 1)[-1],
                "value": value,
                "type": type(value).__name__,
                "is_new": cfg.is_new_field(dotted),
                "is_default": cfg.is_default_value(dotted),
                "read_only": dotted in READ_ONLY_KEYS,
            }
        )
    return [
        {
            "name": name,
            "fields": fields,
            "has_new": any(field["is_new"] for field in fields),
        }
        for name, fields in sorted(grouped.items())
    ]


@bp.get("/config")
def config_form():
    context = ctx()
    return render_template(
        "config.html",
        sections=_sections(context.cfg),
        cfg=context.cfg,
        schema_state=context.cfg.schema_state,
        rotation_allowed=context.cfg.bool_("secrets.allow_web_rotation", True),
        credential_name=context.cfg.str_("vision.anthropic.credential_name", "anthropic_api_key"),
        secret_sources=secrets.describe_sources(
            context.cfg.str_("vision.anthropic.credential_name", "anthropic_api_key"),
            context.cfg.str_("vision.anthropic.api_key_env", "ANTHROPIC_API_KEY"),
        ),
    )


def _coerce(raw: str, original: Any) -> Any:
    """Parse a form value back to the type the config field already had."""
    if isinstance(original, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(original, int):
        try:
            return int(raw.strip())
        except ValueError as exc:
            raise ConfigError(f"{raw!r} is not a whole number") from exc
    if isinstance(original, float):
        try:
            return float(raw.strip())
        except ValueError as exc:
            raise ConfigError(f"{raw!r} is not a number") from exc
    return raw


def _assign(tree: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


@bp.post("/config")
def config_save():
    context = ctx()
    updated = copy.deepcopy(context.cfg.data)
    errors: list[str] = []

    for dotted, original in _flatten(context.cfg.data):
        if dotted in READ_ONLY_KEYS:
            continue
        if isinstance(original, bool):
            _assign(updated, dotted, f"field:{dotted}" in request.form)
            continue
        if f"field:{dotted}" not in request.form:
            continue
        try:
            _assign(updated, dotted, _coerce(request.form[f"field:{dotted}"], original))
        except ConfigError as exc:
            errors.append(f"{dotted}: {exc}")

    if errors:
        return render_template(
            "config.html",
            sections=_sections(context.cfg),
            cfg=context.cfg,
            schema_state=context.cfg.schema_state,
            errors=errors,
            rotation_allowed=context.cfg.bool_("secrets.allow_web_rotation", True),
            credential_name=context.cfg.str_(
                "vision.anthropic.credential_name", "anthropic_api_key"
            ),
            secret_sources=[],
        ), 400

    # Validate the candidate before it is written, so a bad edit cannot take
    # the service down on next start.
    from ..config import Config

    candidate = Config(
        data=updated,
        path=context.cfg.path,
        defaults=context.cfg.defaults,
        schema_state=context.cfg.schema_state,
    )
    try:
        warnings = validate(candidate)
    except ConfigError as exc:
        return render_template(
            "config.html",
            sections=_sections(context.cfg),
            cfg=context.cfg,
            schema_state=context.cfg.schema_state,
            errors=[str(exc)],
            rotation_allowed=context.cfg.bool_("secrets.allow_web_rotation", True),
            credential_name=context.cfg.str_(
                "vision.anthropic.credential_name", "anthropic_api_key"
            ),
            secret_sources=[],
        ), 400

    context.cfg.replace(updated)
    context.warnings = warnings
    log.warning(
        "configuration saved from the web form; restart the service for backend, "
        "binding and embedding-model changes to take effect"
    )
    return redirect(url_for("admin.config_form", saved=1))


@bp.post("/config/acknowledge")
def config_acknowledge():
    ctx().cfg.acknowledge_schema()
    log.info("schema migration acknowledged")
    return redirect(url_for("admin.config_form"))


@bp.post("/config/rotate-secret")
def rotate_secret():
    """Rotate the API credential through two narrow sudo rules. See deploy/."""
    context = ctx()
    if not context.cfg.bool_("secrets.allow_web_rotation", True):
        raise ConflictError("secrets.allow_web_rotation is off")
    new_value = request.form.get("value", "")
    try:
        message = secrets.rotate_systemd_credential(
            credential_name=context.cfg.str_(
                "vision.anthropic.credential_name", "anthropic_api_key"
            ),
            new_value=new_value,
            credentials_dir=context.cfg.path_("secrets.credentials_dir",
                                              "/etc/credstore.encrypted"),
            unit=context.cfg.str_("secrets.systemd_unit", "qr-organizer.service"),
        )
    except secrets.RotationError as exc:
        log.error("credential rotation failed: %s", exc)
        return render_template("rotated.html", ok=False, message=str(exc)), 500
    return render_template("rotated.html", ok=True, message=message)


# -- labels ---------------------------------------------------------------


@bp.get("/labels")
def label_home():
    context = ctx()
    return render_template(
        "labels.html",
        sheets=bins_service.list_sheets(context.db),
        base_url=context.base_url,
        columns=context.cfg.int_("labels.sheet_columns", 3),
        rows=context.cfg.int_("labels.sheet_rows", 8),
    )


@bp.post("/labels/sheet")
def label_sheet():
    """Mint and render a batch of sequential codes as a printable PDF."""
    context = ctx()
    kind = request.form.get("kind", "bin")
    if kind not in {"bin", "location"}:
        raise ConflictError(f"unknown label kind {kind!r}")
    count = int(request.form.get("count", "24") or 24)
    if not 1 <= count <= 500:
        raise ConflictError("count must be between 1 and 500")

    prefix = context.cfg.str_(
        "labels.bin_prefix" if kind == "bin" else "labels.location_prefix",
        "BIN" if kind == "bin" else "LOC",
    )
    sheet_id = f"{kind}-{now_iso().replace(':', '').replace('-', '')}"
    codes = bins_service.reserve_codes(
        context.db,
        kind=kind,
        prefix=prefix,
        count=count,
        digits=context.cfg.int_("labels.code_digits", 4),
        sheet_id=sheet_id,
    )
    pdf = labels_module.render_sheet(
        codes,
        base_url=context.base_url,
        title=f"QR Organizer -- {kind} labels {codes[0]}..{codes[-1]}",
        page_size=context.cfg.str_("labels.sheet_page_size", "letter"),
        columns=context.cfg.int_("labels.sheet_columns", 3),
        rows=context.cfg.int_("labels.sheet_rows", 8),
    )
    return Response(
        pdf,
        mimetype="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{slugify(sheet_id)}.pdf"'},
    )


@bp.get("/labels/sheets/<sheet_id>.pdf")
def reprint_sheet(sheet_id: str):
    """Reprint an existing sheet -- same codes, no new ones minted."""
    context = ctx()
    entries = bins_service.sheet_codes(context.db, sheet_id)
    if not entries:
        raise ConflictError(f"no sheet {sheet_id}")
    pdf = labels_module.render_sheet(
        [entry["code"] for entry in entries],
        base_url=context.base_url,
        title=f"QR Organizer -- reprint of {sheet_id}",
        page_size=context.cfg.str_("labels.sheet_page_size", "letter"),
        columns=context.cfg.int_("labels.sheet_columns", 3),
        rows=context.cfg.int_("labels.sheet_rows", 8),
    )
    return Response(
        pdf,
        mimetype="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{slugify(sheet_id)}.pdf"'},
    )


@bp.get("/labels/single/<code>.pdf")
def single_label(code: str):
    """One oversized placard, for a shed door or a shelf end."""
    context = ctx()
    pdf = labels_module.render_single(
        code.upper(),
        base_url=context.base_url,
        page_size=context.cfg.str_("labels.sheet_page_size", "letter"),
    )
    return Response(
        pdf,
        mimetype="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{slugify(code)}.pdf"'},
    )
