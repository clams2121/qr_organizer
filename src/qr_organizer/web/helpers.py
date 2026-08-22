"""Request-scoped helpers shared by the view modules."""

from __future__ import annotations

import uuid
from typing import Any

from flask import current_app, g, request

from ..runtime import AppContext
from ..services import scans

DEVICE_COOKIE = "qro_device"


def ctx() -> AppContext:
    return current_app.config["APP_CONTEXT"]


def device_key() -> str:
    """A stable per-browser identifier.

    Not authentication -- there is deliberately none. It exists so the active
    location context and the pull list belong to the phone in your hand rather
    than to the whole household at once.
    """
    key = getattr(g, "device_key", None)
    if key:
        return key
    key = request.cookies.get(DEVICE_COOKIE) or uuid.uuid4().hex
    g.device_key = key
    return key


def client_ip() -> str:
    context = ctx()
    if context.cfg.bool_("server.trust_proxy_headers", False):
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.remote_addr or ""


def device_label() -> str:
    """Best-effort human name for whoever is scanning, via Tailscale."""
    label = getattr(g, "device_label", None)
    if label is not None:
        return label
    label = scans.device_name_for_ip(client_ip())
    g.device_label = label
    return label


def log_scan(**kwargs: Any) -> None:
    scans.record(
        ctx().db,
        device_key=device_key(),
        device_ip=client_ip(),
        device_name=device_label(),
        user_agent=request.headers.get("User-Agent", ""),
        **kwargs,
    )


def wants_json() -> bool:
    return (
        request.accept_mimetypes.best == "application/json"
        or request.headers.get("X-Requested-With") == "fetch"
    )
