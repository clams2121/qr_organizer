"""Flask application factory.

The config form and the status/error log live in this same app and share its
navigation, so a config mistake and the runtime error it caused are one click
apart.
"""

from __future__ import annotations

import logging

from flask import Flask, g, make_response, request

from ..errors import (
    ConflictError,
    LocationContextExpired,
    NotFoundError,
    QROrganizerError,
)
from ..runtime import AppContext
from ..util import humanise_age
from .helpers import DEVICE_COOKIE, device_key

log = logging.getLogger(__name__)

#: Photos from a modern phone camera; anything larger is a mistake, not a bin.
MAX_UPLOAD_BYTES = 40 * 1024 * 1024


def create_app(context: AppContext) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["APP_CONTEXT"] = context
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 3600
    app.jinja_env.trim_blocks = True
    app.jinja_env.lstrip_blocks = True
    app.jinja_env.filters["age"] = humanise_age

    from . import views_admin, views_inventory, views_media, views_scan, views_search

    app.register_blueprint(views_scan.bp)
    app.register_blueprint(views_inventory.bp)
    app.register_blueprint(views_search.bp)
    app.register_blueprint(views_admin.bp)
    app.register_blueprint(views_media.bp)

    @app.before_request
    def _bind_device() -> None:
        device_key()

    @app.after_request
    def _persist_device(response):
        key = getattr(g, "device_key", None)
        if key and request.cookies.get(DEVICE_COOKIE) != key:
            response.set_cookie(
                DEVICE_COOKIE, key, max_age=60 * 60 * 24 * 730, samesite="Lax", httponly=True
            )
        return response

    @app.context_processor
    def _globals() -> dict:
        from .views_scan import current_location_banner

        return {
            "app_context": context,
            "active_location": current_location_banner(),
            "nav_path": request.path,
        }

    @app.errorhandler(NotFoundError)
    def _not_found(exc: NotFoundError):
        return _error_response(exc, 404)

    @app.errorhandler(ConflictError)
    def _conflict(exc: ConflictError):
        return _error_response(exc, 409)

    @app.errorhandler(LocationContextExpired)
    def _expired(exc: LocationContextExpired):
        return _error_response(exc, 409)

    @app.errorhandler(QROrganizerError)
    def _app_error(exc: QROrganizerError):
        log.error("request %s failed: %s", request.path, exc)
        return _error_response(exc, 500)

    return app


def _error_response(exc: Exception, status: int):
    from flask import jsonify, render_template

    from .helpers import wants_json

    if wants_json() or request.path.startswith("/api/"):
        return jsonify({"error": str(exc), "type": type(exc).__name__}), status
    response = make_response(
        render_template("error.html", message=str(exc), kind=type(exc).__name__, status=status),
        status,
    )
    return response
