"""Serving stored photos and thumbnails from the data directory."""

from __future__ import annotations

from flask import Blueprint, abort, send_from_directory

from .. import paths
from .helpers import ctx

bp = Blueprint("media", __name__, url_prefix="/media")


@bp.get("/photos/<path:relative>")
def photo(relative: str):
    return _send(paths.photos_dir(ctx().cfg.data_dir), relative)


@bp.get("/thumbnails/<path:relative>")
def thumbnail(relative: str):
    return _send(paths.thumbnails_dir(ctx().cfg.data_dir), relative)


def _send(root, relative: str):
    if ".." in relative:
        abort(404)
    if not (root / relative).is_file():
        abort(404)
    return send_from_directory(root, relative, max_age=86400)
