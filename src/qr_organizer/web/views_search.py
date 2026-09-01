"""Keyword search, visual search, and the pull list.

Search is keyword-first on purpose: the point of the app is finding a thing
without knowing where it is. Location is a sortable column, never a required
input.

In-use and loaned items stay in the results with a status badge rather than
being hidden. Knowing that the 10mm socket exists but Dave has it is almost
always the answer you wanted; hiding it makes the app look like the item
vanished. A filter narrows to in-bin only when you specifically want a list of
things you can actually go and fetch.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from flask import Blueprint, redirect, render_template, request, url_for

from ..errors import EmbeddingError
from ..imaging import open_normalised
from ..services import pulllist
from .helpers import ctx, device_key

log = logging.getLogger(__name__)

bp = Blueprint("search", __name__)

SORT_KEYS = {
    "relevance": lambda hit: -hit.score,
    "label": lambda hit: hit.label.lower(),
    "location": lambda hit: (hit.location_name.lower() or "zzz", hit.container_label.lower()),
    "bin": lambda hit: hit.container_label.lower(),
    "status": lambda hit: hit.status,
}


@bp.get("/search")
def search():
    context = ctx()
    query = request.args.get("q", "").strip()
    sort = request.args.get("sort", "relevance")
    default_include = context.cfg.bool_("search.include_in_use_by_default", True)
    include_in_use = request.args.get("in_use", "1" if default_include else "0") != "0"
    limit = context.cfg.int_("search.default_limit", 50)

    hits = context.search.keyword(query, limit=limit, include_in_use=include_in_use)
    if sort in SORT_KEYS:
        hits = sorted(hits, key=SORT_KEYS[sort])

    total, done = pulllist.counts(context.db, device_key())
    in_list = {
        str(entry["item_id"]) for entry in pulllist.entries(context.db, device_key())
    }
    return render_template(
        "search.html",
        query=query,
        hits=hits,
        sort=sort,
        include_in_use=include_in_use,
        pull_total=total,
        pull_done=done,
        in_list=in_list,
        sources=context.search.sources,
    )


@bp.post("/search/visual")
def visual_search():
    """Find an item by photographing it -- the same embedding space as the RAG."""
    context = ctx()
    upload = request.files.get("photo")
    if upload is None or not upload.filename:
        return redirect(url_for("search.search"))
    suffix = Path(upload.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        upload.save(handle.name)
        staged = Path(handle.name)
    try:
        vector = context.embedder.embed_image(open_normalised(staged))
    except EmbeddingError as exc:
        log.error("visual search unavailable: %s", exc)
        return render_template(
            "search.html", query="", hits=[], sort="relevance", include_in_use=True,
            pull_total=0, pull_done=0, in_list=set(), sources=context.search.sources,
            error=f"Visual search needs the embedding backend: {exc}",
        ), 503
    finally:
        staged.unlink(missing_ok=True)

    hits = context.search.vector(vector, limit=context.cfg.int_("search.default_limit", 50),
                                 min_score=0.4)
    total, done = pulllist.counts(context.db, device_key())
    return render_template(
        "search.html",
        query="(photo)",
        hits=hits,
        sort="relevance",
        include_in_use=True,
        pull_total=total,
        pull_done=done,
        in_list=set(),
        sources=context.search.sources,
        visual=True,
    )


# -- pull list ------------------------------------------------------------


@bp.get("/pull")
def pull_list():
    context = ctx()
    total, done = pulllist.counts(context.db, device_key())
    return render_template(
        "pull.html",
        entries=pulllist.entries(context.db, device_key()),
        pull_total=total,
        pull_done=done,
    )


@bp.post("/pull/add")
def pull_add():
    pulllist.add(ctx().db, device_key(), int(request.form["item_id"]))
    return redirect(request.form.get("next") or url_for("search.search"))


@bp.post("/pull/remove")
def pull_remove():
    pulllist.remove(ctx().db, device_key(), int(request.form["item_id"]))
    return redirect(request.form.get("next") or url_for("search.pull_list"))


@bp.post("/pull/check")
def pull_check():
    """Physically retrieved: the item becomes 'in use' until it is scanned back in."""
    pulllist.check_off(ctx().db, device_key(), int(request.form["item_id"]))
    return redirect(request.form.get("next") or url_for("search.pull_list"))


@bp.post("/pull/uncheck")
def pull_uncheck():
    pulllist.uncheck(ctx().db, device_key(), int(request.form["item_id"]))
    return redirect(request.form.get("next") or url_for("search.pull_list"))


@bp.post("/pull/clear")
def pull_clear():
    pulllist.clear(
        ctx().db, device_key(), only_checked=request.form.get("only_checked") == "1"
    )
    return redirect(url_for("search.pull_list"))
