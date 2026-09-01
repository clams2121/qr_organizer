"""Command line entry point.

`qr-organizer` with no arguments starts the server. The other flags are the
operational ones: create the config, check the config, print a sheet without a
browser, and rebuild the embedding library after changing the model.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from . import APP_NAME, __version__, logging_setup, net, paths, registry
from .config import load_config
from .errors import FatalConfigError, QROrganizerError
from .runtime import AppContext

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description="Self-hosted QR-code storage inventory.",
    )
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    parser.add_argument(
        "--setup", action="store_true",
        help="create the config, data directories and database, then exit",
    )
    parser.add_argument(
        "--validate-config", action="store_true",
        help="check the config (and the backends it names) and exit non-zero on a problem",
    )
    parser.add_argument(
        "--rebuild-embeddings", action="store_true",
        help="recompute every item's visual fingerprint; needed after changing "
             "embeddings.model",
    )
    parser.add_argument(
        "--print-sheet", type=int, metavar="N",
        help="reserve N bin codes and write a printable PDF to --output",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("labels.pdf"),
        help="where --print-sheet writes its PDF (default: labels.pdf)",
    )
    parser.add_argument("--host", help="override server.host for this run")
    parser.add_argument("--port", type=int, help="override server.port for this run")
    parser.add_argument(
        "--debug", action="store_true", help="Flask debug mode; never in production"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging_setup.bootstrap_stderr_logging()

    try:
        cfg = load_config(create_if_missing=True)
    except FatalConfigError as exc:
        # Section 8: no usable config means there is nothing meaningful to run.
        logging.getLogger(__name__).critical("FATAL: %s", exc)
        registry.write_failure_breadcrumb(
            Path("/var/log/service-registry"), f"config load failed: {exc}"
        )
        return 78  # EX_CONFIG

    log_path = logging_setup.configure(
        log_dir=cfg.log_dir,
        data_dir=cfg.data_dir,
        level=cfg.str_("app.log_level", "INFO"),
        max_bytes=cfg.int_("app.log_max_bytes", 10 * 1024 * 1024),
        backup_count=cfg.int_("app.log_backup_count", 5),
    )
    log.info("%s %s starting", APP_NAME, __version__)

    if args.host:
        cfg.data.setdefault("server", {})["host"] = args.host
    if args.port:
        cfg.data.setdefault("server", {})["port"] = args.port

    try:
        context = AppContext.create(cfg, log_path=log_path)
    except QROrganizerError as exc:
        log.critical("could not start: %s", exc)
        registry.write_failure_breadcrumb(
            cfg.path_("registry.dir", "/var/log/service-registry"), f"startup failed: {exc}"
        )
        return 1

    if args.setup:
        return _setup(context)
    if args.validate_config:
        return _validate(context)
    if args.rebuild_embeddings:
        return _rebuild_embeddings(context)
    if args.print_sheet:
        return _print_sheet(context, args.print_sheet, args.output)
    return _serve(context, debug=args.debug)


def _setup(context: AppContext) -> int:
    cfg = context.cfg
    print(f"config:      {cfg.path}{' (created now)' if cfg.created_now else ''}")
    print(f"data:        {cfg.data_dir}")
    print(f"photos:      {paths.photos_dir(cfg.data_dir)}")
    print(f"thumbnails:  {paths.thumbnails_dir(cfg.data_dir)}")
    print(f"database:    {paths.database_path(cfg.data_dir)}")
    print(f"log file:    {context.log_path}")
    print(f"binding:     {context.bind_host} ({context.bind_explanation})")
    print(f"label URLs:  {context.base_url}/s/<CODE>")
    print()
    print("Next: set the API credential (see the README's secrets section), then run")
    print(f"  {APP_NAME} --validate-config")
    context.shutdown()
    return 0


def _validate(context: AppContext) -> int:
    report = context.health.report()
    print(f"config:  {context.cfg.path}")
    print(f"status:  {report.status}")
    worst = 0
    for check in report.checks:
        mark = "ok  " if check.ok else ("ERROR" if check.critical else "warn")
        print(f"  [{mark}] {check.name}: {check.detail}")
        if not check.ok:
            worst = max(worst, 2 if check.critical else 1)
    if context.cfg.schema_state.added_fields and not context.cfg.schema_state.acknowledged:
        print(
            f"\nconfig migration: {len(context.cfg.schema_state.added_fields)} new field(s) "
            f"are using defaults. Review them at {context.base_url}/config"
        )
    context.shutdown()
    return worst


def _rebuild_embeddings(context: AppContext) -> int:
    if not context.embedder.available:
        ok, detail = context.embedder.status()
        print(f"embedding backend unavailable: {detail}", file=sys.stderr)
        context.shutdown()
        return 1
    rows = context.db.query("SELECT id FROM items WHERE thumbnail_path != ''")
    print(f"re-embedding {len(rows)} item(s) with {context.embedder.name}...")
    context.db.set_meta("embedding_dim", str(context.embedder.dim))
    context.db.ensure_vector_index(context.embedder.dim)
    done = failed = 0
    for row in rows:
        try:
            if context.pipeline.reembed_item(int(row["id"])):
                done += 1
            else:
                failed += 1
        except QROrganizerError as exc:
            log.error("item %s could not be re-embedded: %s", row["id"], exc)
            failed += 1
    print(f"done: {done} re-embedded, {failed} skipped or failed")
    context.shutdown()
    return 1 if failed else 0


def _print_sheet(context: AppContext, count: int, output: Path) -> int:
    from . import labels
    from .services import bins as bins_service
    from .util import now_iso

    sheet_id = f"bin-{now_iso().replace(':', '').replace('-', '')}"
    codes = bins_service.reserve_codes(
        context.db,
        kind="bin",
        prefix=context.cfg.str_("labels.bin_prefix", "BIN"),
        count=count,
        digits=context.cfg.int_("labels.code_digits", 4),
        sheet_id=sheet_id,
    )
    pdf = labels.render_sheet(
        codes,
        base_url=context.base_url,
        title=f"QR Organizer -- {codes[0]}..{codes[-1]}",
        page_size=context.cfg.str_("labels.sheet_page_size", "letter"),
        columns=context.cfg.int_("labels.sheet_columns", 3),
        rows=context.cfg.int_("labels.sheet_rows", 8),
    )
    output.write_bytes(pdf)
    print(f"wrote {output} with {len(codes)} labels ({codes[0]}..{codes[-1]})")
    print(f"they point at {context.base_url}/s/<CODE>")
    context.shutdown()
    return 0


def _serve(context: AppContext, *, debug: bool = False) -> int:
    from .web.app import create_app

    app = create_app(context)
    host = context.bind_host
    port = context.cfg.int_("server.port", 8815)

    def _terminate(signum, _frame):
        log.info("received signal %s, shutting down", signum)
        context.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)

    context.start_background()
    log.info("serving on http://%s:%d -- %s", host, port, context.bind_explanation)
    if host == net.LOOPBACK:
        log.info("reach it with: ssh -L %d:localhost:%d %s", port, port, net.local_hostname())

    report = context.health.report()
    if report.status != "ok":
        for check in report.checks:
            if not check.ok:
                log.warning("starting %s: %s -- %s", report.status, check.name, check.detail)

    try:
        app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
    finally:
        context.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
