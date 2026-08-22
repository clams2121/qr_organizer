"""Application wiring.

One `AppContext` owns every long-lived object -- database, vision backend,
embedder, search registry, pipeline, health monitor, registry writer -- and both
the CLI and the web app are built on top of it. Nothing else constructs these.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import APP_NAME, net, paths, registry, watchdog
from .config import Config
from .db import Database
from .embeddings import Embedder, build_embedder
from .health import HealthMonitor, disk_probe
from .pipeline import IdentificationPipeline
from .search import SearchRegistry
from .search.bins_source import BinInventorySource
from .vision import StructuredVisionBackend, build_backend

log = logging.getLogger(__name__)


@dataclass
class AppContext:
    cfg: Config
    db: Database
    embedder: Embedder
    vision: StructuredVisionBackend
    search: SearchRegistry
    bins_source: BinInventorySource
    pipeline: IdentificationPipeline
    health: HealthMonitor
    registry_writer: registry.RegistryWriter
    log_path: Path
    bind_host: str
    bind_explanation: str
    base_url: str
    warnings: list[str] = field(default_factory=list)
    _watchdog: watchdog.Watchdog | None = None

    # -- construction -----------------------------------------------------
    @classmethod
    def create(cls, cfg: Config, *, log_path: Path | None = None) -> "AppContext":
        data_dir = cfg.data_dir
        paths.ensure_dir(data_dir)
        paths.ensure_dir(paths.photos_dir(data_dir))
        paths.ensure_dir(paths.thumbnails_dir(data_dir))

        database = Database(paths.database_path(data_dir))
        database.initialise()

        embedder = build_embedder(cfg)
        vision = build_backend(cfg)

        bins_source = BinInventorySource(database)
        search_registry = SearchRegistry()
        search_registry.register(bins_source)

        pipeline = IdentificationPipeline(
            db=database,
            backend=vision,
            embedder=embedder,
            source=bins_source,
            thumbnails_root=paths.thumbnails_dir(data_dir),
            photos_root=paths.photos_dir(data_dir),
            config=cfg,
        )

        bind_host, explanation = net.resolve_bind_host(cfg.str_("server.host", "auto"))
        port = cfg.int_("server.port", 8815)
        base_url = net.public_base_url(cfg.str_("server.base_url", ""), bind_host, port)

        writer = registry.RegistryWriter(
            directory=cfg.path_("registry.dir", "/var/log/service-registry"),
            name=APP_NAME,
            host=bind_host,
            port=port,
            base_url=base_url,
            log_path=str(log_path or ""),
            refresh_minutes=cfg.int_("registry.refresh_minutes", 7),
        )

        context = cls(
            cfg=cfg,
            db=database,
            embedder=embedder,
            vision=vision,
            search=search_registry,
            bins_source=bins_source,
            pipeline=pipeline,
            health=HealthMonitor(),
            registry_writer=writer,
            log_path=log_path or Path(""),
            bind_host=bind_host,
            bind_explanation=explanation,
            base_url=base_url,
        )
        context._register_health_checks()
        return context

    # -- health -----------------------------------------------------------
    def _register_health_checks(self) -> None:
        self.health.register("database", self.db.health_check, critical=True)
        self.health.register("vision", self.vision.status)
        self.health.register("embeddings", self.embedder.status)
        self.health.register("search", self._search_probe)
        self.health.register("storage", lambda: disk_probe(self.cfg.data_dir))
        self.health.register("service_registry", self._registry_probe)
        self.health.register("config", self._config_probe)

    def _search_probe(self) -> tuple[bool, str]:
        results = self.search.health()
        broken = [f"{sid}: {detail}" for sid, (ok, detail) in results.items() if not ok]
        if broken:
            return False, "; ".join(broken)
        return True, "; ".join(f"{sid}: {detail}" for sid, (_, detail) in results.items())

    def _registry_probe(self) -> tuple[bool, str]:
        directory = self.registry_writer.directory
        if not directory.is_dir():
            # No aggregator installed on this host is a normal state, not a fault.
            return True, f"{directory} absent (no status aggregator on this host)"
        if self.registry_writer.last_error:
            return False, self.registry_writer.last_error
        return True, f"last written {self.registry_writer.last_write}"

    def _config_probe(self) -> tuple[bool, str]:
        state = self.cfg.schema_state
        if state.added_fields and not state.acknowledged:
            return False, (
                f"{len(state.added_fields)} config field(s) added by a schema migration on "
                f"{state.migrated_at} are still using defaults and unacknowledged"
            )
        if self.warnings:
            return False, "; ".join(self.warnings)
        return True, f"loaded from {self.cfg.path}"

    def liveness(self) -> bool:
        """What the systemd watchdog asks: is the critical path still working?"""
        ok, _ = self.db.health_check()
        return ok

    # -- lifecycle --------------------------------------------------------
    def start_background(self) -> None:
        self.registry_writer.start()
        interval = watchdog.watchdog_interval_seconds()
        if interval and self.cfg.bool_("health.watchdog", True):
            self._watchdog = watchdog.Watchdog(self.liveness, interval)
            self._watchdog.start()
        watchdog.ready(f"listening on {self.bind_host}:{self.cfg.int_('server.port', 8815)}")

    def shutdown(self) -> None:
        watchdog.stopping()
        if self._watchdog is not None:
            self._watchdog.stop()
        self.registry_writer.stop()
        self.db.close()
        log.info("shut down cleanly")
