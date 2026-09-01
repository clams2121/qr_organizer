"""TOML configuration: first-run creation, schema migration, typed access.

Two rules drive this module:

* A user's `config.toml` is never overwritten. New releases *merge* their new
  fields in with default values and record that they did so.
* Missing config that cannot be created is fatal -- there is nothing
  meaningful to run without it (project standards, section 8).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

from . import paths
from .errors import ConfigError, FatalConfigError

log = logging.getLogger(__name__)

_SENTINEL = object()


@dataclass
class SchemaState:
    """Which config fields arrived via migration and whether the user has seen them."""

    added_fields: list[str] = field(default_factory=list)
    acknowledged: bool = False
    migrated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "added_fields": self.added_fields,
            "acknowledged": self.acknowledged,
            "migrated_at": self.migrated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SchemaState":
        return cls(
            added_fields=list(raw.get("added_fields") or []),
            acknowledged=bool(raw.get("acknowledged")),
            migrated_at=raw.get("migrated_at"),
        )


@dataclass
class Config:
    """Loaded configuration plus the metadata the config web page needs."""

    data: dict[str, Any]
    path: Path
    defaults: dict[str, Any]
    schema_state: SchemaState
    created_now: bool = False

    # -- access ----------------------------------------------------------
    def get(self, dotted: str, default: Any = _SENTINEL) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _SENTINEL:
                    raise ConfigError(f"missing config key: {dotted}")
                return default
            node = node[part]
        return node

    def str_(self, dotted: str, default: Any = _SENTINEL) -> str:
        return str(self.get(dotted, default))

    def int_(self, dotted: str, default: Any = _SENTINEL) -> int:
        return int(self.get(dotted, default))

    def float_(self, dotted: str, default: Any = _SENTINEL) -> float:
        return float(self.get(dotted, default))

    def bool_(self, dotted: str, default: Any = _SENTINEL) -> bool:
        value = self.get(dotted, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def path_(self, dotted: str, default: Any = _SENTINEL) -> Path:
        return Path(self.str_(dotted, default)).expanduser()

    # -- derived locations ------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.path_("app.data_dir")

    @property
    def log_dir(self) -> Path:
        return self.path_("app.log_dir")

    # -- migration metadata ----------------------------------------------
    def is_new_field(self, dotted: str) -> bool:
        return dotted in self.schema_state.added_fields and not self.schema_state.acknowledged

    def is_default_value(self, dotted: str) -> bool:
        """True when the live value is still exactly what the template ships."""
        default = _lookup(self.defaults, dotted, _SENTINEL)
        if default is _SENTINEL:
            return False
        return self.get(dotted, _SENTINEL) == default

    def acknowledge_schema(self) -> None:
        self.schema_state.acknowledged = True
        _write_schema_state(self.data_dir, self.schema_state)

    # -- persistence ------------------------------------------------------
    def replace(self, new_data: dict[str, Any]) -> None:
        """Persist an edited config (from the web form) atomically."""
        _atomic_write_toml(self.path, new_data)
        self.data = new_data
        log.info("configuration rewritten via web form: %s", self.path)


def _lookup(tree: dict[str, Any], dotted: str, default: Any) -> Any:
    node: Any = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _merge_defaults(
    defaults: dict[str, Any], user: dict[str, Any], prefix: str = ""
) -> tuple[dict[str, Any], list[str]]:
    """Fill in fields the user's config predates. Returns (merged, added_paths)."""
    merged: dict[str, Any] = {}
    added: list[str] = []
    for key, default_value in defaults.items():
        dotted = f"{prefix}{key}"
        if key not in user:
            merged[key] = default_value
            if isinstance(default_value, dict):
                added.extend(_flatten_keys(default_value, f"{dotted}."))
            else:
                added.append(dotted)
        elif isinstance(default_value, dict) and isinstance(user[key], dict):
            sub_merged, sub_added = _merge_defaults(default_value, user[key], f"{dotted}.")
            merged[key] = sub_merged
            added.extend(sub_added)
        else:
            merged[key] = user[key]
    # Keys the user added that the template no longer ships are kept untouched.
    for key, value in user.items():
        if key not in merged:
            merged[key] = value
    return merged, added


def _flatten_keys(tree: dict[str, Any], prefix: str) -> list[str]:
    out: list[str] = []
    for key, value in tree.items():
        if isinstance(value, dict):
            out.extend(_flatten_keys(value, f"{prefix}{key}."))
        else:
            out.append(f"{prefix}{key}")
    return out


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise FatalConfigError(f"{path} is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise FatalConfigError(f"cannot read {path}: {exc}") from exc


def _atomic_write_toml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        tomli_w.dump(data, handle)
    os.replace(tmp, path)


def _read_schema_state(data_dir: Path) -> SchemaState:
    state_path = paths.schema_state_path(data_dir)
    if not state_path.is_file():
        return SchemaState()
    try:
        return SchemaState.from_dict(json.loads(state_path.read_text()))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("ignoring unreadable schema state %s: %s", state_path, exc)
        return SchemaState()


def _write_schema_state(data_dir: Path, state: SchemaState) -> None:
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        paths.schema_state_path(data_dir).write_text(json.dumps(state.to_dict(), indent=2))
    except OSError as exc:
        log.warning("could not persist schema state: %s", exc)


def load_config(*, create_if_missing: bool = True) -> Config:
    """Load config, creating it from the template on first run and migrating it."""
    template_path = paths.default_config_path()
    if not template_path.is_file():
        raise FatalConfigError(
            f"default config template missing at {template_path}; the install is incomplete"
        )
    defaults = _read_toml(template_path)

    target = paths.config_path()
    created_now = False
    if not target.is_file():
        if not create_if_missing:
            raise FatalConfigError(
                f"no configuration at {target}; run `qr-organizer --setup` to create one"
            )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(template_path, target)
        except OSError as exc:
            raise FatalConfigError(f"cannot create {target}: {exc}") from exc
        created_now = True
        log.warning("no config found -- created %s from the shipped defaults", target)

    user_data = _read_toml(target)
    merged, added = _merge_defaults(defaults, user_data)

    data_dir = Path(
        str(_lookup(merged, "app.data_dir", "~/.local/share/qr-organizer"))
    ).expanduser()
    state = _read_schema_state(data_dir)

    if created_now:
        state = SchemaState()
        _write_schema_state(data_dir, state)
    elif added:
        from datetime import datetime
        from datetime import timezone as _tz

        _atomic_write_toml(target, merged)
        state.added_fields = sorted(set(state.added_fields) | set(added))
        state.acknowledged = False
        state.migrated_at = datetime.now(_tz.utc).isoformat(timespec="seconds")
        _write_schema_state(data_dir, state)
        log.warning(
            "CONFIG MIGRATION: %d new field(s) added with defaults and written to %s: %s",
            len(added),
            target,
            ", ".join(added),
        )

    cfg = Config(
        data=merged,
        path=target,
        defaults=defaults,
        schema_state=state,
        created_now=created_now,
    )
    validate(cfg)
    return cfg


def validate(cfg: Config) -> list[str]:
    """Structural validation. Raises on fatal problems, returns non-fatal warnings."""
    problems: list[str] = []

    port = cfg.int_("server.port", 8815)
    if not 1 <= port <= 65535:
        raise FatalConfigError(f"server.port {port} is out of range")

    host = cfg.str_("server.host", "auto")
    if host == "0.0.0.0":  # noqa: S104 -- rejecting it is the whole point
        raise FatalConfigError(
            "server.host = '0.0.0.0' is refused: bind 127.0.0.1, a Tailscale address, or 'auto'"
        )

    backend = cfg.str_("vision.backend", "anthropic")
    if backend not in {"anthropic", "ollama"}:
        raise FatalConfigError(f"vision.backend must be 'anthropic' or 'ollama', got {backend!r}")

    embed_backend = cfg.str_("embeddings.backend", "clip")
    if embed_backend not in {"clip", "none"}:
        raise FatalConfigError(
            f"embeddings.backend must be 'clip' or 'none', got {embed_backend!r}"
        )

    timeout = cfg.int_("scanning.location_context_timeout_minutes", 30)
    if timeout <= 0:
        raise FatalConfigError("scanning.location_context_timeout_minutes must be positive")

    if cfg.float_("embeddings.match_threshold", 0.86) < cfg.float_(
        "embeddings.suggest_threshold", 0.78
    ):
        problems.append(
            "embeddings.match_threshold is below embeddings.suggest_threshold; "
            "no suggestion will ever reach auto-apply confidence"
        )

    if cfg.int_("images.thumbnail_size", 384) > cfg.int_("images.max_dimension", 2048):
        problems.append("images.thumbnail_size exceeds images.max_dimension")

    for message in problems:
        log.warning("config warning: %s", message)
    return problems
