"""Filesystem layout.

Everything the app writes lives under one of three roots: the config dir, the
data dir (database + media), and the log dir. Each is resolved once, here.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import APP_NAME

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent


def _xdg(var: str, fallback: str) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser() if raw else Path(fallback).expanduser()


def config_dir() -> Path:
    """`~/.config/qr-organizer` (honours XDG_CONFIG_HOME)."""
    return _xdg("XDG_CONFIG_HOME", "~/.config") / APP_NAME


def config_path() -> Path:
    override = os.environ.get("QR_ORGANIZER_CONFIG")
    if override:
        return Path(override).expanduser()
    return config_dir() / "config.toml"


def default_config_path() -> Path:
    """The shipped `config.default.toml`, whether installed or run from a checkout."""
    candidates = [
        PACKAGE_ROOT / "config.default.toml",
        REPO_ROOT / "config.default.toml",
        Path("/usr/share") / APP_NAME / "config.default.toml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def media_dir(data_dir: Path) -> Path:
    return data_dir / "media"


def photos_dir(data_dir: Path) -> Path:
    return media_dir(data_dir) / "photos"


def thumbnails_dir(data_dir: Path) -> Path:
    return media_dir(data_dir) / "thumbnails"


def database_path(data_dir: Path) -> Path:
    return data_dir / "inventory.db"


def schema_state_path(data_dir: Path) -> Path:
    """Tracks which config fields arrived in a migration and are unacknowledged."""
    return data_dir / "schema_state.json"


def ensure_dir(path: Path, mode: int = 0o755) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=mode)
    return path
