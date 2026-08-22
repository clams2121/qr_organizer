"""Config creation, schema migration and validation."""

from __future__ import annotations

import tomllib

import pytest
import tomli_w

from qr_organizer import paths
from qr_organizer.config import load_config, validate
from qr_organizer.errors import FatalConfigError


def test_first_run_creates_config_from_template(env):
    cfg = load_config()
    assert cfg.created_now is True
    assert cfg.path.is_file()
    assert cfg.get("vision.backend") == "anthropic"
    assert cfg.schema_state.added_fields == []


def test_existing_config_is_never_overwritten(env):
    cfg = load_config()
    cfg.data["server"]["port"] = 9999
    cfg.replace(cfg.data)

    reloaded = load_config()
    assert reloaded.int_("server.port") == 9999
    assert reloaded.created_now is False


def test_migration_fills_new_fields_and_flags_them(env):
    cfg = load_config()
    # Simulate an older config that predates a whole section and one scalar.
    trimmed = dict(cfg.data)
    trimmed.pop("labels")
    trimmed["scanning"] = {}
    with cfg.path.open("wb") as handle:
        tomli_w.dump(trimmed, handle)

    migrated = load_config()
    assert "labels.bin_prefix" in migrated.schema_state.added_fields
    assert "scanning.location_context_timeout_minutes" in migrated.schema_state.added_fields
    assert migrated.schema_state.acknowledged is False
    assert migrated.is_new_field("labels.bin_prefix")
    # ...and the merged result is actually written back to disk.
    with migrated.path.open("rb") as handle:
        assert tomllib.load(handle)["labels"]["bin_prefix"] == "BIN"


def test_acknowledging_a_migration_persists(env):
    cfg = load_config()
    cfg.schema_state.added_fields = ["labels.bin_prefix"]
    cfg.acknowledge_schema()
    assert load_config().schema_state.acknowledged is True


def test_default_detection_drives_the_web_highlight(env):
    cfg = load_config()
    assert cfg.is_default_value("server.port") is True
    cfg.data["server"]["port"] = 4242
    assert cfg.is_default_value("server.port") is False


def test_binding_all_interfaces_is_refused(env):
    cfg = load_config()
    cfg.data["server"]["host"] = "0.0.0.0"
    with pytest.raises(FatalConfigError, match="refused"):
        validate(cfg)


def test_unknown_vision_backend_is_fatal(env):
    cfg = load_config()
    cfg.data["vision"]["backend"] = "gpt"
    with pytest.raises(FatalConfigError, match="vision.backend"):
        validate(cfg)


def test_inverted_thresholds_warn_but_do_not_kill(env):
    cfg = load_config()
    cfg.data["embeddings"]["match_threshold"] = 0.1
    cfg.data["embeddings"]["suggest_threshold"] = 0.9
    warnings = validate(cfg)
    assert any("match_threshold" in warning for warning in warnings)


def test_broken_toml_is_fatal(env, monkeypatch):
    target = paths.config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("this is not = = toml")
    with pytest.raises(FatalConfigError, match="not valid TOML"):
        load_config()
