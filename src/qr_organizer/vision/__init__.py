"""Vision backends: Claude (hosted) or Ollama (fully local), chosen in config."""

from __future__ import annotations

import logging

from .base import StructuredVisionBackend
from .schemas import Detection, EnumeratedItem, Enumeration, Verification

log = logging.getLogger(__name__)

__all__ = [
    "StructuredVisionBackend",
    "Detection",
    "EnumeratedItem",
    "Enumeration",
    "Verification",
    "build_backend",
]


def build_backend(cfg) -> StructuredVisionBackend:
    """Construct the configured vision backend."""
    backend = cfg.str_("vision.backend", "anthropic")
    if backend == "ollama":
        from .ollama_backend import OllamaVisionBackend

        return OllamaVisionBackend(
            base_url=cfg.str_("vision.ollama.base_url", "http://127.0.0.1:11434"),
            model=cfg.str_("vision.ollama.model", "qwen2.5vl:7b"),
            timeout_seconds=cfg.int_("vision.ollama.timeout_seconds", 300),
            context_length=cfg.int_("vision.ollama.context_length", 8192),
        )

    from .. import secrets as secrets_module
    from .anthropic_backend import AnthropicVisionBackend

    secret = secrets_module.resolve(
        cfg.str_("vision.anthropic.credential_name", "anthropic_api_key"),
        cfg.str_("vision.anthropic.api_key_env", "ANTHROPIC_API_KEY"),
    )
    if secret is None:
        log.error(
            "ANTHROPIC API KEY MISSING: photo identification will fail until it is set. "
            "/health will report degraded. See the secrets section of the README."
        )
    return AnthropicVisionBackend(
        api_key=secret.value if secret else "",
        model=cfg.str_("vision.anthropic.model", "claude-opus-5"),
        max_tokens=cfg.int_("vision.anthropic.max_tokens", 8000),
        effort=cfg.str_("vision.anthropic.effort", "high"),
    )
