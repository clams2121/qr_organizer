"""Fully local vision backend, talking to an Ollama server.

Ollama's `format` parameter takes a JSON schema and constrains generation the
same way `output_config.format` does on the Claude side, so both backends get
schema-valid replies through the same code path in `base.py`.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Sequence

import httpx
from PIL import Image

from ..errors import VisionError, VisionResponseError
from ..imaging import encode_jpeg
from .base import StructuredVisionBackend

log = logging.getLogger(__name__)

MAX_IMAGE_EDGE = 1344


class OllamaVisionBackend(StructuredVisionBackend):
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5vl:7b",
        timeout_seconds: int = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout_seconds
        self.name = f"ollama:{model}"

    def status(self) -> tuple[bool, str]:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
            tags = response.json().get("models") or []
        except (httpx.HTTPError, ValueError) as exc:
            return False, f"{self.base_url} unreachable: {exc}"
        names = {entry.get("name", "") for entry in tags}
        if self.model not in names and f"{self.model}:latest" not in names:
            return False, (
                f"{self.base_url} is up but {self.model!r} is not pulled "
                f"(run `ollama pull {self.model}`)"
            )
        return True, f"{self.model} on {self.base_url}"

    def _ask(
        self,
        *,
        system: str,
        prompt: str,
        images: Sequence[Image.Image],
        schema: dict[str, Any],
        purpose: str,
    ) -> dict[str, Any]:
        encoded = [
            base64.standard_b64encode(encode_jpeg(image, max_dimension=MAX_IMAGE_EDGE)).decode(
                "ascii"
            )
            for image in images
        ]
        body = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt, "images": encoded},
            ],
        }
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat", json=body, timeout=float(self.timeout)
            )
            response.raise_for_status()
            envelope = response.json()
        except httpx.HTTPStatusError as exc:
            raise VisionError(
                f"{purpose}: ollama returned {exc.response.status_code} -- "
                f"{exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise VisionError(
                f"{purpose}: could not reach ollama at {self.base_url} -- {exc}"
            ) from exc
        except ValueError as exc:
            raise VisionResponseError(
                f"{purpose}: ollama sent a non-JSON envelope -- {exc}"
            ) from exc

        content = (envelope.get("message") or {}).get("content", "")
        if not content:
            raise VisionResponseError(f"{purpose}: ollama returned an empty message")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise VisionResponseError(
                f"{purpose}: ollama's reply was not JSON -- {exc}: {content[:300]}"
            ) from exc
