"""Claude vision backend.

Uses `output_config.format` structured outputs so the reply is schema-valid
JSON rather than prose we have to fish JSON out of. Thinking is left adaptive
(the default on Opus 5) because enumerating a cluttered layout genuinely
benefits from it, with `effort` exposed in config for cost control.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from PIL import Image

from ..errors import VisionError, VisionResponseError
from ..imaging import encode_jpeg
from .base import StructuredVisionBackend

log = logging.getLogger(__name__)

# Claude resizes anything larger internally; sending a smaller image just wastes
# less bandwidth for identical results.
MAX_IMAGE_EDGE = 1568


class AnthropicVisionBackend(StructuredVisionBackend):
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "claude-opus-5",
        max_tokens: int = 8000,
        effort: str = "high",
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.name = f"anthropic:{model}"
        self._api_key = api_key
        self._client: Any = None
        self._client_error: str | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover -- a hard dependency
            self._client_error = f"anthropic SDK not installed: {exc}"
            raise VisionError(self._client_error) from exc
        if not self._api_key:
            self._client_error = (
                "no Anthropic API key resolved; see the secrets section of the README"
            )
            raise VisionError(self._client_error)
        self._client = anthropic.Anthropic(api_key=self._api_key, max_retries=3, timeout=180.0)
        return self._client

    def status(self) -> tuple[bool, str]:
        """Cheap readiness: do we have a usable client and key?

        Deliberately does not call the API -- /health is polled, and a paid
        round-trip per poll is the wrong trade for a personal service.
        """
        if not self._api_key:
            return False, "no API key resolved"
        try:
            self._get_client()
        except VisionError as exc:
            return False, str(exc)
        return True, f"{self.model} (key configured, {self.effort} effort)"

    def _ask(
        self,
        *,
        system: str,
        prompt: str,
        images: Sequence[Image.Image],
        schema: dict[str, Any],
        purpose: str,
    ) -> dict[str, Any]:
        import anthropic

        client = self._get_client()
        content: list[dict[str, Any]] = []
        for image in images:
            import base64

            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.standard_b64encode(
                            encode_jpeg(image, max_dimension=MAX_IMAGE_EDGE)
                        ).decode("ascii"),
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        except anthropic.APIStatusError as exc:
            raise VisionError(
                f"{purpose}: Claude returned {exc.status_code} -- {exc.message}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise VisionError(f"{purpose}: could not reach the Claude API -- {exc}") from exc

        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            raise VisionError(
                f"{purpose}: Claude declined the request "
                f"({getattr(detail, 'category', 'unspecified')})"
            )
        if response.stop_reason == "max_tokens":
            raise VisionResponseError(
                f"{purpose}: response hit max_tokens ({self.max_tokens}) and is truncated; "
                "raise vision.anthropic.max_tokens or lower vision.max_items_per_photo"
            )

        text = next((block.text for block in response.content if block.type == "text"), None)
        if not text:
            raise VisionResponseError(f"{purpose}: no text block in the response")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise VisionResponseError(
                f"{purpose}: reply was not JSON -- {exc}: {text[:300]}"
            ) from exc

        usage = getattr(response, "usage", None)
        if usage is not None:
            log.debug("%s: %s in / %s out tokens", purpose, usage.input_tokens, usage.output_tokens)
        return payload
