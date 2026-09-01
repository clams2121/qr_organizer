"""The three-pass identification contract, implemented once for every backend.

Pass 1 (enumerate): one call over the whole photo, listing distinct items.
Pass 2 (locate):    one call over the whole photo, boxing the enumerated names.
Pass 3 (verify):    one call per still-uncertain crop, with RAG candidates.

Only the transport differs between Claude and Ollama, so only `_ask` is
abstract. Everything above it -- prompts, schema validation, retry semantics --
is shared, which is what keeps the two backends behaving identically.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Sequence

from PIL import Image

from ..errors import VisionError, VisionResponseError
from . import prompts, schemas
from .schemas import Detection, Enumeration, Verification

log = logging.getLogger(__name__)


class StructuredVisionBackend(ABC):
    """A vision model that can be asked for schema-constrained JSON."""

    name: str = "vision"
    #: A failed structured call is retried once -- a second identical failure is
    #: a real problem with the prompt, the image or the model, not a blip.
    max_attempts: int = 2

    @abstractmethod
    def _ask(
        self,
        *,
        system: str,
        prompt: str,
        images: Sequence[Image.Image],
        schema: dict[str, Any],
        purpose: str,
    ) -> dict[str, Any]:
        """Send one structured request and return the parsed JSON object."""

    @abstractmethod
    def status(self) -> tuple[bool, str]:
        """(reachable, human-readable detail) for /health and the status page."""

    # -- shared passes ----------------------------------------------------
    def _ask_with_retry(self, **kwargs: Any) -> dict[str, Any]:
        purpose = kwargs.get("purpose", "vision call")
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._ask(**kwargs)
            except VisionResponseError as exc:
                last = exc
                log.warning("%s: %s (attempt %d/%d)", purpose, exc, attempt, self.max_attempts)
                time.sleep(0.5 * attempt)
        raise VisionError(f"{purpose} failed after {self.max_attempts} attempts: {last}") from last

    def enumerate_items(
        self, image: Image.Image, *, max_items: int, pass_index: int = 0
    ) -> Enumeration:
        payload = self._ask_with_retry(
            system=prompts.SYSTEM,
            prompt=prompts.enumerate_prompt(max_items, pass_index),
            images=[image],
            schema=schemas.ENUMERATE_SCHEMA,
            purpose=f"enumerate pass {pass_index + 1}",
        )
        result = schemas.parse_enumeration(payload, max_items=max_items)
        log.info("enumerate pass %d found %d item(s)%s", pass_index + 1, len(result.items),
                 f": {result.notes}" if result.notes else "")
        return result

    def locate_items(self, image: Image.Image, names: Sequence[str]) -> list[Detection]:
        if not names:
            return []
        payload = self._ask_with_retry(
            system=prompts.SYSTEM,
            prompt=prompts.locate_prompt(list(names)),
            images=[image],
            schema=schemas.LOCATE_SCHEMA,
            purpose="locate pass",
        )
        detections = schemas.parse_detections(payload)
        if len(detections) < len(names):
            # Loud, not fatal: the pipeline reports unlocated items to the user
            # rather than dropping them on the floor.
            log.warning(
                "locate pass boxed %d of %d enumerated item(s); the rest will be reported "
                "as unlocated", len(detections), len(names)
            )
        return detections

    def verify_crop(self, crop: Image.Image, candidates: Sequence[str]) -> Verification:
        payload = self._ask_with_retry(
            system=prompts.SYSTEM,
            prompt=prompts.verify_prompt(list(candidates)),
            images=[crop],
            schema=schemas.VERIFY_SCHEMA,
            purpose="verify pass",
        )
        return schemas.parse_verification(payload)
