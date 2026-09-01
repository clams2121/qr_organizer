"""Embedding backends for visual RAG.

The pipeline embeds every item thumbnail and looks it up against the library of
already-labelled thumbnails. That library updates the moment a user labels or
corrects something -- there is no retraining step and no batch job.

Only one real backend ships (local open-clip). `NullEmbedder` exists so the app
can still run and identify items with embeddings switched off or broken, while
reporting `degraded` rather than pretending RAG is working.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns a thumbnail into a comparable unit vector."""

    name: str

    @property
    def dim(self) -> int: ...

    @property
    def available(self) -> bool: ...

    def status(self) -> tuple[bool, str]: ...

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def embed_image(self, image: Image.Image) -> np.ndarray: ...

    def embed_text(self, text: str) -> np.ndarray: ...


class NullEmbedder:
    """Explicitly does nothing, and says so."""

    name = "none"

    def __init__(self, reason: str = "embeddings.backend = 'none'") -> None:
        self.reason = reason

    @property
    def dim(self) -> int:
        return 0

    @property
    def available(self) -> bool:
        return False

    def status(self) -> tuple[bool, str]:
        return False, self.reason

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        raise_unavailable(self.reason)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        raise_unavailable(self.reason)

    def embed_text(self, text: str) -> np.ndarray:
        raise_unavailable(self.reason)


def raise_unavailable(reason: str):
    from ..errors import EmbeddingError

    raise EmbeddingError(f"no embedding backend available: {reason}")


def normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise so a dot product is cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        norm = float(np.linalg.norm(vectors))
        return vectors / norm if norm > 0 else vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build_embedder(cfg) -> Embedder:
    """Construct the configured embedder, degrading loudly rather than crashing."""
    backend = cfg.str_("embeddings.backend", "clip")
    if backend == "none":
        return NullEmbedder()
    if backend != "clip":  # config.validate already rejects this, belt and braces
        return NullEmbedder(f"unknown embeddings.backend {backend!r}")

    from .clip_local import ClipEmbedder

    embedder = ClipEmbedder(
        model=cfg.str_("embeddings.model", "ViT-B-32"),
        pretrained=cfg.str_("embeddings.pretrained", "laion2b_s34b_b79k"),
        device=cfg.str_("embeddings.device", "cpu"),
    )
    ok, detail = embedder.status()
    if not ok:
        log.error(
            "EMBEDDINGS UNAVAILABLE: %s -- visual matching is off and /health will report "
            "degraded. Install the extra with `uv sync --extra embeddings`.",
            detail,
        )
    return embedder
