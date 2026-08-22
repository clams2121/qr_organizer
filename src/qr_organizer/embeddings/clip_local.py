"""Local open-clip embeddings.

Model weights load lazily on first use so that starting the web app doesn't pay
for torch import when no photo has been submitted yet. Load failure is recorded
and reported, never silently swallowed.
"""

from __future__ import annotations

import logging
import threading
from typing import Sequence

import numpy as np
from PIL import Image

from ..errors import EmbeddingError
from . import normalise

log = logging.getLogger(__name__)


class ClipEmbedder:
    """open-clip image/text encoder pinned to one model + pretrained tag."""

    def __init__(self, model: str, pretrained: str, device: str = "cpu") -> None:
        self.model_name = model
        self.pretrained = pretrained
        self.device = device
        self.name = f"open_clip:{model}/{pretrained}"
        self._lock = threading.Lock()
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._dim: int | None = None
        self._error: str | None = None
        self._loaded = False

    # -- loading ----------------------------------------------------------
    def _load(self) -> None:
        if self._loaded or self._error:
            return
        with self._lock:
            if self._loaded or self._error:
                return
            try:
                import open_clip
                import torch
            except ImportError as exc:
                self._error = (
                    f"open-clip/torch not installed ({exc}); "
                    "install with `uv sync --extra embeddings`"
                )
                return
            try:
                log.info("loading CLIP model %s (%s) on %s -- first run downloads weights",
                         self.model_name, self.pretrained, self.device)
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name, pretrained=self.pretrained, device=self.device
                )
                model.eval()
                self._model = model
                self._preprocess = preprocess
                self._tokenizer = open_clip.get_tokenizer(self.model_name)
                self._torch = torch
                # One pass over a dummy image is the only reliable way to learn
                # the output dimension without hardcoding a per-model table.
                probe = self._preprocess(Image.new("RGB", (64, 64))).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    self._dim = int(model.encode_image(probe).shape[-1])
                self._loaded = True
                log.info("CLIP model ready: %s, %d dimensions", self.name, self._dim)
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                log.exception("CLIP model failed to load")

    def ensure_loaded(self) -> None:
        self._load()
        if not self._loaded:
            raise EmbeddingError(f"CLIP backend unavailable: {self._error}")

    # -- interface --------------------------------------------------------
    @property
    def dim(self) -> int:
        self._load()
        return self._dim or 0

    @property
    def available(self) -> bool:
        self._load()
        return self._loaded

    def status(self) -> tuple[bool, str]:
        self._load()
        if self._loaded:
            return True, f"{self.name} ({self._dim}d, {self.device})"
        return False, self._error or "not loaded"

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        self.ensure_loaded()
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        batch = torch.stack([self._preprocess(image.convert("RGB")) for image in images])
        batch = batch.to(self.device)
        with torch.no_grad():
            features = self._model.encode_image(batch)
        return normalise(features.cpu().numpy())

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return self.embed_images([image])[0]

    def embed_text(self, text: str) -> np.ndarray:
        """Text in the same space as the images -- used for text-to-photo search."""
        self.ensure_loaded()
        torch = self._torch
        tokens = self._tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
        return normalise(features.cpu().numpy())[0]
