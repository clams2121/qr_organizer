"""Shared fixtures.

The vision backend and the embedder are both faked so the whole pipeline --
enumerate, locate, crop, embed, RAG lookup, reconcile -- runs offline and
deterministically. The fakes implement the same protocols the real backends do,
so a change that breaks the contract breaks these tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from qr_organizer import paths
from qr_organizer.config import Config, load_config
from qr_organizer.db import Database
from qr_organizer.pipeline import IdentificationPipeline
from qr_organizer.search import SearchRegistry
from qr_organizer.search.bins_source import BinInventorySource
from qr_organizer.vision.base import StructuredVisionBackend
from qr_organizer.vision.schemas import BOX_SCALE

REPO_ROOT = Path(__file__).resolve().parent.parent

#: label -> (box on a 0..1 grid, fill colour)
LAYOUT = {
    "wrench": ((0.05, 0.05, 0.30, 0.45), (200, 30, 30)),
    "roll of tape": ((0.40, 0.05, 0.62, 0.45), (30, 160, 60)),
    "extension cord": ((0.70, 0.05, 0.95, 0.45), (40, 60, 210)),
    "bag of screws": ((0.05, 0.55, 0.30, 0.95), (220, 190, 40)),
}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """An isolated config + data directory for one test."""
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    config_home.mkdir()
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("QR_ORGANIZER_CONFIG", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    return tmp_path


@pytest.fixture()
def cfg(env) -> Config:
    config = load_config()
    config.data["app"]["data_dir"] = str(env / "data")
    config.data["app"]["log_dir"] = str(env / "logs")
    config.data["registry"]["dir"] = str(env / "registry")
    return config


@pytest.fixture()
def db(cfg) -> Database:
    paths.ensure_dir(cfg.data_dir)
    database = Database(paths.database_path(cfg.data_dir))
    database.initialise()
    yield database
    database.close()


@pytest.fixture()
def photo(tmp_path) -> Path:
    """A synthetic 'contents laid out on the floor' photo."""
    image = Image.new("RGB", (1000, 750), (235, 235, 235))
    draw = ImageDraw.Draw(image)
    for (x0, y0, x1, y1), colour in LAYOUT.values():
        draw.rectangle(
            [x0 * 1000, y0 * 750, x1 * 1000, y1 * 750], fill=colour, outline=(10, 10, 10), width=3
        )
    target = tmp_path / "layout.jpg"
    image.save(target, quality=95)
    return target


def make_photo(tmp_path: Path, labels: list[str], name: str = "layout2.jpg") -> Path:
    """A photo containing only the named subset of the standard layout."""
    image = Image.new("RGB", (1000, 750), (235, 235, 235))
    draw = ImageDraw.Draw(image)
    for label in labels:
        (x0, y0, x1, y1), colour = LAYOUT[label]
        draw.rectangle(
            [x0 * 1000, y0 * 750, x1 * 1000, y1 * 750], fill=colour, outline=(10, 10, 10), width=3
        )
    target = tmp_path / name
    image.save(target, quality=95)
    return target


class FakeVisionBackend(StructuredVisionBackend):
    """Answers the three passes from a fixed script."""

    name = "fake"

    def __init__(self, present: list[str], *, confidence: float = 0.9) -> None:
        self.present = present
        self.confidence = confidence
        self.calls: list[str] = []
        self.verify_candidates: list[list[str]] = []

    def status(self):
        return True, "fake backend"

    def _ask(self, *, system, prompt, images, schema, purpose):
        self.calls.append(purpose)
        if purpose.startswith("enumerate"):
            return {
                "items": [
                    {"name": name, "description": "", "position_hint": ""}
                    for name in self.present
                ],
                "notes": "",
            }
        if purpose == "locate pass":
            return {
                "detections": [
                    {
                        "name": name,
                        "box": [int(value * BOX_SCALE) for value in LAYOUT[name][0]],
                        "confidence": self.confidence,
                    }
                    for name in self.present
                ]
            }
        # verify pass: echo the first candidate when one was offered.
        candidates = [line[2:] for line in prompt.splitlines() if line.startswith("- ")]
        self.verify_candidates.append(candidates)
        return {
            "label": candidates[0] if candidates else "mystery object",
            "description": "",
            "confidence": 0.9 if candidates else 0.2,
            "chosen_candidate": candidates[0] if candidates else "",
            "unidentifiable": not candidates,
        }


class FakeEmbedder:
    """A real, deterministic embedding: the crop's downsampled colour signature.

    Crude, but it has the property that matters for these tests -- the same
    physical object photographed twice lands in nearly the same place, and two
    different objects do not.
    """

    name = "fake-colour"

    @property
    def dim(self) -> int:
        return 48

    @property
    def available(self) -> bool:
        return True

    def status(self):
        return True, "fake colour embedder"

    def embed_image(self, image):
        small = image.convert("RGB").resize((4, 4))
        vector = np.asarray(small, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def embed_images(self, images):
        return np.vstack([self.embed_image(image) for image in images])

    def embed_text(self, text):
        raise NotImplementedError("the fake embedder has no text encoder")


@pytest.fixture()
def pipeline_factory(cfg, db, tmp_path):
    """Builds a pipeline wired to fakes, with the thresholds tests rely on."""

    def build(present: list[str], *, confidence: float = 0.9, embedder=None):
        backend = FakeVisionBackend(present, confidence=confidence)
        source = BinInventorySource(db)
        registry = SearchRegistry()
        registry.register(source)
        pipeline = IdentificationPipeline(
            db=db,
            backend=backend,
            embedder=embedder if embedder is not None else FakeEmbedder(),
            source=source,
            thumbnails_root=paths.thumbnails_dir(cfg.data_dir),
            photos_root=paths.photos_dir(cfg.data_dir),
            config=cfg,
        )
        return pipeline, backend, source

    return build
