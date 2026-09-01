"""The Ollama backend, exercised against a stand-in for the real Ollama API.

Nothing here talks to a real model, but everything goes through the actual
`OllamaVisionBackend` and the real HTTP client, so the request shape this app
sends -- endpoint, JSON-schema `format`, base64 image list -- is pinned to what
Ollama documents. A change that would break against a real server breaks here.
"""

from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from conftest import LAYOUT
from PIL import Image

from qr_organizer.errors import VisionError
from qr_organizer.vision.ollama_backend import OllamaVisionBackend
from qr_organizer.vision.schemas import BOX_SCALE

MODEL = "qwen2.5vl:7b"


class FakeOllama:
    """Serves /api/tags and /api/chat the way Ollama does, and records requests."""

    def __init__(self, *, tags: list[str] | None = None, reply=None) -> None:
        self.tags = [{"name": name} for name in (tags if tags is not None else [MODEL])]
        self.reply = reply or (lambda body: {"items": [], "notes": ""})
        self.requests: list[dict] = []
        server_self = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep pytest output clean
                pass

            def do_GET(self):
                if self.path != "/api/tags":
                    self.send_error(404)
                    return
                self._json({"models": server_self.tags})

            def do_POST(self):
                if self.path != "/api/chat":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                server_self.requests.append(body)
                payload = server_self.reply(body)
                if isinstance(payload, int):        # let a test force an HTTP error
                    self.send_error(payload)
                    return
                content = payload if isinstance(payload, str) else json.dumps(payload)
                self._json({"message": {"role": "assistant", "content": content},
                            "done": True})

            def _json(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}"

    def __enter__(self):
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture()
def image():
    return Image.new("RGB", (800, 600), (200, 120, 60))


def _backend(server) -> OllamaVisionBackend:
    return OllamaVisionBackend(base_url=server.url, model=MODEL, timeout_seconds=10)


# -- readiness ------------------------------------------------------------


def test_status_is_ok_when_the_model_is_pulled():
    with FakeOllama() as server:
        ok, detail = _backend(server).status()
    assert ok is True
    assert MODEL in detail


def test_status_names_the_pull_command_when_the_model_is_missing():
    with FakeOllama(tags=["llama3:8b"]) as server:
        ok, detail = _backend(server).status()
    assert ok is False
    assert f"ollama pull {MODEL}" in detail


def test_status_accepts_the_latest_tag_form():
    """`ollama pull qwen2.5vl:7b` can list as-is or with :latest appended."""
    with FakeOllama(tags=[f"{MODEL}:latest"]) as server:
        ok, _ = _backend(server).status()
    assert ok is True


def test_status_reports_an_unreachable_server_rather_than_hanging():
    backend = OllamaVisionBackend(base_url="http://127.0.0.1:1", model=MODEL)
    ok, detail = backend.status()
    assert ok is False
    assert "unreachable" in detail


# -- request shape --------------------------------------------------------


def test_the_request_matches_what_ollama_expects(image):
    with FakeOllama(reply=lambda body: {"items": [], "notes": "nothing"}) as server:
        _backend(server).enumerate_items(image, max_items=30)
        body = server.requests[0]

    assert body["model"] == MODEL
    assert body["stream"] is False
    # Structured outputs: `format` carries the JSON schema itself, not "json".
    assert isinstance(body["format"], dict)
    assert body["format"]["type"] == "object"
    assert "items" in body["format"]["properties"]

    system, user = body["messages"]
    assert system["role"] == "system"
    assert user["role"] == "user"
    # Images ride alongside the prompt as a list of base64 strings.
    assert isinstance(user["images"], list) and len(user["images"]) == 1
    decoded = base64.b64decode(user["images"][0])
    assert Image.open(io.BytesIO(decoded)).format == "JPEG"


def test_images_are_downscaled_before_being_sent():
    """A phone photo shouldn't be shipped at full resolution to a local model."""
    big = Image.new("RGB", (4032, 3024), (10, 10, 10))
    with FakeOllama(reply=lambda body: {"items": [], "notes": ""}) as server:
        _backend(server).enumerate_items(big, max_items=30)
        sent = server.requests[0]["messages"][1]["images"][0]
    decoded = Image.open(io.BytesIO(base64.b64decode(sent)))
    assert max(decoded.size) <= 1344


# -- the three passes -----------------------------------------------------


def test_enumerate_parses_a_real_reply(image):
    reply = {"items": [{"name": "wrench", "description": "red", "position_hint": "left"}],
             "notes": ""}
    with FakeOllama(reply=lambda body: reply) as server:
        result = _backend(server).enumerate_items(image, max_items=30)
    assert [item.name for item in result.items] == ["wrench"]


def test_locate_converts_the_0_1000_grid_to_fractions(image):
    reply = {"detections": [{"name": "wrench", "box": [100, 200, 300, 400], "confidence": 0.9}]}
    with FakeOllama(reply=lambda body: reply) as server:
        detections = _backend(server).locate_items(image, ["wrench"])
    assert detections[0].box == pytest.approx((0.1, 0.2, 0.3, 0.4))


def test_verify_reuses_a_candidate_label(image):
    reply = {"label": "robot kit parts", "description": "", "confidence": 0.8,
             "chosen_candidate": "robot kit parts", "unidentifiable": False}
    with FakeOllama(reply=lambda body: reply) as server:
        verification = _backend(server).verify_crop(image, ["robot kit parts"])
    assert verification.chosen_candidate == "robot kit parts"


# -- failures are loud ----------------------------------------------------


def test_a_non_json_reply_fails_loudly_after_a_retry(image):
    with FakeOllama(reply=lambda body: "I'm afraid I can't do that") as server:
        with pytest.raises(VisionError, match="failed after"):
            _backend(server).enumerate_items(image, max_items=30)
        # Retried once rather than giving up on a single bad reply.
        assert len(server.requests) == 2


def test_an_http_error_is_reported_with_its_status(image):
    with FakeOllama(reply=lambda body: 500) as server:
        with pytest.raises(VisionError, match="500"):
            _backend(server).enumerate_items(image, max_items=30)


def test_an_unreachable_server_is_reported_not_swallowed(image):
    backend = OllamaVisionBackend(base_url="http://127.0.0.1:1", model=MODEL, timeout_seconds=2)
    with pytest.raises(VisionError, match="could not reach ollama"):
        backend.enumerate_items(image, max_items=30)


# -- end to end through the pipeline --------------------------------------


def test_a_full_identification_runs_on_the_ollama_backend(db, cfg, photo, tmp_path):
    """The real backend, the real pipeline, only the model itself stubbed."""
    from conftest import FakeEmbedder

    from qr_organizer import paths
    from qr_organizer.pipeline import IdentificationPipeline
    from qr_organizer.search.bins_source import BinInventorySource
    from qr_organizer.services import bins as bins_service
    from qr_organizer.services import photos as photos_service

    names = list(LAYOUT)

    def reply(body):
        prompt = body["messages"][1]["content"]
        if prompt.startswith("List every distinct item"):
            return {"items": [{"name": n, "description": "", "position_hint": ""} for n in names],
                    "notes": ""}
        return {"detections": [
            {"name": n, "box": [int(v * BOX_SCALE) for v in LAYOUT[n][0]], "confidence": 0.9}
            for n in names
        ]}

    with FakeOllama(reply=reply) as server:
        cfg.data["vision"]["backend"] = "ollama"
        cfg.data["vision"]["ollama"]["base_url"] = server.url
        source = BinInventorySource(db)
        pipeline = IdentificationPipeline(
            db=db, backend=_backend(server), embedder=FakeEmbedder(), source=source,
            thumbnails_root=paths.thumbnails_dir(cfg.data_dir),
            photos_root=paths.photos_dir(cfg.data_dir), config=cfg,
        )
        record = bins_service.create_bin(db, code="BIN-0001")
        stored = photos_service.ingest(
            db, source=photo, photos_root=paths.photos_dir(cfg.data_dir), kind="bin_layout"
        )
        session_id = pipeline.start_session(
            bin_id=int(record["id"]), photo_id=int(stored["id"]), device_key="t"
        )
        result = pipeline.run(
            session_id=session_id, bin_id=int(record["id"]), photo_id=int(stored["id"]),
            photo_path=paths.photos_dir(cfg.data_dir) / stored["path"],
        )

    assert len(result.added_item_ids) == len(LAYOUT)
    assert {i.label for i in result.detected} == set(LAYOUT)


def test_the_factory_builds_an_ollama_backend_from_config(cfg):
    from qr_organizer.vision import build_backend

    cfg.data["vision"]["backend"] = "ollama"
    cfg.data["vision"]["ollama"]["model"] = "llava:13b"
    backend = build_backend(cfg)
    assert isinstance(backend, OllamaVisionBackend)
    assert backend.name == "ollama:llava:13b"


def test_a_context_window_large_enough_for_a_photo_is_requested(image):
    """Ollama's default context is too small here, and the failure is obscure."""
    with FakeOllama(reply=lambda body: {"items": [], "notes": ""}) as server:
        backend = OllamaVisionBackend(base_url=server.url, model=MODEL, context_length=16384)
        backend.enumerate_items(image, max_items=30)
    assert server.requests[0]["options"]["num_ctx"] == 16384


def test_the_context_length_comes_from_config(cfg):
    from qr_organizer.vision import build_backend

    cfg.data["vision"]["backend"] = "ollama"
    cfg.data["vision"]["ollama"]["context_length"] = 32768
    assert build_backend(cfg).context_length == 32768
