"""The AI identification pipeline: enumerate -> locate -> verify, plus visual RAG.

Why this shape, and what it costs
---------------------------------
Pass 1 asks the model to *enumerate* the distinct items in the whole photo. A
single whole-image pass is what gets the count right, because the model can see
that the three sockets in the corner are one set and not three items.

Pass 2 asks the same model to *locate* those named items with bounding boxes.
Splitting naming from boxing matters: models are much better at boxing a thing
you have already named than at naming and boxing in one breath, and every item
needs a real box because the crop becomes the thumbnail and the embedding.

Between the passes, every crop is embedded and looked up against the library of
already-labelled thumbnails. A close match reuses that exact label -- which is
how a generic catch-all like "robot kit parts" gets reapplied to visually
similar parts forever after, without asking again.

Pass 3 runs *only* on crops that are still uncertain: a focused look at the
single crop, offered the RAG near-misses as candidates. Cheap, because most
crops never reach it.

Cost: 2 whole-image calls plus one small call per uncertain crop. On a busy tote
photo with Claude that lands around 5-15 US cents. Raising
`vision.enumerate_passes` unions repeated independent enumerations for better
recall on very cluttered layouts, at one extra whole-image call each.

Nothing here silently swallows a failure. A pass that will not parse raises, a
box that cannot be cropped is reported, and an item the model will not name is
handed to the user to name rather than being guessed at or dropped.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .db import Database
from .embeddings import Embedder
from .errors import EmbeddingError, QROrganizerError, VisionError
from .imaging import BBox, bbox_iou, crop, open_normalised, save_thumbnail
from .search.bins_source import BinInventorySource
from .services import items as items_service
from .util import now_iso, slugify
from .vision import StructuredVisionBackend
from .vision.schemas import Detection

log = logging.getLogger(__name__)

#: Two boxes overlapping more than this are treated as the same physical object.
DUPLICATE_IOU = 0.55
#: A box covering more of the frame than this is the whole layout, not an item.
MAX_BOX_AREA = 0.85


@dataclass
class DetectedItem:
    """One item found in a photo, before it is reconciled with the bin's contents."""

    label: str
    description: str
    bbox: BBox
    thumbnail_path: str
    confidence: float
    label_source: str
    needs_review: bool
    suggestion_label: str = ""
    suggestion_score: float = 0.0
    embedding: np.ndarray | None = None
    matched_item_id: int | None = None


@dataclass
class IdentificationResult:
    session_id: int
    bin_id: int
    photo_id: int
    detected: list[DetectedItem] = field(default_factory=list)
    added_item_ids: list[int] = field(default_factory=list)
    matched_item_ids: list[int] = field(default_factory=list)
    missing_item_ids: list[int] = field(default_factory=list)
    unlocated_names: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def summary(self) -> str:
        parts = [f"{len(self.detected)} detected"]
        if self.added_item_ids:
            parts.append(f"{len(self.added_item_ids)} new")
        if self.matched_item_ids:
            parts.append(f"{len(self.matched_item_ids)} already known")
        if self.missing_item_ids:
            parts.append(f"{len(self.missing_item_ids)} missing")
        return ", ".join(parts)


class IdentificationPipeline:
    """Runs identification for one photo against one bin."""

    def __init__(
        self,
        *,
        db: Database,
        backend: StructuredVisionBackend,
        embedder: Embedder,
        source: BinInventorySource,
        thumbnails_root: Path,
        photos_root: Path,
        config,
    ) -> None:
        self.db = db
        self.backend = backend
        self.embedder = embedder
        self.source = source
        self.thumbnails_root = thumbnails_root
        self.photos_root = photos_root
        self.cfg = config
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="identify")
        self._lock = threading.Lock()

    # -- entry points -----------------------------------------------------
    def start_session(self, *, bin_id: int, photo_id: int, device_key: str) -> int:
        with self.db.write() as conn:
            cursor = conn.execute(
                "INSERT INTO inventory_sessions(bin_id, kind, status, device_key, started_at) "
                "VALUES(?, 'bin', 'pending', ?, ?)",
                (bin_id, device_key, now_iso()),
            )
            session_id = int(cursor.lastrowid)
            conn.execute("UPDATE photos SET session_id = ? WHERE id = ?", (session_id, photo_id))
        return session_id

    def submit(self, *, session_id: int, bin_id: int, photo_id: int, photo_path: Path) -> None:
        """Queue identification. Photos are processed one at a time, in order."""
        self._executor.submit(self._run_guarded, session_id, bin_id, photo_id, photo_path)

    def _run_guarded(self, session_id: int, bin_id: int, photo_id: int, photo_path: Path) -> None:
        try:
            self.run(session_id=session_id, bin_id=bin_id, photo_id=photo_id,
                     photo_path=photo_path)
        except QROrganizerError as exc:
            log.error("identification session %d failed: %s", session_id, exc)
            self._fail_session(session_id, str(exc))
        except Exception as exc:  # pragma: no cover -- last-resort breadcrumb
            log.exception("identification session %d crashed", session_id)
            self._fail_session(session_id, f"{type(exc).__name__}: {exc}")

    def _fail_session(self, session_id: int, message: str) -> None:
        with self.db.write() as conn:
            conn.execute(
                "UPDATE inventory_sessions SET status = 'error', error = ?, finished_at = ? "
                "WHERE id = ?",
                (message[:1000], now_iso(), session_id),
            )

    # -- the pipeline itself ----------------------------------------------
    def run(
        self, *, session_id: int, bin_id: int, photo_id: int, photo_path: Path
    ) -> IdentificationResult:
        with self.db.write() as conn:
            conn.execute(
                "UPDATE inventory_sessions SET status = 'running' WHERE id = ?", (session_id,)
            )

        result = IdentificationResult(session_id=session_id, bin_id=bin_id, photo_id=photo_id)
        image = open_normalised(photo_path)
        max_items = self.cfg.int_("vision.max_items_per_photo", 60)

        # -- pass 1: enumerate ------------------------------------------
        passes = max(1, self.cfg.int_("vision.enumerate_passes", 1))
        names: list[str] = []
        seen_names: set[str] = set()
        for index in range(passes):
            enumeration = self.backend.enumerate_items(image, max_items=max_items,
                                                       pass_index=index)
            if enumeration.notes:
                result.notes = " ".join(filter(None, [result.notes, enumeration.notes]))
            for entry in enumeration.items:
                key = entry.name.strip().lower()
                if key and key not in seen_names:
                    seen_names.add(key)
                    names.append(entry.name.strip())
        if not names:
            raise VisionError(
                "the enumeration pass found no items in this photo -- if there really are "
                "items in it, the photo may be too dark or too far away"
            )
        log.info("session %d: enumerated %d distinct item name(s)", session_id, len(names))

        # -- pass 2: locate ---------------------------------------------
        detections = self.backend.locate_items(image, names[:max_items])
        detections = _deduplicate(detections)
        located = {detection.name.strip().lower() for detection in detections}
        result.unlocated_names = [name for name in names if name.strip().lower() not in located]
        if result.unlocated_names:
            result.warnings.append(
                f"{len(result.unlocated_names)} enumerated item(s) could not be located in the "
                f"photo and were not added: {', '.join(result.unlocated_names[:8])}"
            )

        # -- crops, embeddings, RAG, and pass 3 -------------------------
        for detection in detections:
            detected = self._process_detection(
                session_id=session_id, image=image, detection=detection, photo_id=photo_id
            )
            if detected is not None:
                result.detected.append(detected)

        # -- reconcile against what the bin already holds ----------------
        self._reconcile(result)
        self._finish_session(result)
        log.info("session %d complete: %s", session_id, result.summary)
        return result

    def _process_detection(
        self, *, session_id: int, image: Image.Image, detection: Detection, photo_id: int
    ) -> DetectedItem | None:
        from .imaging import bbox_area

        if bbox_area(detection.box) > MAX_BOX_AREA:
            log.warning(
                "session %d: dropping %r -- its box covers %.0f%% of the frame, which is the "
                "whole layout rather than one item",
                session_id, detection.name, bbox_area(detection.box) * 100,
            )
            return None
        try:
            piece = crop(image, detection.box)
        except QROrganizerError as exc:
            log.warning("session %d: could not crop %r: %s", session_id, detection.name, exc)
            return None

        unique = abs(hash(detection.box)) % 10**8
        thumb_rel = f"{session_id}/{slugify(detection.name)}-{unique}.jpg"
        save_thumbnail(
            piece,
            self.thumbnails_root / thumb_rel,
            size=self.cfg.int_("images.thumbnail_size", 384),
            jpeg_quality=self.cfg.int_("images.jpeg_quality", 88),
        )

        vector: np.ndarray | None = None
        suggestion_label = ""
        suggestion_score = 0.0
        if self.embedder.available:
            try:
                vector = self.embedder.embed_image(piece)
                suggestion_label, suggestion_score = self._rag_lookup(vector)
            except EmbeddingError as exc:
                log.error("session %d: embedding failed for %r: %s", session_id,
                          detection.name, exc)

        match_threshold = self.cfg.float_("embeddings.match_threshold", 0.86)
        suggest_threshold = self.cfg.float_("embeddings.suggest_threshold", 0.78)
        low_confidence = self.cfg.float_("vision.low_confidence_threshold", 0.55)

        label = detection.name
        description = ""
        source = "ai"
        confidence = detection.confidence
        needs_review = False

        if suggestion_label and suggestion_score >= match_threshold:
            # Close visual match to something already labelled: reuse that exact
            # label, but flag it so the user can see and correct the assist.
            label = suggestion_label
            source = "rag"
            confidence = suggestion_score
            needs_review = True
        elif (
            self.cfg.bool_("vision.verify_low_confidence", True)
            and confidence < low_confidence
        ):
            candidates = (
                self._rag_candidates(vector, suggest_threshold) if vector is not None else []
            )
            verification = self.backend.verify_crop(piece, candidates)
            if verification.unidentifiable or not verification.label:
                label = detection.name or "unidentified item"
                source = "unidentified"
                confidence = 0.0
                needs_review = True
                log.info("session %d: %r could not be identified confidently; asking the user",
                         session_id, detection.name)
            else:
                label = verification.chosen_candidate or verification.label
                description = verification.description
                source = "ai_verified"
                confidence = verification.confidence
                needs_review = verification.confidence < low_confidence

        return DetectedItem(
            label=label,
            description=description,
            bbox=detection.box,
            thumbnail_path=thumb_rel,
            confidence=confidence,
            label_source=source,
            needs_review=needs_review,
            suggestion_label=suggestion_label,
            suggestion_score=suggestion_score,
            embedding=vector,
        )

    def _rag_lookup(self, vector: np.ndarray) -> tuple[str, float]:
        neighbours = self.source.nearest(vector, limit=1)
        if not neighbours:
            return "", 0.0
        item_id, score = neighbours[0]
        labels = items_service.labels_for_items(self.db, [item_id])
        return labels.get(item_id, ""), score

    def _rag_candidates(self, vector: np.ndarray | None, threshold: float) -> list[str]:
        if vector is None:
            return []
        neighbours = self.source.nearest(vector, limit=6, min_score=threshold)
        labels = items_service.labels_for_items(self.db, [item_id for item_id, _ in neighbours])
        seen: list[str] = []
        for item_id, _score in neighbours:
            label = labels.get(item_id, "")
            if label and label not in seen:
                seen.append(label)
        return seen

    # -- diff against the bin's known contents -----------------------------
    def _reconcile(self, result: IdentificationResult) -> None:
        """Match detections to existing items; add the rest; flag what's gone."""
        # Items already flagged missing stay in the candidate set: a re-inventory
        # is exactly when a missing thing turns up again, and it must be
        # recognised as the same item rather than added as a duplicate.
        existing = items_service.list_bin_items(self.db, result.bin_id, include_missing=True)
        existing_by_id = {int(row["id"]): row for row in existing}
        vectors = self._existing_vectors(list(existing_by_id))
        match_threshold = self.cfg.float_("embeddings.match_threshold", 0.86)
        claimed: set[int] = set()

        for detected in result.detected:
            match_id = self._match_existing(
                detected, existing_by_id, vectors, claimed, match_threshold
            )
            if match_id is not None:
                claimed.add(match_id)
                detected.matched_item_id = match_id
                result.matched_item_ids.append(match_id)
                prior_status = existing_by_id[match_id]["status"]
                if prior_status != items_service.STATUS_IN_BIN:
                    # Seeing it in the bin photo is the item being scanned back
                    # into a bin, so it stops being missing / in use / on loan.
                    items_service.check_into_bin(
                        self.db, match_id, result.bin_id,
                        detail="seen again in the latest bin photo",
                    )
                continue
            item_id = items_service.create_item(
                self.db,
                label=detected.label,
                description=detected.description,
                bin_id=result.bin_id,
                thumbnail_path=detected.thumbnail_path,
                source_photo_id=result.photo_id,
                bbox=detected.bbox,
                label_source=detected.label_source,
                label_confidence=detected.confidence,
                needs_review=detected.needs_review,
            )
            detected.matched_item_id = item_id
            result.added_item_ids.append(item_id)
            if detected.embedding is not None:
                self._store_vector(item_id, detected.embedding)

        items_service.touch_seen(self.db, result.matched_item_ids)

        # Anything previously in this bin and not seen this time is *flagged*,
        # never deleted -- missing usually means moved, loaned or misplaced.
        for item_id, row in existing_by_id.items():
            if item_id in claimed:
                continue
            if row["status"] != items_service.STATUS_IN_BIN:
                # Already accounted for: out on loan, in use, or flagged missing
                # by an earlier run. Nothing new to say about it.
                continue
            items_service.mark_missing(
                self.db, item_id, detail="not detected in the latest bin photo"
            )
            result.missing_item_ids.append(item_id)

    def _match_existing(
        self,
        detected: DetectedItem,
        existing_by_id: dict[int, dict[str, Any]],
        vectors: dict[int, np.ndarray],
        claimed: set[int],
        threshold: float,
    ) -> int | None:
        best_id: int | None = None
        best_score = 0.0
        if detected.embedding is not None:
            for item_id, vector in vectors.items():
                if item_id in claimed:
                    continue
                score = float(np.dot(vector, detected.embedding))
                if score > best_score:
                    best_id, best_score = item_id, score
        if best_id is not None and best_score >= threshold:
            return best_id
        # No usable vectors (embeddings off) or no close visual match: fall back
        # to an exact label match, which is the only other signal we can trust.
        target = detected.label.strip().lower()
        for item_id, row in existing_by_id.items():
            if item_id not in claimed and row["label"].strip().lower() == target:
                return item_id
        return None

    def _existing_vectors(self, item_ids: list[int]) -> dict[int, np.ndarray]:
        if not item_ids:
            return {}
        placeholders = ",".join("?" for _ in item_ids)
        rows = self.db.query(
            f"SELECT item_id, vector FROM item_embeddings WHERE item_id IN ({placeholders})",
            tuple(item_ids),
        )
        return {
            int(row["item_id"]): np.frombuffer(row["vector"], dtype=np.float32) for row in rows
        }

    def _store_vector(self, item_id: int, vector: np.ndarray) -> None:
        try:
            self.db.ensure_vector_index(int(np.asarray(vector).ravel().size))
            items_service.store_embedding(
                self.db, item_id, model=self.embedder.name, vector=vector
            )
        except QROrganizerError as exc:
            log.error("could not store the embedding for item %d: %s", item_id, exc)

    def _finish_session(self, result: IdentificationResult) -> None:
        with self.db.write() as conn:
            conn.execute(
                "UPDATE inventory_sessions SET status = 'complete', detected_count = ?, "
                "added_count = ?, matched_count = ?, missing_count = ?, error = ?, "
                "finished_at = ? WHERE id = ?",
                (
                    len(result.detected),
                    len(result.added_item_ids),
                    len(result.matched_item_ids),
                    len(result.missing_item_ids),
                    " | ".join(result.warnings)[:1000],
                    now_iso(),
                    result.session_id,
                ),
            )

    # -- re-embedding -----------------------------------------------------
    def reembed_item(self, item_id: int) -> bool:
        """Recompute one item's vector from its stored thumbnail."""
        if not self.embedder.available:
            return False
        item = items_service.get_item(self.db, item_id)
        if not item or not item["thumbnail_path"]:
            return False
        path = self.thumbnails_root / item["thumbnail_path"]
        if not path.is_file():
            log.warning("item %d has no thumbnail on disk at %s", item_id, path)
            return False
        vector = self.embedder.embed_image(open_normalised(path))
        self._store_vector(item_id, vector)
        return True


def _deduplicate(detections: list[Detection]) -> list[Detection]:
    """Collapse boxes that clearly describe the same physical object."""
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if any(bbox_iou(detection.box, other.box) > DUPLICATE_IOU for other in kept):
            log.debug("dropping duplicate detection %r", detection.name)
            continue
        kept.append(detection)
    return kept


def session_status(db: Database, session_id: int) -> dict[str, Any] | None:
    row = db.query_one("SELECT * FROM inventory_sessions WHERE id = ?", (session_id,))
    return dict(row) if row else None
