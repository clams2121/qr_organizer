"""Structured shapes exchanged with the vision backends.

Every model response is constrained by a JSON schema and then validated again
here. A reply that doesn't fit raises `VisionResponseError` -- it is never
coerced, defaulted, or half-accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import VisionResponseError

# Boxes travel over the wire as integers on a 0-1000 grid. Models are markedly
# more reliable with that convention than with floats, and it round-trips to the
# normalised 0..1 floats the rest of the app uses without precision loss worth
# caring about at thumbnail scale.
BOX_SCALE = 1000


ENUMERATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short everyday name, e.g. 'wrench', 'roll of tape'.",
                    },
                    "description": {
                        "type": "string",
                        "description": "A few words of distinguishing detail, or empty.",
                    },
                    "position_hint": {
                        "type": "string",
                        "description": "Where it sits, e.g. 'top-left', 'centre, on the blue rag'.",
                    },
                },
                "required": ["name", "description", "position_hint"],
                "additionalProperties": False,
            },
        },
        "notes": {
            "type": "string",
            "description": "Anything unclear about the photo, or empty.",
        },
    },
    "required": ["items", "notes"],
    "additionalProperties": False,
}


LOCATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "box": {
                        "type": "array",
                        "description": "[x0, y0, x1, y1] on a 0-1000 grid, origin top-left.",
                        "items": {"type": "integer"},
                    },
                    "confidence": {
                        "type": "number",
                        "description": "0.0-1.0 confidence that this box holds this item.",
                    },
                },
                "required": ["name", "box", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["detections"],
    "additionalProperties": False,
}


VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "description": "Short everyday name for this one object."},
        "description": {"type": "string"},
        "confidence": {"type": "number"},
        "chosen_candidate": {
            "type": "string",
            "description": "Exact candidate label reused, or empty if none fit.",
        },
        "unidentifiable": {
            "type": "boolean",
            "description": "True if this crop cannot be named without guessing.",
        },
    },
    "required": ["label", "description", "confidence", "chosen_candidate", "unidentifiable"],
    "additionalProperties": False,
}


@dataclass
class EnumeratedItem:
    name: str
    description: str = ""
    position_hint: str = ""


@dataclass
class Enumeration:
    items: list[EnumeratedItem] = field(default_factory=list)
    notes: str = ""


@dataclass
class Detection:
    name: str
    box: tuple[float, float, float, float]
    confidence: float


@dataclass
class Verification:
    label: str
    description: str
    confidence: float
    chosen_candidate: str = ""
    unidentifiable: bool = False


def _require(payload: Any, key: str, kind: type, context: str) -> Any:
    if not isinstance(payload, dict) or key not in payload:
        raise VisionResponseError(f"{context}: response is missing {key!r} -- got {payload!r:.400}")
    value = payload[key]
    if not isinstance(value, kind):
        raise VisionResponseError(
            f"{context}: {key!r} should be {kind.__name__}, got {type(value).__name__}"
        )
    return value


def parse_enumeration(payload: Any, *, max_items: int) -> Enumeration:
    raw_items = _require(payload, "items", list, "enumeration pass")
    items: list[EnumeratedItem] = []
    for entry in raw_items:
        name = str(_require(entry, "name", str, "enumeration pass")).strip()
        if not name:
            continue
        items.append(
            EnumeratedItem(
                name=name,
                description=str(entry.get("description") or "").strip(),
                position_hint=str(entry.get("position_hint") or "").strip(),
            )
        )
    if len(items) > max_items:
        items = items[:max_items]
    return Enumeration(items=items, notes=str(payload.get("notes") or "").strip())


def parse_detections(payload: Any) -> list[Detection]:
    raw = _require(payload, "detections", list, "locate pass")
    detections: list[Detection] = []
    for entry in raw:
        name = str(_require(entry, "name", str, "locate pass")).strip()
        box = _require(entry, "box", list, "locate pass")
        if len(box) != 4:
            raise VisionResponseError(
                f"locate pass: box for {name!r} has {len(box)} values, need 4"
            )
        try:
            coords = [float(value) / BOX_SCALE for value in box]
        except (TypeError, ValueError) as exc:
            raise VisionResponseError(
                f"locate pass: box for {name!r} is not numeric: {box!r}"
            ) from exc
        x0, y0, x1, y1 = coords
        if x1 <= x0 or y1 <= y0:
            # An inverted or empty box is a real failure to locate, not a
            # rounding artefact -- drop the detection and let the caller notice
            # the count mismatch.
            continue
        detections.append(
            Detection(
                name=name,
                box=(x0, y0, x1, y1),
                confidence=float(entry.get("confidence", 0.0) or 0.0),
            )
        )
    return detections


def parse_verification(payload: Any) -> Verification:
    label = str(_require(payload, "label", str, "verify pass")).strip()
    return Verification(
        label=label,
        description=str(payload.get("description") or "").strip(),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        chosen_candidate=str(payload.get("chosen_candidate") or "").strip(),
        unidentifiable=bool(payload.get("unidentifiable", False)),
    )
