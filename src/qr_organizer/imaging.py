"""Photo intake, cropping and thumbnails.

Bounding boxes travel through the pipeline in a single normalised form:
`(x0, y0, x1, y1)` as floats in 0..1 relative to the *displayed* (EXIF-rotated)
image. Converting a model's own coordinate convention into that form is the
backend's job, not this module's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import StorageError

log = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class StoredImage:
    path: Path
    width: int
    height: int


def open_normalised(source: Path) -> Image.Image:
    """Open an image with EXIF rotation applied and no alpha channel."""
    try:
        image = Image.open(source)
    except (UnidentifiedImageError, OSError) as exc:
        raise StorageError(f"{source} is not a readable image: {exc}") from exc
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")
    return image


def store_photo(
    source: Path, destination: Path, *, max_dimension: int, jpeg_quality: int
) -> StoredImage:
    """Normalise, downscale and write a photo to its permanent home."""
    image = open_normalised(source)
    image.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=jpeg_quality, optimize=True)
    return StoredImage(path=destination, width=image.width, height=image.height)


def clamp_bbox(bbox: BBox) -> BBox:
    x0, y0, x1, y1 = bbox
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    return x0, y0, x1, y1


def bbox_area(bbox: BBox) -> float:
    x0, y0, x1, y1 = clamp_bbox(bbox)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_iou(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = clamp_bbox(a)
    bx0, by0, bx1, by1 = clamp_bbox(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    union = bbox_area(a) + bbox_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def crop(image: Image.Image, bbox: BBox, *, pad: float = 0.04) -> Image.Image:
    """Crop a normalised bbox with a little padding so context survives."""
    x0, y0, x1, y1 = clamp_bbox(bbox)
    width, height = image.size
    pad_x = (x1 - x0) * pad
    pad_y = (y1 - y0) * pad
    box = (
        int(max(0.0, x0 - pad_x) * width),
        int(max(0.0, y0 - pad_y) * height),
        int(min(1.0, x1 + pad_x) * width),
        int(min(1.0, y1 + pad_y) * height),
    )
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        raise StorageError(f"degenerate crop {bbox} on a {width}x{height} image")
    return image.crop(box)


def save_thumbnail(image: Image.Image, destination: Path, *, size: int, jpeg_quality: int) -> None:
    thumb = image.copy()
    thumb.thumbnail((size, size), Image.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(destination, format="JPEG", quality=jpeg_quality, optimize=True)


def encode_jpeg(image: Image.Image, *, max_dimension: int = 1568, quality: int = 85) -> bytes:
    """JPEG bytes sized for a vision API request."""
    import io

    buffer = io.BytesIO()
    working = image.copy()
    working.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
    working.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()
