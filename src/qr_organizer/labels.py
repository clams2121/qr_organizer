"""Printable QR labels.

Bin labels are produced as sheets of sequential codes, printed ahead of time and
peeled off as bins get filled. Every label carries the QR code *and* the human
readable code next to it, so a bin can be identified across the room without a
phone.

The QR payload is the full scan URL rather than the bare code. The in-app
scanner strips it back to a code either way, and encoding a URL means a phone's
own camera app also opens the right page -- useful when someone else's phone
is the one in the shed.
"""

from __future__ import annotations

import io
import logging

import qrcode
from qrcode.image.pil import PilImage
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

log = logging.getLogger(__name__)

PAGE_SIZES = {"letter": letter, "a4": A4}


def scan_url(base_url: str, code: str) -> str:
    return f"{base_url.rstrip('/')}/s/{code}"


def qr_image(payload: str, *, box_size: int = 10, border: int = 2) -> PilImage:
    encoder = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    encoder.add_data(payload)
    encoder.make(fit=True)
    return encoder.make_image(fill_color="black", back_color="white")


def qr_png_bytes(payload: str, *, box_size: int = 8) -> bytes:
    buffer = io.BytesIO()
    qr_image(payload, box_size=box_size).save(buffer, format="PNG")
    return buffer.getvalue()


def render_sheet(
    codes: list[str],
    *,
    base_url: str,
    title: str = "",
    page_size: str = "letter",
    columns: int = 3,
    rows: int = 8,
) -> bytes:
    """Lay codes out on a grid, one page at a time, and return the PDF bytes."""
    if not codes:
        raise ValueError("no codes to print")
    size = PAGE_SIZES.get(page_size.lower(), letter)
    width, height = size
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=size)
    pdf.setTitle(title or "QR Organizer labels")

    margin = 12 * mm
    header = 10 * mm
    cell_width = (width - 2 * margin) / columns
    cell_height = (height - 2 * margin - header) / rows
    per_page = columns * rows

    for index, code in enumerate(codes):
        position = index % per_page
        if position == 0:
            if index:
                pdf.showPage()
            _draw_header(pdf, width, height, margin, title, index // per_page + 1)

        column = position % columns
        row = position // columns
        x = margin + column * cell_width
        y = height - margin - header - (row + 1) * cell_height
        _draw_label(pdf, x, y, cell_width, cell_height, code, scan_url(base_url, code))

    pdf.showPage()
    pdf.save()
    log.info("rendered a %d-label sheet (%s, %dx%d per page)", len(codes), page_size, columns, rows)
    return buffer.getvalue()


def _draw_header(canvas_obj, width, height, margin, title, page_number) -> None:
    canvas_obj.setFont("Helvetica-Bold", 10)
    canvas_obj.drawString(margin, height - margin, title or "QR Organizer labels")
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.drawRightString(width - margin, height - margin, f"page {page_number}")


def _draw_label(canvas_obj, x, y, cell_width, cell_height, code, payload) -> None:
    pad = 3 * mm
    canvas_obj.setLineWidth(0.3)
    canvas_obj.setDash(1, 2)
    canvas_obj.rect(x + 1 * mm, y + 1 * mm, cell_width - 2 * mm, cell_height - 2 * mm)
    canvas_obj.setDash()

    qr_side = min(cell_width, cell_height) - 2 * pad - 6 * mm
    qr_x = x + pad + 1 * mm
    qr_y = y + (cell_height - qr_side) / 2 + 2 * mm
    canvas_obj.drawImage(
        ImageReader(io.BytesIO(qr_png_bytes(payload))),
        qr_x, qr_y, width=qr_side, height=qr_side,
        preserveAspectRatio=True, mask="auto",
    )

    text_x = qr_x + qr_side + 3 * mm
    available = x + cell_width - pad - text_x
    font_size = 16 if available > 30 * mm else 12
    canvas_obj.setFont("Helvetica-Bold", font_size)
    canvas_obj.drawString(text_x, y + cell_height / 2, code)
    canvas_obj.setFont("Helvetica", 6.5)
    canvas_obj.drawString(text_x, y + cell_height / 2 - 5 * mm, "qr-organizer")


def render_single(code: str, *, base_url: str, page_size: str = "letter") -> bytes:
    """One big label, for a location placard stuck on a shed door."""
    size = PAGE_SIZES.get(page_size.lower(), letter)
    width, height = size
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=size)
    pdf.setTitle(code)

    side = min(width, height) * 0.55
    x = (width - side) / 2
    y = (height - side) / 2
    pdf.drawImage(
        ImageReader(io.BytesIO(qr_png_bytes(scan_url(base_url, code), box_size=12))),
        x, y, width=side, height=side, preserveAspectRatio=True, mask="auto",
    )
    pdf.setFont("Helvetica-Bold", 36)
    pdf.drawCentredString(width / 2, y - 20 * mm, code)
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()
