"""PDF helpers built around PyMuPDF."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import fitz
except ImportError:  # pragma: no cover - import validated at runtime
    fitz = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency failure path
    Image = Any

from medical_extraction.core.exceptions import CorruptedDocumentError, MissingDependencyError


def open_pdf_document(path: str):
    if fitz is None:
        raise MissingDependencyError("PyMuPDF is required to open PDFs.")
    try:
        return fitz.open(path)
    except Exception as exc:  # pragma: no cover - library exception surface
        raise CorruptedDocumentError(f"Unable to open PDF: {path}") from exc


def page_to_image(page, dpi: int = 150) -> Image:
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    if Image is Any:
        raise MissingDependencyError("Pillow is required for page rendering.")
    mode = "RGB" if pixmap.n < 4 else "RGBA"
    return Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)


def save_page_render(page, output_path: str, dpi: int = 150) -> str:
    image = page_to_image(page, dpi=dpi)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)


def scale_bbox_to_image(page, image: Image, bbox: list[float]) -> list[float]:
    x_scale = image.width / page.rect.width
    y_scale = image.height / page.rect.height
    x0, y0, x1, y1 = bbox
    return [x0 * x_scale, y0 * y_scale, x1 * x_scale, y1 * y_scale]
