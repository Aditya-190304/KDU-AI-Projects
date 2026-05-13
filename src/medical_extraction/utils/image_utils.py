"""Image helpers used by scanned and crop extraction flows."""

from __future__ import annotations

from typing import Any

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - optional dependency failure path
    Image = Any
    ImageOps = None


def preprocess_image(image: Image) -> Image:
    if ImageOps is None:
        return image
    grayscale = ImageOps.grayscale(image)
    return ImageOps.autocontrast(grayscale)


def safe_crop(image: Image, bbox: list[float]) -> Image:
    return image.crop(tuple(int(value) for value in bbox))
