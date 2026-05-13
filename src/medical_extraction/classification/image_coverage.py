"""Image-coverage helpers for page classification."""

from __future__ import annotations


def compute_image_coverage(
    page_area: float,
    image_bboxes: list[list[float]],
    tiny_image_area_ratio: float = 0.02,
) -> tuple[float, int]:
    if not page_area:
        return 0.0, 0

    total_area = 0.0
    image_count = 0
    for x0, y0, x1, y1 in image_bboxes:
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area / page_area < tiny_image_area_ratio:
            continue
        total_area += area
        image_count += 1

    return round(total_area / page_area, 4), image_count
