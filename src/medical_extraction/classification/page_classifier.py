"""Page classification for digital, mixed, scanned, and handwritten pages."""

from __future__ import annotations

import re
from pathlib import Path

from medical_extraction.classification.image_coverage import compute_image_coverage
from medical_extraction.classification.text_quality import analyze_text_quality
from medical_extraction.core.constants import (
    DEFAULT_THRESHOLDS,
    PAGE_CLASS_COPYABLE,
    PAGE_CLASS_HANDWRITTEN,
    PAGE_CLASS_MIXED,
    PAGE_CLASS_SCANNED,
    PAGE_CLASS_UNKNOWN,
    PRESCRIPTION_HINTS,
)
from medical_extraction.core.types import PageClassification


class PageClassifier:
    def __init__(self, thresholds: dict | None = None) -> None:
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def classify(self, page, input_path: str = "") -> PageClassification:
        text = page.get_text("text") or ""
        text_metrics = analyze_text_quality(text, self.thresholds)
        page_area = page.rect.width * page.rect.height
        blocks = page.get_text("dict").get("blocks", [])
        image_bboxes = [block["bbox"] for block in blocks if block.get("type") == 1 and "bbox" in block]
        image_coverage, image_count = compute_image_coverage(
            page_area=page_area,
            image_bboxes=image_bboxes,
            tiny_image_area_ratio=self.thresholds["tiny_image_area_ratio"],
        )

        has_useful_text = text_metrics.quality == "good"
        is_mostly_image = image_coverage >= self.thresholds["scanned_image_coverage_ratio"]
        is_handwritten_candidate = self._is_handwritten_candidate(
            input_path=input_path,
            text=text,
            has_useful_text=has_useful_text,
            is_mostly_image=is_mostly_image,
        )

        warnings: list[str] = []
        if text_metrics.quality == "poor":
            warnings.append("Selectable text exists but appears noisy.")

        if is_handwritten_candidate:
            page_class = PAGE_CLASS_HANDWRITTEN
        elif has_useful_text and image_count == 0:
            page_class = PAGE_CLASS_COPYABLE
        elif has_useful_text and image_count > 0:
            page_class = PAGE_CLASS_MIXED
        elif not has_useful_text and is_mostly_image:
            page_class = PAGE_CLASS_SCANNED
        else:
            page_class = PAGE_CLASS_UNKNOWN
            warnings.append("Page routed to hybrid/unknown fallback.")

        return PageClassification(
            page_number=page.number + 1,
            has_selectable_text=bool(text.strip()),
            selectable_text_chars=text_metrics.char_count,
            text_quality=text_metrics.quality,
            has_images=image_count > 0,
            image_count=image_count,
            image_coverage=image_coverage,
            is_mostly_image=is_mostly_image,
            is_handwritten_candidate=is_handwritten_candidate,
            page_class=page_class,
            route=page_class,
            warnings=warnings,
        )

    def _is_handwritten_candidate(
        self,
        input_path: str,
        text: str,
        has_useful_text: bool,
        is_mostly_image: bool,
    ) -> bool:
        file_hints = {part.lower() for part in Path(input_path).stem.replace("-", "_").split("_")}
        if PRESCRIPTION_HINTS.intersection(file_hints) and is_mostly_image:
            return True

        lowered = text.lower()
        if any(hint in lowered for hint in ("rx", "sig", "tab", "caps")) and not has_useful_text:
            return True

        # Scanned prescriptions often have a printed header plus a handwritten body.
        # When the page is almost entirely image-based, route to the handwritten flow
        # if we see prescription-style cues in low-quality OCR text.
        prescription_markers = (
            "adv:",
            "cc:",
            "diagnosis",
            "dose",
            "stat",
            "od",
            "bd",
            "days",
        )
        has_dose_like_pattern = bool(re.search(r"\b\d+\s?(mg|mcg|ml|g)\b", lowered))
        if is_mostly_image and (any(marker in lowered for marker in prescription_markers) or has_dose_like_pattern):
            return True

        return False
