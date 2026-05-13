"""Heuristics for judging whether extracted PDF text is usable."""

from __future__ import annotations

import re

from medical_extraction.core.constants import DEFAULT_THRESHOLDS
from medical_extraction.core.types import TextQualityMetrics


def analyze_text_quality(text: str, thresholds: dict | None = None) -> TextQualityMetrics:
    config = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    stripped = text.strip()
    char_count = len(stripped)
    words = re.findall(r"\b[\w/%.-]+\b", stripped)
    word_count = len(words)
    total_chars = len(stripped) or 1
    alpha_ratio = sum(char.isalpha() for char in stripped) / total_chars
    allowed_symbols = ".,:;-/()%"
    weird_char_ratio = (
        sum(
            1
            for char in stripped
            if not char.isalnum() and not char.isspace() and char not in allowed_symbols
        )
        / total_chars
    )

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    duplicate_line_ratio = 0.0
    if lines:
        duplicate_line_ratio = 1 - (len(set(lines)) / len(lines))

    mostly_whitespace = char_count == 0
    meets_length_requirement = char_count >= config["min_selectable_text_chars"] or word_count >= 6
    looks_good = (
        meets_length_requirement
        and alpha_ratio >= config["min_alpha_ratio"]
        and weird_char_ratio <= config["max_weird_char_ratio"]
        and duplicate_line_ratio <= config["max_duplicate_line_ratio"]
    )
    quality = "good" if looks_good else "poor" if char_count else "empty"

    return TextQualityMetrics(
        char_count=char_count,
        word_count=word_count,
        alpha_ratio=round(alpha_ratio, 4),
        weird_char_ratio=round(weird_char_ratio, 4),
        duplicate_line_ratio=round(duplicate_line_ratio, 4),
        mostly_whitespace=mostly_whitespace,
        quality=quality,
    )


def looks_readable(text: str, thresholds: dict | None = None) -> bool:
    return analyze_text_quality(text, thresholds=thresholds).quality == "good"
