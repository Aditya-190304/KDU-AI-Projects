"""Validation helpers for parsed medical structures."""

from __future__ import annotations


def apply_review_policy(items: list[dict], review_threshold: float = 0.70) -> list[dict]:
    for item in items:
        confidence = item.get("confidence", 0.0)
        if confidence < review_threshold:
            item["needs_review"] = True
    return items
