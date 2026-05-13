"""Confidence policy helpers."""

from __future__ import annotations


def needs_review(confidence: float, threshold: float = 0.70) -> bool:
    return confidence < threshold
