"""Backward-compatible alias for the renamed Paddle OCR extractor."""

from __future__ import annotations

from medical_extraction.models.paddle_extractor import PaddleExtractor

# Backward compatibility for existing imports/tests/config references.
SuryaExtractor = PaddleExtractor

