"""Dataclasses describing pipeline results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TextQualityMetrics:
    char_count: int
    word_count: int
    alpha_ratio: float
    weird_char_ratio: float
    duplicate_line_ratio: float
    mostly_whitespace: bool
    quality: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageClassification:
    page_number: int
    has_selectable_text: bool
    selectable_text_chars: int
    text_quality: str
    has_images: bool
    image_count: int
    image_coverage: float
    is_mostly_image: bool
    is_handwritten_candidate: bool
    page_class: str
    route: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedBlock:
    block_id: str
    type: str
    text: str
    source: str
    confidence: float
    page_number: int
    bbox: list[float] | None = None
    needs_review: bool = False
    title: str | None = None
    fields: dict[str, Any] | None = None
    structured_data: dict[str, Any] | None = None
    crop_classifier: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedPage:
    page_number: int
    page_type: str
    classification: dict[str, Any]
    timing: dict[str, Any]
    blocks: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    needs_review: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MedicalEntity:
    text: str
    type: str
    page_number: int
    block_id: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionResult:
    document_id: str
    input_file: str
    extraction_version: str
    created_at: str
    summary: dict[str, Any]
    pages: list[dict[str, Any]]
    medical_entities: list[dict[str, Any]]
    medications: list[dict[str, Any]]
    labs: list[dict[str, Any]]
    warnings: list[str]
    debug_artifacts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
