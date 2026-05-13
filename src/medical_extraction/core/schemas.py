"""Factory helpers for the output schema."""

from __future__ import annotations

from datetime import datetime, timezone

from medical_extraction.core.constants import EXTRACTION_VERSION
from medical_extraction.core.types import ExtractionResult


def build_result(
    document_id: str,
    input_file: str,
    summary: dict,
    pages: list[dict],
    medical_entities: list[dict],
    medications: list[dict],
    labs: list[dict],
    warnings: list[str],
    debug_artifacts: dict,
) -> ExtractionResult:
    return ExtractionResult(
        document_id=document_id,
        input_file=input_file,
        extraction_version=EXTRACTION_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        summary=summary,
        pages=pages,
        medical_entities=medical_entities,
        medications=medications,
        labs=labs,
        warnings=warnings,
        debug_artifacts=debug_artifacts,
    )
