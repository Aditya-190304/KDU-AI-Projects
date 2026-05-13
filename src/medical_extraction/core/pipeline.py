"""Pipeline orchestration."""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from medical_extraction.classification.page_classifier import PageClassifier
from medical_extraction.core.constants import (
    DEFAULT_THRESHOLDS,
    PAGE_CLASS_COPYABLE,
    PAGE_CLASS_HANDWRITTEN,
    PAGE_CLASS_MIXED,
    PAGE_CLASS_SCANNED,
)
from medical_extraction.core.schemas import build_result
from medical_extraction.core.types import ExtractedPage, PageClassification
from medical_extraction.extraction.copyable_pdf_extractor import CopyablePdfExtractor
from medical_extraction.extraction.handwritten_prescription_extractor import HandwrittenPrescriptionExtractor
from medical_extraction.extraction.mixed_pdf_extractor import MixedPdfExtractor
from medical_extraction.extraction.scanned_page_extractor import ScannedPageExtractor
from medical_extraction.models.model_registry import ModelRegistry
from medical_extraction.parsing.clinical_validator import apply_review_policy
from medical_extraction.parsing.lab_parser import parse_labs
from medical_extraction.parsing.medication_parser import parse_medications
from medical_extraction.quality.quality_checker import QualityChecker
from medical_extraction.storage.local_storage import LocalInputAdapter, LocalOutputAdapter
from medical_extraction.utils.logging_utils import get_logger
from medical_extraction.utils.pdf_utils import open_pdf_document
from medical_extraction.utils.rag_text import build_rag_text, derive_rag_text_path
from medical_extraction.utils.timing import capture_elapsed_ms


class ExtractionPipeline:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        thresholds = self.config.get("thresholds", DEFAULT_THRESHOLDS)
        device = self.config.get("pipeline", {}).get("device", "cpu")
        self.logger = get_logger(__name__)
        self.input_adapter = LocalInputAdapter()
        self.output_adapter = LocalOutputAdapter()
        self.model_registry = ModelRegistry(device=device)
        self.page_classifier = PageClassifier(thresholds=thresholds)
        self.copyable_extractor = CopyablePdfExtractor()
        self.mixed_extractor = MixedPdfExtractor(self.model_registry)
        self.scanned_extractor = ScannedPageExtractor(self.model_registry)
        self.handwritten_extractor = HandwrittenPrescriptionExtractor(self.model_registry)
        self.quality_checker = QualityChecker()

    def run(
        self,
        input_path: str,
        output_path: str,
        debug_dir: str | None = None,
        save_debug_images: bool = False,
        enable_medical_ner: bool = False,
        text_only: bool = False,
    ) -> dict:
        validated_input = self.input_adapter.validate(input_path)
        document_id = self.input_adapter.document_id(input_path)
        self.model_registry.reset_ocr_usage()

        pages: list[dict] = []
        warnings: list[str] = []
        medical_entities: list[dict] = []
        medications: list[dict] = []
        labs: list[dict] = []

        started = time.perf_counter()
        if validated_input.suffix.lower() == ".pdf":
            document = open_pdf_document(str(validated_input))
            for page in document:
                extracted_page, page_entities, page_medications, page_labs, page_warnings = self._process_page(
                    page=page,
                    input_path=str(validated_input),
                    debug_dir=debug_dir,
                    save_debug_images=save_debug_images,
                    enable_medical_ner=enable_medical_ner,
                )
                pages.append(self.quality_checker.enrich_page(extracted_page))
                medical_entities.extend(page_entities)
                medications.extend(page_medications)
                labs.extend(page_labs)
                warnings.extend(page_warnings)
        else:
            extracted_page, page_entities, page_medications, page_labs, page_warnings = self._process_image_page(
                image_path=str(validated_input),
                debug_dir=debug_dir,
                save_debug_images=save_debug_images,
                enable_medical_ner=enable_medical_ner,
            )
            pages.append(self.quality_checker.enrich_page(extracted_page))
            medical_entities.extend(page_entities)
            medications.extend(page_medications)
            labs.extend(page_labs)
            warnings.extend(page_warnings)

        elapsed_seconds = round(time.perf_counter() - started, 2)
        ocr_usage = self.model_registry.get_ocr_usage_summary()
        summary = self._build_summary(pages, elapsed_seconds)
        summary["ocr_requests"] = ocr_usage["totals"]["requests"]
        summary["ocr_estimated_cost_usd"] = ocr_usage["totals"]["estimated_cost_usd"]
        self.logger.info(
            "Document OCR usage requests=%s input_tokens=%s output_tokens=%s estimated_cost_usd=%.6f",
            ocr_usage["totals"]["requests"],
            ocr_usage["totals"]["input_tokens"],
            ocr_usage["totals"]["output_tokens"],
            ocr_usage["totals"]["estimated_cost_usd"],
        )
        payload = build_result(
            document_id=document_id,
            input_file=str(validated_input),
            summary=summary,
            pages=pages,
            medical_entities=medical_entities,
            medications=apply_review_policy(medications),
            labs=labs,
            warnings=warnings,
            debug_artifacts={
                "debug_folder": debug_dir,
                "ocr_usage": ocr_usage,
            },
        ).to_dict()
        rag_text_output_path = output_path if text_only else derive_rag_text_path(output_path)
        payload.setdefault("debug_artifacts", {})["rag_text_file"] = rag_text_output_path
        if not text_only:
            self.output_adapter.write_result(output_path, payload)
        self._write_rag_text_file(rag_text_output_path, build_rag_text(payload))
        return payload

    def _write_rag_text_file(self, rag_text_output_path: str, text_payload: str) -> None:
        path = Path(rag_text_output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text_payload, encoding="utf-8")

    def _process_image_page(
        self,
        image_path: str,
        debug_dir: str | None,
        save_debug_images: bool,
        enable_medical_ner: bool,
    ) -> tuple[dict, list[dict], list[dict], list[dict], list[str]]:
        page_number = 1
        page_warnings: list[str] = []
        medical_entities: list[dict] = []
        medications: list[dict] = []
        labs: list[dict] = []

        image = Image.open(image_path).convert("RGB")

        with capture_elapsed_ms() as classification_ms:
            classification = self._classify_standalone_image(image=image, input_path=image_path)

        try:
            with capture_elapsed_ms() as extraction_ms:
                if classification.page_class == PAGE_CLASS_HANDWRITTEN:
                    blocks, route_warnings = self.handwritten_extractor.extract_from_image(
                        image=image,
                        page_number=page_number,
                        debug_dir=debug_dir,
                        save_debug_images=save_debug_images,
                    )
                else:
                    blocks, route_warnings = self.scanned_extractor.extract_from_image(
                        image=image,
                        page_number=page_number,
                        debug_dir=debug_dir,
                        save_debug_images=save_debug_images,
                    )
            page_warnings.extend(classification.warnings)
            page_warnings.extend(route_warnings)

            with capture_elapsed_ms() as ner_ms:
                if enable_medical_ner:
                    for block in blocks:
                        medical_entities.extend(
                            self.model_registry.biomedical_ner.extract_entities(
                                block.get("text", ""),
                                page_number=page_number,
                                block_id=block["block_id"],
                            )
                        )
                for block in blocks:
                    medications.extend(
                        self._collect_block_medications(
                            block=block,
                            page_number=page_number,
                            page_class=classification.page_class,
                        )
                    )
                    labs.extend(
                        parse_labs(
                            block.get("text", ""),
                            page_number=page_number,
                            block_id=block["block_id"],
                            confidence=block.get("confidence", 0.8),
                        )
                    )

            page_payload = ExtractedPage(
                page_number=page_number,
                page_type=classification.page_class,
                classification=classification.to_dict(),
                timing={
                    "classification_ms": classification_ms[0],
                    "extraction_ms": extraction_ms[0],
                    "medical_ner_ms": ner_ms[0],
                },
                blocks=blocks,
                warnings=page_warnings,
            ).to_dict()
        except Exception as exc:
            page_payload = ExtractedPage(
                page_number=page_number,
                page_type="error",
                classification=classification.to_dict(),
                timing={
                    "classification_ms": classification_ms[0],
                    "extraction_ms": 0.0,
                    "medical_ner_ms": 0.0,
                },
                blocks=[],
                error=f"Extraction failed on image input: {exc}",
                warnings=page_warnings + [str(exc)],
            ).to_dict()

        return page_payload, medical_entities, medications, labs, page_warnings

    def _classify_standalone_image(self, image, input_path: str) -> PageClassification:
        crop_classification = self.model_registry.crop_classifier.classify(image)
        predicted_class = str(crop_classification.get("predicted_class", "unknown/review")).strip().lower()

        warnings: list[str] = []
        warning_message = crop_classification.get("warning")
        if warning_message:
            warnings.append(str(warning_message))

        if predicted_class == "handwritten-like image":
            page_class = PAGE_CLASS_HANDWRITTEN
        elif predicted_class == "logo/stamp/signature/noise":
            page_class = PAGE_CLASS_UNKNOWN
            warnings.append("Image classified as logo/noise; OCR may be limited.")
        else:
            page_class = PAGE_CLASS_SCANNED

        warnings.append(
            f"Standalone image routed by classifier: {crop_classification.get('predicted_class', 'unknown/review')}."
        )

        return PageClassification(
            page_number=1,
            has_selectable_text=False,
            selectable_text_chars=0,
            text_quality="empty",
            has_images=True,
            image_count=1,
            image_coverage=1.0,
            is_mostly_image=True,
            is_handwritten_candidate=page_class == PAGE_CLASS_HANDWRITTEN,
            page_class=page_class,
            route=page_class,
            warnings=warnings,
        )

    def _process_page(
        self,
        page,
        input_path: str,
        debug_dir: str | None,
        save_debug_images: bool,
        enable_medical_ner: bool,
    ) -> tuple[dict, list[dict], list[dict], list[dict], list[str]]:
        page_number = page.number + 1
        page_warnings: list[str] = []
        medical_entities: list[dict] = []
        medications: list[dict] = []
        labs: list[dict] = []

        with capture_elapsed_ms() as classification_ms:
            classification = self.page_classifier.classify(page, input_path=input_path)

        try:
            with capture_elapsed_ms() as extraction_ms:
                blocks, route_warnings = self._extract_blocks_for_page(
                    page=page,
                    input_path=input_path,
                    page_number=page_number,
                    page_class=classification.page_class,
                    debug_dir=debug_dir,
                    save_debug_images=save_debug_images,
                )
            page_warnings.extend(classification.warnings)
            page_warnings.extend(route_warnings)

            with capture_elapsed_ms() as ner_ms:
                if enable_medical_ner:
                    for block in blocks:
                        medical_entities.extend(
                            self.model_registry.biomedical_ner.extract_entities(
                                block.get("text", ""),
                                page_number=page_number,
                                block_id=block["block_id"],
                            )
                        )
                for block in blocks:
                    medications.extend(
                        self._collect_block_medications(
                            block=block,
                            page_number=page_number,
                            page_class=classification.page_class,
                        )
                    )
                    labs.extend(
                        parse_labs(
                            block.get("text", ""),
                            page_number=page_number,
                            block_id=block["block_id"],
                            confidence=block.get("confidence", 0.8),
                        )
                    )

            page_payload = ExtractedPage(
                page_number=page_number,
                page_type=classification.page_class,
                classification=classification.to_dict(),
                timing={
                    "classification_ms": classification_ms[0],
                    "extraction_ms": extraction_ms[0],
                    "medical_ner_ms": ner_ms[0],
                },
                blocks=blocks,
                warnings=page_warnings,
            ).to_dict()
        except Exception as exc:
            page_payload = ExtractedPage(
                page_number=page_number,
                page_type="error",
                classification=classification.to_dict(),
                timing={
                    "classification_ms": classification_ms[0],
                    "extraction_ms": 0.0,
                    "medical_ner_ms": 0.0,
                },
                blocks=[],
                error=f"Extraction failed on page {page_number}: {exc}",
                warnings=page_warnings + [str(exc)],
            ).to_dict()

        return page_payload, medical_entities, medications, labs, page_warnings

    def _collect_block_medications(self, block: dict, page_number: int, page_class: str) -> list[dict]:
        if block.get("type") == "prescription_item":
            fields = block.get("fields") or {}
            if fields:
                return [
                    {
                        "page_number": page_number,
                        "block_id": block["block_id"],
                        "medication": fields.get("medication", {}).get("value"),
                        "dose": fields.get("dose", {}).get("value"),
                        "route": fields.get("route", {}).get("value"),
                        "frequency": fields.get("frequency", {}).get("value"),
                        "duration": fields.get("duration", {}).get("value"),
                        "form": fields.get("form", {}).get("value"),
                        "instructions": fields.get("instructions", {}).get("value"),
                        "text": block.get("text", ""),
                        "confidence": block.get("confidence", 0.75),
                        "needs_review": block.get("needs_review", False),
                    }
                ]
            return parse_medications(
                block.get("text", ""),
                page_number=page_number,
                block_id=block["block_id"],
                confidence=block.get("confidence", 0.75),
            )

        if page_class == PAGE_CLASS_HANDWRITTEN:
            return []

        return parse_medications(
            block.get("text", ""),
            page_number=page_number,
            block_id=block["block_id"],
            confidence=block.get("confidence", 0.75),
        )

    def _extract_blocks_for_page(
        self,
        page,
        input_path: str,
        page_number: int,
        page_class: str,
        debug_dir: str | None,
        save_debug_images: bool,
    ) -> tuple[list[dict], list[str]]:
        if page_class == PAGE_CLASS_COPYABLE:
            return self.copyable_extractor.extract(page, input_path, page_number), []
        if page_class == PAGE_CLASS_MIXED:
            return self.mixed_extractor.extract(page, input_path, page_number, debug_dir, save_debug_images)
        if page_class == PAGE_CLASS_SCANNED:
            return self.scanned_extractor.extract(page, page_number, debug_dir, save_debug_images)
        if page_class == PAGE_CLASS_HANDWRITTEN:
            return self.handwritten_extractor.extract(page, page_number, debug_dir, save_debug_images)

        blocks = self.copyable_extractor.extract(page, input_path, page_number, include_tables=False)
        if blocks:
            for block in blocks:
                block["needs_review"] = True
            return blocks, ["Hybrid fallback used copyable-text extraction with review flag."]
        return self.scanned_extractor.extract(page, page_number, debug_dir, save_debug_images)

    def _build_summary(self, pages: list[dict], processing_time_seconds: float) -> dict:
        page_types = [page["page_type"] for page in pages]
        total_blocks = sum(len(page.get("blocks", [])) for page in pages)
        return {
            "total_pages": len(pages),
            "copyable_pages": page_types.count(PAGE_CLASS_COPYABLE),
            "mixed_pages": page_types.count(PAGE_CLASS_MIXED),
            "scanned_pages": page_types.count(PAGE_CLASS_SCANNED),
            "handwritten_pages": page_types.count(PAGE_CLASS_HANDWRITTEN),
            "total_blocks": total_blocks,
            "total_tables": sum(
                1 for page in pages for block in page.get("blocks", []) if block.get("type") == "table"
            ),
            "total_forms": sum(
                1 for page in pages for block in page.get("blocks", []) if block.get("type") == "form"
            ),
            "total_prescriptions": sum(
                1
                for page in pages
                for block in page.get("blocks", [])
                if block.get("type") == "prescription_item"
            ),
            "needs_review_blocks": self.quality_checker.count_review_blocks(pages),
            "processing_time_seconds": processing_time_seconds,
        }
