"""Pipeline orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

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
from medical_extraction.storage.processed_registry import ProcessedFileRegistry
from medical_extraction.storage.rag_ingestion import RagIngestionService
from medical_extraction.utils.chunking import LocalMedicalChunker, derive_chunk_path
from medical_extraction.utils.json_utils import read_json, write_json
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
        self.chunker = LocalMedicalChunker(config=self.config.get("chunking", {}), device=device)
        self.ingestion_service = RagIngestionService(config=self.config, device=device)
        self.processed_registry = ProcessedFileRegistry()
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
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict:
        validated_input = self.input_adapter.validate(input_path)
        document_id = self.input_adapter.document_id(input_path)
        file_hash = self.processed_registry.compute_file_hash(validated_input)
        self.model_registry.reset_ocr_usage()
        self._notify_progress(progress_callback, 2, "preparing", "Validated input and initialized extraction.")
        cached_payload = self._load_cached_payload(
            file_hash=file_hash,
            requested_output_path=output_path,
            text_only=text_only,
            progress_callback=progress_callback,
        )
        if cached_payload is not None:
            return cached_payload

        pages: list[dict] = []
        warnings: list[str] = []
        medical_entities: list[dict] = []
        medications: list[dict] = []
        labs: list[dict] = []

        started = time.perf_counter()
        if validated_input.suffix.lower() == ".pdf":
            document = open_pdf_document(str(validated_input))
            total_pages = len(document)
            self._notify_progress(
                progress_callback,
                6,
                "preparing",
                f"Opened PDF with {total_pages} page{'s' if total_pages != 1 else ''}.",
                total_pages=total_pages,
            )
            for index, page in enumerate(document, start=1):
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
                page_progress = 10 + int((index / max(total_pages, 1)) * 48)
                self._notify_progress(
                    progress_callback,
                    page_progress,
                    "extracting",
                    f"Processed page {index} of {total_pages}.",
                    processed_pages=index,
                    total_pages=total_pages,
                )
        else:
            self._notify_progress(progress_callback, 10, "extracting", "Processing uploaded image.")
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
            self._notify_progress(progress_callback, 58, "extracting", "Finished OCR and parsing for the uploaded image.")

        elapsed_seconds = round(time.perf_counter() - started, 2)
        self._notify_progress(progress_callback, 62, "finalizing", "Compiling extracted document payload.")
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
        payload["file_hash"] = file_hash
        rag_text_output_path = output_path if text_only else derive_rag_text_path(output_path)
        payload.setdefault("debug_artifacts", {})["rag_text_file"] = rag_text_output_path
        rag_text_payload = build_rag_text(payload)
        self._notify_progress(progress_callback, 68, "writing_text", "Writing retrieval text output.")
        self._write_rag_text_file(rag_text_output_path, rag_text_payload)
        self._notify_progress(progress_callback, 74, "chunking", "Building medical chunks.")
        raw_chunks = self._write_chunk_file(payload, rag_text_output_path)
        self._notify_progress(
            progress_callback,
            80,
            "chunking",
            f"Built {len(raw_chunks)} chunk{'s' if len(raw_chunks) != 1 else ''}.",
            total_chunks=len(raw_chunks),
        )
        self._run_ingestion(payload, rag_text_payload, raw_chunks, progress_callback=progress_callback)
        cache_payload_path = self._resolve_cache_payload_path(output_path, text_only=text_only)
        if not text_only:
            self._notify_progress(progress_callback, 98, "writing_result", "Writing extraction result file.")
            self.output_adapter.write_result(output_path, payload)
        if Path(cache_payload_path) != Path(output_path) or text_only:
            write_json(cache_payload_path, payload)
        self._store_processed_record(
            file_hash=file_hash,
            input_file=str(validated_input),
            output_path=cache_payload_path,
            rag_text_output_path=rag_text_output_path,
            chunk_output_path=str((payload.get("debug_artifacts") or {}).get("chunk_file", "")),
            payload=payload,
        )
        self._notify_progress(progress_callback, 100, "complete", "Document is ready for querying.")
        return payload

    def _write_rag_text_file(self, rag_text_output_path: str, text_payload: str) -> None:
        path = Path(rag_text_output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text_payload, encoding="utf-8")

    def _write_chunk_file(self, payload: dict, rag_text_output_path: str) -> list[dict]:
        chunking_config = self.config.get("chunking", {})
        if not bool(chunking_config.get("enabled", True)):
            return []
        result = self.chunker.build_chunks(payload, source_text_path=rag_text_output_path)
        if result.warnings:
            payload.setdefault("warnings", []).extend(result.warnings)
            payload.setdefault("debug_artifacts", {})["chunking_warnings"] = result.warnings
        payload.setdefault("summary", {})["total_chunks"] = len(result.chunks)
        chunk_output_path = derive_chunk_path(rag_text_output_path)
        payload.setdefault("debug_artifacts", {})["chunk_file"] = chunk_output_path
        if not bool(chunking_config.get("write_chunk_file", True)):
            return result.chunks
        write_json(chunk_output_path, result.chunks)
        return result.chunks

    def _run_ingestion(
        self,
        payload: dict,
        rag_text_payload: str,
        raw_chunks: list[dict],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        refresh_indexes_only: bool = False,
    ) -> None:
        if not raw_chunks:
            self._notify_progress(progress_callback, 92, "indexing", "No chunks generated, skipping embedding and indexing.")
            return
        self._notify_progress(progress_callback, 84, "indexing", "Preparing chunks for storage and embedding.")
        result = self.ingestion_service.ingest(
            payload=payload,
            rag_text=rag_text_payload,
            raw_chunks=raw_chunks,
            progress_callback=progress_callback,
            refresh_indexes_only=refresh_indexes_only,
        )
        if result.get("warnings"):
            payload.setdefault("warnings", []).extend(result["warnings"])
            payload.setdefault("debug_artifacts", {})["ingestion_warnings"] = result["warnings"]
        payload.setdefault("summary", {})["indexed_chunks"] = int(result.get("indexed", 0))
        payload.setdefault("debug_artifacts", {})["ingestion_artifacts"] = result.get("artifacts", {})
        self._notify_progress(
            progress_callback,
            96,
            "indexing",
            f"Indexed {int(result.get('indexed', 0))} chunk{'s' if int(result.get('indexed', 0)) != 1 else ''}.",
            indexed_chunks=int(result.get("indexed", 0)),
        )

    def _notify_progress(
        self,
        callback: Callable[[dict[str, Any]], None] | None,
        percent: int,
        stage: str,
        detail: str,
        **extra: Any,
    ) -> None:
        if callback is None:
            return
        payload: dict[str, Any] = {
            "percent": max(0, min(int(percent), 100)),
            "stage": str(stage),
            "detail": str(detail),
        }
        payload.update(extra)
        callback(payload)

    def _load_cached_payload(
        self,
        *,
        file_hash: str,
        requested_output_path: str,
        text_only: bool,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict | None:
        record = self.processed_registry.get(file_hash)
        if record is None:
            return None
        output_path = Path(record.output_file)
        rag_text_path = Path(record.rag_text_file)
        chunk_path = Path(record.chunk_file)
        if not output_path.exists() or not rag_text_path.exists() or not chunk_path.exists():
            return None
        cached_payload = read_json(output_path)
        if not isinstance(cached_payload, dict):
            return None
        raw_chunks = read_json(chunk_path)
        if not isinstance(raw_chunks, list):
            return None
        rag_text_payload = rag_text_path.read_text(encoding="utf-8")
        cached_payload["file_hash"] = file_hash
        cached_payload.setdefault("debug_artifacts", {})["cache_hit"] = True
        cached_payload.setdefault("debug_artifacts", {})["rag_text_file"] = str(rag_text_path)
        cached_payload.setdefault("debug_artifacts", {})["chunk_file"] = str(chunk_path)
        cached_payload.setdefault("summary", {}).setdefault("indexed_chunks", int(record.summary.get("indexed_chunks", 0) or 0))
        self._notify_progress(
            progress_callback,
            68,
            "cached",
            "Loaded cached extraction. Refreshing retrieval indexes from saved chunks.",
        )
        self._run_ingestion(
            cached_payload,
            rag_text_payload,
            raw_chunks,
            progress_callback=progress_callback,
            refresh_indexes_only=True,
        )
        write_json(output_path, cached_payload)
        if not text_only and Path(requested_output_path) != output_path:
            self.output_adapter.write_result(requested_output_path, cached_payload)
        self.processed_registry.upsert(
            file_hash=file_hash,
            document_id=str(cached_payload.get("document_id", "")).strip(),
            input_file=record.input_file,
            output_file=str(output_path),
            rag_text_file=str(rag_text_path),
            chunk_file=str(chunk_path),
            summary=dict(cached_payload.get("summary") or {}),
            warnings=list(cached_payload.get("warnings") or []),
        )
        self._notify_progress(
            progress_callback,
            100,
            "cached",
            "This exact file was already processed earlier. Loaded cached extraction and refreshed retrieval indexes.",
        )
        return cached_payload

    def _store_processed_record(
        self,
        *,
        file_hash: str,
        input_file: str,
        output_path: str,
        rag_text_output_path: str,
        chunk_output_path: str,
        payload: dict[str, Any],
    ) -> None:
        self.processed_registry.upsert(
            file_hash=file_hash,
            document_id=str(payload.get("document_id", "")).strip(),
            input_file=input_file,
            output_file=output_path,
            rag_text_file=rag_text_output_path,
            chunk_file=chunk_output_path,
            summary=dict(payload.get("summary") or {}),
            warnings=list(payload.get("warnings") or []),
        )

    def _resolve_cache_payload_path(self, output_path: str, text_only: bool) -> str:
        target = Path(output_path)
        if not text_only and target.suffix.lower() == ".json":
            return str(target)
        return str(target.with_name(f"{target.stem}_payload.json"))

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
