"""Hybrid local chunking for medical extraction outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from medical_extraction.embeddings.local_embedder import LocalTextEmbedder
from medical_extraction.models.biomedical_ner import BiomedicalNerModel
from medical_extraction.models.runtime import resolve_torch_device
from medical_extraction.parsing.lab_parser import parse_labs
from medical_extraction.parsing.medication_parser import parse_medications
from medical_extraction.utils.rag_text import _render_block_text


HEADING_WITH_BODY_PATTERN = re.compile(
    r"^\s*(?P<section>history|chief complaint|complaints|findings|impression|plan|assessment|diagnosis|"
    r"prescribed\s+medications?|medications?|prescription|labs?|investigations?|advice|"
    r"patient\s+instructions?|instructions?|follow[\s-]?up|discharge summary)"
    r"(?:\s*\([^)]*\))?\s*[:\-]?\s*(?P<body>.+)?$",
    re.IGNORECASE,
)
SECTION_ONLY_PATTERN = re.compile(
    r"^\s*(history|chief complaint|complaints|findings|impression|plan|assessment|diagnosis|"
    r"prescribed\s+medications?|medications?|prescription|labs?|investigations?|advice|"
    r"patient\s+instructions?|instructions?|follow[\s-]?up|discharge summary)"
    r"(?:\s*\([^)]*\))?\s*[:\-]?\s*$",
    re.IGNORECASE,
)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
TOKEN_PATTERN = re.compile(r"\S+")
FORM_LABEL_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z\s/_-]{0,40}):?\s*$")

FORM_LABEL_ALIASES = {
    "date": "date",
    "patient name": "patient_name",
    "name": "patient_name",
    "address": "address",
    "dob": "dob",
    "date of birth": "dob",
    "allergies": "allergies",
    "allergy": "allergies",
    "weight": "weight",
    "mrn": "mrn",
    "patient id": "patient_id",
    "id": "patient_id",
    "rx": "rx",
    "prescription": "rx",
}

DEMOGRAPHIC_LABELS = {"patient_name", "dob", "allergies", "weight", "address", "date", "mrn", "patient_id"}


@dataclass(slots=True)
class ChunkingResult:
    chunks: list[dict[str, Any]]
    warnings: list[str]


class LocalMedicalChunker:
    """Chunk extraction payloads with entity, section, and optional semantic logic."""

    def __init__(self, config: dict[str, Any] | None = None, device: str = "cpu") -> None:
        self.config = config or {}
        self.device = resolve_torch_device(device)
        self.ner = BiomedicalNerModel(device=self.device)
        self.section_headings = {
            _normalize_section_name(str(item))
            for item in self.config.get("section_headings", [])
            if str(item).strip()
        }
        self.max_tokens = int(self.config.get("max_tokens", 220))
        self.hard_max_tokens = int(self.config.get("hard_max_tokens", 320))
        self.min_chunk_tokens = int(self.config.get("min_chunk_tokens", 80))
        self.overlap_tokens = int(self.config.get("overlap_tokens", 40))
        self.semantic_similarity_threshold = float(self.config.get("semantic_similarity_threshold", 0.72))
        self.semantic_min_sentences = int(self.config.get("semantic_min_sentences", 3))
        self.use_semantic_layer = bool(self.config.get("use_semantic_layer", True))
        self._sentence_embedder = LocalTextEmbedder(
            model_name=str(self.config.get("embedding_model", "BAAI/bge-small-en-v1.5")),
            device=self.device,
            local_files_only=bool(self.config.get("local_files_only", True)),
            max_length=256,
        )
        self._warnings: list[str] = []

    def build_chunks(self, payload: dict[str, Any], source_text_path: str) -> ChunkingResult:
        self._warnings = []
        chunks: list[dict[str, Any]] = []
        document_id = str(payload.get("document_id", "")).strip()
        input_file = str(payload.get("input_file", "")).strip()
        created_at = str(payload.get("created_at", "")).strip()

        for page in _sorted_pages(payload.get("pages")):
            chunks.extend(
                self._build_form_field_chunks(
                    page=page,
                    document_id=document_id,
                    input_file=input_file,
                    source_text_path=source_text_path,
                    created_at=created_at,
                )
            )
            current_section = ""
            prescription_buffer: list[dict[str, Any]] = []
            page_number = int(page.get("page_number", 0) or 0)
            page_type = str(page.get("page_type", "")).strip()
            for block in _sorted_blocks(page.get("blocks")):
                block_text = _render_block_text(block)
                if not block_text:
                    continue

                detected_section, stripped_text, section_only = self._detect_section(block_text)
                if detected_section:
                    if current_section == "prescription" and detected_section != "prescription":
                        chunks.extend(
                            self._flush_prescription_buffer(
                                prescription_buffer,
                                document_id=document_id,
                                input_file=input_file,
                                source_text_path=source_text_path,
                                created_at=created_at,
                                page_number=page_number,
                                page_type=page_type,
                            )
                        )
                        prescription_buffer = []
                    current_section = detected_section
                if section_only:
                    continue

                effective_text = stripped_text or block_text
                effective_section = detected_section or current_section or self._infer_section_from_block(block)
                if effective_section == "prescription":
                    prescription_buffer.append(
                        {
                            "block": block,
                            "text": effective_text,
                        }
                    )
                    continue

                if prescription_buffer:
                    chunks.extend(
                        self._flush_prescription_buffer(
                            prescription_buffer,
                            document_id=document_id,
                            input_file=input_file,
                            source_text_path=source_text_path,
                            created_at=created_at,
                            page_number=page_number,
                            page_type=page_type,
                        )
                    )
                    prescription_buffer = []

                block_chunks = self._chunk_block(
                    block=block,
                    text=effective_text,
                    document_id=document_id,
                    input_file=input_file,
                    source_text_path=source_text_path,
                    created_at=created_at,
                    page_number=page_number,
                    page_type=page_type,
                    section=effective_section,
                )
                chunks.extend(block_chunks)

            if prescription_buffer:
                chunks.extend(
                    self._flush_prescription_buffer(
                        prescription_buffer,
                        document_id=document_id,
                        input_file=input_file,
                        source_text_path=source_text_path,
                        created_at=created_at,
                        page_number=page_number,
                        page_type=page_type,
                    )
                )

        chunks.extend(
            self._build_prescription_summary_chunks(
                payload=payload,
                all_chunks=chunks,
                document_id=document_id,
                input_file=input_file,
                source_text_path=source_text_path,
                created_at=created_at,
            )
        )

        for index, chunk in enumerate(chunks):
            chunk["chunk_index"] = index
            chunk["chunk_id"] = f"{document_id}:chunk:{index:04d}"

        return ChunkingResult(chunks=chunks, warnings=list(dict.fromkeys(self._warnings)))

    def _build_form_field_chunks(
        self,
        page: dict[str, Any],
        document_id: str,
        input_file: str,
        source_text_path: str,
        created_at: str,
    ) -> list[dict[str, Any]]:
        blocks = _sorted_blocks(page.get("blocks"))
        if not blocks:
            return []

        page_number = int(page.get("page_number", 0) or 0)
        page_type = str(page.get("page_type", "")).strip()
        paired_fields: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()

        for block in blocks:
            label_lines = _extract_form_labels_from_text(str(block.get("text", "")))
            if len(label_lines) < 2:
                continue
            value_lines = self._collect_right_side_value_lines(block, blocks)
            if not value_lines:
                continue
            for label_key, label_display, value_text, source_block_ids in _pair_form_labels_with_values(label_lines, value_lines):
                pair_key = (label_key, value_text)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                paired_fields.append(
                    {
                        "label_key": label_key,
                        "label_display": label_display,
                        "value_text": value_text,
                        "source_block_ids": source_block_ids,
                        "block": block,
                    }
                )

        if not paired_fields:
            return []

        chunks: list[dict[str, Any]] = []
        demographic_lines: list[str] = []
        demographic_block_ids: list[str] = []
        for field in paired_fields:
            line_text = f"{field['label_display']}: {field['value_text']}"
            if field["label_key"] in DEMOGRAPHIC_LABELS:
                demographic_lines.append(line_text)
                demographic_block_ids.extend(field["source_block_ids"])
            chunks.append(
                self._make_chunk(
                    chunk_text=_with_section_prefix("demographics", line_text),
                    document_id=document_id,
                    input_file=input_file,
                    source_text_path=source_text_path,
                    created_at=created_at,
                    page_number=page_number,
                    page_type=page_type,
                    section="demographics",
                    strategy="form_fields",
                    block=field["block"],
                    metadata_updates={
                        "entity_focus": "form_field",
                        "form_field_key": field["label_key"],
                        "source_block_ids": field["source_block_ids"],
                    },
                )
            )

        if len(demographic_lines) >= 2:
            summary_block = paired_fields[0]["block"]
            chunks.append(
                self._make_chunk(
                    chunk_text=_with_section_prefix("demographics", " ".join(demographic_lines)),
                    document_id=document_id,
                    input_file=input_file,
                    source_text_path=source_text_path,
                    created_at=created_at,
                    page_number=page_number,
                    page_type=page_type,
                    section="demographics",
                    strategy="form_fields",
                    block=summary_block,
                    metadata_updates={
                        "entity_focus": "demographic_profile",
                        "source_block_ids": list(dict.fromkeys(demographic_block_ids)),
                    },
                )
            )

        return chunks

    def _collect_right_side_value_lines(self, label_block: dict[str, Any], blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        label_bbox = label_block.get("bbox") or []
        if len(label_bbox) != 4:
            return []
        label_order = int((label_block.get("metadata") or {}).get("reading_order", 10**9) or 10**9)
        label_top = float(label_bbox[1])
        label_bottom = float(label_bbox[3])
        label_right = float(label_bbox[2])
        expanded_bottom = label_bottom + max(120.0, (label_bottom - label_top) * 0.75)

        lines: list[dict[str, Any]] = []
        for block in blocks:
            if block is label_block:
                continue
            block_bbox = block.get("bbox") or []
            if len(block_bbox) != 4:
                continue
            block_order = int((block.get("metadata") or {}).get("reading_order", 10**9) or 10**9)
            if block_order <= label_order:
                continue
            if float(block_bbox[0]) <= label_right - 8.0:
                continue
            if float(block_bbox[1]) < label_top - 18.0:
                continue
            if float(block_bbox[1]) > expanded_bottom:
                continue
            if str(block.get("type", "")).lower() not in {"paragraph", "form"}:
                continue
            for line in _split_block_lines(str(block.get("text", ""))):
                if not line:
                    continue
                lines.append(
                    {
                        "text": line,
                        "block_id": str(block.get("block_id", "")),
                    }
                )
        return lines

    def _chunk_block(
        self,
        block: dict[str, Any],
        text: str,
        document_id: str,
        input_file: str,
        source_text_path: str,
        created_at: str,
        page_number: int,
        page_type: str,
        section: str,
    ) -> list[dict[str, Any]]:
        block_type = str(block.get("type", "paragraph")).lower()
        if block_type == "table":
            table_chunks = self._build_table_chunks(block, document_id, input_file, source_text_path, created_at, page_number, page_type, section)
            if table_chunks:
                return table_chunks
        if block_type == "form":
            return [
                self._make_chunk(
                    chunk_text=_with_section_prefix(section, text),
                    document_id=document_id,
                    input_file=input_file,
                    source_text_path=source_text_path,
                    created_at=created_at,
                    page_number=page_number,
                    page_type=page_type,
                    section=section,
                    strategy="section_aware",
                    block=block,
                )
            ]

        medication_chunks = self._build_medication_chunks(
            block, text, document_id, input_file, source_text_path, created_at, page_number, page_type, section
        )
        if medication_chunks:
            return medication_chunks

        lab_chunks = self._build_lab_chunks(
            block, text, document_id, input_file, source_text_path, created_at, page_number, page_type, section
        )
        if lab_chunks:
            return lab_chunks

        narrative_texts = self._split_narrative_text(text)
        if not narrative_texts:
            return []

        if self.use_semantic_layer:
            semantic_chunks = self._semantic_group_narrative(narrative_texts)
            if semantic_chunks:
                return [
                    self._make_chunk(
                        chunk_text=_with_section_prefix(section, chunk_text),
                        document_id=document_id,
                        input_file=input_file,
                        source_text_path=source_text_path,
                        created_at=created_at,
                        page_number=page_number,
                        page_type=page_type,
                        section=section,
                        strategy="hybrid_semantic",
                        block=block,
                    )
                    for chunk_text in semantic_chunks
                ]

        return [
            self._make_chunk(
                chunk_text=_with_section_prefix(section, chunk_text),
                document_id=document_id,
                input_file=input_file,
                source_text_path=source_text_path,
                created_at=created_at,
                page_number=page_number,
                page_type=page_type,
                section=section,
                strategy="entity_preserving",
                block=block,
            )
            for chunk_text in _pack_units_by_token_budget(
                narrative_texts,
                max_tokens=self.max_tokens,
                hard_max_tokens=self.hard_max_tokens,
                overlap_tokens=self.overlap_tokens,
            )
        ]

    def _flush_prescription_buffer(
        self,
        buffered_items: list[dict[str, Any]],
        document_id: str,
        input_file: str,
        source_text_path: str,
        created_at: str,
        page_number: int,
        page_type: str,
    ) -> list[dict[str, Any]]:
        if not buffered_items:
            return []

        grouped_orders: list[dict[str, Any]] = []
        current_order: dict[str, Any] | None = None
        note_lines: list[str] = []
        note_block_ids: list[str] = []

        for item in buffered_items:
            block = item["block"]
            raw_line = _normalize_text(item["text"])
            if not raw_line or _is_prescription_boilerplate(raw_line):
                continue

            if _is_prescription_order_start(raw_line):
                if current_order:
                    grouped_orders.append(current_order)
                current_order = {
                    "lines": [raw_line],
                    "block_ids": [str(block.get("block_id", ""))],
                    "bbox": block.get("bbox"),
                    "confidence": float(block.get("confidence", 0.0) or 0.0),
                    "source": str(block.get("source", "")),
                    "needs_review": bool(block.get("needs_review", False)),
                    "type": "paragraph",
                }
                continue

            if current_order and (_is_prescription_continuation(raw_line) or _is_continuation_fragment(raw_line, current_order["lines"][-1])):
                current_order["lines"].append(raw_line)
                current_order["block_ids"].append(str(block.get("block_id", "")))
                current_order["needs_review"] = current_order["needs_review"] or bool(block.get("needs_review", False))
                continue

            note_lines.append(raw_line)
            note_block_ids.append(str(block.get("block_id", "")))

        if current_order:
            grouped_orders.append(current_order)

        chunks: list[dict[str, Any]] = []
        for order in grouped_orders:
            order_text = _normalize_text(" ".join(order["lines"]))
            base_block = {
                "block_id": order["block_ids"][0] if order["block_ids"] else "",
                "bbox": order.get("bbox"),
                "confidence": order.get("confidence", 0.0),
                "source": order.get("source", ""),
                "needs_review": order.get("needs_review", False),
                "type": order.get("type", "paragraph"),
            }
            chunks.append(
                self._make_chunk(
                    chunk_text=_with_section_prefix("prescription", order_text),
                    document_id=document_id,
                    input_file=input_file,
                    source_text_path=source_text_path,
                    created_at=created_at,
                    page_number=page_number,
                    page_type=page_type,
                    section="prescription",
                    strategy="entity_preserving",
                    block=base_block,
                    metadata_updates={"entity_focus": "medication_order", "source_block_ids": order["block_ids"]},
                )
            )

        if note_lines:
            note_text = _normalize_text(" ".join(note_lines))
            if note_text:
                base_block = {
                    "block_id": note_block_ids[0] if note_block_ids else "",
                    "bbox": buffered_items[0]["block"].get("bbox"),
                    "confidence": float(buffered_items[0]["block"].get("confidence", 0.0) or 0.0),
                    "source": str(buffered_items[0]["block"].get("source", "")),
                    "needs_review": any(bool(item["block"].get("needs_review", False)) for item in buffered_items),
                    "type": "paragraph",
                }
                chunks.append(
                    self._make_chunk(
                        chunk_text=_with_section_prefix("prescription", note_text),
                        document_id=document_id,
                        input_file=input_file,
                        source_text_path=source_text_path,
                        created_at=created_at,
                        page_number=page_number,
                        page_type=page_type,
                        section="prescription",
                        strategy="section_aware",
                        block=base_block,
                        metadata_updates={"entity_focus": "prescription_notes", "source_block_ids": note_block_ids},
                    )
                )
        return chunks

    def _build_prescription_summary_chunks(
        self,
        payload: dict[str, Any],
        all_chunks: list[dict[str, Any]],
        document_id: str,
        input_file: str,
        source_text_path: str,
        created_at: str,
    ) -> list[dict[str, Any]]:
        """Build a document-level prescription summary chunk combining patient context with all medications."""
        doc_context = _extract_document_context_from_id(document_id)
        if not doc_context.get("patient_name"):
            return []

        diagnosis = ""
        for page in _sorted_pages(payload.get("pages")):
            for block in _sorted_blocks(page.get("blocks")):
                text = str(block.get("text", "")).strip()
                if text.lower().startswith("diagnosis:"):
                    diagnosis = text.split(":", 1)[1].strip()
                    break
            if diagnosis:
                break

        medication_lines: list[str] = []
        medication_block_ids: list[str] = []
        seen_texts: set[str] = set()
        for chunk in all_chunks:
            metadata = chunk.get("metadata", {})
            entity_focus = str(metadata.get("entity_focus", "")).lower()
            section = str(chunk.get("section", "")).lower()
            if entity_focus in ("medication", "medication_order") or (
                section == "prescription" and entity_focus not in ("prescription_notes", "prescription_summary", "form_field", "demographic_profile")
            ):
                raw = str(chunk.get("chunk_text", "")).strip()
                clean = re.sub(r"^(?:Prescription|Diagnosis|Medications?|Table):\s*", "", raw, flags=re.IGNORECASE).strip()
                if clean and clean.lower() not in seen_texts and not _is_prescription_boilerplate(clean):
                    seen_texts.add(clean.lower())
                    medication_lines.append(clean)
                    medication_block_ids.extend(metadata.get("source_block_ids", []))

        if not medication_lines:
            for med in (payload.get("medications") or []):
                parts = []
                if med.get("medication"):
                    parts.append(str(med["medication"]))
                if med.get("dose"):
                    parts.append(str(med["dose"]))
                if med.get("frequency"):
                    parts.append(str(med["frequency"]))
                if parts:
                    line = " ".join(parts)
                    if line.lower() not in seen_texts:
                        seen_texts.add(line.lower())
                        medication_lines.append(line)

        if not medication_lines:
            return []

        summary_parts = []
        summary_parts.append(f"Patient: {doc_context['patient_name']}")
        if doc_context.get("doctor_name"):
            summary_parts.append(f"Prescribing Physician: {doc_context['doctor_name']}")
        if doc_context.get("hospital"):
            summary_parts.append(f"Hospital: {doc_context['hospital']}")
        if doc_context.get("mrn"):
            summary_parts.append(f"MRN: {doc_context['mrn']}")
        if diagnosis:
            summary_parts.append(f"Diagnosis: {diagnosis}")
        summary_parts.append("Prescribed Medications:")
        for line in medication_lines:
            summary_parts.append(f"  {line}")

        summary_text = " ".join(summary_parts)

        first_page = 1
        first_page_type = "derived"
        for page in _sorted_pages(payload.get("pages")):
            first_page = int(page.get("page_number", 1) or 1)
            first_page_type = str(page.get("page_type", "derived")).strip()
            break

        synthetic_block = {
            "block_id": "prescription_summary",
            "bbox": None,
            "confidence": 1.0,
            "source": "derived_summary",
            "needs_review": False,
            "type": "paragraph",
        }
        return [
            self._make_chunk(
                chunk_text=f"Prescription: {summary_text}",
                document_id=document_id,
                input_file=input_file,
                source_text_path=source_text_path,
                created_at=created_at,
                page_number=first_page,
                page_type=first_page_type,
                section="prescription",
                strategy="prescription_summary",
                block=synthetic_block,
                metadata_updates={
                    "entity_focus": "prescription_summary",
                    "source_block_ids": list(dict.fromkeys(medication_block_ids)),
                },
            )
        ]

    def _build_medication_chunks(
        self,
        block: dict[str, Any],
        text: str,
        document_id: str,
        input_file: str,
        source_text_path: str,
        created_at: str,
        page_number: int,
        page_type: str,
        section: str,
    ) -> list[dict[str, Any]]:
        medications = parse_medications(
            text,
            page_number=page_number,
            block_id=str(block.get("block_id", "")),
            confidence=float(block.get("confidence", 0.75) or 0.75),
        )
        if not medications and str(block.get("type", "")).lower() == "prescription_item":
            medications = [
                {
                    "text": text,
                    "medication": (block.get("fields") or {}).get("medication", {}).get("value"),
                }
            ]
        chunks: list[dict[str, Any]] = []
        for medication in medications:
            medication_text = str(medication.get("text", "")).strip()
            if not medication_text:
                continue
            chunks.append(
                self._make_chunk(
                    chunk_text=_with_section_prefix(section or "medications", medication_text),
                    document_id=document_id,
                    input_file=input_file,
                    source_text_path=source_text_path,
                    created_at=created_at,
                    page_number=page_number,
                    page_type=page_type,
                    section=section or "medications",
                    strategy="entity_preserving",
                    block=block,
                    metadata_updates={"entity_focus": "medication", "medication": medication},
                )
            )
        return chunks

    def _build_lab_chunks(
        self,
        block: dict[str, Any],
        text: str,
        document_id: str,
        input_file: str,
        source_text_path: str,
        created_at: str,
        page_number: int,
        page_type: str,
        section: str,
    ) -> list[dict[str, Any]]:
        labs = parse_labs(
            text,
            page_number=page_number,
            block_id=str(block.get("block_id", "")),
            confidence=float(block.get("confidence", 0.8) or 0.8),
        )
        chunks: list[dict[str, Any]] = []
        for lab in labs:
            lab_text = f"{lab.get('test_name')}: {lab.get('value') or ''} {lab.get('unit') or ''}".strip()
            chunks.append(
                self._make_chunk(
                    chunk_text=_with_section_prefix(section or "labs", lab_text),
                    document_id=document_id,
                    input_file=input_file,
                    source_text_path=source_text_path,
                    created_at=created_at,
                    page_number=page_number,
                    page_type=page_type,
                    section=section or "labs",
                    strategy="entity_preserving",
                    block=block,
                    metadata_updates={"entity_focus": "lab", "lab": lab},
                )
            )
        return chunks

    def _build_table_chunks(
        self,
        block: dict[str, Any],
        document_id: str,
        input_file: str,
        source_text_path: str,
        created_at: str,
        page_number: int,
        page_type: str,
        section: str,
    ) -> list[dict[str, Any]]:
        structured_data = block.get("structured_data")
        if not isinstance(structured_data, dict):
            return []
        rows = structured_data.get("rows")
        if not isinstance(rows, list):
            return []
        columns = [str(col).strip().lower() for col in (structured_data.get("columns") or [])]
        is_medication_table = any(
            keyword in col for col in columns for keyword in ("medication", "drug", "dosage", "dose", "frequency")
        )
        doc_context = _extract_document_context_from_id(document_id) if is_medication_table else {}
        context_prefix = ""
        if doc_context:
            parts = []
            if doc_context.get("patient_name"):
                parts.append(f"Patient: {doc_context['patient_name']}")
            if doc_context.get("doctor_name"):
                parts.append(f"Prescribed by: {doc_context['doctor_name']}")
            if parts:
                context_prefix = ". ".join(parts) + ". "

        effective_section = section or ("prescription" if is_medication_table else "table")
        row_chunks: list[dict[str, Any]] = []
        all_row_texts: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            values = [str(value).strip() for value in row.values() if str(value).strip()]
            if not values:
                continue
            row_text = " | ".join(values)
            all_row_texts.append(row_text)
            enriched_text = f"{context_prefix}{row_text}" if context_prefix else row_text
            entity_focus = "medication_order" if is_medication_table else "table_row"
            row_chunks.append(
                self._make_chunk(
                    chunk_text=_with_section_prefix(effective_section, enriched_text),
                    document_id=document_id,
                    input_file=input_file,
                    source_text_path=source_text_path,
                    created_at=created_at,
                    page_number=page_number,
                    page_type=page_type,
                    section=effective_section,
                    strategy="section_aware",
                    block=block,
                    metadata_updates={"entity_focus": entity_focus, "row": row},
                )
            )

        if is_medication_table and len(all_row_texts) >= 2:
            summary_lines = []
            if doc_context.get("patient_name"):
                summary_lines.append(f"Patient: {doc_context['patient_name']}")
            if doc_context.get("doctor_name"):
                summary_lines.append(f"Prescribed by: {doc_context['doctor_name']}")
            if doc_context.get("hospital"):
                summary_lines.append(f"Hospital: {doc_context['hospital']}")
            if doc_context.get("mrn"):
                summary_lines.append(f"MRN: {doc_context['mrn']}")
            summary_lines.append("Prescribed Medications:")
            for row_text in all_row_texts:
                summary_lines.append(f"  {row_text}")
            row_chunks.append(
                self._make_chunk(
                    chunk_text=_with_section_prefix("prescription", " ".join(summary_lines)),
                    document_id=document_id,
                    input_file=input_file,
                    source_text_path=source_text_path,
                    created_at=created_at,
                    page_number=page_number,
                    page_type=page_type,
                    section="prescription",
                    strategy="prescription_summary",
                    block=block,
                    metadata_updates={
                        "entity_focus": "prescription_summary",
                        "source_block_ids": [str(block.get("block_id", ""))],
                    },
                )
            )
        return row_chunks

    def _semantic_group_narrative(self, sentences: list[str]) -> list[str]:
        if len(sentences) < self.semantic_min_sentences:
            return []
        embeddings = self._sentence_embedder.encode(sentences)
        if embeddings is None:
            if self._sentence_embedder.load_error:
                self._warnings.append(
                    f"Semantic chunking fallback used because local embedding model was unavailable: {self._sentence_embedder.load_error}"
                )
            return []

        groups: list[list[str]] = []
        current_group = [sentences[0]]
        current_tokens = _count_tokens(sentences[0])
        for index in range(1, len(sentences)):
            similarity = float(torch.dot(embeddings[index - 1], embeddings[index]).item())
            sentence_tokens = _count_tokens(sentences[index])
            should_split = (
                current_tokens >= self.min_chunk_tokens
                and similarity < self.semantic_similarity_threshold
            ) or current_tokens + sentence_tokens > self.hard_max_tokens
            if should_split:
                groups.append(current_group)
                current_group = [sentences[index]]
                current_tokens = sentence_tokens
                continue
            current_group.append(sentences[index])
            current_tokens += sentence_tokens
        if current_group:
            groups.append(current_group)

        packed_groups: list[str] = []
        for group in groups:
            packed_groups.extend(
                _pack_units_by_token_budget(
                    group,
                    max_tokens=self.max_tokens,
                    hard_max_tokens=self.hard_max_tokens,
                    overlap_tokens=self.overlap_tokens,
                )
            )
        return packed_groups

    def _make_chunk(
        self,
        chunk_text: str,
        document_id: str,
        input_file: str,
        source_text_path: str,
        created_at: str,
        page_number: int,
        page_type: str,
        section: str,
        strategy: str,
        block: dict[str, Any],
        metadata_updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        chunk_text = _normalize_text(chunk_text)
        entities = self.ner.extract_entities(chunk_text, page_number=page_number, block_id=str(block.get("block_id", "")))
        metadata = {
            "source_block_ids": [str(block.get("block_id", ""))],
            "source": str(block.get("source", "")),
            "block_type": str(block.get("type", "")),
            "strategy": strategy,
            "bbox": block.get("bbox"),
            "ocr_confidence": float(block.get("confidence", 0.0) or 0.0),
            "needs_review": bool(block.get("needs_review", False)),
            "entity_types": sorted({str(entity.get("type", "")).upper() for entity in entities if entity.get("type")}),
        }
        if metadata_updates:
            metadata.update(metadata_updates)
        return {
            "chunk_id": "",
            "document_id": document_id,
            "source_path": input_file,
            "page_number": page_number,
            "chunk_index": -1,
            "page_type": page_type,
            "section": section,
            "chunk_text": chunk_text,
            "chunk_char_count": len(chunk_text),
            "metadata": metadata,
            "embedding_model": self._sentence_embedder.model_name if strategy == "hybrid_semantic" else None,
            "embedding_dimension": None,
            "created_at": created_at,
            "source_text_file": source_text_path,
        }

    def _detect_section(self, text: str) -> tuple[str, str, bool]:
        normalized = _normalize_text(text)
        heading_match = HEADING_WITH_BODY_PATTERN.match(normalized)
        if heading_match:
            section_name = _normalize_section_name(heading_match.group("section"))
            if self.section_headings and section_name not in self.section_headings:
                return "", text, False
            body = _normalize_text(heading_match.group("body") or "")
            return section_name, body, not bool(body)

        if SECTION_ONLY_PATTERN.match(normalized):
            section_name = _normalize_section_name(normalized)
            if self.section_headings and section_name not in self.section_headings:
                return "", text, False
            return section_name, "", True
        return "", text, False

    def _infer_section_from_block(self, block: dict[str, Any]) -> str:
        title = str(block.get("title", "")).strip()
        if title:
            return _normalize_section_name(title)
        block_type = str(block.get("type", "")).lower()
        if block_type == "table":
            return "table"
        if block_type == "form":
            return "form"
        return ""

    def _split_narrative_text(self, text: str) -> list[str]:
        normalized = _normalize_text(text)
        if not normalized:
            return []
        line_units = [_normalize_text(part) for part in re.split(r"\n+", text) if _normalize_text(part)]
        if len(line_units) > 1 and any(_count_tokens(unit) >= 4 for unit in line_units):
            units: list[str] = []
            for line in line_units:
                units.extend(_split_long_sentence(line, self.hard_max_tokens))
            return units

        sentence_units = [_normalize_text(part) for part in SENTENCE_SPLIT_PATTERN.split(normalized) if _normalize_text(part)]
        if sentence_units:
            units = []
            for sentence in sentence_units:
                units.extend(_split_long_sentence(sentence, self.hard_max_tokens))
            return units
        return [normalized]


def derive_chunk_path(text_output_path: str) -> str:
    path = Path(text_output_path)
    return str(path.with_name(f"{path.stem}_chunks.json"))


def _sorted_pages(pages: Any) -> list[dict[str, Any]]:
    if not isinstance(pages, list):
        return []
    return sorted(
        [page for page in pages if isinstance(page, dict)],
        key=lambda page: (int(page.get("page_number", 10**9) or 10**9), str(page.get("page_number", ""))),
    )


def _sorted_blocks(blocks: Any) -> list[dict[str, Any]]:
    if not isinstance(blocks, list):
        return []
    return sorted(
        [block for block in blocks if isinstance(block, dict)],
        key=lambda block: (
            int((block.get("metadata") or {}).get("reading_order", 10**9) or 10**9),
            float((block.get("bbox") or [10**9, 10**9, 10**9, 10**9])[1]),
            float((block.get("bbox") or [10**9, 10**9, 10**9, 10**9])[0]),
            str(block.get("block_id", "")),
        ),
    )


def _pack_units_by_token_budget(units: list[str], max_tokens: int, hard_max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    chunks: list[str] = []
    current_units: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = _count_tokens(unit)
        if current_units and (current_tokens + unit_tokens > max_tokens or current_tokens >= hard_max_tokens):
            chunks.append(_normalize_text(" ".join(current_units)))
            overlap_units = _tail_units_for_overlap(current_units, overlap_tokens)
            current_units = overlap_units + [unit]
            current_tokens = sum(_count_tokens(value) for value in current_units)
            continue
        current_units.append(unit)
        current_tokens += unit_tokens
    if current_units:
        chunks.append(_normalize_text(" ".join(current_units)))
    return [chunk for chunk in chunks if chunk]


def _tail_units_for_overlap(units: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []
    selected: list[str] = []
    running_tokens = 0
    for unit in reversed(units):
        selected.insert(0, unit)
        running_tokens += _count_tokens(unit)
        if running_tokens >= overlap_tokens:
            break
    return selected


def _split_long_sentence(text: str, hard_max_tokens: int) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text)
    if len(tokens) <= hard_max_tokens:
        return [_normalize_text(text)]
    parts: list[str] = []
    current: list[str] = []
    for token in tokens:
        current.append(token)
        if len(current) >= hard_max_tokens:
            parts.append(" ".join(current))
            current = []
    if current:
        parts.append(" ".join(current))
    return [_normalize_text(part) for part in parts if _normalize_text(part)]


def _count_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _normalize_section_name(section: str) -> str:
    normalized = _normalize_text(section).lower()
    normalized = re.sub(r"\s*\([^)]*\)\s*$", "", normalized)
    normalized = normalized.rstrip(":")
    if normalized == "medication":
        normalized = "medications"
    if normalized == "lab":
        normalized = "labs"
    if normalized in ("prescribed medications", "prescribed medication"):
        normalized = "prescription"
    if normalized in ("patient instructions", "patient instruction", "instructions", "instruction"):
        normalized = "instructions"
    normalized = normalized.replace("follow-up", "follow up")
    return normalized


def _extract_document_context_from_id(document_id: str) -> dict[str, str]:
    """Parse patient name, doctor, hospital, MRN from a document_id like 'John_Scott-Jennifer_Kim-Mercy_General-MRN100008'."""
    context: dict[str, str] = {}
    if not document_id:
        return context
    parts = document_id.split("-")
    if len(parts) < 4:
        return context
    mrn_index = -1
    for i, part in enumerate(parts):
        if re.match(r"^MRN\d+$", part, re.IGNORECASE):
            mrn_index = i
            break
    if mrn_index < 0:
        return context
    context["patient_name"] = parts[0].replace("_", " ")
    context["mrn"] = parts[mrn_index]
    if mrn_index >= 3:
        context["doctor_name"] = "Dr. " + parts[1].replace("_", " ")
        hospital_parts = parts[2:mrn_index]
        context["hospital"] = "-".join(hospital_parts).replace("_", " ")
    elif mrn_index == 2:
        context["doctor_name"] = "Dr. " + parts[1].replace("_", " ")
    return context


def _split_block_lines(text: str) -> list[str]:
    return [_normalize_text(part) for part in str(text or "").splitlines() if _normalize_text(part)]


def _extract_form_labels_from_text(text: str) -> list[tuple[str, str]]:
    labels: list[tuple[str, str]] = []
    for line in _split_block_lines(text):
        matched = FORM_LABEL_PATTERN.match(line)
        if not matched:
            continue
        raw_label = _normalize_text(matched.group(1))
        key = FORM_LABEL_ALIASES.get(raw_label.lower())
        if not key:
            continue
        labels.append((key, _display_form_label(key)))
    return labels


def _display_form_label(label_key: str) -> str:
    labels = {
        "date": "Date",
        "patient_name": "Patient Name",
        "address": "Address",
        "dob": "DOB",
        "allergies": "Allergies",
        "weight": "Weight",
        "mrn": "MRN",
        "patient_id": "Patient ID",
        "rx": "RX",
    }
    return labels.get(label_key, " ".join(part.capitalize() for part in label_key.split("_")))


def _pair_form_labels_with_values(
    label_lines: list[tuple[str, str]],
    value_lines: list[dict[str, Any]],
) -> list[tuple[str, str, str, list[str]]]:
    pairs: list[tuple[str, str, str, list[str]]] = []
    label_index = 0
    value_index = 0
    while label_index < len(label_lines) and value_index < len(value_lines):
        label_key, label_display = label_lines[label_index]
        while value_index < len(value_lines) and not str(value_lines[value_index].get("text", "")).strip():
            value_index += 1
        if value_index >= len(value_lines):
            break

        value_text = _normalize_text(str(value_lines[value_index].get("text", "")))
        source_block_id = str(value_lines[value_index].get("block_id", "")).strip()
        if not value_text:
            value_index += 1
            continue

        if _value_matches_label(label_key, value_text):
            pairs.append((label_key, label_display, value_text, [source_block_id] if source_block_id else []))
            label_index += 1
            value_index += 1
            continue

        later_match_exists = any(
            _value_matches_label(next_label_key, value_text)
            for next_label_key, _next_label_display in label_lines[label_index + 1 :]
        )
        if later_match_exists:
            label_index += 1
            continue

        value_index += 1
    return pairs


def _with_section_prefix(section: str, text: str) -> str:
    clean_text = _normalize_text(text)
    clean_section = _normalize_section_name(section)
    if not clean_section:
        return clean_text
    title = " ".join(part.capitalize() for part in clean_section.split())
    if clean_text.lower().startswith(clean_section):
        return clean_text
    return f"{title}: {clean_text}".strip()


PRESCRIPTION_ORDER_START_PATTERN = re.compile(
    r"^(?:prescription:\s*)?(?:\d+[\.\)]\s*)?(?=.*\b(?:\d+mg|tablet|tablets|capsule|capsules|therapy|cbt)\b).+",
    re.IGNORECASE,
)
PRESCRIPTION_CONTINUATION_PATTERN = re.compile(
    r"^(?:prescription:\s*)?(?:-+\s*)?(take|dispense|refills?|schedule|monitor|watch|follow[-\s]?up|continue|please take|instructions?)\b",
    re.IGNORECASE,
)
PRESCRIPTION_BOILERPLATE_PATTERN = re.compile(
    r"^(?:prescription:\s*)?(?:rx|medication and instructions|patient instructions|hello\b|mercy general hospital\b|"
    r"\d+\s+health plaza|prescribing physician\b|signature\b|date:\s*\d{4}-\d{2}-\d{2}|page\s+\d+|\[your name\]|"
    r"take care and stay healthy)\b",
    re.IGNORECASE,
)


def _is_prescription_order_start(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized or _is_prescription_boilerplate(normalized):
        return False
    if _is_prescription_continuation(normalized):
        return False
    return bool(PRESCRIPTION_ORDER_START_PATTERN.match(normalized))


def _is_prescription_continuation(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return bool(PRESCRIPTION_CONTINUATION_PATTERN.match(normalized))


def _is_prescription_boilerplate(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    return bool(PRESCRIPTION_BOILERPLATE_PATTERN.match(normalized))


def _looks_like_medication_or_instruction(text: str) -> bool:
    normalized = _normalize_text(text).lower()
    if not normalized:
        return False
    medication_hints = (
        "mg",
        "ml",
        "tablet",
        "capsule",
        "solution",
        "syrup",
        "dispense",
        "take ",
        "day 1",
        "day 2",
        "refill",
    )
    return any(hint in normalized for hint in medication_hints)


def _value_matches_label(label_key: str, value_text: str) -> bool:
    normalized = _normalize_text(value_text)
    lowered = normalized.lower()
    if not normalized:
        return False
    is_date_like = bool(re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", normalized)) or bool(
        re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", lowered)
    )
    if label_key == "patient_name":
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z\s.'-]+", normalized)) and len(normalized.split()) >= 2
    if label_key == "address":
        if is_date_like:
            return False
        return any(token in lowered for token in ("street", "road", "avenue", "lane", "drive", ",")) or (
            bool(re.search(r"\d", normalized)) and len(normalized.split()) >= 3
        )
    if label_key in {"dob", "date"}:
        return is_date_like
    if label_key == "allergies":
        return lowered in {"nkda", "nka"} or bool(re.fullmatch(r"[A-Za-z\s,/-]+", normalized))
    if label_key == "weight":
        return bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:kg|kgs|lb|lbs)\b", lowered))
    if label_key in {"mrn", "patient_id"}:
        return bool(re.search(r"\d", normalized))
    if label_key == "rx":
        return _looks_like_medication_or_instruction(normalized)
    return False


def _is_continuation_fragment(text: str, previous_line: str) -> bool:
    normalized = _normalize_text(text)
    previous = _normalize_text(previous_line)
    if not normalized or not previous:
        return False
    if previous.endswith(("of", "to", "for", "with", "and", ",")):
        return True
    if normalized[0].islower():
        return True
    return normalized.endswith(".") and len(normalized.split()) <= 4
