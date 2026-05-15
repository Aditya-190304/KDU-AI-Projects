"""Extractor for handwritten prescription pages."""

from __future__ import annotations

import os
from pathlib import Path
import re

from PIL import ImageEnhance

from medical_extraction.core.types import ExtractedBlock
from medical_extraction.parsing.medication_parser import (
    clean_prescription_line,
    has_medication_anchor,
    is_footer_noise_line,
    looks_like_prescription_line,
    looks_like_continuation_line,
    parse_medications,
)
from medical_extraction.utils.image_utils import preprocess_image
from medical_extraction.utils.pdf_utils import page_to_image


class HandwrittenPrescriptionExtractor:
    def __init__(self, model_registry) -> None:
        self.model_registry = model_registry

    def extract(
        self,
        page,
        page_number: int,
        debug_dir: str | None = None,
        save_debug_images: bool = False,
    ) -> tuple[list[dict], list[str]]:
        page_image = page_to_image(page)
        image = preprocess_image(page_image)
        return self.extract_from_image(
            image=image,
            page_number=page_number,
            debug_dir=debug_dir,
            save_debug_images=save_debug_images,
        )

    def extract_from_image(
        self,
        image,
        page_number: int,
        debug_dir: str | None = None,
        save_debug_images: bool = False,
    ) -> tuple[list[dict], list[str]]:
        warnings: list[str] = []

        handwritten_layout_backend = os.environ.get("MEDICAL_HANDWRITTEN_LAYOUT_BACKEND", "paddle").strip().lower()
        if handwritten_layout_backend in {"rule", "fullpage"}:
            warnings.append(
                "Using rule-based handwritten layout backend (region layout disabled for handwritten route)."
            )
            fallback_blocks = self._build_full_page_qwen_fallback(image=image, page_number=page_number)
            if fallback_blocks:
                return fallback_blocks, warnings
            warnings.append("Rule-based handwritten OCR returned no text.")
            return [], warnings

        if save_debug_images and debug_dir:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            image.save(Path(debug_dir) / f"page_{page_number}_handwritten.png")

        page_ocr = self.model_registry.scanned_ocr.ocr_image(image)
        if not page_ocr.get("lines"):
            surya_error = None
            get_last_ocr_error = getattr(self.model_registry.scanned_ocr, "get_last_ocr_error", None)
            if callable(get_last_ocr_error):
                surya_error = get_last_ocr_error()
            if surya_error:
                warnings.append(f"Line detection unavailable ({surya_error}); using full-page handwritten OCR fallback.")
            else:
                warnings.append("Line detection unavailable; using full-page handwritten OCR fallback.")
            fallback_blocks = self._build_full_page_qwen_fallback(image=image, page_number=page_number)
            if fallback_blocks:
                return fallback_blocks, warnings
            warnings.append("Full-page handwritten OCR fallback returned no text.")
            return [], warnings

        header_lines, body_lines, footer_lines = self._segment_page_lines(
            page_ocr.get("lines", []),
            image.height,
        )
        body_lines = self._merge_body_lines(body_lines, image.width, image.height)

        header_crop, body_crop, body_bbox = self._build_body_region(image, body_lines)
        if save_debug_images and debug_dir:
            body_crop.save(Path(debug_dir) / f"page_{page_number}_handwritten_body.png")

        if not body_lines:
            warnings.append("No prescription body lines detected; falling back to lower-page split.")
            header_crop, body_crop, body_bbox = self._fallback_split(image)
            header_lines = [line for line in page_ocr.get("lines", []) if line.get("bbox", [0, 0, 0, 0])[1] < body_bbox[1]]
            body_lines = [line for line in page_ocr.get("lines", []) if line.get("bbox", [0, 0, 0, 0])[1] >= body_bbox[1]]
            body_lines = self._merge_body_lines(body_lines, image.width, image.height)

        header_blocks = self._build_header_blocks(header_lines, page_number)
        prescription_blocks = self._build_prescription_blocks(
            image=image,
            body_lines=body_lines,
            page_number=page_number,
        )
        if not prescription_blocks:
            prescription_blocks = self._build_body_text_fallback_blocks(
                body_lines=body_lines,
                body_crop=body_crop,
                body_bbox=body_bbox,
                page_number=page_number,
            )

        if footer_lines:
            warnings.append(f"Suppressed {len(footer_lines)} footer lines from handwritten OCR.")
        if not prescription_blocks:
            warnings.append("No prescription lines could be converted into structured items.")

        return header_blocks + prescription_blocks, warnings

    def _build_full_page_qwen_fallback(self, image, page_number: int) -> list[dict]:
        text, confidence = self.model_registry.handwriting_ocr.extract_text(image)
        text = clean_prescription_line(text)
        if not text:
            return []
        return [
            ExtractedBlock(
                block_id=f"p{page_number}_body_full_1",
                type="paragraph",
                text=text,
                source="qwen_vision_fullpage_fallback",
                confidence=round(max(0.55, confidence), 2),
                page_number=page_number,
                bbox=[0.0, 0.0, float(image.width), float(image.height)],
                needs_review=confidence < 0.72,
                metadata={"fallback": "surya_unavailable"},
            ).to_dict()
        ]

    def _fallback_split(self, image):
        width = image.width
        height = image.height
        body_top = int(height * 0.35)
        header_crop = image.crop((0, 0, width, body_top))
        body_crop = image.crop((0, body_top, width, height))
        body_bbox = [0.0, float(body_top), float(width), float(height)]
        return header_crop, body_crop, body_bbox

    def _segment_page_lines(self, lines: list[dict], image_height: int) -> tuple[list[dict], list[dict], list[dict]]:
        sorted_lines = sorted(
            (line for line in lines if line.get("bbox") and clean_prescription_line(line.get("text", ""))),
            key=lambda item: (item["bbox"][1], item["bbox"][0]),
        )
        if not sorted_lines:
            return [], [], []

        footer_start_index = len(sorted_lines)
        for index, line in enumerate(sorted_lines):
            bbox = line["bbox"]
            text = clean_prescription_line(line["text"])
            if bbox[1] >= image_height * 0.80 and is_footer_noise_line(text):
                footer_start_index = index
                break

        prescription_start_index = None
        for index, line in enumerate(sorted_lines[:footer_start_index]):
            if self._body_line_score(line, image_height) >= 4:
                prescription_start_index = index
                break

        if prescription_start_index is None:
            for index, line in enumerate(sorted_lines[:footer_start_index]):
                if line["bbox"][1] >= image_height * 0.35:
                    prescription_start_index = index
                    break

        if prescription_start_index is None:
            prescription_start_index = footer_start_index

        header_lines = [
            line for line in sorted_lines[:prescription_start_index]
            if not is_footer_noise_line(line["text"])
        ]
        body_lines = [
            line for line in sorted_lines[prescription_start_index:footer_start_index]
            if self._should_keep_body_line(line, image_height)
        ]
        footer_lines = sorted_lines[footer_start_index:]
        return header_lines, body_lines, footer_lines

    def _merge_body_lines(self, body_lines: list[dict], image_width: int, image_height: int) -> list[dict]:
        if not body_lines:
            return []

        merged_lines: list[dict] = []
        columns = self._cluster_lines_by_column(body_lines, image_width)

        for column_index, column_lines in enumerate(columns, start=1):
            pending = [dict(line) for line in column_lines]
            index = 0
            while index < len(pending):
                current = pending[index]
                current["text"] = clean_prescription_line(current.get("text", ""))
                current["column_index"] = column_index
                while index + 1 < len(pending):
                    candidate = dict(pending[index + 1])
                    candidate["text"] = clean_prescription_line(candidate.get("text", ""))
                    candidate["column_index"] = column_index
                    if self._prefer_attaching_to_next(current, candidate, pending, index, image_height):
                        break
                    if not self._should_merge_lines(current, candidate, image_height):
                        break
                    current = self._combine_lines(current, candidate)
                    index += 1
                merged_lines.append(current)
                index += 1

        merged_lines.sort(
            key=lambda item: (
                int(item.get("column_index", 99)),
                float(item.get("bbox", [0, 0, 0, 0])[1]),
                float(item.get("bbox", [0, 0, 0, 0])[0]),
            )
        )
        return [line for line in merged_lines if self._should_keep_body_line(line, image_height)]

    def _body_line_score(self, line: dict, image_height: int) -> int:
        text = clean_prescription_line(line.get("text", ""))
        bbox = line.get("bbox", [0, 0, 0, 0])
        lowered = text.lower()
        score = 0
        if looks_like_prescription_line(text):
            score += 4
        if any(marker in lowered for marker in ("adv", "rx", "tab", "cap", "syp", "gel")):
            score += 2
        if re.search(r"\b\d+\s?(mg|mcg|g|ml)\b", lowered):
            score += 2
        if re.search(r"\b(od|bd|tds|stat|sos|1-0-1|1-1-1|0-1-0)\b", lowered):
            score += 2
        if bbox[1] >= image_height * 0.30:
            score += 1
        if any(marker in lowered for marker in ("hospital", "dental", "doctor", "dr.", "www", "email", "@")):
            score -= 3
        return score

    def _should_keep_body_line(self, line: dict, image_height: int) -> bool:
        text = clean_prescription_line(line.get("text", ""))
        bbox = line.get("bbox", [0, 0, 0, 0])
        if not text or is_footer_noise_line(text):
            return False
        if bbox[1] >= image_height * 0.88 and not looks_like_prescription_line(text):
            return False
        return self._body_line_score(line, image_height) >= 1

    def _should_merge_lines(self, current: dict, candidate: dict, image_height: int) -> bool:
        current_text = clean_prescription_line(current.get("text", ""))
        candidate_text = clean_prescription_line(candidate.get("text", ""))
        if not current_text or not candidate_text:
            return False

        current_bbox = current.get("bbox", [0, 0, 0, 0])
        candidate_bbox = candidate.get("bbox", [0, 0, 0, 0])
        vertical_gap = max(0, candidate_bbox[1] - current_bbox[3])
        max_gap = max(42, int(image_height * 0.035))
        if vertical_gap > max_gap:
            return False
        if self._x_overlap_ratio(current_bbox, candidate_bbox) < 0.55:
            return False

        current_anchor = has_medication_anchor(current_text)
        candidate_anchor = has_medication_anchor(candidate_text)
        candidate_continuation = looks_like_continuation_line(candidate_text)
        current_continuation = looks_like_continuation_line(current_text)

        if current_anchor and candidate_continuation:
            return True
        if candidate_anchor and current_continuation:
            return True
        if current_anchor and not candidate_anchor and len(candidate_text.split()) <= 3:
            return True
        if candidate_anchor and not current_anchor and len(current_text.split()) <= 3:
            return True
        return False

    def _prefer_attaching_to_next(self, current: dict, candidate: dict, pending: list[dict], index: int, image_height: int) -> bool:
        candidate_text = clean_prescription_line(candidate.get("text", ""))
        if not has_medication_anchor(current.get("text", "")):
            return False
        if not looks_like_continuation_line(candidate_text):
            return False
        if len(candidate_text.split()) > 2 or index + 2 >= len(pending):
            return False

        next_line = dict(pending[index + 2])
        next_text = clean_prescription_line(next_line.get("text", ""))
        if not has_medication_anchor(next_text):
            return False

        candidate_bbox = candidate.get("bbox", [0, 0, 0, 0])
        next_bbox = next_line.get("bbox", [0, 0, 0, 0])
        if self._x_overlap_ratio(candidate_bbox, next_bbox) < 0.55:
            return False
        max_gap = max(42, int(image_height * 0.035))
        next_gap = max(0, next_bbox[1] - candidate_bbox[3])
        return next_gap <= max_gap

    def _combine_lines(self, current: dict, candidate: dict) -> dict:
        current_text = clean_prescription_line(current.get("text", ""))
        candidate_text = clean_prescription_line(candidate.get("text", ""))
        merged_text = clean_prescription_line(f"{current_text} {candidate_text}")
        current_bbox = current.get("bbox", [0, 0, 0, 0])
        candidate_bbox = candidate.get("bbox", [0, 0, 0, 0])
        current_words = list(current.get("words", []))
        candidate_words = list(candidate.get("words", []))
        return {
            "text": merged_text,
            "bbox": [
                min(current_bbox[0], candidate_bbox[0]),
                min(current_bbox[1], candidate_bbox[1]),
                max(current_bbox[2], candidate_bbox[2]),
                max(current_bbox[3], candidate_bbox[3]),
            ],
            "confidence": max(float(current.get("confidence", 0.6)), float(candidate.get("confidence", 0.6))),
            "words": current_words + candidate_words,
            "column_index": current.get("column_index", candidate.get("column_index")),
        }

    def _cluster_lines_by_column(self, lines: list[dict], image_width: int) -> list[list[dict]]:
        if not lines:
            return []

        threshold = max(52.0, float(image_width) * 0.14)
        sorted_lines = sorted(lines, key=lambda item: (float(item["bbox"][0]), float(item["bbox"][1])))
        clusters: list[dict] = []

        for line in sorted_lines:
            bbox = line.get("bbox", [0, 0, 0, 0])
            x_left = float(bbox[0])
            best_cluster = None
            best_distance = None
            for cluster in clusters:
                distance = abs(x_left - float(cluster["x_left"]))
                if distance <= threshold and (best_distance is None or distance < best_distance):
                    best_cluster = cluster
                    best_distance = distance
            if best_cluster is None:
                clusters.append({"x_left": x_left, "items": [line]})
                continue
            best_cluster["items"].append(line)
            count = len(best_cluster["items"])
            best_cluster["x_left"] = ((best_cluster["x_left"] * (count - 1)) + x_left) / count

        clusters.sort(key=lambda cluster: float(cluster["x_left"]))
        ordered_columns: list[list[dict]] = []
        for cluster in clusters:
            ordered_columns.append(
                sorted(
                    cluster["items"],
                    key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0])),
                )
            )
        return ordered_columns

    def _x_overlap_ratio(self, first_bbox: list[float], second_bbox: list[float]) -> float:
        first_left, first_right = float(first_bbox[0]), float(first_bbox[2])
        second_left, second_right = float(second_bbox[0]), float(second_bbox[2])
        overlap = max(0.0, min(first_right, second_right) - max(first_left, second_left))
        first_width = max(1.0, first_right - first_left)
        second_width = max(1.0, second_right - second_left)
        min_width = min(first_width, second_width)
        return overlap / min_width

    def _build_body_region(self, image, body_lines: list[dict]):
        if not body_lines:
            return self._fallback_split(image)
        x0 = min(line["bbox"][0] for line in body_lines)
        y0 = min(line["bbox"][1] for line in body_lines)
        x1 = max(line["bbox"][2] for line in body_lines)
        y1 = max(line["bbox"][3] for line in body_lines)
        padded_bbox = (
            max(0, int(x0) - 20),
            max(0, int(y0) - 20),
            min(image.width, int(x1) + 20),
            min(image.height, int(y1) + 20),
        )
        header_crop = image.crop((0, 0, image.width, padded_bbox[1]))
        body_crop = image.crop(padded_bbox)
        body_bbox = [float(padded_bbox[0]), float(padded_bbox[1]), float(padded_bbox[2]), float(padded_bbox[3])]
        return header_crop, body_crop, body_bbox

    def _build_header_blocks(self, header_lines: list[dict], page_number: int) -> list[dict]:
        blocks: list[dict] = []
        for index, line in enumerate(header_lines, start=1):
            text = clean_prescription_line(line.get("text", ""))
            if not text:
                continue
            blocks.append(
                ExtractedBlock(
                    block_id=f"p{page_number}_hdr_{index}",
                    type="paragraph",
                    text=text,
                    source="surya_ocr",
                    confidence=round(float(line.get("confidence", 0.75)), 2),
                    page_number=page_number,
                    bbox=line.get("bbox"),
                    needs_review=float(line.get("confidence", 0.75)) < 0.70,
                    metadata={"words": line.get("words", [])},
                ).to_dict()
            )
        return blocks

    def _build_prescription_blocks(self, image, body_lines: list[dict], page_number: int) -> list[dict]:
        if not body_lines:
            return []

        columns = self._cluster_lines_by_column(body_lines, image.width)
        blocks: list[dict] = []
        block_index = 1

        for column_index, column_lines in enumerate(columns, start=1):
            region_bbox = self._column_region_bbox(column_lines, image.width, image.height)
            if region_bbox is None:
                continue
            region_crop = image.crop(region_bbox)
            region_text, region_confidence = self._ocr_region_with_augmentations(region_crop)
            region_text = clean_prescription_line(region_text)

            if not region_text:
                fallback_lines = [
                    clean_prescription_line(line.get("text", ""))
                    for line in column_lines
                    if clean_prescription_line(line.get("text", ""))
                ]
                if not fallback_lines:
                    continue
                region_text = " ".join(fallback_lines).strip()
                region_confidence = max(float(line.get("confidence", 0.6)) for line in column_lines)
                source = "surya_column_ocr"
            else:
                source = "qwen_vision_region_ocr"

            fragments = self._split_region_text(region_text)
            if not fragments:
                fragments = [region_text]
            allow_structured_items = self._column_supports_prescription_items(column_lines, fragments)

            if not allow_structured_items:
                narrative_text = self._build_narrative_column_text(column_lines)
                if not narrative_text:
                    narrative_text = "\n".join(fragment for fragment in fragments if fragment).strip()
                if narrative_text:
                    blocks.append(
                        ExtractedBlock(
                            block_id=f"p{page_number}_body_region_{block_index}",
                            type="paragraph",
                            text=narrative_text,
                            source="paddle_layout_paragraph",
                            confidence=round(max(0.55, float(region_confidence)), 2),
                            page_number=page_number,
                            bbox=[float(value) for value in region_bbox],
                            needs_review=float(region_confidence) < 0.72,
                            metadata={"column_index": column_index, "layout_strategy": "paddle_lines"},
                        ).to_dict()
                    )
                    block_index += 1
                    continue

            for fragment in fragments:
                if allow_structured_items:
                    medications = parse_medications(
                        fragment,
                        page_number=page_number,
                        block_id=f"p{page_number}_rx{block_index}",
                        confidence=region_confidence,
                    )
                    if medications:
                        for medication in medications:
                            fields = {
                                "medication": {
                                    "value": medication["medication"],
                                    "confidence": medication["confidence"],
                                },
                                "dose": {
                                    "value": medication["dose"],
                                    "confidence": medication["confidence"],
                                },
                                "route": {
                                    "value": medication["route"],
                                    "confidence": medication["confidence"],
                                },
                                "frequency": {
                                    "value": medication["frequency"],
                                    "confidence": medication["confidence"],
                                },
                                "duration": {
                                    "value": medication["duration"],
                                    "confidence": medication["confidence"],
                                },
                                "form": {
                                    "value": medication["form"],
                                    "confidence": medication["confidence"],
                                },
                                "instructions": {
                                    "value": medication["instructions"],
                                    "confidence": medication["confidence"],
                                },
                            }
                            blocks.append(
                                ExtractedBlock(
                                    block_id=medication["block_id"],
                                    type="prescription_item",
                                    text=medication["text"],
                                    source=source,
                                    confidence=medication["confidence"],
                                    page_number=page_number,
                                    bbox=[float(value) for value in region_bbox],
                                    needs_review=medication["needs_review"],
                                    fields=fields,
                                    metadata={"column_index": column_index},
                                ).to_dict()
                            )
                        block_index += 1
                        continue

                blocks.append(
                    ExtractedBlock(
                        block_id=f"p{page_number}_body_region_{block_index}",
                        type="paragraph",
                        text=fragment,
                        source=source,
                        confidence=round(max(0.55, float(region_confidence)), 2),
                        page_number=page_number,
                        bbox=[float(value) for value in region_bbox],
                        needs_review=float(region_confidence) < 0.72,
                        metadata={"column_index": column_index},
                    ).to_dict()
                )
                block_index += 1

        return blocks

    def _column_supports_prescription_items(self, column_lines: list[dict], fragments: list[str]) -> bool:
        candidates = [clean_prescription_line(line.get("text", "")) for line in column_lines]
        candidates.extend(clean_prescription_line(fragment) for fragment in fragments)
        return any(
            text and (has_medication_anchor(text) or looks_like_prescription_line(text))
            for text in candidates
        )

    def _build_narrative_column_text(self, column_lines: list[dict]) -> str:
        lines: list[str] = []
        for line in column_lines:
            text = clean_prescription_line(line.get("text", ""))
            if not text:
                continue
            lines.append(text)
        return "\n".join(lines).strip()

    def _column_region_bbox(self, column_lines: list[dict], width: int, height: int) -> tuple[int, int, int, int] | None:
        if not column_lines:
            return None
        x0 = min(float(line.get("bbox", [0, 0, 0, 0])[0]) for line in column_lines)
        y0 = min(float(line.get("bbox", [0, 0, 0, 0])[1]) for line in column_lines)
        x1 = max(float(line.get("bbox", [0, 0, 0, 0])[2]) for line in column_lines)
        y1 = max(float(line.get("bbox", [0, 0, 0, 0])[3]) for line in column_lines)
        return self._expand_bbox([x0, y0, x1, y1], width, height, pad_x=28, pad_y=20)

    def _ocr_region_with_augmentations(self, region_crop) -> tuple[str, float]:
        candidates: list[tuple[str, float]] = []
        base_text, base_confidence = self.model_registry.handwriting_ocr.extract_text(region_crop)
        if base_text:
            candidates.append((base_text, base_confidence))

        if hasattr(region_crop, "convert"):
            enhanced = ImageEnhance.Contrast(region_crop.convert("RGB")).enhance(1.4)
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.3)
            enhanced_text, enhanced_confidence = self.model_registry.handwriting_ocr.extract_text(enhanced)
            if enhanced_text:
                candidates.append((enhanced_text, enhanced_confidence))

        if not candidates:
            return "", 0.0
        best_text, best_confidence = max(candidates, key=lambda item: (len(clean_prescription_line(item[0])), item[1]))
        return best_text, best_confidence

    def _split_region_text(self, text: str) -> list[str]:
        normalized = clean_prescription_line(text)
        if not normalized:
            return []
        chunks = re.split(r"(?<=[.!?])\s+|\s{2,}|\n+", normalized)
        return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    def _ocr_prescription_line(
        self,
        image,
        line: dict,
        previous_line: dict | None = None,
        next_line: dict | None = None,
        image_height: int = 0,
    ) -> tuple[str, str, float]:
        vision_candidates: list[tuple[str, float]] = []
        for variant_bbox in self._line_crop_variants(line["bbox"], image.width, image.height):
            line_crop = image.crop(variant_bbox)
            variant_text, variant_confidence = self._ocr_line_with_augmentations(line_crop)
            variant_text = clean_prescription_line(variant_text)
            if variant_text:
                vision_candidates.append((variant_text, variant_confidence))

        if self._should_try_neighbor_join(line, next_line, image_height):
            joined_bbox = self._union_bbox(line["bbox"], next_line["bbox"])
            line_crop = image.crop(self._expand_bbox(joined_bbox, image.width, image.height, pad_x=24, pad_y=14))
            joined_text, joined_confidence = self._ocr_line_with_augmentations(line_crop)
            joined_text = clean_prescription_line(joined_text)
            if joined_text:
                vision_candidates.append((joined_text, joined_confidence))

        if self._should_try_neighbor_join(previous_line, line, image_height):
            joined_bbox = self._union_bbox(previous_line["bbox"], line["bbox"])
            line_crop = image.crop(self._expand_bbox(joined_bbox, image.width, image.height, pad_x=24, pad_y=14))
            joined_text, joined_confidence = self._ocr_line_with_augmentations(line_crop)
            joined_text = clean_prescription_line(joined_text)
            if joined_text:
                vision_candidates.append((joined_text, joined_confidence))

        vision_text, vision_confidence, vision_score = self._best_vision_candidate(vision_candidates)

        surya_text = clean_prescription_line(line.get("text", ""))
        surya_confidence = float(line.get("confidence", 0.6))
        surya_score = self._prescription_text_score(surya_text)

        if vision_text and self._is_truncated_candidate(vision_text) and surya_score > vision_score:
            vision_text = ""
            vision_confidence = 0.0
            vision_score = -10

        if vision_text and (vision_score >= surya_score or vision_confidence >= 0.72):
            return vision_text, "qwen_vision_ocr", round(max(vision_confidence, 0.55), 2)
        if surya_text:
            return surya_text, "surya_line_ocr", round(max(surya_confidence, 0.55), 2)
        if vision_text:
            return vision_text, "qwen_vision_ocr", round(max(vision_confidence, 0.55), 2)
        return surya_text, "surya_line_ocr", round(max(surya_confidence, 0.55), 2)

    def _prescription_text_score(self, text: str) -> int:
        cleaned = clean_prescription_line(text)
        if not cleaned or is_footer_noise_line(cleaned):
            return -10
        score = 0
        if looks_like_prescription_line(cleaned):
            score += 5
        if re.search(r"\b\d+\s?(mg|mcg|g|ml)\b", cleaned, re.IGNORECASE):
            score += 2
        if re.search(r"\b(od|bd|tds|stat|sos|1-0-1|1-1-1|0-1-0)\b", cleaned, re.IGNORECASE):
            score += 2
        score += min(len(cleaned.split()), 6)
        return score

    def _padded_bbox(self, bbox, width: int, height: int):
        x0, y0, x1, y1 = [int(value) for value in bbox]
        return (
            max(0, x0 - 16),
            max(0, y0 - 10),
            min(width, x1 + 16),
            min(height, y1 + 10),
        )

    def _line_crop_variants(self, bbox: list[float], width: int, height: int) -> list[tuple[int, int, int, int]]:
        tight = self._expand_bbox(bbox, width, height, pad_x=10, pad_y=6)
        medium = self._expand_bbox(bbox, width, height, pad_x=16, pad_y=10)
        wide = self._expand_bbox(bbox, width, height, pad_x=26, pad_y=14)
        variants: list[tuple[int, int, int, int]] = []
        for variant in (medium, tight, wide):
            if variant not in variants:
                variants.append(variant)
        return variants

    def _expand_bbox(
        self,
        bbox: list[float],
        width: int,
        height: int,
        pad_x: int,
        pad_y: int,
    ) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = [int(value) for value in bbox]
        return (
            max(0, x0 - pad_x),
            max(0, y0 - pad_y),
            min(width, x1 + pad_x),
            min(height, y1 + pad_y),
        )

    def _best_vision_candidate(self, candidates: list[tuple[str, float]]) -> tuple[str, float, int]:
        if not candidates:
            return "", 0.0, -10
        best_text, best_confidence = max(
            candidates,
            key=lambda candidate: (self._prescription_text_score(candidate[0]), candidate[1], len(candidate[0])),
        )
        return best_text, best_confidence, self._prescription_text_score(best_text)

    def _should_try_neighbor_join(self, first_line: dict | None, second_line: dict | None, image_height: int) -> bool:
        if not first_line or not second_line:
            return False
        first_bbox = first_line.get("bbox", [0, 0, 0, 0])
        second_bbox = second_line.get("bbox", [0, 0, 0, 0])
        vertical_gap = max(0.0, float(second_bbox[1]) - float(first_bbox[3]))
        max_gap = max(32.0, float(image_height) * 0.03)
        if vertical_gap > max_gap:
            return False
        return self._x_overlap_ratio(first_bbox, second_bbox) >= 0.40

    def _union_bbox(self, first_bbox: list[float], second_bbox: list[float]) -> list[float]:
        return [
            min(float(first_bbox[0]), float(second_bbox[0])),
            min(float(first_bbox[1]), float(second_bbox[1])),
            max(float(first_bbox[2]), float(second_bbox[2])),
            max(float(first_bbox[3]), float(second_bbox[3])),
        ]

    def _ocr_line_with_augmentations(self, line_crop) -> tuple[str, float]:
        candidates: list[tuple[str, float]] = []
        base_text, base_confidence = self.model_registry.handwriting_ocr.extract_text(line_crop)
        if base_text:
            candidates.append((base_text, base_confidence))

        if hasattr(line_crop, "convert"):
            enhanced = ImageEnhance.Contrast(line_crop.convert("RGB")).enhance(1.5)
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.4)
            enhanced_text, enhanced_confidence = self.model_registry.handwriting_ocr.extract_text(enhanced)
            if enhanced_text:
                candidates.append((enhanced_text, enhanced_confidence))

            enhanced_alt = ImageEnhance.Contrast(line_crop.convert("RGB")).enhance(1.8)
            enhanced_alt = ImageEnhance.Sharpness(enhanced_alt).enhance(1.8)
            enhanced_alt_text, enhanced_alt_confidence = self.model_registry.handwriting_ocr.extract_text(enhanced_alt)
            if enhanced_alt_text:
                candidates.append((enhanced_alt_text, enhanced_alt_confidence))

        if not candidates:
            return "", 0.0

        best_text, best_confidence = max(
            candidates,
            key=lambda candidate: (self._prescription_text_score(candidate[0]), candidate[1]),
        )
        return best_text, best_confidence

    def _is_truncated_candidate(self, text: str) -> bool:
        cleaned = clean_prescription_line(text)
        if not cleaned:
            return True
        if cleaned.endswith(("-", ":", "/", "(", ",")):
            return True
        tokens = cleaned.split()
        return len(tokens) <= 2 and not has_medication_anchor(cleaned)

    def _build_body_text_fallback_blocks(
        self,
        body_lines: list[dict],
        body_crop,
        body_bbox: list[float],
        page_number: int,
    ) -> list[dict]:
        blocks: list[dict] = []
        for index, line in enumerate(body_lines, start=1):
            text = clean_prescription_line(line.get("text", ""))
            if not text:
                continue
            blocks.append(
                ExtractedBlock(
                    block_id=f"p{page_number}_body_{index}",
                    type="paragraph",
                    text=text,
                    source="surya_line_ocr",
                    confidence=round(float(line.get("confidence", 0.6)), 2),
                    page_number=page_number,
                    bbox=line.get("bbox"),
                    needs_review=True,
                    metadata={"fallback": "body_line"},
                ).to_dict()
            )
        if blocks:
            return blocks

        body_payload = self.model_registry.scanned_ocr.ocr_image(body_crop)
        body_payload_lines = body_payload.get("lines", [])
        x_offset, y_offset = float(body_bbox[0]), float(body_bbox[1])
        for index, line in enumerate(body_payload_lines, start=1):
            text = clean_prescription_line(line.get("text", ""))
            bbox = line.get("bbox")
            if not text or not bbox:
                continue
            page_bbox = [
                round(x_offset + float(bbox[0]), 2),
                round(y_offset + float(bbox[1]), 2),
                round(x_offset + float(bbox[2]), 2),
                round(y_offset + float(bbox[3]), 2),
            ]
            blocks.append(
                ExtractedBlock(
                    block_id=f"p{page_number}_body_ocr_{index}",
                    type="paragraph",
                    text=text,
                    source="surya_crop_ocr",
                    confidence=round(float(line.get("confidence", 0.6)), 2),
                    page_number=page_number,
                    bbox=page_bbox,
                    needs_review=True,
                    metadata={"fallback": "body_crop_ocr"},
                ).to_dict()
            )
        if blocks:
            return blocks

        body_text, body_confidence = self.model_registry.handwriting_ocr.extract_text(body_crop)
        body_text = clean_prescription_line(body_text)
        if not body_text:
            return []
        return [
            ExtractedBlock(
                block_id=f"p{page_number}_body_full_1",
                type="paragraph",
                text=body_text,
                source="qwen_vision_body_fallback",
                confidence=round(max(0.55, body_confidence), 2),
                page_number=page_number,
                bbox=body_bbox,
                needs_review=True,
                metadata={"fallback": "qwen_body"},
            ).to_dict()
        ]
