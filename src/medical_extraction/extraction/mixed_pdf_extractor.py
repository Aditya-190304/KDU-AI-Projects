"""Extractor for copyable pages that also contain embedded image regions."""

from __future__ import annotations

from pathlib import Path

from medical_extraction.extraction.copyable_pdf_extractor import CopyablePdfExtractor
from medical_extraction.extraction.merge_engine import merge_blocks
from medical_extraction.models.model_registry import ModelRegistry
from medical_extraction.parsing.medication_parser import clean_prescription_line
from medical_extraction.utils.image_utils import preprocess_image, safe_crop
from medical_extraction.utils.pdf_utils import page_to_image, scale_bbox_to_image


class MixedPdfExtractor:
    def __init__(self, model_registry: ModelRegistry) -> None:
        self.model_registry = model_registry
        self.copyable_extractor = CopyablePdfExtractor()

    def extract(
        self,
        page,
        input_path: str,
        page_number: int,
        debug_dir: str | None = None,
        save_debug_images: bool = False,
    ) -> tuple[list[dict], list[str]]:
        blocks = self.copyable_extractor.extract(page, input_path, page_number, include_tables=True)
        warnings: list[str] = []

        rendered_page = page_to_image(page)
        image_blocks = [block for block in page.get_text("dict").get("blocks", []) if block.get("type") == 1]

        for index, image_block in enumerate(image_blocks, start=1):
            page_bbox = image_block.get("bbox")
            if not page_bbox:
                continue
            crop_bbox = scale_bbox_to_image(page, rendered_page, page_bbox)
            crop = preprocess_image(safe_crop(rendered_page, crop_bbox))
            if save_debug_images and debug_dir:
                Path(debug_dir).mkdir(parents=True, exist_ok=True)
                crop.save(Path(debug_dir) / f"page_{page_number}_crop_{index}.png")

            crop_classification = self.model_registry.crop_classifier.classify(crop)
            ocr_payload = self.model_registry.scanned_ocr.ocr_image(crop)
            ocr_blocks = self.model_registry.scanned_ocr.extract_text(
                crop,
                page_number=page_number,
                block_id=f"p{page_number}_img{index}",
                bbox=[round(value, 2) for value in page_bbox],
            )
            extracted_text = ocr_payload["text"]
            layout_boxes = self.model_registry.scanned_ocr.detect_layout(crop)
            absolute_layout_boxes = self._to_absolute_layout_boxes(
                layout_boxes=layout_boxes,
                page_bbox=[round(value, 2) for value in page_bbox],
            )

            predicted_class = crop_classification["predicted_class"]
            if predicted_class == "logo/stamp/signature/noise" and not extracted_text:
                continue

            if predicted_class in {"report/letter/memo-like image", "printed text image"} or any(
                str(box.get("label", "")).lower() == "table" for box in layout_boxes
            ):
                table_block = self.model_registry.scanned_ocr.extract_table(
                    crop,
                    page_number=page_number,
                    block_id=f"p{page_number}_t{index}",
                    bbox=[round(value, 2) for value in page_bbox],
                )
                if table_block:
                    table_block["crop_classifier"] = crop_classification
                    metadata = table_block.get("metadata") or {}
                    metadata["layout_label"] = "Table"
                    table_block["metadata"] = metadata
                    blocks.append(table_block)

            if predicted_class == "form-like image":
                form_block = self.model_registry.form_extractor.extract(
                    crop,
                    extracted_text,
                    page_number=page_number,
                    block_id=f"p{page_number}_f{index}",
                    bbox=[round(value, 2) for value in page_bbox],
                    words=ocr_payload["words"],
                )
                if form_block:
                    form_block["crop_classifier"] = crop_classification
                    metadata = form_block.get("metadata") or {}
                    metadata["layout_label"] = "Form"
                    form_block["metadata"] = metadata
                    blocks.append(form_block)
                    continue

            if predicted_class == "handwritten-like image":
                handwritten_blocks = self._extract_handwritten_crop_lines(
                    crop=crop,
                    page_number=page_number,
                    crop_index=index,
                    page_bbox=[round(value, 2) for value in page_bbox],
                    crop_classifier=crop_classification,
                )
                if handwritten_blocks:
                    blocks.extend(handwritten_blocks)
                continue

            if not ocr_blocks:
                warnings.append(f"Embedded image crop {index} produced no OCR text.")
                continue

            for block in ocr_blocks:
                block["crop_classifier"] = crop_classification
                self._attach_layout_metadata(block, absolute_layout_boxes)
                blocks.append(block)

        return merge_blocks(blocks), warnings

    def _extract_handwritten_crop_lines(
        self,
        crop,
        page_number: int,
        crop_index: int,
        page_bbox: list[float],
        crop_classifier: dict,
    ) -> list[dict]:
        payload = self.model_registry.scanned_ocr.ocr_image(crop)
        lines = sorted(payload.get("lines", []), key=lambda item: (item["bbox"][1], item["bbox"][0]))
        if not lines:
            text, confidence = self.model_registry.handwriting_ocr.extract_text(crop)
            text = clean_prescription_line(text)
            if not text:
                return []
            return [
                {
                    "block_id": f"p{page_number}_rx{crop_index}",
                    "type": "image_ocr",
                    "text": text,
                    "source": "qwen_vision_ocr",
                    "confidence": round(confidence, 2),
                    "page_number": page_number,
                    "bbox": page_bbox,
                    "needs_review": confidence < 0.70,
                    "crop_classifier": crop_classifier,
                }
            ]

        extracted_blocks: list[dict] = []
        for line_index, line in enumerate(lines, start=1):
            bbox = line.get("bbox")
            if not bbox:
                continue
            line_crop = crop.crop(self._padded_bbox(bbox, crop.width, crop.height))
            text, confidence = self.model_registry.handwriting_ocr.extract_text(line_crop)
            text = clean_prescription_line(text) if text else clean_prescription_line(line.get("text", ""))
            if not text:
                continue
            mapped_bbox = self._map_crop_bbox_to_page_bbox(bbox, page_bbox)
            extracted_blocks.append(
                {
                    "block_id": f"p{page_number}_rx{crop_index}_{line_index}",
                    "type": "image_ocr",
                    "text": text,
                    "source": "qwen_vision_ocr",
                    "confidence": round(max(confidence, float(line.get("confidence", 0.6))), 2),
                    "page_number": page_number,
                    "bbox": mapped_bbox,
                    "needs_review": confidence < 0.70,
                    "crop_classifier": crop_classifier,
                    "metadata": {"line_index": line_index},
                }
            )
        return extracted_blocks

    def _map_crop_bbox_to_page_bbox(self, line_bbox: list[float], page_bbox: list[float]) -> list[float]:
        x0, y0 = page_bbox[0], page_bbox[1]
        return [
            round(x0 + float(line_bbox[0]), 2),
            round(y0 + float(line_bbox[1]), 2),
            round(x0 + float(line_bbox[2]), 2),
            round(y0 + float(line_bbox[3]), 2),
        ]

    def _padded_bbox(self, bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = [int(value) for value in bbox]
        return (
            max(0, x0 - 16),
            max(0, y0 - 10),
            min(width, x1 + 16),
            min(height, y1 + 10),
        )

    def _to_absolute_layout_boxes(self, layout_boxes: list[dict], page_bbox: list[float]) -> list[dict]:
        x0, y0 = page_bbox[0], page_bbox[1]
        absolute_boxes: list[dict] = []
        for box in layout_boxes:
            bbox = box.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            absolute_boxes.append(
                {
                    **box,
                    "bbox": [
                        round(x0 + float(bbox[0]), 2),
                        round(y0 + float(bbox[1]), 2),
                        round(x0 + float(bbox[2]), 2),
                        round(y0 + float(bbox[3]), 2),
                    ],
                }
            )
        return absolute_boxes

    def _attach_layout_metadata(self, block: dict, layout_boxes: list[dict]) -> None:
        bbox = block.get("bbox")
        if not bbox:
            return
        best = None
        best_score = 0.0
        for box in layout_boxes:
            candidate_bbox = box.get("bbox")
            if not candidate_bbox:
                continue
            score = self._intersection_ratio(bbox, candidate_bbox)
            if score > best_score:
                best_score = score
                best = box
        if not best or best_score <= 0.15:
            return
        metadata = block.get("metadata") or {}
        metadata["layout_label"] = best.get("label", "Other")
        metadata["layout_position"] = best.get("position")
        block["metadata"] = metadata

    def _intersection_ratio(self, first: list[float], second: list[float]) -> float:
        x_left = max(float(first[0]), float(second[0]))
        y_top = max(float(first[1]), float(second[1]))
        x_right = min(float(first[2]), float(second[2]))
        y_bottom = min(float(first[3]), float(second[3]))
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        first_area = max(1.0, (float(first[2]) - float(first[0])) * (float(first[3]) - float(first[1])))
        return intersection / first_area
