"""Extractor for scanned report, table, and form pages."""

from __future__ import annotations

from pathlib import Path

from medical_extraction.extraction.merge_engine import merge_blocks
from medical_extraction.models.model_registry import ModelRegistry
from medical_extraction.utils.image_utils import preprocess_image
from medical_extraction.utils.pdf_utils import page_to_image


class ScannedPageExtractor:
    def __init__(self, model_registry: ModelRegistry) -> None:
        self.model_registry = model_registry

    def extract(
        self,
        page,
        page_number: int,
        debug_dir: str | None = None,
        save_debug_images: bool = False,
    ) -> tuple[list[dict], list[str]]:
        image = preprocess_image(page_to_image(page))
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

        if save_debug_images and debug_dir:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            image.save(Path(debug_dir) / f"page_{page_number}_render.png")

        page_bbox = [0.0, 0.0, float(image.width), float(image.height)]
        ocr_payload = self.model_registry.scanned_ocr.ocr_image(image)
        blocks = self.model_registry.scanned_ocr.extract_text(
            image,
            page_number=page_number,
            block_id=f"p{page_number}_b1",
            bbox=page_bbox,
        )
        layout_boxes = self.model_registry.scanned_ocr.detect_layout(image)
        self._attach_layout_metadata(blocks, layout_boxes)

        if not blocks:
            warnings.append("Scanned page OCR returned no text.")
        else:
            for layout_index, layout_box in enumerate(layout_boxes, start=1):
                if str(layout_box.get("label", "")).lower() == "table":
                    table_block = self.model_registry.scanned_ocr.extract_table(
                        image.crop(tuple(int(value) for value in layout_box["bbox"])),
                        page_number=page_number,
                        block_id=f"p{page_number}_t{layout_index}",
                        bbox=layout_box["bbox"],
                    )
                    if table_block:
                        blocks.append(table_block)

        extracted_text = ocr_payload["text"]
        form_block = self.model_registry.form_extractor.extract(
            image,
            extracted_text,
            page_number=page_number,
            block_id=f"p{page_number}_f1",
            bbox=page_bbox,
            words=ocr_payload["words"],
        )
        if form_block:
            blocks.append(form_block)

        return merge_blocks(blocks), warnings

    def _attach_layout_metadata(self, blocks: list[dict], layout_boxes: list[dict]) -> None:
        for block in blocks:
            bbox = block.get("bbox")
            if not bbox:
                continue
            match = self._best_layout_match(bbox, layout_boxes)
            if not match:
                continue
            metadata = block.get("metadata") or {}
            metadata["layout_label"] = match.get("label", "Other")
            metadata["layout_position"] = match.get("position")
            block["metadata"] = metadata

    def _best_layout_match(self, bbox: list[float], layout_boxes: list[dict]) -> dict | None:
        best = None
        best_score = 0.0
        for candidate in layout_boxes:
            candidate_bbox = candidate.get("bbox")
            if not candidate_bbox:
                continue
            score = self._intersection_ratio(bbox, candidate_bbox)
            if score > best_score:
                best_score = score
                best = candidate
        return best if best_score > 0.15 else None

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
