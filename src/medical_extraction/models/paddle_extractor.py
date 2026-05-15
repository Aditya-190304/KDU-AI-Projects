"""Paddle-backed OCR, layout, and table recognition helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from medical_extraction.core.types import ExtractedBlock


class PaddleExtractor:
    model_name = "paddleocr_v5"

    def __init__(self, device: str = "cpu") -> None:
        temp_root = os.environ.get("TEMP") or os.environ.get("TMP")
        if temp_root:
            default_cache_home = Path(temp_root) / "medical_extraction_paddlex_cache"
        else:
            default_cache_home = Path(os.getcwd()) / ".model_cache" / "paddlex"
        cache_home = os.environ.get("MEDICAL_PADDLE_CACHE_HOME", str(default_cache_home))
        Path(cache_home).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", cache_home)
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        self.device = device
        self._paddle_device = self._to_paddle_device(device)
        self._ocr_engine = None
        self._structure_engine = None
        self._ocr_error: str | None = None
        self._layout_error: str | None = None
        self._table_error: str | None = None
        self._ocr_cache: dict[int, dict[str, Any]] = {}
        self._use_structure = os.environ.get("MEDICAL_PADDLE_USE_STRUCTURE", "true").strip().lower() == "true"
        self._offline_only = os.environ.get("MEDICAL_PADDLE_OFFLINE_ONLY", "true").strip().lower() == "true"
        self._paddle_cache_home = Path(os.environ["PADDLE_PDX_CACHE_HOME"])

    def ocr_image(self, image: Any) -> dict[str, Any]:
        cache_key = id(image)
        cached = self._ocr_cache.get(cache_key)
        if cached is not None:
            return cached

        if not self._ensure_ocr_engine():
            return {"text": "", "lines": [], "words": []}

        try:
            image_pil = self._to_pil(image)
            result_list = self._ocr_engine.predict(np.asarray(image_pil))
            lines, words = self._parse_ocr_result_list(result_list)
        except Exception as exc:
            self._ocr_error = str(exc)
            return {"text": "", "lines": [], "words": []}

        lines.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        text = "\n".join(line["text"] for line in lines).strip()
        payload = {"text": text, "lines": lines, "words": words}
        self._ocr_cache[cache_key] = payload
        return payload

    def detect_layout(self, image: Any) -> list[dict[str, Any]]:
        image_pil = self._to_pil(image)
        width, height = image_pil.size

        if self._ensure_structure_engine():
            try:
                result_list = self._structure_engine.predict(
                    np.asarray(image_pil),
                    use_table_recognition=False,
                    use_formula_recognition=False,
                    use_chart_recognition=False,
                    use_seal_recognition=False,
                )
                boxes = self._parse_structure_layout_boxes(result_list, width=width, height=height)
                if boxes:
                    boxes.sort(key=lambda item: (int(item.get("position", 9999)), item["bbox"][1], item["bbox"][0]))
                    return boxes
            except Exception as exc:
                self._layout_error = str(exc)

        ocr_payload = self.ocr_image(image_pil)
        return self._layout_from_ocr_lines(ocr_payload.get("lines", []), width=width, height=height)

    def extract_table(self, image: Any, page_number: int, block_id: str, bbox: list[float] | None = None) -> dict | None:
        image_pil = self._to_pil(image)

        if self._ensure_structure_engine():
            try:
                result_list = self._structure_engine.predict(
                    np.asarray(image_pil),
                    use_table_recognition=True,
                    use_formula_recognition=False,
                    use_chart_recognition=False,
                    use_seal_recognition=False,
                )
                table_block = self._table_block_from_structure_result(
                    result_list=result_list,
                    page_number=page_number,
                    block_id=block_id,
                    bbox=bbox,
                )
                if table_block:
                    return table_block
            except Exception as exc:
                self._table_error = str(exc)

        return None

    def extract_text(self, image: Any, page_number: int, block_id: str, bbox: list[float] | None = None) -> list[dict]:
        payload = self.ocr_image(image)
        if not payload["lines"]:
            return []

        blocks = []
        multi_line = len(payload["lines"]) > 1
        for index, line in enumerate(payload["lines"], start=1):
            block_bbox = self._merge_bbox(bbox, line["bbox"]) if bbox else line["bbox"]
            blocks.append(
                ExtractedBlock(
                    block_id=block_id if not multi_line else f"{block_id}_{index}",
                    type="paragraph",
                    text=line["text"],
                    source="paddle_ocr",
                    confidence=round(float(line["confidence"]), 2),
                    page_number=page_number,
                    bbox=block_bbox,
                    needs_review=float(line["confidence"]) < 0.70,
                    metadata={"words": line.get("words", [])},
                ).to_dict()
            )
        return blocks

    def usage_summary(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "requests": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "billable_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    def reset_usage(self) -> None:
        self._ocr_cache.clear()

    def get_last_ocr_error(self) -> str | None:
        return self._ocr_error

    def _ensure_ocr_engine(self) -> bool:
        if self._ocr_engine is not None:
            return True
        if self._ocr_error:
            return False
        if self._offline_only and not self._has_local_ocr_models():
            self._ocr_error = (
                "Required local Paddle OCR models not found/readable in cache. "
                "Set MEDICAL_PADDLE_OFFLINE_ONLY=false to allow auto-download."
            )
            return False
        try:
            from paddleocr import PaddleOCR

            self._ocr_engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang="en",
                device=self._paddle_device,
            )
            return True
        except Exception as exc:
            self._ocr_error = str(exc)
            return False

    def _has_local_ocr_models(self) -> bool:
        required = [
            self._paddle_cache_home / "official_models" / "PP-OCRv5_server_det" / "inference.yml",
            self._paddle_cache_home / "official_models" / "en_PP-OCRv5_mobile_rec" / "inference.yml",
        ]
        for file_path in required:
            try:
                if not file_path.exists():
                    return False
                with file_path.open("r", encoding="utf-8") as handle:
                    handle.read(1)
            except Exception:
                return False
        return True

    def _ensure_structure_engine(self) -> bool:
        if not self._use_structure:
            return False
        if self._structure_engine is not None:
            return True
        if self._layout_error and self._table_error:
            return False
        try:
            from paddleocr import PPStructureV3

            self._structure_engine = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_formula_recognition=False,
                use_chart_recognition=False,
                use_seal_recognition=False,
                device=self._paddle_device,
            )
            return True
        except Exception as exc:
            message = str(exc)
            self._layout_error = message
            self._table_error = message
            return False

    def _parse_ocr_result_list(self, result_list: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        lines: list[dict[str, Any]] = []
        words: list[dict[str, Any]] = []

        for result in list(result_list or []):
            result_dict = self._result_to_dict(result)
            payload = result_dict.get("res", result_dict)
            texts = list(payload.get("rec_texts") or [])
            scores = list(payload.get("rec_scores") or [])
            boxes = self._extract_boxes_from_payload(payload)

            for index, text in enumerate(texts):
                cleaned = str(text).strip()
                if not cleaned:
                    continue
                confidence = self._score_at(scores, index, default=0.70)
                bbox = boxes[index] if index < len(boxes) else None
                if bbox is None:
                    continue
                word_payload = {
                    "text": cleaned,
                    "confidence": round(confidence, 4),
                    "bbox": bbox,
                }
                words.append(word_payload)
                lines.append(
                    {
                        "text": cleaned,
                        "confidence": round(confidence, 4),
                        "bbox": bbox,
                        "words": [word_payload],
                    }
                )

        return lines, words

    def _parse_structure_layout_boxes(self, result_list: Any, width: int, height: int) -> list[dict[str, Any]]:
        boxes: list[dict[str, Any]] = []

        for result in list(result_list or []):
            result_dict = self._result_to_dict(result)
            payload = result_dict.get("res", result_dict)

            parsing_items = list(payload.get("parsing_res_list") or [])
            for index, item in enumerate(parsing_items, start=1):
                bbox = self._coerce_bbox(item.get("block_bbox") or item.get("layout_bbox"))
                if bbox is None:
                    continue
                label = self._normalize_layout_label(item.get("block_label") or item.get("layout") or "Other")
                position = item.get("block_order")
                if position is None:
                    position = item.get("index")
                if position is None:
                    position = index
                boxes.append(
                    {
                        "label": label,
                        "confidence": 0.85,
                        "bbox": self._clip_bbox(bbox, width=width, height=height),
                        "position": int(position),
                    }
                )

            layout_det = payload.get("layout_det_res")
            if isinstance(layout_det, dict):
                layout_boxes = list(layout_det.get("boxes") or [])
                for index, item in enumerate(layout_boxes, start=1):
                    bbox = self._coerce_bbox(item.get("coordinate"))
                    if bbox is None:
                        continue
                    boxes.append(
                        {
                            "label": self._normalize_layout_label(item.get("label", "Other")),
                            "confidence": round(float(item.get("score", 0.85) or 0.85), 2),
                            "bbox": self._clip_bbox(bbox, width=width, height=height),
                            "position": index,
                        }
                    )

        if boxes:
            return boxes
        return []

    def _table_block_from_structure_result(
        self,
        result_list: Any,
        page_number: int,
        block_id: str,
        bbox: list[float] | None,
    ) -> dict | None:
        for result in list(result_list or []):
            result_dict = self._result_to_dict(result)
            payload = result_dict.get("res", result_dict)
            table_items = list(payload.get("table_res_list") or [])
            for table in table_items:
                table_ocr = table.get("table_ocr_pred") if isinstance(table, dict) else None
                if not isinstance(table_ocr, dict):
                    continue
                rec_texts = [str(value).strip() for value in list(table_ocr.get("rec_texts") or []) if str(value).strip()]
                rec_scores = list(table_ocr.get("rec_scores") or [])
                if not rec_texts:
                    continue
                confidence = round(float(np.mean(rec_scores)), 2) if rec_scores else 0.75
                html = str(table.get("pred_html", "")).strip() if isinstance(table, dict) else ""
                return ExtractedBlock(
                    block_id=block_id,
                    type="table",
                    text="\n".join(rec_texts),
                    source="paddle_table_recognition",
                    confidence=confidence,
                    page_number=page_number,
                    bbox=bbox,
                    needs_review=confidence < 0.70,
                    title="Detected Table",
                    structured_data={"html": html} if html else None,
                ).to_dict()
        return None

    def _layout_from_ocr_lines(self, lines: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
        if not lines:
            return []

        y_values = [float(line["bbox"][1]) for line in lines if line.get("bbox")]
        bottom_values = [float(line["bbox"][3]) for line in lines if line.get("bbox")]
        if not y_values or not bottom_values:
            return []

        header_limit = max(40.0, height * 0.18)
        header_lines = [line for line in lines if float(line["bbox"][1]) <= header_limit]
        boxes: list[dict[str, Any]] = []

        if header_lines:
            boxes.append(
                {
                    "label": "Header",
                    "confidence": 0.78,
                    "bbox": [
                        0.0,
                        0.0,
                        float(width),
                        round(min(float(height), max(float(line["bbox"][3]) for line in header_lines) + 8.0), 2),
                    ],
                    "position": 1,
                }
            )

        boxes.append(
            {
                "label": "Paragraph",
                "confidence": 0.72,
                "bbox": [0.0, round(min(y_values), 2), float(width), round(max(bottom_values), 2)],
                "position": 2,
            }
        )
        return boxes

    def _extract_boxes_from_payload(self, payload: dict[str, Any]) -> list[list[float]]:
        boxes: list[list[float]] = []
        raw_boxes = payload.get("rec_boxes")
        if raw_boxes is not None:
            try:
                box_array = np.asarray(raw_boxes)
                if box_array.ndim == 2 and box_array.shape[1] >= 4:
                    for row in box_array:
                        bbox = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
                        boxes.append([round(value, 2) for value in bbox])
            except Exception:
                pass
        if boxes:
            return boxes

        raw_polys = payload.get("rec_polys") or payload.get("dt_polys") or []
        for poly in list(raw_polys):
            bbox = self._polygon_to_bbox(poly)
            if bbox is not None:
                boxes.append(bbox)
        return boxes

    def _result_to_dict(self, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return result

        if hasattr(result, "json"):
            try:
                value = result.json
                if callable(value):
                    value = value()
                if isinstance(value, str):
                    return json.loads(value)
                if isinstance(value, dict):
                    return value
            except Exception:
                pass

        if hasattr(result, "res"):
            res_value = getattr(result, "res")
            if isinstance(res_value, dict):
                return {"res": res_value}

        if hasattr(result, "to_dict"):
            try:
                value = result.to_dict()
                if isinstance(value, dict):
                    return value
            except Exception:
                pass

        return {}

    def _score_at(self, scores: list[Any], index: int, default: float) -> float:
        if index >= len(scores):
            return default
        try:
            return float(scores[index])
        except (TypeError, ValueError):
            return default

    def _coerce_bbox(self, value: Any) -> list[float] | None:
        if value is None:
            return None
        try:
            points = np.asarray(value, dtype=float)
        except Exception:
            return None

        if points.ndim == 1 and points.size >= 4:
            return [round(float(points[0]), 2), round(float(points[1]), 2), round(float(points[2]), 2), round(float(points[3]), 2)]

        if points.ndim == 2 and points.shape[1] >= 2:
            x_values = points[:, 0]
            y_values = points[:, 1]
            return [
                round(float(np.min(x_values)), 2),
                round(float(np.min(y_values)), 2),
                round(float(np.max(x_values)), 2),
                round(float(np.max(y_values)), 2),
            ]
        return None

    def _clip_bbox(self, bbox: list[float], width: int, height: int) -> list[float]:
        return [
            round(max(0.0, min(float(width), float(bbox[0]))), 2),
            round(max(0.0, min(float(height), float(bbox[1]))), 2),
            round(max(0.0, min(float(width), float(bbox[2]))), 2),
            round(max(0.0, min(float(height), float(bbox[3]))), 2),
        ]

    def _to_paddle_device(self, device: str) -> str:
        lowered = str(device).strip().lower()
        if lowered in {"cuda", "gpu"} or lowered.startswith("cuda"):
            return "gpu"
        return "cpu"

    def _to_pil(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if hasattr(image, "convert"):
            return image.convert("RGB")
        return Image.fromarray(np.asarray(image)).convert("RGB")

    def _polygon_to_bbox(self, polygon: Any) -> list[float] | None:
        if polygon is None:
            return None
        try:
            points = np.asarray(polygon, dtype=float)
        except Exception:
            return None

        if points.ndim == 2 and points.shape[1] >= 2:
            x_values = points[:, 0]
            y_values = points[:, 1]
            return [
                round(float(np.min(x_values)), 2),
                round(float(np.min(y_values)), 2),
                round(float(np.max(x_values)), 2),
                round(float(np.max(y_values)), 2),
            ]
        if points.ndim == 1 and points.size >= 4:
            return [round(float(points[0]), 2), round(float(points[1]), 2), round(float(points[2]), 2), round(float(points[3]), 2)]
        return None

    def _normalize_layout_label(self, raw_label: str) -> str:
        normalized = str(raw_label).strip().lower()
        if normalized == "table":
            return "Table"
        if normalized in {"title", "header", "section_header", "page_header", "doc_title", "paragraph_title"}:
            return "Header"
        if normalized in {"footer", "page_footer", "footer_image"}:
            return "Footer"
        if normalized in {"figure", "image", "chart", "figure_title"}:
            return "Figure"
        if normalized in {"text", "paragraph", "list"}:
            return "Paragraph"
        if normalized in {"form", "key_value", "question_answer"}:
            return "Form"
        return "Other"

    def _merge_bbox(self, base_bbox: list[float] | None, relative_bbox: list[float] | None) -> list[float] | None:
        if not relative_bbox:
            return base_bbox
        if not base_bbox:
            return relative_bbox
        x0, y0, _, _ = base_bbox
        return [
            round(x0 + relative_bbox[0], 2),
            round(y0 + relative_bbox[1], 2),
            round(x0 + relative_bbox[2], 2),
            round(y0 + relative_bbox[3], 2),
        ]
