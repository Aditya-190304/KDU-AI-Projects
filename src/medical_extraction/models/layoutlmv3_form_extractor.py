"""Form extraction backed by LayoutLMv3 FUNSD with rule fallback."""

from __future__ import annotations

import re
from typing import Any

import torch
from PIL import Image

from medical_extraction.core.types import ExtractedBlock


class LayoutLmV3FormExtractor:
    model_name = "nielsr/layoutlmv3-finetuned-funsd"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._processor = None
        self._model = None
        self._load_attempted = False

    def extract(
        self,
        image,
        text: str,
        page_number: int,
        block_id: str,
        bbox: list[float] | None = None,
        words: list[dict] | None = None,
    ) -> dict | None:
        words = words or []
        fields = self._extract_with_model(image, words) or {}
        fallback = self._fallback_fields(text)
        for key, value in fallback.items():
            fields.setdefault(key, value)
        if not fields:
            return None

        confidence = self._confidence_from_fields(fields)
        return ExtractedBlock(
            block_id=block_id,
            type="form",
            text=". ".join(f"{key}: {value['value']}" for key, value in fields.items()),
            source="layoutlmv3_funsd",
            confidence=confidence,
            page_number=page_number,
            bbox=bbox,
            needs_review=confidence < 0.70,
            fields=fields,
        ).to_dict()

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
        return None

    def _extract_with_model(self, image, words: list[dict]) -> dict[str, dict] | None:
        if not words:
            return None

        processor, model = self._load_model()
        if processor is None or model is None:
            return None

        image_pil = self._to_pil(image)
        page_width, page_height = image_pil.size
        ordered_words = self._prepare_words(words, page_width, page_height)
        if not ordered_words:
            return None

        texts = [item["text"] for item in ordered_words]
        boxes = [item["box_1000"] for item in ordered_words]

        try:
            encoded = processor(
                images=image_pil,
                words=texts,
                boxes=boxes,
                truncation=True,
                return_tensors="pt",
            )
            encoded = encoded.to(self.device)
            with torch.no_grad():
                output = model(**encoded)
            logits = output.logits[0]
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1).tolist()
            word_ids = encoded.word_ids(batch_index=0)
        except Exception:
            return None

        word_labels: dict[int, list[tuple[str, float]]] = {}
        for token_index, word_index in enumerate(word_ids):
            if word_index is None or word_index >= len(ordered_words):
                continue
            label_id = int(predictions[token_index])
            label = model.config.id2label.get(label_id, "O")
            confidence = float(probabilities[token_index, label_id].item())
            word_labels.setdefault(word_index, []).append((label, confidence))

        for index, word in enumerate(ordered_words):
            labels = word_labels.get(index, [])
            if not labels:
                word["label"] = "O"
                word["label_confidence"] = 0.0
                continue
            best_label, best_conf = max(labels, key=lambda item: item[1])
            word["label"] = best_label
            word["label_confidence"] = round(best_conf, 4)

        questions = self._collect_spans(ordered_words, "QUESTION")
        answers = self._collect_spans(ordered_words, "ANSWER")
        if not answers:
            return None

        fields: dict[str, dict] = {}
        answer_index = 0
        for question in questions:
            while answer_index < len(answers):
                answer = answers[answer_index]
                answer_index += 1
                if self._is_plausible_pair(question, answer):
                    key = self._normalize_field_key(question["text"], len(fields) + 1)
                    fields[key] = {
                        "value": answer["text"],
                        "confidence": round((question["confidence"] + answer["confidence"]) / 2.0, 2),
                    }
                    break

        if not fields:
            for index, answer in enumerate(answers, start=1):
                key = f"field_{index}"
                fields[key] = {"value": answer["text"], "confidence": round(answer["confidence"], 2)}

        return fields or None

    def _load_model(self):
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        if self._load_attempted:
            return None, None

        self._load_attempted = True
        try:
            from transformers import AutoModelForTokenClassification, AutoProcessor
        except Exception:
            return None, None

        for local_only in (False, True):
            try:
                processor = AutoProcessor.from_pretrained(self.model_name, apply_ocr=False, local_files_only=local_only)
                model = AutoModelForTokenClassification.from_pretrained(
                    self.model_name,
                    local_files_only=local_only,
                ).to(self.device).eval()
                self._processor = processor
                self._model = model
                return processor, model
            except Exception:
                continue
        return None, None

    def _prepare_words(self, words: list[dict], page_width: int, page_height: int) -> list[dict]:
        prepared: list[dict] = []
        for word in words:
            text = str(word.get("text", "")).strip()
            bbox = word.get("bbox")
            if not text or not bbox or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(value) for value in bbox]
            if x1 <= x0 or y1 <= y0:
                continue
            prepared.append(
                {
                    "text": text,
                    "bbox": [x0, y0, x1, y1],
                    "box_1000": self._normalize_bbox_1000([x0, y0, x1, y1], page_width, page_height),
                }
            )

        prepared.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        return prepared

    def _normalize_bbox_1000(self, bbox: list[float], page_width: int, page_height: int) -> list[int]:
        width = max(1.0, float(page_width))
        height = max(1.0, float(page_height))
        return [
            max(0, min(1000, int(round((bbox[0] / width) * 1000)))),
            max(0, min(1000, int(round((bbox[1] / height) * 1000)))),
            max(0, min(1000, int(round((bbox[2] / width) * 1000)))),
            max(0, min(1000, int(round((bbox[3] / height) * 1000)))),
        ]

    def _collect_spans(self, words: list[dict], target: str) -> list[dict]:
        spans: list[dict] = []
        current: dict | None = None
        for word in words:
            label = str(word.get("label", "O"))
            if not label.endswith(target):
                if current:
                    spans.append(current)
                    current = None
                continue

            confidence = float(word.get("label_confidence", 0.0))
            if current is None:
                current = {
                    "text": word["text"],
                    "bbox": word["bbox"][:],
                    "confidence_scores": [confidence],
                }
            else:
                current["text"] = f"{current['text']} {word['text']}".strip()
                current["bbox"] = [
                    min(current["bbox"][0], word["bbox"][0]),
                    min(current["bbox"][1], word["bbox"][1]),
                    max(current["bbox"][2], word["bbox"][2]),
                    max(current["bbox"][3], word["bbox"][3]),
                ]
                current["confidence_scores"].append(confidence)

        if current:
            spans.append(current)

        finalized: list[dict] = []
        for span in spans:
            span_text = str(span.get("text", "")).strip()
            if not span_text:
                continue
            scores = span.get("confidence_scores", [])
            finalized.append(
                {
                    "text": span_text,
                    "bbox": span["bbox"],
                    "confidence": sum(scores) / len(scores) if scores else 0.65,
                }
            )
        return finalized

    def _is_plausible_pair(self, question: dict, answer: dict) -> bool:
        qx0, qy0, qx1, qy1 = [float(value) for value in question["bbox"]]
        ax0, ay0, ax1, ay1 = [float(value) for value in answer["bbox"]]
        horizontal_alignment = ax0 >= qx0 - 30
        vertical_distance = abs(((ay0 + ay1) / 2.0) - ((qy0 + qy1) / 2.0))
        return horizontal_alignment and vertical_distance <= 80.0

    def _normalize_field_key(self, text: str, fallback_index: int) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "_", text.strip().lower())
        cleaned = cleaned.strip("_")
        return cleaned or f"field_{fallback_index}"

    def _fallback_fields(self, text: str) -> dict[str, dict]:
        fields: dict[str, dict] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                fields[self._normalize_field_key(key, len(fields) + 1)] = {"value": value, "confidence": 0.65}
        return fields

    def _confidence_from_fields(self, fields: dict[str, dict]) -> float:
        scores = [value.get("confidence", 0.65) for value in fields.values()]
        return round(sum(scores) / len(scores), 2) if scores else 0.65

    def _to_pil(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if hasattr(image, "convert"):
            return image.convert("RGB")
        return Image.fromarray(image).convert("RGB")
