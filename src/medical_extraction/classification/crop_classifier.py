"""DiT-backed crop classifier with heuristic fallback."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from medical_extraction.core.constants import FORM_FIELD_HINTS
from medical_extraction.models.runtime import resolve_torch_device


class CropClassifier:
    model_name = "microsoft/dit-base-finetuned-rvlcdip"

    def __init__(self, device: str = "cpu") -> None:
        self.device = resolve_torch_device(device)
        self._processor = None
        self._model = None
        self._load_error: str | None = None

    def classify(self, image: Any, extracted_text: str = "") -> dict[str, Any]:
        raw_label = None
        confidence = 0.0
        try:
            processor, model = self._load()
            inputs = processor(images=image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1)[0]
                score, predicted_id = probabilities.max(dim=-1)
            raw_label = model.config.id2label[int(predicted_id)].lower()
            confidence = float(score.item())
        except Exception as exc:  # pragma: no cover - network/model dependent
            self._load_error = str(exc)

        if raw_label:
            predicted_class = self._map_label(raw_label)
            return {
                "model": self.model_name,
                "predicted_class": predicted_class,
                "raw_label": raw_label,
                "confidence": round(confidence, 2),
                "fallback_used": False,
            }

        predicted_class, heuristic_confidence = self._heuristic_classification(image, extracted_text)
        payload = {
            "model": self.model_name,
            "predicted_class": predicted_class,
            "confidence": round(heuristic_confidence, 2),
            "fallback_used": True,
        }
        if self._load_error:
            payload["warning"] = self._load_error
        return payload

    def _load(self):
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        self._processor = AutoImageProcessor.from_pretrained(self.model_name, local_files_only=True)
        self._model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            local_files_only=True,
        ).to(self.device)
        self._model.eval()
        return self._processor, self._model

    def _map_label(self, raw_label: str) -> str:
        if any(token in raw_label for token in ("handwritten",)):
            return "handwritten-like image"
        if any(token in raw_label for token in ("questionnaire", "form")):
            return "form-like image"
        if any(token in raw_label for token in ("memo", "letter", "scientific", "report", "file_folder")):
            return "report/letter/memo-like image"
        if any(token in raw_label for token in ("invoice", "budget", "presentation", "advertisement", "news_article")):
            return "printed text image"
        return "unknown/review"

    def _heuristic_classification(self, image: Any, extracted_text: str = "") -> tuple[str, float]:
        width = getattr(image, "width", 0)
        height = getattr(image, "height", 0)
        area = width * height
        lowered = extracted_text.lower()

        if area < 15_000:
            predicted_class = "logo/stamp/signature/noise"
            confidence = 0.65
        elif any(hint in lowered for hint in FORM_FIELD_HINTS):
            predicted_class = "form-like image"
            confidence = 0.78
        elif any(char.isdigit() for char in lowered) and ":" not in lowered and width > height:
            predicted_class = "report/letter/memo-like image"
            confidence = 0.72
        elif ":" in lowered:
            predicted_class = "printed text image"
            confidence = 0.70
        else:
            predicted_class = "unknown/review"
            confidence = 0.55

        return predicted_class, confidence
