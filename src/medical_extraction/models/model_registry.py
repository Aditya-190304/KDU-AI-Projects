"""Central place for lazy model-wrapper construction."""

from __future__ import annotations

from typing import Any

from medical_extraction.models.biomedical_ner import BiomedicalNerModel
from medical_extraction.models.dit_classifier import DitClassifier
from medical_extraction.models.layoutlmv3_form_extractor import LayoutLmV3FormExtractor
from medical_extraction.models.paddle_extractor import PaddleExtractor
from medical_extraction.models.trocr_extractor import TrOcrExtractor


class ModelRegistry:
    def __init__(self, device: str = "cpu") -> None:
        self.crop_classifier = DitClassifier(device=device)
        self.scanned_ocr = PaddleExtractor(device=device)
        self.form_extractor = LayoutLmV3FormExtractor(device=device)
        self.handwriting_ocr = TrOcrExtractor(device=device)
        self.biomedical_ner = BiomedicalNerModel(device=device)

    def reset_ocr_usage(self) -> None:
        self.scanned_ocr.reset_usage()
        self.form_extractor.reset_usage()
        self.handwriting_ocr.reset_usage()

    def get_ocr_usage_summary(self) -> dict[str, Any]:
        summaries = [
            self.scanned_ocr.usage_summary(),
            self.form_extractor.usage_summary(),
            self.handwriting_ocr.usage_summary(),
        ]
        total = {
            "requests": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "billable_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
        for summary in summaries:
            total["requests"] += int(summary.get("requests", 0))
            total["input_tokens"] += int(summary.get("input_tokens", 0))
            total["cached_input_tokens"] += int(summary.get("cached_input_tokens", 0))
            total["billable_input_tokens"] += int(summary.get("billable_input_tokens", 0))
            total["output_tokens"] += int(summary.get("output_tokens", 0))
            total["total_tokens"] += int(summary.get("total_tokens", 0))
            total["estimated_cost_usd"] = round(
                total["estimated_cost_usd"] + float(summary.get("estimated_cost_usd", 0.0)),
                6,
            )
        return {"totals": total, "models": summaries}
