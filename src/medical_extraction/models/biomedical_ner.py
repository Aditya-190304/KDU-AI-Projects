"""Biomedical NER model wrapper with rule-based fallback."""

from __future__ import annotations

import re

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from medical_extraction.models.runtime import inference_device_index


class BiomedicalNerModel:
    model_name = "d4data/biomedical-ner-all"

    SYMPTOMS = {"chest pain", "fatigue", "thirst", "fever", "cough", "headache"}
    DISEASES = {"diabetes", "hypertension", "asthma"}
    MEDICATIONS = {"metformin", "warfarin", "amoxicillin", "insulin"}
    LABS = {"hba1c", "glucose", "ldl", "creatinine"}

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._pipeline = None

    def extract_entities(self, text: str, page_number: int, block_id: str) -> list[dict]:
        model_entities = self._extract_with_model(text, page_number, block_id)
        if model_entities:
            return model_entities

        lowered = text.lower()
        entities: list[dict] = []
        for label, terms in (
            ("SYMPTOM", self.SYMPTOMS),
            ("DISEASE", self.DISEASES),
            ("MEDICATION", self.MEDICATIONS),
            ("LAB_TEST", self.LABS),
        ):
            for term in terms:
                if re.search(rf"\b{re.escape(term)}\b", lowered):
                    entities.append(
                        {
                            "text": term,
                            "type": label,
                            "page_number": page_number,
                            "block_id": block_id,
                            "confidence": 0.72,
                        }
                    )
        return entities

    def _extract_with_model(self, text: str, page_number: int, block_id: str) -> list[dict]:
        if not text.strip():
            return []
        try:
            ner_pipeline = self._load()
            predictions = ner_pipeline(text)
            entities = []
            for prediction in predictions:
                label = str(prediction.get("entity_group", prediction.get("entity", "UNKNOWN"))).upper()
                entities.append(
                    {
                        "text": prediction["word"],
                        "type": label,
                        "page_number": page_number,
                        "block_id": block_id,
                        "confidence": round(float(prediction["score"]), 2),
                    }
                )
            return entities
        except Exception:  # pragma: no cover - model/network dependent
            return []

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            local_files_only=True,
        )
        self._pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=inference_device_index(self.device),
        )
        return self._pipeline
