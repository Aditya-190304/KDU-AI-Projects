"""Local cross-encoder reranker for hydrated candidate chunks."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from medical_extraction.models.runtime import resolve_torch_device


class LocalCrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        local_files_only: bool = False,
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_torch_device(device)
        self.local_files_only = local_files_only
        self.max_length = max_length
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None

    def rerank(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        text_field: str = "redacted_text",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None

        scored_candidates: list[dict[str, Any]] = []
        pairs = [(query_text, str(candidate.get(text_field, ""))) for candidate in candidates]

        for batch_start in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[batch_start : batch_start + self.batch_size]
            encoded = self._tokenizer(
                [pair[0] for pair in batch_pairs],
                [pair[1] for pair in batch_pairs],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self._model(**encoded).logits
            batch_scores = logits.squeeze(-1).detach().cpu().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            for candidate, score in zip(candidates[batch_start : batch_start + self.batch_size], batch_scores):
                candidate_with_score = dict(candidate)
                candidate_with_score["rerank_score"] = float(score)
                scored_candidates.append(candidate_with_score)

        scored_candidates.sort(key=lambda candidate: candidate.get("rerank_score", float("-inf")), reverse=True)
        if top_k is not None:
            return scored_candidates[:top_k]
        return scored_candidates

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=self.local_files_only)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        self._model.to(self.device)
        self._model.eval()
