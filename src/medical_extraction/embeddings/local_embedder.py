"""Local Hugging Face embedding wrapper."""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

from medical_extraction.models.runtime import resolve_torch_device


class LocalTextEmbedder:
    def __init__(self, model_name: str, device: str = "cpu", local_files_only: bool = False, max_length: int = 512) -> None:
        self.model_name = model_name
        self.device = resolve_torch_device(device)
        self.local_files_only = local_files_only
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self.load_error = ""

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            self._load()
        except Exception as exc:  # pragma: no cover - depends on local cache/network
            self.load_error = str(exc)
            raise
        assert self._tokenizer is not None
        assert self._model is not None

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self._model(**encoded)
        embeddings = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()

    def encode(self, texts: list[str]) -> torch.Tensor | None:
        try:
            vectors = self.encode_texts(texts)
        except Exception:  # pragma: no cover
            return None
        if not vectors:
            return None
        return torch.tensor(vectors, dtype=torch.float32)

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=self.local_files_only)
        self._model = AutoModel.from_pretrained(self.model_name, local_files_only=self.local_files_only)
        self._model.to(self.device)
        self._model.eval()


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts
