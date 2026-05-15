"""OpenAI embeddings wrapper for persisted retrieval indexes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib import error, request


@dataclass(slots=True)
class OpenAIEmbeddingSettings:
    api_key: str
    model: str = "text-embedding-3-small"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 60
    dimensions: int | None = 1536
    batch_size: int = 32

    @classmethod
    def from_env_and_config(cls, config: dict[str, Any] | None = None) -> "OpenAIEmbeddingSettings":
        config = config or {}
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or your shell environment.")

        env_dimensions = os.environ.get("OPENAI_EMBEDDING_DIMENSIONS", "").strip()
        config_dimensions = config.get("embedding_dimensions")
        dimensions: int | None
        if env_dimensions:
            dimensions = int(env_dimensions)
        elif config_dimensions is not None:
            dimensions = int(config_dimensions)
        else:
            dimensions = 1536

        return cls(
            api_key=api_key,
            model=str(
                os.environ.get("OPENAI_EMBEDDING_MODEL", config.get("embedding_model", "text-embedding-3-small"))
            ).strip()
            or "text-embedding-3-small",
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip() or "https://api.openai.com/v1",
            timeout_seconds=int(os.environ.get("OPENAI_TIMEOUT_SECONDS", str(config.get("timeout_seconds", 60)))),
            dimensions=dimensions,
            batch_size=int(config.get("batch_size", 32)),
        )


class OpenAITextEmbedder:
    def __init__(self, settings: OpenAIEmbeddingSettings | None = None, config: dict[str, Any] | None = None) -> None:
        self._settings = settings
        self._config = config or {}
        self.model_name = settings.model if settings is not None else str(self._config.get("embedding_model", "text-embedding-3-small"))

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        settings = self._resolve_settings()
        normalized_texts = [str(text or "").strip() for text in texts if str(text or "").strip()]
        if not normalized_texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(normalized_texts), settings.batch_size):
            batch = normalized_texts[start : start + settings.batch_size]
            payload: dict[str, Any] = {
                "model": settings.model,
                "input": batch,
                "encoding_format": "float",
            }
            if settings.dimensions:
                payload["dimensions"] = settings.dimensions
            response = self._post_json(f"{settings.base_url}/embeddings", payload, settings)
            data = response.get("data") or []
            if len(data) != len(batch):
                raise RuntimeError("OpenAI embeddings API returned an unexpected number of vectors.")
            embeddings.extend([list(item.get("embedding") or []) for item in data])
        return embeddings

    def encode_query(self, text: str) -> list[float]:
        vectors = self.encode_texts([text])
        if not vectors:
            raise RuntimeError("Failed to create a query embedding.")
        return vectors[0]

    def _resolve_settings(self) -> OpenAIEmbeddingSettings:
        if self._settings is None:
            self._settings = OpenAIEmbeddingSettings.from_env_and_config(self._config)
            self.model_name = self._settings.model
        return self._settings

    def _post_json(self, url: str, payload: dict[str, Any], settings: OpenAIEmbeddingSettings) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=settings.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI embeddings API error {exc.code}: {details}") from exc
