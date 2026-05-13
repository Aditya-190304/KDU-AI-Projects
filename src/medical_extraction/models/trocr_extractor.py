"""Handwriting OCR wrapper backed by local Ollama vision models."""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from typing import Any


class TrOcrExtractor:
    model_name = "qwen2.5vl:3b"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.model_name = os.environ.get("MEDICAL_HANDWRITING_OLLAMA_MODEL", self.model_name).strip()
        self.base_url = os.environ.get("MEDICAL_OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
        self.timeout_seconds = int(os.environ.get("MEDICAL_OLLAMA_TIMEOUT_SECONDS", "180"))

    def extract_text(self, image: Any) -> tuple[str, float]:
        if not hasattr(image, "convert"):
            return "", 0.0
        try:
            encoded_image = self._encode_image(image)
            text = self._request_ollama(encoded_image)
            text = self._normalize_ocr_text(text)
            if not text:
                return "", 0.0
            return text, self._estimate_confidence(text)
        except Exception:
            return "", 0.0

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

    def _request_ollama(self, encoded_image: str) -> str:
        prompt = (
            "Read this medical image and return only the extracted text in natural reading order. "
            "Do not explain. Do not return JSON or markdown."
        )
        payload = {
            "model": self.model_name,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encoded_image],
                }
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama connection error: {exc}") from exc

        data = json.loads(raw)
        message = data.get("message") if isinstance(data, dict) else None
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str):
                return content
        if isinstance(data, dict) and isinstance(data.get("response"), str):
            return data["response"]
        return ""

    def _encode_image(self, image) -> str:
        rgb_image = image.convert("RGB")
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _normalize_ocr_text(self, raw: Any) -> str:
        if isinstance(raw, str):
            return " ".join(part.strip() for part in raw.splitlines() if part.strip()).strip()
        if isinstance(raw, (list, tuple)):
            joined = " ".join(str(item).strip() for item in raw if str(item).strip())
            return " ".join(part.strip() for part in joined.splitlines() if part.strip()).strip()
        return str(raw).strip()

    def _estimate_confidence(self, text: str) -> float:
        if not text:
            return 0.0
        token_count = len(text.split())
        if token_count >= 20:
            return 0.85
        if token_count >= 8:
            return 0.75
        return 0.65
