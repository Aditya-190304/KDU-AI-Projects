"""OpenAI-backed answer generation for retrieved medical context."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib import error, request

from medical_extraction.answering.prompting import (
    build_authorized_system_prompt,
    build_context_block,
    build_question_instruction,
    build_unauthorized_system_prompt,
)


@dataclass(slots=True)
class OpenAIAnswererSettings:
    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 60
    max_tokens: int = 500
    temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "OpenAIAnswererSettings":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or your shell environment.")
        return cls(
            api_key=api_key,
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
            or "https://api.openai.com/v1",
            timeout_seconds=int(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60")),
            max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "500")),
            temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.2")),
        )


class OpenAIAnswerer:
    def __init__(self, settings: OpenAIAnswererSettings | None = None) -> None:
        self.settings = settings or OpenAIAnswererSettings.from_env()

    def answer_question(
        self,
        question: str,
        context_chunks: list[dict[str, Any]],
        authorized: bool,
        role: str,
    ) -> dict[str, Any]:
        context_block = build_context_block(context_chunks, authorized=authorized)
        system_prompt = build_authorized_system_prompt() if authorized else build_unauthorized_system_prompt()
        user_prompt = (
            f"Role: {role}\n"
            f"Question: {question}\n\n"
            f"Context chunks:\n{context_block}\n\n"
            f"{build_question_instruction(question, authorized=authorized)}"
        )
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        response = self._post_json(f"{self.settings.base_url}/chat/completions", payload)
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI API returned no choices.")
        message = choices[0].get("message") or {}
        answer_text = str(message.get("content", "")).strip()
        return {
            "model": self.settings.model,
            "answer": answer_text,
            "authorized": authorized,
            "role": role,
        }

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=self.settings.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API error {exc.code}: {details}") from exc
