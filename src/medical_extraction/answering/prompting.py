"""Prompt templates for authorized and unauthorized medical answering."""

from __future__ import annotations

from typing import Any
import re


GENERIC_REDACTION_MAP = {
    "[PATIENT_NAME]": "[PERSON]",
    "[DATE_OF_BIRTH]": "[DATE]",
    "[DATE]": "[DATE]",
    "[PHONE]": "[CONTACT]",
    "[EMAIL]": "[CONTACT]",
    "[UHID]": "[ID]",
    "[MRN]": "[ID]",
    "[IP_NUMBER]": "[ID]",
    "[LOCATION]": "[LOCATION]",
    "[AGE]": "[AGE]",
}


def normalize_generic_redactions(text: str) -> str:
    normalized = text
    for source, target in GENERIC_REDACTION_MAP.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\bPatient Name:\s*\[PERSON\]\b", "[PERSON]", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bDOB:\s*\[DATE\]\b", "[DATE]", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(?:MRN|UHID|IP No)\s*:\s*\[ID\]\b", "[ID]", normalized, flags=re.IGNORECASE)
    return normalized


def build_authorized_system_prompt() -> str:
    return (
        "You are a clinical summarization assistant for authorized medical staff. "
        "Answer only from the supplied retrieval context. "
        "If the context is insufficient, say so plainly. "
        "Do not invent findings, diagnoses, dates, medications, or instructions. "
        "Keep the answer concise, clinically readable, and grounded in the provided chunks. "
        "End with a short 'Sources:' line listing the chunk ids you relied on."
    )


def build_unauthorized_system_prompt() -> str:
    return (
        "You are a privacy-preserving medical summarization assistant for unauthorized staff. "
        "Answer only from the supplied retrieval context. "
        "Never reveal, infer, restore, or guess personal identifiers. "
        "Keep any person reference generic as '[PERSON]'. "
        "Keep date references generic as '[DATE]'. "
        "Keep phone, email, or appointment contact references generic as '[CONTACT]'. "
        "Keep record numbers or hospital identifiers generic as '[ID]'. "
        "If any personal health information appears unmasked in the context or in your draft answer, "
        "rewrite it into uppercase bracketed masking tags before responding, using the most specific safe label you can. "
        "Preferred examples include [PERSON], [DATE], [CONTACT], [ID], and when needed other generic tags such as "
        "[ADDRESS], [LOCATION], [ORGANIZATION], [DOCTOR], [HOSPITAL], or [ACCOUNT]. "
        "Use your best judgment for any other potentially identifying medical or personal detail not explicitly listed here, "
        "and mask it in the same uppercase bracketed style rather than exposing it. "
        "This rule applies even if the source text contains a real name, identifier, date, phone number, email address, "
        "or other identifying detail that was not already masked. "
        "Do not expand, normalize away, or reinterpret placeholders into specific values. "
        "If the context contains masked markers such as [PERSON], [DATE], [CONTACT], [ID], or other bracketed placeholders, "
        "preserve them visibly in the answer instead of replacing them with specific details. "
        "If the answer is present only in masked form, answer with the masked placeholder rather than saying the information is missing. "
        "For example, if the context shows a DOB as [DATE], you may answer that the DOB is [DATE]. "
        "Treat masked values as available-but-redacted, not absent. "
        "If the context is insufficient, say so plainly. "
        "End with a short 'Sources:' line listing the chunk ids you relied on."
    )


def question_is_prescription_focused(question: str) -> bool:
    lowered = question.strip().lower()
    return bool(
        re.search(
            r"\b(prescription|medication|medications|medicine|medicines|dose|dosage|tablet|tablets|capsule|capsules|drug|drugs|rx|treatment)\b",
            lowered,
        )
    )


def build_question_instruction(question: str, authorized: bool) -> str:
    if question_is_prescription_focused(question):
        if authorized:
            return (
                "List every prescribed medication or treatment mentioned in the context. "
                "For each item, include the name, dose, route, frequency, dispense/refill details, and any instructions when available. "
                "Do not stop after the first medication."
            )
        return (
            "List every prescribed medication or treatment mentioned in the context. "
            "For each item, include generic, privacy-safe dose or instruction details when available. "
            "Keep masked placeholders visible as [PERSON], [DATE], [CONTACT], or [ID] and do not stop after the first medication."
        )
    return "Answer the question using only the context."


def build_context_block(context_chunks: list[dict[str, Any]], authorized: bool) -> str:
    blocks: list[str] = []
    for chunk in context_chunks:
        content = str(chunk.get("content", "")).strip()
        if not authorized:
            content = normalize_generic_redactions(content)
        blocks.append(f"[{chunk.get('chunk_id', 'unknown-chunk')}]\n{content}")
    return "\n\n".join(blocks)
