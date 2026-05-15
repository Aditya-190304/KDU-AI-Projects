"""Presidio-based PHI redaction with deterministic HMAC metadata."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from dataclasses import dataclass
from typing import Any

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


@dataclass(frozen=True, slots=True)
class _EntitySpec:
    entity_type: str
    placeholder: str
    hash_key: str
    label_prefix: str | None = None


ENTITY_SPECS: dict[str, _EntitySpec] = {
    "PATIENT_NAME": _EntitySpec("PATIENT_NAME", "[PATIENT_NAME]", "patient_name_hmac", "Patient Name"),
    "PERSON": _EntitySpec("PERSON", "[PATIENT_NAME]", "patient_name_hmac"),
    "UHID": _EntitySpec("UHID", "[UHID]", "uhid_hmac", "UHID"),
    "IP_NUMBER": _EntitySpec("IP_NUMBER", "[IP_NUMBER]", "ip_number_hmac", "IP No"),
    "MRN": _EntitySpec("MRN", "[MRN]", "mrn_hmac", "MRN"),
    "DATE_OF_BIRTH": _EntitySpec("DATE_OF_BIRTH", "[DATE_OF_BIRTH]", "dob_hmac", "DOB"),
    "DATE_TIME": _EntitySpec("DATE_TIME", "[DATE]", "date_hmacs"),
    "EMAIL_ADDRESS": _EntitySpec("EMAIL_ADDRESS", "[EMAIL]", "email_hmacs"),
    "PHONE_NUMBER": _EntitySpec("PHONE_NUMBER", "[PHONE]", "phone_hmacs"),
    "LOCATION": _EntitySpec("LOCATION", "[LOCATION]", "location_hmacs"),
    "AGE": _EntitySpec("AGE", "[AGE]", "age_hmacs"),
}

ENTITY_PRIORITY = {
    "PATIENT_NAME": 100,
    "PERSON": 85,
    "DATE_OF_BIRTH": 95,
    "UHID": 90,
    "IP_NUMBER": 90,
    "MRN": 90,
    "PHONE_NUMBER": 80,
    "EMAIL_ADDRESS": 80,
    "DATE_TIME": 70,
    "LOCATION": 65,
    "AGE": 60,
}


class ChunkRedactor:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.secret = self._resolve_secret()
        self.language = str(self.config.get("presidio_language", "en")).strip() or "en"
        self.spacy_model = str(self.config.get("presidio_spacy_model", "en_core_web_sm")).strip() or "en_core_web_sm"
        self._analyzer = self._build_analyzer()
        self._anonymizer = AnonymizerEngine()

    def redact_text(self, text: str) -> str:
        """Redact PHI from arbitrary text. Used for post-generation filtering."""
        raw = str(text or "").strip()
        if not raw:
            return raw
        results = self._analyzer.analyze(
            text=raw,
            language=self.language,
            entities=list(ENTITY_SPECS.keys()),
        )
        selected = self._select_results(results)
        if not selected:
            return raw
        operators: dict[str, OperatorConfig] = {}
        for result in selected:
            spec = ENTITY_SPECS.get(result.entity_type)
            if spec:
                operators[spec.entity_type] = OperatorConfig("replace", {"new_value": self._replacement_text(spec)})
        anonymized = self._anonymizer.anonymize(text=raw, analyzer_results=selected, operators=operators)
        return anonymized.text

    def redact_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        raw_text = str(chunk.get("chunk_text", "")).strip()
        if not raw_text:
            redacted_chunk = dict(chunk)
            metadata = dict(redacted_chunk.get("metadata") or {})
            metadata["identity_hmacs"] = {}
            metadata["redactions"] = []
            metadata["redaction_applied"] = False
            redacted_chunk["metadata"] = metadata
            redacted_chunk["chunk_char_count"] = 0
            return redacted_chunk

        raw_results = self._analyzer.analyze(
            text=raw_text,
            language=self.language,
            entities=list(ENTITY_SPECS.keys()),
        )
        selected_results = self._select_results(raw_results)

        identity_hmacs: dict[str, str] = {}
        redactions: list[dict[str, str]] = []
        operators: dict[str, OperatorConfig] = {}

        for result in selected_results:
            spec = ENTITY_SPECS.get(result.entity_type)
            if not spec:
                continue
            matched_text = raw_text[result.start : result.end]
            hashed_value = self._hash_source_value(spec.entity_type, matched_text)
            if hashed_value:
                if spec.hash_key.endswith("_hmac"):
                    identity_hmacs[spec.hash_key] = self._hmac(hashed_value)
                else:
                    existing = identity_hmacs.get(spec.hash_key, "")
                    digest = self._hmac(hashed_value)
                    identity_hmacs[spec.hash_key] = "|".join(part for part in [existing, digest] if part)
            operators[spec.entity_type] = OperatorConfig("replace", {"new_value": self._replacement_text(spec)})
            redactions.append({"type": spec.hash_key, "placeholder": spec.placeholder})

        if selected_results:
            anonymized = self._anonymizer.anonymize(
                text=raw_text,
                analyzer_results=selected_results,
                operators=operators,
            )
            redacted_text = anonymized.text
        else:
            redacted_text = raw_text

        redacted_chunk = dict(chunk)
        metadata = dict(redacted_chunk.get("metadata") or {})
        metadata["identity_hmacs"] = identity_hmacs
        metadata["redactions"] = redactions
        metadata["redaction_applied"] = redacted_text != raw_text
        redacted_chunk["metadata"] = metadata
        redacted_chunk["chunk_text"] = redacted_text
        redacted_chunk["chunk_char_count"] = len(redacted_text)
        return redacted_chunk

    def _build_analyzer(self) -> AnalyzerEngine:
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": self.language, "model_name": self.spacy_model}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[self.language])
        for recognizer in self._custom_recognizers():
            analyzer.registry.add_recognizer(recognizer)
        return analyzer

    def _custom_recognizers(self) -> list[PatternRecognizer]:
        return [
            PatternRecognizer(
                supported_entity="PATIENT_NAME",
                patterns=[
                    Pattern(
                        "patient-name-labeled",
                        r"\b(?:patient\s*name|name)\s*:\s*[A-Za-z][A-Za-z\s.'-]{1,80}(?=\s+(?:uhid|uid|patient\s*id|ip\s*no|ip\s*number|mrn|dob|date\s*of\s*birth|phone|email)\b|[.;,]|$)",
                        0.9,
                    )
                ],
                supported_language=self.language,
            ),
            PatternRecognizer(
                supported_entity="UHID",
                patterns=[Pattern("uhid-labeled", r"\b(?:uhid|uid|patient\s*id)\s*[:#-]?\s*[A-Z0-9/-]{4,}", 0.9)],
                supported_language=self.language,
            ),
            PatternRecognizer(
                supported_entity="IP_NUMBER",
                patterns=[Pattern("ip-number-labeled", r"\b(?:ip\s*no|ip\s*number)\s*[:#-]?\s*[A-Z0-9/-]{3,}", 0.9)],
                supported_language=self.language,
            ),
            PatternRecognizer(
                supported_entity="MRN",
                patterns=[
                    Pattern(
                        "mrn-labeled",
                        r"\b(?:mrn|medical\s*record\s*number)\s*[:#-]?\s*[A-Z0-9/-]{3,}",
                        0.9,
                    )
                ],
                supported_language=self.language,
            ),
            PatternRecognizer(
                supported_entity="DATE_OF_BIRTH",
                patterns=[
                    Pattern(
                        "dob-labeled",
                        r"\b(?:dob|date\s*of\s*birth)\s*[:#-]?\s*[A-Za-z0-9,/\-\s]{4,}(?=[.;,]|$)",
                        0.9,
                    )
                ],
                supported_language=self.language,
            ),
            PatternRecognizer(
                supported_entity="AGE",
                patterns=[
                    Pattern(
                        "age-years",
                        r"\b\d{1,3}\s*(?:yr|yrs|year|years|y/?o|year[\s-]*old)\b",
                        0.85,
                    ),
                    Pattern(
                        "age-months",
                        r"\b\d{1,3}\s*(?:mo|mos|month|months|month[\s-]*old)\b",
                        0.85,
                    ),
                    Pattern(
                        "age-days",
                        r"\b\d{1,3}\s*(?:day|days|day[\s-]*old)\b",
                        0.70,
                    ),
                    Pattern(
                        "age-labeled",
                        r"\b(?:age|aged)\s*[:#-]?\s*\d{1,3}\b",
                        0.90,
                    ),
                ],
                supported_language=self.language,
            ),
        ]

    def _resolve_secret(self) -> str:
        env_var = str(self.config.get("hmac_secret_env_var", "MEDICAL_RAG_HMAC_SECRET")).strip()
        secret = os.environ.get(env_var, "").strip()
        if secret:
            return secret
        fallback = str(self.config.get("dev_fallback_secret", "")).strip()
        if fallback:
            return fallback
        raise RuntimeError(
            "No HMAC secret configured. Set the configured environment variable or provide a dev fallback secret."
        )

    def _select_results(self, results: list[Any]) -> list[Any]:
        ordered = sorted(
            results,
            key=lambda result: (
                int(result.start),
                -(int(result.end) - int(result.start)),
                -ENTITY_PRIORITY.get(str(result.entity_type), 0),
                -float(result.score),
            ),
        )
        selected: list[Any] = []
        for candidate in ordered:
            if any(self._overlaps(candidate, existing) for existing in selected):
                continue
            selected.append(candidate)
        selected.sort(key=lambda result: int(result.start))
        return selected

    def _replacement_text(self, spec: _EntitySpec) -> str:
        if spec.label_prefix:
            return f"{spec.label_prefix}: {spec.placeholder}"
        return spec.placeholder

    def _hash_source_value(self, entity_type: str, matched_text: str) -> str:
        normalized = matched_text.strip()
        if not normalized:
            return ""
        if entity_type in {"PATIENT_NAME", "PERSON"}:
            return self._value_after_label(normalized)
        if entity_type in {"UHID", "IP_NUMBER", "MRN", "DATE_OF_BIRTH"}:
            return self._value_after_label(normalized)
        return normalized

    def _value_after_label(self, matched_text: str) -> str:
        parts = re.split(r"\s*[:#-]\s*", matched_text, maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
        return matched_text.strip()

    def _hmac(self, value: str) -> str:
        digest = hmac.new(self.secret.encode("utf-8"), value.strip().lower().encode("utf-8"), hashlib.sha256)
        return digest.hexdigest()

    def _overlaps(self, first: Any, second: Any) -> bool:
        return int(first.start) < int(second.end) and int(second.start) < int(first.end)
