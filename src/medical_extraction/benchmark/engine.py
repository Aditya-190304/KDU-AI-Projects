"""Benchmark engine for OCR accuracy, retrieval accuracy, and processing time.

OCR accuracy:  Averaged across all documents that have ground truth.
Retrieval accuracy:  Per-question (reset for each question asked).
Processing time:  Per-document, tracked from upload job timing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


GROUND_TRUTH_PATH = Path(__file__).resolve().parents[3] / "data" / "benchmark" / "ground_truth.json"
BENCHMARK_STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "benchmark" / "state.json"


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for fuzzy comparison."""
    text = str(text or "").lower()
    text = re.sub(r"[.'',;:!?\-\"/()\\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_ground_truth() -> dict[str, Any]:
    if not GROUND_TRUTH_PATH.exists():
        return {}
    try:
        return json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _load_state() -> dict[str, Any]:
    if not BENCHMARK_STATE_PATH.exists():
        return {"ocr_scores": {}, "processing_times": {}}
    try:
        return json.loads(BENCHMARK_STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"ocr_scores": {}, "processing_times": {}}


def _save_state(state: dict[str, Any]) -> None:
    BENCHMARK_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")


# ─── OCR Accuracy ───────────────────────────────────────────────────────────


def compute_ocr_accuracy(document_id: str, extracted_text: str) -> float | None:
    """Compare extracted text against ground truth key phrases.

    Returns accuracy as a percentage (0-100), or None if no ground truth exists.
    The score is the percentage of key phrases found in the extracted text.
    """
    gt = _load_ground_truth()
    doc_gt = (gt.get("documents") or {}).get(document_id)
    if not doc_gt:
        return None

    key_phrases: list[str] = doc_gt.get("key_phrases", [])
    if not key_phrases:
        return None

    normalized_extracted = _normalize(extracted_text)
    found = 0
    for phrase in key_phrases:
        normalized_phrase = _normalize(phrase)
        if normalized_phrase in normalized_extracted:
            found += 1

    accuracy = round((found / len(key_phrases)) * 100, 1)

    # Persist to running average
    state = _load_state()
    state.setdefault("ocr_scores", {})[document_id] = {
        "accuracy": accuracy,
        "found": found,
        "total": len(key_phrases),
    }
    _save_state(state)
    return accuracy


def get_average_ocr_accuracy() -> dict[str, Any]:
    """Return the running average OCR accuracy across all benchmarked documents."""
    state = _load_state()
    scores = state.get("ocr_scores", {})
    if not scores:
        return {"average_ocr_accuracy": None, "documents_evaluated": 0, "per_document": {}}

    total_accuracy = sum(s["accuracy"] for s in scores.values())
    avg = round(total_accuracy / len(scores), 1)
    return {
        "average_ocr_accuracy": avg,
        "documents_evaluated": len(scores),
        "per_document": dict(scores),
    }


# ─── Retrieval / Answer Accuracy ────────────────────────────────────────────


def evaluate_answer(question: str, rag_answer: str, document_id: str | None = None) -> dict[str, Any]:
    """Compare the RAG answer against the answer key for this specific question.

    This is per-question (not averaged). It checks all documents' answer keys
    for the best matching question and scores the answer.
    """
    gt = _load_ground_truth()
    documents = gt.get("documents", {})

    # Score ALL candidate answer keys and pick the best match
    best_match: dict[str, Any] | None = None
    best_score: float = 0.0
    for doc_id, doc_gt in documents.items():
        if document_id and doc_id != document_id:
            continue
        answer_keys: dict[str, str] = doc_gt.get("answer_keys", {})
        for key_question, expected_answer in answer_keys.items():
            score = _question_similarity(question, key_question)
            if score > best_score and score >= 0.35:
                best_score = score
                best_match = {
                    "document_id": doc_id,
                    "matched_question": key_question,
                    "expected_answer": expected_answer,
                }

    if not best_match:
        return {
            "retrieval_accuracy": None,
            "matched": False,
            "reason": "No answer key found for this question.",
        }

    expected = best_match["expected_answer"]
    score = _score_answer(rag_answer, expected)

    return {
        "retrieval_accuracy": score,
        "matched": True,
        "matched_question": best_match["matched_question"],
        "expected_answer": expected,
        "document_id": best_match["document_id"],
    }


def _question_similarity(user_question: str, key_question: str) -> float:
    """Return similarity score (0.0 to 1.0) between user question and answer key question."""
    norm_user = _normalize(user_question)
    norm_key = _normalize(key_question)

    # Extract meaningful words (skip stop words)
    stop_words = {"what", "is", "the", "who", "how", "many", "does", "for", "to", "of",
                  "a", "an", "in", "on", "at", "by", "and", "or", "with", "mr", "mrs", "ms", "s"}
    user_words = set(norm_user.split()) - stop_words
    key_words = set(norm_key.split()) - stop_words

    if not key_words or not user_words:
        return 0.0

    overlap = user_words & key_words
    lexical_score = len(overlap) / len(user_words | key_words)
    intent_score = _intent_similarity(norm_user, norm_key)
    phrase_score = _phrase_overlap_score(norm_user, norm_key)

    user_intents = _extract_question_intents(norm_user)
    key_intents = _extract_question_intents(norm_key)
    if user_intents and key_intents and not (user_intents & key_intents):
        lexical_score *= 0.35
        phrase_score *= 0.35

    return round((lexical_score * 0.35) + (intent_score * 0.50) + (phrase_score * 0.15), 4)


QUESTION_INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "doctor": ("doctor", "dr", "physician", "consultant", "clinician"),
    "diagnosis": ("diagnosis", "condition", "disease", "assessment", "impression", "diagnosed"),
    "ip_number": ("ip number", "ip no", "ip", "patient id", "record number"),
    "hospital_stay": ("hospital stay", "days", "how many days", "stay", "recovery"),
    "medication": ("medication", "medicine", "drug", "prescribed", "prescription", "tablet", "dose"),
    "dob": ("dob", "date of birth", "born", "birth"),
    "mrn": ("mrn", "medical record number"),
}


def _extract_question_intents(normalized_question: str) -> set[str]:
    intents: set[str] = set()
    for intent, patterns in QUESTION_INTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern in normalized_question:
                intents.add(intent)
                break
    return intents


def _intent_similarity(norm_user: str, norm_key: str) -> float:
    user_intents = _extract_question_intents(norm_user)
    key_intents = _extract_question_intents(norm_key)
    if not user_intents and not key_intents:
        return 0.0
    if user_intents and key_intents:
        overlap = user_intents & key_intents
        if overlap:
            return len(overlap) / len(user_intents | key_intents)
        return 0.0
    return 0.0


def _phrase_overlap_score(norm_user: str, norm_key: str) -> float:
    user_bigrams = _ngrams(norm_user.split(), 2)
    key_bigrams = _ngrams(norm_key.split(), 2)
    if user_bigrams and key_bigrams:
        overlap = user_bigrams & key_bigrams
        if overlap:
            return len(overlap) / len(user_bigrams | key_bigrams)
    user_trigrams = _ngrams(norm_user.split(), 3)
    key_trigrams = _ngrams(norm_key.split(), 3)
    if user_trigrams and key_trigrams:
        overlap = user_trigrams & key_trigrams
        if overlap:
            return len(overlap) / len(user_trigrams | key_trigrams)
    return 0.0


def _ngrams(tokens: list[str], size: int) -> set[str]:
    if len(tokens) < size:
        return set()
    return {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _score_answer(rag_answer: str, expected_answer: str) -> float:
    """Score the RAG answer vs expected answer.

    Uses key-phrase overlap: split the expected answer into significant tokens
    and check what fraction appear in the RAG answer.
    Returns percentage 0-100.
    """
    norm_rag = _normalize(rag_answer)
    norm_expected = _normalize(expected_answer)

    # Check if the full expected answer appears
    if norm_expected in norm_rag:
        return 100.0

    # Otherwise score by token overlap
    expected_tokens = [t for t in norm_expected.split() if len(t) > 2]
    if not expected_tokens:
        return 0.0

    found = sum(1 for t in expected_tokens if t in norm_rag)
    return round((found / len(expected_tokens)) * 100, 1)


# ─── Processing Time ────────────────────────────────────────────────────────


def record_processing_time(document_id: str, elapsed_seconds: float) -> None:
    """Record the processing time for a document."""
    state = _load_state()
    state.setdefault("processing_times", {})[document_id] = round(elapsed_seconds, 1)
    _save_state(state)


def get_processing_times() -> dict[str, Any]:
    """Return all processing times and their average."""
    state = _load_state()
    times = state.get("processing_times", {})
    if not times:
        return {"average_seconds": None, "documents": {}}

    avg = round(sum(times.values()) / len(times), 1)
    return {
        "average_seconds": avg,
        "documents": dict(times),
    }


# ─── Full Benchmark Summary ─────────────────────────────────────────────────


def get_benchmark_summary() -> dict[str, Any]:
    """Return the full benchmark summary for the UI."""
    ocr = get_average_ocr_accuracy()
    proc = get_processing_times()
    return {
        "ocr_accuracy": ocr,
        "processing_time": proc,
    }
