"""Regex-based medication parser."""

from __future__ import annotations

import re


DOSE_PATTERN = re.compile(r"\b(?P<dose>\d+(?:\.\d+)?)\s?(?P<unit>mg|mcg|g|ml)\b", re.IGNORECASE)
FREQUENCY_PATTERN = re.compile(
    r"\b(once daily|twice daily|three times daily|tid|tds|bid|qhs|daily|od|bd|qid|hs|sos|stat|1-0-1|1-1-1|0-1-0|0-0-1|1-0-0)\b",
    re.IGNORECASE,
)
ROUTE_PATTERN = re.compile(r"\b(orally|oral|iv|po|topical|subcutaneous)\b", re.IGNORECASE)
DURATION_PATTERN = re.compile(r"\b(?:x|for)?\s*(\d+\s*(?:day|days|week|weeks|month|months))\b", re.IGNORECASE)
FORM_PATTERN = re.compile(
    r"^\s*(?P<form>tab(?:let)?|cap(?:sule)?|syp|syrup|inj|ointment|cream|gel|drop(?:s)?|mouthwash|paint)\.?\s+",
    re.IGNORECASE,
)
INSTRUCTION_PATTERN = re.compile(
    r"\b(before meals?|after meals?|with meals?|massage|gargle|apply|bedtime)\b",
    re.IGNORECASE,
)
CONTINUATION_PATTERN = re.compile(
    r"\b(od|bd|tds|qid|hs|sos|stat|1-0-1|1-1-1|0-1-0|0-0-1|1-0-0|x\s*\d+\s*(?:day|days|week|weeks|month|months)|for\s*\d+\s*(?:day|days|week|weeks|month|months)|before meals?|after meals?|with meals?|massage|gargle|apply|bedtime)\b",
    re.IGNORECASE,
)

KNOWN_MEDICATIONS = {
    "metformin",
    "warfarin",
    "amoxicillin",
    "augmentin",
    "insulin",
    "atorvastatin",
    "remdesivir",
    "remdec",
    "enzoflora",
    "enzoflorn",
    "hexigel",
    "paracetamol",
    "dolo",
    "pantop",
    "pan d",
    "pand",
    "pantocid",
    "rabeprazole",
    "aceclofenac",
}

FOOTER_NOISE_MARKERS = {
    "www",
    ".com",
    "email",
    "@",
    "philanthropy",
    "rajouri",
    "garden",
    "new delhi",
    "+91",
    "contact",
    "info@",
}

NARRATIVE_MARKERS = {
    "to whom",
    "concern",
    "this is to",
    "under my care",
    "admitted",
    "undergoing treatment",
    "hospital stay",
    "complete recovery",
    "kindly do the needful",
    "thanking you",
    "icu",
    "sepsis",
    "pneumonia",
}

SIGNATURE_MARKERS = {
    "appointments",
    "apollo hospitals",
    "consultant",
    "gastroenterologist",
    "hepatologist",
    "jubilee hills",
    "regd.no",
    "regd no",
    "mrcp",
    "frcp",
    "ccst",
    "extn",
}

def _normalize_digit_token(value: str) -> str:
    lowered = value.lower()
    if lowered in {"i", "l"}:
        return "1"
    return value


def configure_medication_parser(**_kwargs) -> None:
    return None


def clean_prescription_line(text: str) -> str:
    normalized = (text or "").replace("<br>", " ").replace("|", " | ")
    normalized = re.sub(r"<math[^>]*>", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"</math>", " ", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("\\times", " x ")
    normalized = normalized.replace("Lab.", "Tab.").replace("lab.", "tab.")
    normalized = normalized.replace("Stab.", "Tab.").replace("stab.", "tab.")
    normalized = normalized.replace("grum", "gum").replace("weith", "with")
    normalized = normalized.replace("neals", "meals").replace("meal5", "meals")
    normalized = normalized.replace("voyage", "lavage").replace("Other-ritensk", "other")
    normalized = re.sub(r"[{}\[\]<>]+", " ", normalized)
    normalized = re.sub(
        r"\b([01Iil])\s*-\s*([01Iil])\s*-\s*([01Iil])\b",
        lambda match: f"{_normalize_digit_token(match.group(1))}-{_normalize_digit_token(match.group(2))}-{_normalize_digit_token(match.group(3))}",
        normalized,
    )
    normalized = re.sub(
        r"\b[xX]\s*([Ii1l])\s*(day|days|week|weeks|month|months)\b",
        lambda match: f"x {_normalize_digit_token(match.group(1))} {match.group(2)}",
        normalized,
    )
    normalized = re.sub(r"\b(\d+)\s*(day|days|week|weeks|month|months)\b", r"\1 \2", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[\"`]+", "", normalized)
    normalized = re.sub(r"[â€¢Â·]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" .,:;-")


def is_footer_noise_line(text: str) -> bool:
    lowered = clean_prescription_line(text).lower()
    if not lowered:
        return True
    return any(marker in lowered for marker in FOOTER_NOISE_MARKERS)


def has_explicit_medication_signal(text: str) -> bool:
    cleaned = clean_prescription_line(text)
    if not cleaned or is_footer_noise_line(cleaned):
        return False
    lowered = cleaned.lower()
    if any(name in lowered for name in KNOWN_MEDICATIONS):
        return True
    if FORM_PATTERN.search(cleaned) or DOSE_PATTERN.search(cleaned):
        return True
    if FREQUENCY_PATTERN.search(cleaned) or ROUTE_PATTERN.search(cleaned):
        return True
    if lowered.startswith(("adv", "rx")):
        return True
    return bool(INSTRUCTION_PATTERN.search(cleaned) and len(cleaned.split()) <= 8)


def is_likely_non_medication_text(text: str) -> bool:
    cleaned = clean_prescription_line(text)
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if any(marker in lowered for marker in SIGNATURE_MARKERS):
        return True
    if any(lowered.startswith(prefix) for prefix in ("this is to", "he is", "he was", "he further", "thanking you")):
        return True
    if any(marker in lowered for marker in NARRATIVE_MARKERS) and not has_explicit_medication_signal(cleaned):
        return True
    if len(cleaned.split()) >= 8 and not has_explicit_medication_signal(cleaned):
        return True
    return False


def looks_like_prescription_line(text: str) -> bool:
    cleaned = clean_prescription_line(text)
    if not cleaned or is_footer_noise_line(cleaned) or is_likely_non_medication_text(cleaned):
        return False
    lowered = cleaned.lower()
    score = 0
    if FORM_PATTERN.search(cleaned):
        score += 3
    if DOSE_PATTERN.search(cleaned):
        score += 3
    if FREQUENCY_PATTERN.search(cleaned):
        score += 2
    if DURATION_PATTERN.search(cleaned):
        score += 1
    if lowered.startswith(("adv", "rx")):
        score += 2
    if any(name in lowered for name in KNOWN_MEDICATIONS):
        score += 4
    return score >= 3


def has_medication_anchor(text: str) -> bool:
    cleaned = clean_prescription_line(text)
    if not cleaned or is_footer_noise_line(cleaned) or is_likely_non_medication_text(cleaned):
        return False
    lowered = cleaned.lower()
    if any(name in lowered for name in KNOWN_MEDICATIONS):
        return True
    if FORM_PATTERN.search(cleaned) and _extract_medication_name(cleaned):
        return True
    return bool(DOSE_PATTERN.search(cleaned) and _extract_medication_name(cleaned))


def looks_like_continuation_line(text: str) -> bool:
    cleaned = clean_prescription_line(text)
    if not cleaned or is_footer_noise_line(cleaned) or is_likely_non_medication_text(cleaned):
        return False
    if has_medication_anchor(cleaned):
        return False
    if len(cleaned.split()) <= 2 and cleaned.lower() in {"paint", "massage", "gargle", "lavage"}:
        return True
    return bool(CONTINUATION_PATTERN.search(cleaned))


def _extract_medication_name(text: str) -> str | None:
    lowered = text.lower()
    for medication_name in sorted(KNOWN_MEDICATIONS, key=len, reverse=True):
        if medication_name in lowered:
            return medication_name.title()

    if not has_explicit_medication_signal(text):
        return None

    working = re.sub(r"^\s*adv[:\-]?\s*", "", text, flags=re.IGNORECASE)
    working = re.sub(r"^\s*(before|after|with)\s+meals?\b", "", working, flags=re.IGNORECASE)
    form_match = FORM_PATTERN.match(working)
    if form_match:
        working = working[form_match.end() :]

    stopping_tokens = re.split(
        r"\b(?:\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml)|once daily|twice daily|three times daily|tid|tds|bid|od|bd|qid|hs|sos|stat|x\s*\d+\s*(?:day|days|week|weeks)|for\s*\d+\s*(?:day|days|week|weeks)|before|after|with|massage|gargle|apply)\b",
        working,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", stopping_tokens)
    words = [
        word for word in words
        if word.lower() not in {"before", "after", "with", "meals", "meal", "tab", "cap", "syp", "adv", "rx"}
    ]
    if not words:
        return None
    return " ".join(words[:3]).title()


def parse_prescription_line(line: str, page_number: int, block_id: str, confidence: float = 0.75) -> dict | None:
    cleaned = clean_prescription_line(line)
    if not cleaned or is_footer_noise_line(cleaned) or is_likely_non_medication_text(cleaned):
        return None

    dose_match = DOSE_PATTERN.search(cleaned)
    frequency_match = FREQUENCY_PATTERN.search(cleaned)
    route_match = ROUTE_PATTERN.search(cleaned)
    duration_match = DURATION_PATTERN.search(cleaned)
    form_match = FORM_PATTERN.search(cleaned)
    instruction_matches = INSTRUCTION_PATTERN.findall(cleaned)
    medication_name = _extract_medication_name(cleaned)
    explicit_signal = has_explicit_medication_signal(cleaned)

    if not medication_name and not explicit_signal and not looks_like_prescription_line(cleaned):
        return None
    if not medication_name and not any((dose_match, frequency_match, route_match, form_match, instruction_matches)):
        return None

    parsed_confidence = round(confidence, 2)
    needs_review = (
        parsed_confidence < 0.70
        or medication_name is None
        or dose_match is None
    )
    return {
        "page_number": page_number,
        "block_id": block_id,
        "medication": medication_name,
        "dose": dose_match.group(0) if dose_match else None,
        "route": route_match.group(0) if route_match else None,
        "frequency": frequency_match.group(0) if frequency_match else None,
        "duration": duration_match.group(1) if duration_match else None,
        "form": form_match.group("form").title() if form_match else None,
        "instructions": ", ".join(instruction_matches) if instruction_matches else None,
        "text": cleaned,
        "confidence": parsed_confidence,
        "needs_review": needs_review,
    }


def parse_medications(text: str, page_number: int, block_id: str, confidence: float = 0.75) -> list[dict]:
    medications: list[dict] = []
    for line in text.splitlines():
        parsed = parse_prescription_line(line, page_number=page_number, block_id=block_id, confidence=confidence)
        if parsed is not None:
            medications.append(parsed)
    return medications
