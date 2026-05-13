"""Rule-based lab parser."""

from __future__ import annotations

import re


LAB_PATTERN = re.compile(
    r"\b(?P<name>HbA1c|Glucose|LDL|Creatinine)\b\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|mg/dL|mmol/L)?",
    re.IGNORECASE,
)


def parse_labs(text: str, page_number: int, block_id: str, confidence: float = 0.8) -> list[dict]:
    labs: list[dict] = []
    for match in LAB_PATTERN.finditer(text):
        labs.append(
            {
                "page_number": page_number,
                "block_id": block_id,
                "test_name": match.group("name"),
                "value": match.group("value"),
                "unit": match.group("unit"),
                "reference_range": None,
                "abnormal_flag": None,
                "confidence": round(confidence, 2),
            }
        )
    return labs
