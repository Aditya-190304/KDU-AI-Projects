"""Review-flag helpers for pages and blocks."""

from __future__ import annotations


def page_needs_review(blocks: list[dict], page_type: str, error: str | None = None) -> bool:
    if error:
        return True
    if page_type in {"unknown_or_hybrid", "error"}:
        return True
    return any(block.get("needs_review") for block in blocks)
