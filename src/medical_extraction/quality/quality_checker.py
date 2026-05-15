"""Final quality pass for blocks and summary values."""

from __future__ import annotations

from medical_extraction.quality.review_flags import page_needs_review


class QualityChecker:
    def enrich_page(self, page: dict) -> dict:
        page["needs_review"] = page_needs_review(
            blocks=page.get("blocks", []),
            page_type=page.get("page_type", "unknown_or_hybrid"),
            error=page.get("error"),
        )
        return page

    def count_review_blocks(self, pages: list[dict]) -> int:
        return sum(1 for page in pages for block in page.get("blocks", []) if block.get("needs_review"))
