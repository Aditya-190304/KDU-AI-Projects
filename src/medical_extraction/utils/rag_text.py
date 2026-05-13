"""RAG-ready plain-text rendering from extraction payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_rag_text(payload: dict[str, Any]) -> str:
    pages = payload.get("pages", []) if isinstance(payload, dict) else []
    rendered_blocks: list[str] = []

    for page in sorted(pages, key=_page_sort_key):
        blocks = page.get("blocks", []) if isinstance(page, dict) else []
        ordered_blocks = sorted(blocks, key=_block_sort_key)
        for block in ordered_blocks:
            block_text = _render_block_text(block)
            if not block_text:
                continue
            if rendered_blocks and _normalize_whitespace(rendered_blocks[-1]) == _normalize_whitespace(block_text):
                continue
            rendered_blocks.append(block_text)

    return "\n\n".join(rendered_blocks).strip()


def derive_rag_text_path(json_output_path: str) -> str:
    path = Path(json_output_path)
    return str(path.with_name(f"{path.stem}_rag.txt"))


def _page_sort_key(page: dict[str, Any]) -> tuple[int, str]:
    page_number = page.get("page_number")
    if isinstance(page_number, int):
        return page_number, ""
    return 10**9, str(page_number or "")


def _block_sort_key(block: dict[str, Any]) -> tuple[int, float, float, str]:
    metadata = block.get("metadata") if isinstance(block, dict) else None
    if isinstance(metadata, dict):
        reading_order = metadata.get("reading_order")
        if isinstance(reading_order, int):
            bbox = block.get("bbox") if isinstance(block, dict) else None
            if isinstance(bbox, list) and len(bbox) == 4:
                return reading_order, float(bbox[1]), float(bbox[0]), str(block.get("block_id", ""))
            return reading_order, 10**9, 10**9, str(block.get("block_id", ""))
    bbox = block.get("bbox") if isinstance(block, dict) else None
    if isinstance(bbox, list) and len(bbox) == 4:
        return 10**9, float(bbox[1]), float(bbox[0]), str(block.get("block_id", ""))
    return 10**9, 10**9, 10**9, str(block.get("block_id", ""))


def _render_block_text(block: dict[str, Any]) -> str:
    block_type = str(block.get("type", "")).lower()
    if block_type == "table":
        table_text = _render_table_text(block.get("structured_data"))
        if table_text:
            return table_text
    if block_type == "form":
        form_text = _render_form_text(block.get("fields"))
        if form_text:
            return form_text

    text = str(block.get("text", "")).strip()
    return _normalize_whitespace(text)


def _render_table_text(structured_data: Any) -> str:
    if not isinstance(structured_data, dict):
        return ""
    rows = structured_data.get("rows")
    if not isinstance(rows, list):
        return ""

    lines: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            values = [str(value).strip() for value in row.values() if str(value).strip()]
            if values:
                lines.append(" | ".join(values))
    return "\n".join(lines).strip()


def _render_form_text(fields: Any) -> str:
    if not isinstance(fields, dict):
        return ""
    lines: list[str] = []
    for key, value_payload in fields.items():
        if isinstance(value_payload, dict):
            value = str(value_payload.get("value", "")).strip()
            if value:
                lines.append(f"{str(key).strip()}: {value}")
        elif value_payload:
            lines.append(f"{str(key).strip()}: {str(value_payload).strip()}")
    return "\n".join(lines).strip()


def _normalize_whitespace(text: str) -> str:
    compact_lines = []
    for line in text.splitlines():
        normalized = " ".join(line.split()).strip()
        if normalized:
            compact_lines.append(normalized)
    return "\n".join(compact_lines).strip()
