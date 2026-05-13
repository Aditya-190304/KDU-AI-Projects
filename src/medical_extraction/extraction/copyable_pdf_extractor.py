"""Extractor for copyable PDFs."""

from __future__ import annotations

from typing import Any

try:
    import camelot
except ImportError:  # pragma: no cover - optional dependency
    camelot = None

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional dependency
    pdfplumber = None

from medical_extraction.core.types import ExtractedBlock
from medical_extraction.extraction.merge_engine import merge_blocks


class CopyablePdfExtractor:
    def __init__(self, model_registry=None) -> None:
        self.model_registry = model_registry

    def extract(
        self,
        page,
        input_path: str,
        page_number: int,
        include_tables: bool = True,
    ) -> list[dict]:
        blocks = self.extract_text_blocks(page, input_path, page_number)
        if include_tables:
            blocks.extend(self.extract_table_blocks(page, input_path, page_number))
        return merge_blocks(blocks)

    def extract_text_blocks(self, page, input_path: str, page_number: int) -> list[dict]:
        blocks = self._extract_page_text_blocks(page, page_number)
        if blocks:
            return blocks

        text = self._extract_text(page, input_path, page_number)
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        if not paragraphs:
            paragraphs = [line.strip() for line in text.splitlines() if line.strip()]

        return [
            ExtractedBlock(
                block_id=f"p{page_number}_b{index}",
                type="paragraph",
                text=paragraph,
                source="pdf_text",
                confidence=1.0,
                page_number=page_number,
                bbox=None,
                needs_review=False,
            ).to_dict()
            for index, paragraph in enumerate(paragraphs, start=1)
        ]

    def extract_table_blocks(self, page, input_path: str, page_number: int) -> list[dict]:
        if camelot is None:
            return []

        blocks: list[dict] = []
        try:
            tables = camelot.read_pdf(input_path, pages=str(page_number))
        except Exception:
            return []

        for index, table in enumerate(tables, start=1):
            data_frame = table.df.fillna("")
            if data_frame.empty:
                continue
            columns = data_frame.iloc[0].tolist() if len(data_frame.index) else []
            rows = []
            for row_index in range(1, len(data_frame.index)):
                row_values = data_frame.iloc[row_index].tolist()
                if columns and len(columns) == len(row_values):
                    rows.append(dict(zip(columns, row_values)))
                else:
                    rows.append({f"column_{idx+1}": value for idx, value in enumerate(row_values)})

            blocks.append(
                ExtractedBlock(
                    block_id=f"p{page_number}_t{index}",
                    type="table",
                    text=" ".join(" ".join(str(cell) for cell in row.values()) for row in rows).strip(),
                    source="digital_table_extraction",
                    confidence=1.0,
                    page_number=page_number,
                    bbox=self._camelot_bbox_to_page_bbox(getattr(table, "_bbox", None), float(page.rect.height)),
                    title=f"Table {index}",
                    structured_data={"columns": columns, "rows": rows},
                    needs_review=False,
                ).to_dict()
            )
        return blocks

    def _extract_text(self, page, input_path: str, page_number: int) -> str:
        if pdfplumber is None:
            return page.get_text("text") or ""

        try:
            with pdfplumber.open(input_path) as pdf:
                pdf_page = pdf.pages[page_number - 1]
                return (pdf_page.extract_text() or "").strip()
        except Exception:
            return page.get_text("text") or ""

    def _extract_page_text_blocks(self, page, page_number: int) -> list[dict]:
        payload = page.get_text("dict") or {}
        raw_blocks = payload.get("blocks", []) if isinstance(payload, dict) else []
        blocks: list[dict] = []
        for index, raw in enumerate(raw_blocks, start=1):
            if not isinstance(raw, dict):
                continue
            if raw.get("type") != 0:
                continue
            bbox = raw.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            lines = raw.get("lines", [])
            rendered_lines: list[str] = []
            for line in lines:
                if not isinstance(line, dict):
                    continue
                spans = line.get("spans", [])
                span_texts = [str(span.get("text", "")).strip() for span in spans if str(span.get("text", "")).strip()]
                if span_texts:
                    rendered_lines.append(" ".join(span_texts).strip())
            text = "\n".join(item for item in rendered_lines if item).strip()
            if not text:
                continue
            blocks.append(
                ExtractedBlock(
                    block_id=f"p{page_number}_b{index}",
                    type="paragraph",
                    text=text,
                    source="pdf_text",
                    confidence=1.0,
                    page_number=page_number,
                    bbox=[round(float(value), 2) for value in bbox],
                    needs_review=False,
                ).to_dict()
            )
        return blocks

    def _camelot_bbox_to_page_bbox(self, table_bbox: Any, page_height: float) -> list[float] | None:
        if not isinstance(table_bbox, (list, tuple)) or len(table_bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(value) for value in table_bbox]
        except (TypeError, ValueError):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        top = max(0.0, page_height - y2)
        bottom = max(0.0, page_height - y1)
        return [round(x1, 2), round(top, 2), round(x2, 2), round(bottom, 2)]
