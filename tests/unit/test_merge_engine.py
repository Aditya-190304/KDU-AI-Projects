from medical_extraction.extraction.merge_engine import merge_blocks


def test_merge_blocks_preserves_line_breaks_within_layout_region():
    merged = merge_blocks(
        [
            {
                "block_id": "b1",
                "type": "paragraph",
                "text": "Line one",
                "source": "paddle_ocr",
                "confidence": 0.9,
                "page_number": 1,
                "bbox": [10.0, 100.0, 210.0, 130.0],
                "metadata": {"layout_label": "Paragraph", "layout_position": 1},
            },
            {
                "block_id": "b2",
                "type": "paragraph",
                "text": "Line two",
                "source": "paddle_ocr",
                "confidence": 0.88,
                "page_number": 1,
                "bbox": [12.0, 134.0, 212.0, 164.0],
                "metadata": {"layout_label": "Paragraph", "layout_position": 1},
            },
        ]
    )

    assert len(merged) == 1
    assert merged[0]["text"] == "Line one\nLine two"
