from medical_extraction.extraction.merge_engine import merge_blocks


def test_merge_blocks_applies_column_aware_reading_order():
    blocks = [
        {
            "block_id": "p1_b1",
            "type": "paragraph",
            "text": "Left column line 1",
            "source": "paddle_ocr",
            "confidence": 0.9,
            "page_number": 1,
            "bbox": [10.0, 100.0, 180.0, 130.0],
        },
        {
            "block_id": "p1_b2",
            "type": "paragraph",
            "text": "Right column line 1",
            "source": "paddle_ocr",
            "confidence": 0.9,
            "page_number": 1,
            "bbox": [300.0, 90.0, 480.0, 120.0],
        },
        {
            "block_id": "p1_b3",
            "type": "paragraph",
            "text": "Left column line 2",
            "source": "paddle_ocr",
            "confidence": 0.9,
            "page_number": 1,
            "bbox": [12.0, 140.0, 182.0, 168.0],
        },
        {
            "block_id": "p1_b4",
            "type": "paragraph",
            "text": "Right column line 2",
            "source": "paddle_ocr",
            "confidence": 0.9,
            "page_number": 1,
            "bbox": [302.0, 130.0, 482.0, 160.0],
        },
    ]

    merged = merge_blocks(blocks)
    ordered_ids = [item["block_id"] for item in merged]
    assert ordered_ids[:2] == ["p1_b1", "p1_b3"]
    assert ordered_ids[2:] == ["p1_b2", "p1_b4"]
    assert merged[0]["metadata"]["reading_order"] == 1
    assert merged[3]["metadata"]["reading_order"] == 4


def test_merge_blocks_only_merges_adjacent_paragraph_lines():
    blocks = [
        {
            "block_id": "p1_b1",
            "type": "paragraph",
            "text": "Hospital Name",
            "source": "paddle_ocr",
            "confidence": 0.9,
            "page_number": 1,
            "bbox": [10.0, 12.0, 200.0, 30.0],
            "metadata": {"layout_label": "Paragraph", "layout_position": 1},
        },
        {
            "block_id": "p1_b2",
            "type": "paragraph",
            "text": "Address line",
            "source": "paddle_ocr",
            "confidence": 0.88,
            "page_number": 1,
            "bbox": [10.0, 34.0, 200.0, 52.0],
            "metadata": {"layout_label": "Paragraph", "layout_position": 1},
        },
        {
            "block_id": "p1_t1",
            "type": "table",
            "text": "A B",
            "source": "paddle_table_recognition",
            "confidence": 0.8,
            "page_number": 1,
            "bbox": [20.0, 120.0, 350.0, 220.0],
        },
    ]

    merged = merge_blocks(blocks)
    assert len(merged) == 2
    assert merged[0]["type"] == "paragraph"
    assert "Hospital Name Address line" in merged[0]["text"]
    assert merged[1]["type"] == "table"


def test_merge_blocks_keeps_close_but_non_overlapping_columns_separate():
    blocks = [
        {
            "block_id": "p1_c1_l1",
            "type": "paragraph",
            "text": "Column A line 1",
            "source": "surya_ocr",
            "confidence": 0.9,
            "page_number": 1,
            "bbox": [20.0, 100.0, 110.0, 130.0],
        },
        {
            "block_id": "p1_c2_l1",
            "type": "paragraph",
            "text": "Column B line 1",
            "source": "surya_ocr",
            "confidence": 0.9,
            "page_number": 1,
            "bbox": [120.0, 80.0, 210.0, 110.0],
        },
        {
            "block_id": "p1_c1_l2",
            "type": "paragraph",
            "text": "Column A line 2",
            "source": "surya_ocr",
            "confidence": 0.9,
            "page_number": 1,
            "bbox": [22.0, 140.0, 108.0, 168.0],
        },
        {
            "block_id": "p1_c2_l2",
            "type": "paragraph",
            "text": "Column B line 2",
            "source": "surya_ocr",
            "confidence": 0.9,
            "page_number": 1,
            "bbox": [122.0, 118.0, 208.0, 148.0],
        },
    ]

    merged = merge_blocks(blocks)
    ordered_ids = [item["block_id"] for item in merged]
    assert ordered_ids == ["p1_c1_l1", "p1_c1_l2", "p1_c2_l1", "p1_c2_l2"]
