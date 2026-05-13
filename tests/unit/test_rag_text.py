from medical_extraction.utils.rag_text import build_rag_text, derive_rag_text_path


def test_build_rag_text_uses_reading_order_without_prefixes():
    payload = {
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {
                        "block_id": "b2",
                        "type": "paragraph",
                        "text": "Second line",
                        "metadata": {"reading_order": 2},
                    },
                    {
                        "block_id": "b1",
                        "type": "paragraph",
                        "text": "First line",
                        "metadata": {"reading_order": 1},
                    },
                ],
            }
        ]
    }

    text = build_rag_text(payload)
    assert text == "First line\n\nSecond line"
    assert "PAGE" not in text
    assert "ORDER" not in text


def test_build_rag_text_renders_form_and_table_as_plain_text():
    payload = {
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {
                        "block_id": "f1",
                        "type": "form",
                        "fields": {
                            "Name": {"value": "John"},
                            "Age": {"value": "42"},
                        },
                        "metadata": {"reading_order": 1},
                    },
                    {
                        "block_id": "t1",
                        "type": "table",
                        "structured_data": {"rows": [{"c1": "Hb", "c2": "12.4"}, {"c1": "WBC", "c2": "8.2"}]},
                        "metadata": {"reading_order": 2},
                    },
                ],
            }
        ]
    }

    text = build_rag_text(payload)
    assert "Name: John" in text
    assert "Age: 42" in text
    assert "Hb | 12.4" in text
    assert "WBC | 8.2" in text


def test_derive_rag_text_path_adds_rag_suffix():
    assert derive_rag_text_path(r"C:\temp\doc_extraction.json").endswith("doc_extraction_rag.txt")
