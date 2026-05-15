from medical_extraction.utils.chunking import LocalMedicalChunker, derive_chunk_path


def test_derive_chunk_path_adds_chunks_suffix():
    assert derive_chunk_path(r"C:\temp\report_rag.txt").endswith("report_rag_chunks.json")


def test_hybrid_chunker_preserves_medications_and_sections():
    payload = {
        "document_id": "doc-1",
        "input_file": r"C:\uploads\report.pdf",
        "created_at": "2026-05-13T00:00:00Z",
        "pages": [
            {
                "page_number": 1,
                "page_type": "copyable_pdf",
                "blocks": [
                    {
                        "block_id": "b1",
                        "type": "paragraph",
                        "text": "Medications:",
                        "source": "pdfplumber",
                        "confidence": 0.99,
                        "metadata": {"reading_order": 1},
                    },
                    {
                        "block_id": "b2",
                        "type": "paragraph",
                        "text": "Warfarin 2.5mg twice daily\nMetformin 500mg once daily",
                        "source": "pdfplumber",
                        "confidence": 0.93,
                        "metadata": {"reading_order": 2},
                    },
                    {
                        "block_id": "b3",
                        "type": "paragraph",
                        "text": "Findings: Patient reports chest pain. ECG is normal. Troponin pending.",
                        "source": "pdfplumber",
                        "confidence": 0.92,
                        "metadata": {"reading_order": 3},
                    },
                ],
            }
        ],
    }

    chunker = LocalMedicalChunker(config={"use_semantic_layer": False}, device="cpu")
    result = chunker.build_chunks(payload, source_text_path=r"C:\output\report_rag.txt")

    assert len(result.chunks) == 3
    warfarin_chunk = result.chunks[0]
    metformin_chunk = result.chunks[1]
    findings_chunk = result.chunks[2]

    assert warfarin_chunk["section"] == "medications"
    assert warfarin_chunk["metadata"]["strategy"] == "entity_preserving"
    assert "Warfarin 2.5mg twice daily" in warfarin_chunk["chunk_text"]
    assert "Metformin 500mg once daily" in metformin_chunk["chunk_text"]
    assert findings_chunk["section"] == "findings"
    assert findings_chunk["chunk_text"].startswith("Findings:")


def test_hybrid_chunker_splits_table_rows_into_chunks():
    payload = {
        "document_id": "doc-2",
        "input_file": r"C:\uploads\labs.pdf",
        "created_at": "2026-05-13T00:00:00Z",
        "pages": [
            {
                "page_number": 1,
                "page_type": "copyable_pdf",
                "blocks": [
                    {
                        "block_id": "t1",
                        "type": "table",
                        "text": "",
                        "source": "camelot",
                        "confidence": 0.95,
                        "structured_data": {
                            "rows": [
                                {"test": "HbA1c", "value": "7.2", "unit": "%"},
                                {"test": "Creatinine", "value": "1.1", "unit": "mg/dL"},
                            ]
                        },
                        "metadata": {"reading_order": 1},
                    }
                ],
            }
        ],
    }

    chunker = LocalMedicalChunker(config={"use_semantic_layer": False}, device="cpu")
    result = chunker.build_chunks(payload, source_text_path=r"C:\output\labs_rag.txt")

    assert len(result.chunks) == 2
    assert result.chunks[0]["metadata"]["entity_focus"] == "table_row"
    assert "HbA1c | 7.2 | %" in result.chunks[0]["chunk_text"]
    assert "Creatinine | 1.1 | mg/dL" in result.chunks[1]["chunk_text"]


def test_prescription_section_groups_medication_with_instructions():
    payload = {
        "document_id": "doc-3",
        "input_file": r"C:\uploads\prescription.pdf",
        "created_at": "2026-05-14T00:00:00Z",
        "pages": [
            {
                "page_number": 1,
                "page_type": "copyable_pdf",
                "blocks": [
                    {
                        "block_id": "p1",
                        "type": "paragraph",
                        "text": "Prescription (Rx)",
                        "source": "pdf_text",
                        "confidence": 1.0,
                        "metadata": {"reading_order": 1},
                    },
                    {
                        "block_id": "p2",
                        "type": "paragraph",
                        "text": "1. Sertraline 50mg tablets",
                        "source": "pdf_text",
                        "confidence": 1.0,
                        "metadata": {"reading_order": 2},
                    },
                    {
                        "block_id": "p3",
                        "type": "paragraph",
                        "text": "Take one tablet orally once daily in the morning",
                        "source": "pdf_text",
                        "confidence": 1.0,
                        "metadata": {"reading_order": 3},
                    },
                    {
                        "block_id": "p4",
                        "type": "paragraph",
                        "text": "- Dispense 30 tablets.",
                        "source": "pdf_text",
                        "confidence": 1.0,
                        "metadata": {"reading_order": 4},
                    },
                    {
                        "block_id": "p5",
                        "type": "paragraph",
                        "text": "2. Bupropion SR 150mg tablets",
                        "source": "pdf_text",
                        "confidence": 1.0,
                        "metadata": {"reading_order": 5},
                    },
                    {
                        "block_id": "p6",
                        "type": "paragraph",
                        "text": "Take one tablet orally twice daily, in the morning and in the evening",
                        "source": "pdf_text",
                        "confidence": 1.0,
                        "metadata": {"reading_order": 6},
                    },
                    {
                        "block_id": "p7",
                        "type": "paragraph",
                        "text": "- Dispense 60 tablets.",
                        "source": "pdf_text",
                        "confidence": 1.0,
                        "metadata": {"reading_order": 7},
                    },
                ],
            }
        ],
    }

    chunker = LocalMedicalChunker(config={"use_semantic_layer": False}, device="cpu")
    result = chunker.build_chunks(payload, source_text_path=r"C:\output\prescription_rag.txt")

    assert len(result.chunks) == 2
    assert "Sertraline 50mg tablets" in result.chunks[0]["chunk_text"]
    assert "Take one tablet orally once daily in the morning" in result.chunks[0]["chunk_text"]
    assert "Dispense 30 tablets" in result.chunks[0]["chunk_text"]
    assert result.chunks[0]["metadata"]["entity_focus"] == "medication_order"
    assert "Bupropion SR 150mg tablets" in result.chunks[1]["chunk_text"]
    assert "Take one tablet orally twice daily" in result.chunks[1]["chunk_text"]
    assert "Dispense 60 tablets" in result.chunks[1]["chunk_text"]


def test_form_like_page_emits_demographics_chunks_alongside_medication_chunks():
    payload = {
        "document_id": "9",
        "input_file": r"C:\uploads\9.jpg",
        "created_at": "2026-05-14T00:00:00Z",
        "pages": [
            {
                "page_number": 1,
                "page_type": "fully_scanned_report_form_table",
                "blocks": [
                    {
                        "block_id": "labels",
                        "type": "paragraph",
                        "text": "Patient Name:\nAddress:\nDOB:\nAllergies:\nWeight:\nRX:",
                        "source": "paddle_ocr",
                        "confidence": 1.0,
                        "bbox": [11.0, 107.0, 128.0, 289.0],
                        "metadata": {"reading_order": 4},
                    },
                    {
                        "block_id": "name",
                        "type": "paragraph",
                        "text": "Joseph McIntyre",
                        "source": "paddle_ocr",
                        "confidence": 1.0,
                        "bbox": [163.0, 109.0, 348.0, 134.0],
                        "metadata": {"reading_order": 5},
                    },
                    {
                        "block_id": "values",
                        "type": "paragraph",
                        "text": "12/26/1998\nNKDA\n65 kg\nAzithromycin 200 mg/5mL\nDay 1: 15 mL\nDay 2: 7.5 mL",
                        "source": "paddle_ocr",
                        "confidence": 0.98,
                        "bbox": [151.0, 165.0, 463.0, 343.0],
                        "metadata": {"reading_order": 6},
                    },
                ],
            }
        ],
    }

    chunker = LocalMedicalChunker(config={"use_semantic_layer": False}, device="cpu")
    result = chunker.build_chunks(payload, source_text_path=r"C:\output\9_result_rag.txt")

    chunk_texts = [chunk["chunk_text"] for chunk in result.chunks]
    assert any("DOB: 12/26/1998" in chunk_text for chunk_text in chunk_texts)
    assert any("Patient Name: Joseph McIntyre" in chunk_text for chunk_text in chunk_texts)
    assert any("Allergies: NKDA" in chunk_text for chunk_text in chunk_texts)
    assert any("Weight: 65 kg" in chunk_text for chunk_text in chunk_texts)
    assert any("Azithromycin 200 mg/5mL" in chunk_text for chunk_text in chunk_texts)
    assert any(chunk["metadata"].get("entity_focus") == "demographic_profile" for chunk in result.chunks)
