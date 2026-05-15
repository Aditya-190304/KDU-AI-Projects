from medical_extraction.retrieval import ChromaRetriever


def _base_config() -> dict:
    return {
        "chroma": {
            "enabled": True,
            "persist_directory": "data/test-chroma",
            "collection_name": "test-collection",
        },
        "privacy": {
            "hmac_secret_env_var": "TEST_MEDICAL_RAG_HMAC_SECRET",
            "dev_fallback_secret": "unit-test-secret",
        },
        "retrieval": {
            "embedding_model": "text-embedding-3-small",
            "embedding_dimensions": 1536,
            "candidate_k": 10,
            "prescription_candidate_k": 30,
            "prescription_candidate_floor": 8,
            "prescription_rerank_k": 20,
            "hybrid": {
                "rrf_k": 60,
                "semantic_weight": 1.0,
                "keyword_weight": 1.0,
                "section_boost": 0.25,
                "medication_text_boost": 0.2,
            },
            "rerank": {
                "enabled": False,
            },
        },
    }


def test_rrf_merge_prefers_document_present_in_both_rankings():
    retriever = ChromaRetriever(config=_base_config())

    semantic_candidates = [
        {"chunk_id": "c1", "document": "general note", "metadata": {}, "semantic_score": 0.9},
        {"chunk_id": "c2", "document": "apollo phone number", "metadata": {}, "semantic_score": 0.8},
    ]
    keyword_candidates = [
        {"chunk_id": "c3", "document": "generic", "metadata": {}, "keyword_score": 3.0},
        {"chunk_id": "c2", "document": "apollo phone number", "metadata": {}, "keyword_score": 2.5},
    ]

    merged = retriever._merge_with_rrf(semantic_candidates, keyword_candidates)
    merged.sort(key=lambda candidate: candidate["hybrid_score"], reverse=True)

    assert merged[0]["chunk_id"] == "c2"


def test_prescription_queries_boost_medication_order_chunks():
    retriever = ChromaRetriever(config=_base_config())
    candidates = [
        {
            "chunk_id": "generic",
            "document": "Patient: Oliver Johnson",
            "metadata": {"section": "none", "entity_focus": ""},
            "hybrid_score": 1.0,
        },
        {
            "chunk_id": "rx",
            "document": "Prescription: Sertraline 50mg tablets",
            "metadata": {"section": "prescription", "entity_focus": "medication_order"},
            "hybrid_score": 0.9,
        },
    ]

    retriever._apply_query_intent_boosts(candidates, "What medicines are prescribed?")
    candidates.sort(key=lambda candidate: candidate["hybrid_score"], reverse=True)

    assert candidates[0]["chunk_id"] == "rx"


def test_prescription_final_selector_prefers_medication_orders():
    retriever = ChromaRetriever(config=_base_config())
    candidates = [
        {
            "chunk_id": "generic-1",
            "raw_text": "Patient Instructions:",
            "section": "prescription",
            "metadata": {"entity_focus": ""},
        },
        {
            "chunk_id": "med-1",
            "raw_text": "Prescription: Sertraline 50mg tablets - Take one tablet orally once daily",
            "section": "prescription",
            "metadata": {"entity_focus": "medication_order"},
        },
        {
            "chunk_id": "med-2",
            "raw_text": "Prescription: Bupropion SR 150mg tablets - Take one tablet orally twice daily",
            "section": "prescription",
            "metadata": {"entity_focus": "medication_order"},
        },
        {
            "chunk_id": "generic-2",
            "raw_text": "Take care and stay healthy,",
            "section": "prescription",
            "metadata": {"entity_focus": ""},
        },
    ]

    selected = retriever._select_final_candidates("What medicines are prescribed?", candidates, top_k=2)
    selected_ids = [candidate["chunk_id"] for candidate in selected]

    assert selected_ids == ["med-1", "med-2"]


def test_prescription_queries_expand_candidate_and_rerank_windows():
    retriever = ChromaRetriever(config=_base_config())

    assert retriever._resolve_candidate_k("what medicines are prescribed?", 10) == 30
    assert retriever._resolve_candidate_k("what is the diagnosis?", 10) == 10
    assert retriever._resolve_rerank_k("what medicines are prescribed?", 5, 30) == 20
    assert retriever._resolve_rerank_k("what is the diagnosis?", 5, 30) == 5
    assert retriever._preferred_sections_for_query("what medicines are prescribed?") == {"prescription", "medications"}


def test_prescription_query_boosts_correct_document_medication_rows():
    retriever = ChromaRetriever(config=_base_config())
    candidates = [
        {
            "chunk_id": "sofia-generic",
            "document": "Diagnosis: Prescribing Physician: Dr. Jennifer Kim",
            "metadata": {
                "document_id": "Sofia_Allen-Jennifer_Kim-Mercy_General_Hospital-MRN100007",
                "section": "diagnosis",
                "entity_focus": "",
            },
            "hybrid_score": 1.0,
        },
        {
            "chunk_id": "john-name",
            "document": "Prescription: Name: John Scott Date: 2025-06-20 Age: 37 MRN: MRN100008",
            "metadata": {
                "document_id": "John_Scott-Jennifer_Kim-Mercy_General_Hospital-MRN100008",
                "section": "prescription",
                "entity_focus": "prescription_notes",
            },
            "hybrid_score": 0.9,
        },
        {
            "chunk_id": "john-med",
            "document": "Medications: 1. Amlodipine | 5mg | 1 tablet daily",
            "metadata": {
                "document_id": "John_Scott-Jennifer_Kim-Mercy_General_Hospital-MRN100008",
                "section": "medications",
                "entity_focus": "table_row",
            },
            "hybrid_score": 0.85,
        },
    ]

    retriever._apply_query_intent_boosts(
        candidates,
        "what medication is prescribed to john scott by dr. jennifer kim at mercy general hospital",
    )
    candidates.sort(key=lambda candidate: candidate["hybrid_score"], reverse=True)

    assert candidates[0]["chunk_id"] == "john-med"


def test_prescription_queries_preserve_medication_anchor_candidates_after_rerank():
    retriever = ChromaRetriever(config=_base_config())
    prepared_candidates = [
        {
            "chunk_id": "john-header",
            "raw_text": "Prescription: Name: John Scott Date: 2025-06-20 Age: 37 MRN: MRN100008",
            "section": "prescription",
            "metadata": {
                "document_id": "John_Scott-Jennifer_Kim-Mercy_General_Hospital-MRN100008",
                "entity_focus": "prescription_notes",
            },
            "hybrid_score": 2.0,
        },
        {
            "chunk_id": "john-med-1",
            "raw_text": "Medications: 1. Amlodipine | 5mg | 1 tablet daily",
            "section": "medications",
            "metadata": {
                "document_id": "John_Scott-Jennifer_Kim-Mercy_General_Hospital-MRN100008",
                "entity_focus": "table_row",
            },
            "hybrid_score": 1.8,
        },
        {
            "chunk_id": "john-med-2",
            "raw_text": "Medications: 2. Losartan | 50mg | 1 tablet daily",
            "section": "medications",
            "metadata": {
                "document_id": "John_Scott-Jennifer_Kim-Mercy_General_Hospital-MRN100008",
                "entity_focus": "table_row",
            },
            "hybrid_score": 1.7,
        },
    ]

    reranked_candidates = [
        {
            **prepared_candidates[0],
            "rerank_score": 9.5,
        }
    ]

    preserved = retriever._ensure_prescription_anchor_candidates(
        query_text="what medication is prescribed to john scott by dr. jennifer kim at mercy general hospital",
        prepared_candidates=prepared_candidates,
        reranked_candidates=reranked_candidates,
    )
    preserved_ids = [candidate["chunk_id"] for candidate in preserved]

    assert "john-med-1" in preserved_ids
    assert "john-med-2" in preserved_ids


def test_retriever_returns_redacted_or_raw_context_based_on_authorization():
    config = _base_config()
    config["retrieval"]["rerank"]["enabled"] = True
    retriever = ChromaRetriever(config=config)
    retriever.hybrid_search = lambda *args, **kwargs: {
        "candidates": [
            {
                "chunk_id": "doc-1:chunk:0001",
                "document": "Patient Name: John Doe. Severe abdominal pain.",
                "metadata": {
                    "document_id": "doc-1",
                    "page_number": 1,
                    "section": "history",
                    "raw_chunk_s3_uri": "s3://demo-bucket/doc-1/chunks/raw/doc-1:chunk:0001.txt",
                    "entity_focus": "",
                    "identity_hmacs": {},
                },
                "semantic_score": 0.42,
                "keyword_score": 0.9,
                "hybrid_score": 0.75,
            }
        ]
    }
    observed = {}

    def _fake_rerank(query_text, candidates, text_field, top_k):
        observed["text_field"] = text_field
        observed["first_raw_text"] = candidates[0]["raw_text"]
        return [{**candidates[0], "rerank_score": 9.9}]

    retriever.reranker.rerank = _fake_rerank

    unauthorized = retriever.retrieve_for_generation("abdominal pain", authorized=False)
    authorized = retriever.retrieve_for_generation("abdominal pain", authorized=True)

    assert observed["text_field"] == "raw_text"
    assert "John Doe" in observed["first_raw_text"]
    assert "[PATIENT_NAME]" in unauthorized["context_chunks"][0]["content"]
    assert "John Doe" in authorized["context_chunks"][0]["content"]
