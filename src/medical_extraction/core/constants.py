"""Shared constants for the extraction pipeline."""

EXTRACTION_VERSION = "local_mvp_v1"
DEFAULT_ROUTE = "unknown_or_hybrid"
PAGE_CLASS_COPYABLE = "copyable_pdf"
PAGE_CLASS_MIXED = "copyable_pdf_with_images"
PAGE_CLASS_SCANNED = "fully_scanned_report_form_table"
PAGE_CLASS_HANDWRITTEN = "handwritten_scanned_prescription"
PAGE_CLASS_UNKNOWN = "unknown_or_hybrid"
PAGE_CLASS_ERROR = "error"

DEFAULT_THRESHOLDS = {
    "min_selectable_text_chars": 100,
    "min_alpha_ratio": 0.45,
    "max_weird_char_ratio": 0.25,
    "max_duplicate_line_ratio": 0.40,
    "tiny_image_area_ratio": 0.02,
    "ocr_candidate_image_area_ratio": 0.10,
    "scanned_image_coverage_ratio": 0.70,
    "handwritten_review_threshold": 0.70,
    "default_review_threshold": 0.70,
    "caution_threshold": 0.90,
}

DEFAULT_CHROMA_CONFIG = {
    "enabled": True,
    "persist_directory": "data/chroma",
    "collection_name": "medical-document-chunks",
    "distance_metric": "cosine",
}

DEFAULT_KEYWORD_INDEX_CONFIG = {
    "enabled": True,
    "persist_path": "data/keyword_index.db",
    "table_name": "medical_document_chunks",
}

DEFAULT_CHUNKING_CONFIG = {
    "enabled": True,
    "write_chunk_file": True,
    "max_tokens": 220,
    "hard_max_tokens": 320,
    "min_chunk_tokens": 80,
    "overlap_tokens": 40,
    "semantic_similarity_threshold": 0.72,
    "semantic_min_sentences": 3,
    "use_semantic_layer": False,
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "local_files_only": False,
    "section_headings": [
        "history",
        "chief complaint",
        "complaints",
        "findings",
        "impression",
        "plan",
        "assessment",
        "diagnosis",
        "medications",
        "prescription",
        "labs",
        "investigations",
        "advice",
        "follow up",
        "discharge summary",
    ],
}

DEFAULT_STORAGE_CONFIG = {
    "s3": {
        "enabled": False,
        "bucket": "",
        "region": "ap-southeast-1",
        "profile": "",
        "kms_key_id": "",
        "prefix": "documents",
    }
}

DEFAULT_PRIVACY_CONFIG = {
    "enabled": True,
    "hmac_secret_env_var": "MEDICAL_RAG_HMAC_SECRET",
    "dev_fallback_secret": "medical-rag-dev-secret-change-me",
    "presidio_language": "en",
    "presidio_spacy_model": "en_core_web_sm",
}

DEFAULT_INDEXING_CONFIG = {
    "enabled": False,
    "upload_extraction_artifacts_to_s3": True,
    "upload_raw_chunks_to_s3": True,
    "index_embeddings": True,
    "embedding_model": "text-embedding-3-small",
    "embedding_dimensions": 1536,
    "batch_size": 32,
    "timeout_seconds": 60,
}

DEFAULT_RETRIEVAL_CONFIG = {
    "enabled": True,
    "embedding_model": "text-embedding-3-small",
    "embedding_dimensions": 1536,
    "timeout_seconds": 60,
    "candidate_k": 20,
    "prescription_candidate_k": 30,
    "prescription_candidate_floor": 8,
    "prescription_rerank_k": 20,
    "prescription_anchor_k": 5,
    "top_k": 5,
    "hybrid": {
        "enabled": True,
        "rrf_k": 60,
        "semantic_weight": 1.0,
        "keyword_weight": 1.0,
        "section_boost": 0.25,
        "medication_text_boost": 0.2,
        "document_affinity_boost": 0.35,
        "document_exact_phrase_boost": 0.35,
        "prescription_boilerplate_penalty": 0.2,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
    },
    "rerank": {
        "enabled": True,
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "local_files_only": False,
        "max_length": 512,
        "batch_size": 8,
    },
}

DEFAULT_AUDIT_CONFIG = {
    "enabled": True,
    "required": False,
    "backend": "dynamodb",
    "table_name": "medical-rag-access-audit",
    "region": "us-east-1",
    "endpoint_url": "http://127.0.0.1:8000",
    "tenant_key": "audit",
    "page_size": 10,
}

PRESCRIPTION_HINTS = {
    "rx",
    "prescription",
    "tablet",
    "capsule",
    "syrup",
    "tab",
    "handwritten",
    "script",
}

FORM_FIELD_HINTS = {
    "name",
    "dob",
    "date",
    "mrn",
    "id",
    "symptoms",
    "diagnosis",
}
