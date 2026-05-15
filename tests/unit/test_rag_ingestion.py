from medical_extraction.storage.rag_ingestion import RagIngestionService
from medical_extraction.storage.keyword_index import KeywordIndexSettings, SqliteKeywordIndexManager


def test_rag_ingestion_builds_index_documents_with_raw_chunk_text_and_hmacs():
    service = RagIngestionService(
        config={
            "storage": {"s3": {"enabled": False}},
            "chroma": {"enabled": False},
            "privacy": {
                "hmac_secret_env_var": "TEST_MEDICAL_RAG_HMAC_SECRET",
                "dev_fallback_secret": "unit-test-secret",
            },
            "indexing": {"enabled": True, "index_embeddings": False},
        },
        device="cpu",
    )

    payload = {"document_id": "doc-1"}
    raw_chunks = [
        {
            "chunk_id": "doc-1:chunk:0001",
            "document_id": "doc-1",
            "chunk_text": "Patient Name: John Doe. Warfarin 2.5mg twice daily.",
            "metadata": {},
        }
    ]

    result = service.ingest(payload=payload, rag_text="sample text", raw_chunks=raw_chunks)

    assert result["indexed"] == 0
    assert len(result["index_documents"]) == 1
    assert result["index_documents"][0]["chunk_text"] == "Patient Name: John Doe. Warfarin 2.5mg twice daily."
    assert "patient_name_hmac" in result["index_documents"][0]["metadata"]["identity_hmacs"]


def test_rag_ingestion_s3_uploads_only_text_artifacts():
    service = RagIngestionService(
        config={
            "storage": {
                "s3": {
                    "enabled": True,
                    "bucket": "demo-bucket",
                    "region": "ap-southeast-1",
                }
            },
            "chroma": {"enabled": False},
            "privacy": {
                "hmac_secret_env_var": "TEST_MEDICAL_RAG_HMAC_SECRET",
                "dev_fallback_secret": "unit-test-secret",
            },
            "indexing": {"enabled": True, "index_embeddings": False},
        },
        device="cpu",
    )

    uploads: list[tuple[str, str]] = []

    class _FakeS3Store:
        settings = type("Settings", (), {"enabled": True})()

        def build_key(self, document_id: str, *parts: str) -> str:
            return "/".join((document_id, *parts))

        def put_text(self, key: str, text: str) -> str:
            uploads.append((key, text))
            return f"s3://demo-bucket/{key}"

    service.s3_store = _FakeS3Store()

    payload = {"document_id": "doc-2"}
    raw_chunks = [
        {
            "chunk_id": "doc-2:chunk:0001",
            "document_id": "doc-2",
            "chunk_text": "Raw chunk one",
            "metadata": {},
        },
        {
            "chunk_id": "doc-2:chunk:0002",
            "document_id": "doc-2",
            "chunk_text": "Raw chunk two",
            "metadata": {},
        },
    ]

    result = service.ingest(payload=payload, rag_text="document text", raw_chunks=raw_chunks)

    assert result["artifacts"]["rag_text_s3_uri"] == "s3://demo-bucket/doc-2/extraction/rag.txt"
    assert result["artifacts"]["raw_chunk_count"] == 2
    assert [key for key, _ in uploads] == [
        "doc-2/extraction/rag.txt",
        "doc-2/chunks/raw/doc-2:chunk:0001.txt",
        "doc-2/chunks/raw/doc-2:chunk:0002.txt",
    ]
    assert all(not key.endswith(".json") for key, _ in uploads)
    assert result["index_documents"][0]["metadata"]["raw_chunk_s3_uri"] == "s3://demo-bucket/doc-2/chunks/raw/doc-2:chunk:0001.txt"


def test_rag_ingestion_writes_chunk_text_into_persistent_keyword_index(tmp_path):
    keyword_db = tmp_path / "keyword-index.db"
    service = RagIngestionService(
        config={
            "storage": {"s3": {"enabled": False}},
            "chroma": {"enabled": False},
            "keyword_index": {
                "enabled": True,
                "persist_path": str(keyword_db),
                "table_name": "unit_chunks",
            },
            "privacy": {
                "hmac_secret_env_var": "TEST_MEDICAL_RAG_HMAC_SECRET",
                "dev_fallback_secret": "unit-test-secret",
            },
            "indexing": {"enabled": True, "index_embeddings": False},
        },
        device="cpu",
    )

    payload = {"document_id": "doc-3"}
    raw_chunks = [
        {
            "chunk_id": "doc-3:chunk:0001",
            "document_id": "doc-3",
            "chunk_text": "Bupropion SR 150mg tablets twice daily.",
            "metadata": {"entity_focus": "medication_order"},
        }
    ]

    service.ingest(payload=payload, rag_text="sample text", raw_chunks=raw_chunks)

    manager = SqliteKeywordIndexManager(
        KeywordIndexSettings(enabled=True, persist_path=str(keyword_db), table_name="unit_chunks")
    )
    connection = manager.create_connection()
    try:
        results = manager.keyword_search(connection, "bupropion", size=5)
    finally:
        connection.close()

    assert results
    assert results[0]["chunk_id"] == "doc-3:chunk:0001"
