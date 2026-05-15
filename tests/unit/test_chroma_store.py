import json

from medical_extraction.storage.chroma_store import ChromaChunkIndexManager, ChromaSettings


def test_chroma_settings_read_config():
    settings = ChromaSettings.from_config(
        {
            "enabled": True,
            "persist_directory": "data/chroma-dev",
            "collection_name": "chunks-dev",
            "distance_metric": "cosine",
        }
    )

    assert settings.enabled is True
    assert settings.persist_directory == "data/chroma-dev"
    assert settings.collection_name == "chunks-dev"
    assert settings.distance_metric == "cosine"


def test_chroma_metadata_round_trip():
    manager = ChromaChunkIndexManager(
        ChromaSettings(enabled=True, persist_directory="data/test-chroma", collection_name="test-collection")
    )

    original = {
        "chunk_id": "doc-1:chunk:0001",
        "document_id": "doc-1",
        "source_path": r"C:\docs\report.pdf",
        "page_number": 1,
        "chunk_index": 1,
        "page_type": "copyable_pdf",
        "section": "prescription",
        "chunk_char_count": 42,
        "created_at": "2026-05-14T00:00:00Z",
        "metadata": {
            "raw_chunk_s3_uri": "s3://bucket/raw/doc-1:chunk:0001.txt",
            "entity_focus": "medication_order",
            "identity_hmacs": {"mrn_hmac": "abc123"},
        },
    }

    chroma_metadata = manager._to_chroma_metadata(original)
    restored = manager._from_chroma_metadata(chroma_metadata)

    assert restored["document_id"] == "doc-1"
    assert restored["section"] == "prescription"
    assert restored["entity_focus"] == "medication_order"
    assert restored["raw_chunk_s3_uri"] == "s3://bucket/raw/doc-1:chunk:0001.txt"
    assert restored["identity_hmacs"] == {"mrn_hmac": "abc123"}
    assert json.loads(chroma_metadata["identity_hmacs_json"]) == {"mrn_hmac": "abc123"}
