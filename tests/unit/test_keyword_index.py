from medical_extraction.storage.keyword_index import KeywordIndexSettings, SqliteKeywordIndexManager


def test_keyword_index_upserts_and_searches_documents(tmp_path):
    manager = SqliteKeywordIndexManager(
        KeywordIndexSettings(
            enabled=True,
            persist_path=str(tmp_path / "keyword-index.db"),
            table_name="unit_chunks",
        )
    )
    connection = manager.create_connection()
    try:
        manager.upsert_documents(
            connection,
            [
                {
                    "chunk_id": "doc-1:chunk:0001",
                    "document_id": "doc-1",
                    "chunk_text": "Azithromycin 200 mg per 5 mL. Day 1: 15 mL.",
                    "page_number": 1,
                    "chunk_index": 1,
                    "section": "prescription",
                    "metadata": {"entity_focus": "medication_order", "identity_hmacs": {}},
                },
                {
                    "chunk_id": "doc-2:chunk:0001",
                    "document_id": "doc-2",
                    "chunk_text": "Sertraline 50mg tablets once daily in the morning.",
                    "page_number": 1,
                    "chunk_index": 1,
                    "section": "prescription",
                    "metadata": {"entity_focus": "medication_order", "identity_hmacs": {}},
                },
            ],
        )

        all_results = manager.keyword_search(connection, "sertraline tablets", size=5)
        filtered_results = manager.keyword_search(connection, "sertraline tablets", size=5, document_id="doc-2")
    finally:
        connection.close()

    assert all_results
    assert all_results[0]["chunk_id"] == "doc-2:chunk:0001"
    assert filtered_results
    assert filtered_results[0]["metadata"]["document_id"] == "doc-2"

