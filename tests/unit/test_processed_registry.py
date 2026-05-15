from pathlib import Path

from medical_extraction.storage.processed_registry import ProcessedFileRegistry


def test_processed_registry_round_trips_records(tmp_path: Path):
    registry = ProcessedFileRegistry(tmp_path / "processed_registry.json")
    source = tmp_path / "sample.txt"
    source.write_text("hello world", encoding="utf-8")

    file_hash = registry.compute_file_hash(source)
    registry.upsert(
        file_hash=file_hash,
        document_id="doc-1",
        input_file=str(source),
        output_file=str(tmp_path / "sample_payload.json"),
        rag_text_file=str(tmp_path / "sample_rag.txt"),
        chunk_file=str(tmp_path / "sample_chunks.json"),
        summary={"indexed_chunks": 3},
        warnings=["cached"],
    )

    record = registry.get(file_hash)

    assert record is not None
    assert record.document_id == "doc-1"
    assert record.summary["indexed_chunks"] == 3
    assert record.warnings == ["cached"]
