from pathlib import Path

from medical_extraction.core.pipeline import ExtractionPipeline
from medical_extraction.storage.processed_registry import ProcessedFileRecord
from medical_extraction.utils.json_utils import write_json


def test_cached_payload_refreshes_indexes_from_saved_chunk_file(tmp_path: Path):
    output_path = tmp_path / "cached_payload.json"
    rag_text_path = tmp_path / "cached_rag.txt"
    chunk_path = tmp_path / "cached_chunks.json"

    write_json(
        output_path,
        {
            "document_id": "doc-1",
            "summary": {"indexed_chunks": 0},
            "warnings": [],
            "debug_artifacts": {},
        },
    )
    rag_text_path.write_text("sample rag text", encoding="utf-8")
    write_json(
        chunk_path,
        [
            {
                "chunk_id": "doc-1:chunk:0001",
                "document_id": "doc-1",
                "chunk_text": "Amlodipine 5mg daily",
                "metadata": {},
            }
        ],
    )

    record = ProcessedFileRecord(
        file_hash="abc123",
        document_id="doc-1",
        input_file=str(tmp_path / "source.pdf"),
        output_file=str(output_path),
        rag_text_file=str(rag_text_path),
        chunk_file=str(chunk_path),
        summary={"indexed_chunks": 1},
        warnings=[],
        updated_at="2026-05-14T00:00:00+00:00",
    )

    class _Registry:
        def __init__(self):
            self.upserts = []

        def get(self, file_hash: str):
            return record if file_hash == "abc123" else None

        def upsert(self, **kwargs):
            self.upserts.append(kwargs)

    class _OutputAdapter:
        def __init__(self):
            self.writes = []

        def write_result(self, output_file: str, payload: dict):
            self.writes.append((output_file, payload))

    pipeline = object.__new__(ExtractionPipeline)
    pipeline.processed_registry = _Registry()
    pipeline.output_adapter = _OutputAdapter()
    observed = {"ingestion_calls": [], "progress": []}
    pipeline._notify_progress = lambda callback, percent, stage, detail, **extra: observed["progress"].append(
        {"percent": percent, "stage": stage, "detail": detail, **extra}
    )
    pipeline._run_ingestion = lambda payload, rag_text_payload, raw_chunks, progress_callback=None, refresh_indexes_only=False: observed[
        "ingestion_calls"
    ].append(
        {
            "document_id": payload.get("document_id"),
            "rag_text_payload": rag_text_payload,
            "raw_chunks": raw_chunks,
            "refresh_indexes_only": refresh_indexes_only,
        }
    )

    cached_payload = pipeline._load_cached_payload(
        file_hash="abc123",
        requested_output_path=str(tmp_path / "requested.json"),
        text_only=False,
    )

    assert cached_payload is not None
    assert cached_payload["debug_artifacts"]["cache_hit"] is True
    assert observed["ingestion_calls"]
    assert observed["ingestion_calls"][0]["refresh_indexes_only"] is True
    assert observed["ingestion_calls"][0]["rag_text_payload"] == "sample rag text"
    assert observed["ingestion_calls"][0]["raw_chunks"][0]["chunk_id"] == "doc-1:chunk:0001"
    assert pipeline.processed_registry.upserts
