"""Persistent registry for deduplicating already-processed source files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass(slots=True)
class ProcessedFileRecord:
    file_hash: str
    document_id: str
    input_file: str
    output_file: str
    rag_text_file: str
    chunk_file: str
    summary: dict[str, Any]
    warnings: list[str]
    updated_at: str


class ProcessedFileRegistry:
    def __init__(self, registry_path: str | Path = "data/processed_registry.json") -> None:
        self.registry_path = Path(registry_path)
        self._lock = Lock()

    def compute_file_hash(self, file_path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(file_path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def get(self, file_hash: str) -> ProcessedFileRecord | None:
        with self._lock:
            payload = self._read_registry()
        record = payload.get(str(file_hash).strip())
        if not isinstance(record, dict):
            return None
        try:
            return ProcessedFileRecord(
                file_hash=str(record.get("file_hash", "")).strip(),
                document_id=str(record.get("document_id", "")).strip(),
                input_file=str(record.get("input_file", "")).strip(),
                output_file=str(record.get("output_file", "")).strip(),
                rag_text_file=str(record.get("rag_text_file", "")).strip(),
                chunk_file=str(record.get("chunk_file", "")).strip(),
                summary=dict(record.get("summary") or {}),
                warnings=list(record.get("warnings") or []),
                updated_at=str(record.get("updated_at", "")).strip(),
            )
        except Exception:
            return None

    def upsert(
        self,
        *,
        file_hash: str,
        document_id: str,
        input_file: str,
        output_file: str,
        rag_text_file: str,
        chunk_file: str,
        summary: dict[str, Any],
        warnings: list[str],
    ) -> ProcessedFileRecord:
        record = ProcessedFileRecord(
            file_hash=str(file_hash).strip(),
            document_id=str(document_id).strip(),
            input_file=str(input_file).strip(),
            output_file=str(output_file).strip(),
            rag_text_file=str(rag_text_file).strip(),
            chunk_file=str(chunk_file).strip(),
            summary=dict(summary or {}),
            warnings=list(warnings or []),
            updated_at=datetime.now(UTC).isoformat(),
        )
        with self._lock:
            payload = self._read_registry()
            payload[record.file_hash] = {
                "file_hash": record.file_hash,
                "document_id": record.document_id,
                "input_file": record.input_file,
                "output_file": record.output_file,
                "rag_text_file": record.rag_text_file,
                "chunk_file": record.chunk_file,
                "summary": record.summary,
                "warnings": record.warnings,
                "updated_at": record.updated_at,
            }
            self._write_registry(payload)
        return record

    def list_all(self) -> list[ProcessedFileRecord]:
        """Return all processed file records."""
        with self._lock:
            payload = self._read_registry()
        records: list[ProcessedFileRecord] = []
        for record_data in payload.values():
            if not isinstance(record_data, dict):
                continue
            try:
                records.append(ProcessedFileRecord(
                    file_hash=str(record_data.get("file_hash", "")).strip(),
                    document_id=str(record_data.get("document_id", "")).strip(),
                    input_file=str(record_data.get("input_file", "")).strip(),
                    output_file=str(record_data.get("output_file", "")).strip(),
                    rag_text_file=str(record_data.get("rag_text_file", "")).strip(),
                    chunk_file=str(record_data.get("chunk_file", "")).strip(),
                    summary=dict(record_data.get("summary") or {}),
                    warnings=list(record_data.get("warnings") or []),
                    updated_at=str(record_data.get("updated_at", "")).strip(),
                ))
            except Exception:
                continue
        return records

    def _read_registry(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {}
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_registry(self, payload: dict[str, Any]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
