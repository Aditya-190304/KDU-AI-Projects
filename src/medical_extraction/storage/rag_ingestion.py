"""Post-extraction storage, HMAC metadata extraction, embedding, and Chroma indexing flow."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable

from medical_extraction.embeddings.openai_embedder import OpenAITextEmbedder
from medical_extraction.privacy.redaction import ChunkRedactor
from medical_extraction.storage.chroma_store import ChromaChunkIndexManager, ChromaSettings
from medical_extraction.storage.keyword_index import KeywordIndexSettings, SqliteKeywordIndexManager
from medical_extraction.storage.s3_storage import S3ArtifactStore, S3Settings


class RagIngestionService:
    def __init__(self, config: dict[str, Any] | None = None, device: str = "cpu") -> None:
        self.config = config or {}
        self.device = device
        self.indexing_config = self.config.get("indexing", {})
        self.privacy_config = self.config.get("privacy", {})
        self.storage_config = self.config.get("storage", {})
        self.chroma_config = self.config.get("chroma", {})
        self.keyword_index_config = self.config.get("keyword_index", {})

        s3_config = self.storage_config.get("s3", {})
        self.s3_store = S3ArtifactStore(S3Settings.from_config(s3_config))
        self.redactor = ChunkRedactor(self.privacy_config)
        self.embedder = OpenAITextEmbedder(config=self.indexing_config)
        self.chroma_manager = ChromaChunkIndexManager(ChromaSettings.from_config(self.chroma_config))
        self.keyword_index_manager = SqliteKeywordIndexManager(KeywordIndexSettings.from_config(self.keyword_index_config))

    def ingest(
        self,
        payload: dict[str, Any],
        rag_text: str,
        raw_chunks: list[dict[str, Any]],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        refresh_indexes_only: bool = False,
    ) -> dict[str, Any]:
        if not bool(self.indexing_config.get("enabled", False)):
            return {"warnings": [], "artifacts": {}, "indexed": 0}

        warnings: list[str] = []
        artifacts: dict[str, Any] = {}
        document_id = str(payload.get("document_id", "")).strip()

        raw_chunks_for_storage = [dict(chunk) for chunk in raw_chunks]
        if (
            not refresh_indexes_only
            and bool(self.indexing_config.get("upload_extraction_artifacts_to_s3", True))
            and self.s3_store.settings.enabled
        ):
            self._notify_progress(progress_callback, 86, "uploading_artifacts", "Uploading retrieval text to S3.")
            try:
                artifacts["rag_text_s3_uri"] = self.s3_store.put_text(
                    self.s3_store.build_key(document_id, "extraction", "rag.txt"),
                    rag_text,
                )
            except Exception as exc:
                warnings.append(f"Failed to upload extraction artifacts to S3: {exc}")

        if (
            not refresh_indexes_only
            and bool(self.indexing_config.get("upload_raw_chunks_to_s3", True))
            and self.s3_store.settings.enabled
        ):
            self._notify_progress(progress_callback, 88, "uploading_chunks", "Uploading raw chunk files to S3.")
            try:
                for chunk in raw_chunks_for_storage:
                    chunk_key = self.s3_store.build_key(document_id, "chunks", "raw", f"{chunk['chunk_id']}.txt")
                    chunk["raw_chunk_s3_uri"] = self.s3_store.put_text(chunk_key, str(chunk.get("chunk_text", "")))
                artifacts["raw_chunk_count"] = len(raw_chunks_for_storage)
            except Exception as exc:
                warnings.append(f"Failed to upload raw chunks to S3: {exc}")

        index_documents = [self._build_index_document(chunk) for chunk in raw_chunks_for_storage]

        if bool(self.keyword_index_config.get("enabled", False)):
            try:
                self._notify_progress(progress_callback, 89, "lexical_indexing", "Writing chunk text into the persistent keyword index.")
                keyword_connection = self.keyword_index_manager.create_connection()
                try:
                    self.keyword_index_manager.upsert_documents(keyword_connection, index_documents)
                finally:
                    keyword_connection.close()
                artifacts["keyword_index_table"] = self.keyword_index_manager.settings.table_name
            except Exception as exc:
                warnings.append(f"Failed to index chunk text in the keyword index: {exc}")

        indexed = 0
        if bool(self.indexing_config.get("index_embeddings", True)) and bool(self.chroma_config.get("enabled", False)):
            try:
                self._notify_progress(
                    progress_callback,
                    90,
                    "embedding",
                    f"Creating embeddings for {len(raw_chunks_for_storage)} chunk{'s' if len(raw_chunks_for_storage) != 1 else ''}.",
                    total_chunks=len(raw_chunks_for_storage),
                )
                embeddings = self.embedder.encode_texts([str(chunk.get("chunk_text", "")) for chunk in raw_chunks_for_storage])
                for chunk, embedding in zip(index_documents, embeddings):
                    chunk["embedding"] = embedding
                    chunk["embedding_model"] = self.embedder.model_name
                    chunk["embedding_dimension"] = len(embedding)
                self._notify_progress(progress_callback, 94, "indexing", "Writing chunks and embeddings into Chroma.")
                client = self.chroma_manager.create_client()
                self.chroma_manager.ensure_collection(client)
                indexed = self.chroma_manager.upsert_documents(client, index_documents)["indexed"]
                artifacts["chroma_collection"] = self.chroma_manager.settings.collection_name
            except Exception as exc:
                warnings.append(f"Failed to index chunk embeddings in Chroma: {exc}")

        return {
            "warnings": warnings,
            "artifacts": artifacts,
            "indexed": indexed,
            "raw_chunks": raw_chunks_for_storage,
            "index_documents": index_documents,
        }

    def _notify_progress(
        self,
        callback: Callable[[dict[str, Any]], None] | None,
        percent: int,
        stage: str,
        detail: str,
        **extra: Any,
    ) -> None:
        if callback is None:
            return
        payload: dict[str, Any] = {
            "percent": max(0, min(int(percent), 100)),
            "stage": str(stage),
            "detail": str(detail),
        }
        payload.update(extra)
        callback(payload)

    def _build_index_document(self, raw_chunk: dict[str, Any]) -> dict[str, Any]:
        redacted_chunk = self.redactor.redact_chunk(raw_chunk)
        metadata = {
            "identity_hmacs": dict((redacted_chunk.get("metadata") or {}).get("identity_hmacs") or {}),
            "entity_focus": str((raw_chunk.get("metadata") or {}).get("entity_focus", "")).strip(),
        }
        raw_uri = raw_chunk.get("raw_chunk_s3_uri")
        if raw_uri:
            metadata["raw_chunk_s3_uri"] = raw_uri
        section = raw_chunk.get("section")
        return {
            "chunk_id": raw_chunk.get("chunk_id"),
            "document_id": raw_chunk.get("document_id"),
            "source_path": raw_chunk.get("source_path"),
            "page_number": raw_chunk.get("page_number"),
            "chunk_index": raw_chunk.get("chunk_index"),
            "page_type": raw_chunk.get("page_type"),
            "section": section if section is not None else "",
            "chunk_text": str(raw_chunk.get("chunk_text", "")),
            "chunk_char_count": len(str(raw_chunk.get("chunk_text", ""))),
            "metadata": metadata,
            "created_at": datetime.now(UTC).isoformat(),
        }
