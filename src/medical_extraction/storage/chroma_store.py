"""Persistent Chroma helpers for chunk and embedding storage."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection


@dataclass(slots=True)
class ChromaSettings:
    enabled: bool
    persist_directory: str
    collection_name: str
    distance_metric: str = "cosine"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ChromaSettings":
        return cls(
            enabled=bool(config.get("enabled", False)),
            persist_directory=str(config.get("persist_directory", "data/chroma")).strip() or "data/chroma",
            collection_name=str(config.get("collection_name", "medical-document-chunks")).strip() or "medical-document-chunks",
            distance_metric=str(config.get("distance_metric", "cosine")).strip() or "cosine",
        )


class ChromaChunkIndexManager:
    def __init__(self, settings: ChromaSettings) -> None:
        self.settings = settings

    def create_client(self) -> chromadb.PersistentClient:
        persist_directory = Path(self.settings.persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(persist_directory))

    def ensure_collection(self, client: chromadb.PersistentClient) -> Collection:
        return client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": self.settings.distance_metric},
        )

    def reset_collection(self, client: chromadb.PersistentClient) -> None:
        existing = {collection.name for collection in client.list_collections()}
        if self.settings.collection_name in existing:
            client.delete_collection(self.settings.collection_name)
        self.ensure_collection(client)

    def upsert_documents(self, client: chromadb.PersistentClient, documents: list[dict[str, Any]]) -> dict[str, Any]:
        collection = self.ensure_collection(client)
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        texts: list[str] = []
        embeddings: list[list[float]] = []

        for document in documents:
            chunk_id = str(document.get("chunk_id", "")).strip()
            text = str(document.get("chunk_text", "")).strip()
            embedding = document.get("embedding")
            if not chunk_id or not text or not isinstance(embedding, list):
                continue
            ids.append(chunk_id)
            texts.append(text)
            embeddings.append([float(value) for value in embedding])
            metadatas.append(self._to_chroma_metadata(document))

        if ids:
            collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        return {"indexed": len(ids), "collection_name": self.settings.collection_name}

    def semantic_search(
        self,
        client: chromadb.PersistentClient,
        query_embedding: list[float],
        size: int,
    ) -> list[dict[str, Any]]:
        collection = self.ensure_collection(client)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=size,
            include=["documents", "metadatas", "distances"],
        )
        ids = list((result.get("ids") or [[]])[0])
        docs = list((result.get("documents") or [[]])[0])
        metadatas = list((result.get("metadatas") or [[]])[0])
        distances = list((result.get("distances") or [[]])[0])
        candidates: list[dict[str, Any]] = []
        for rank, (chunk_id, document, metadata, distance) in enumerate(zip(ids, docs, metadatas, distances), start=1):
            decoded_metadata = self._from_chroma_metadata(metadata or {})
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "document": str(document or ""),
                    "metadata": decoded_metadata,
                    "semantic_distance": float(distance if distance is not None else 1.0),
                    "semantic_score": 1.0 - float(distance if distance is not None else 1.0),
                    "rank": rank,
                }
            )
        return candidates

    def get_all_documents(self, client: chromadb.PersistentClient) -> list[dict[str, Any]]:
        collection = self.ensure_collection(client)
        result = collection.get(include=["documents", "metadatas"])
        ids = list(result.get("ids") or [])
        docs = list(result.get("documents") or [])
        metadatas = list(result.get("metadatas") or [])
        records: list[dict[str, Any]] = []
        for chunk_id, document, metadata in zip(ids, docs, metadatas):
            records.append(
                {
                    "chunk_id": str(chunk_id),
                    "document": str(document or ""),
                    "metadata": self._from_chroma_metadata(metadata or {}),
                }
            )
        return records

    def _to_chroma_metadata(self, document: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(document.get("metadata") or {})
        return {
            "document_id": str(document.get("document_id", "")).strip(),
            "source_path": str(document.get("source_path", "")).strip(),
            "page_number": int(document.get("page_number", 0) or 0),
            "chunk_index": int(document.get("chunk_index", 0) or 0),
            "page_type": str(document.get("page_type", "")).strip(),
            "section": str(document.get("section", "")).strip(),
            "chunk_char_count": int(document.get("chunk_char_count", 0) or 0),
            "created_at": str(document.get("created_at", "")).strip(),
            "raw_chunk_s3_uri": str(metadata.get("raw_chunk_s3_uri", "")).strip(),
            "entity_focus": str(metadata.get("entity_focus", "")).strip(),
            "identity_hmacs_json": json.dumps(metadata.get("identity_hmacs") or {}, sort_keys=True),
        }

    def _from_chroma_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        identity_hmacs_raw = str(metadata.get("identity_hmacs_json", "")).strip()
        try:
            identity_hmacs = json.loads(identity_hmacs_raw) if identity_hmacs_raw else {}
        except json.JSONDecodeError:
            identity_hmacs = {}
        return {
            "document_id": str(metadata.get("document_id", "")).strip(),
            "source_path": str(metadata.get("source_path", "")).strip(),
            "page_number": int(metadata.get("page_number", 0) or 0),
            "chunk_index": int(metadata.get("chunk_index", 0) or 0),
            "page_type": str(metadata.get("page_type", "")).strip(),
            "section": str(metadata.get("section", "")).strip(),
            "chunk_char_count": int(metadata.get("chunk_char_count", 0) or 0),
            "created_at": str(metadata.get("created_at", "")).strip(),
            "raw_chunk_s3_uri": str(metadata.get("raw_chunk_s3_uri", "")).strip(),
            "entity_focus": str(metadata.get("entity_focus", "")).strip().lower(),
            "identity_hmacs": identity_hmacs if isinstance(identity_hmacs, dict) else {},
        }
