"""Persistent SQLite FTS keyword index for lexical retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any


@dataclass(slots=True)
class KeywordIndexSettings:
    enabled: bool
    persist_path: str
    table_name: str = "medical_document_chunks"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "KeywordIndexSettings":
        return cls(
            enabled=bool(config.get("enabled", False)),
            persist_path=str(config.get("persist_path", "data/keyword_index.db")).strip() or "data/keyword_index.db",
            table_name=str(config.get("table_name", "medical_document_chunks")).strip() or "medical_document_chunks",
        )


class SqliteKeywordIndexManager:
    def __init__(self, settings: KeywordIndexSettings) -> None:
        self.settings = settings
        self._chunks_table = self.settings.table_name
        self._fts_table = f"{self.settings.table_name}_fts"

    def create_connection(self) -> sqlite3.Connection:
        persist_path = Path(self.settings.persist_path)
        persist_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(persist_path))
        connection.row_factory = sqlite3.Row
        self.ensure_schema(connection)
        return connection

    def ensure_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._chunks_table} (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                source_path TEXT,
                page_number INTEGER,
                chunk_index INTEGER,
                page_type TEXT,
                section TEXT,
                chunk_char_count INTEGER,
                created_at TEXT,
                raw_chunk_s3_uri TEXT,
                entity_focus TEXT,
                identity_hmacs_json TEXT
            )
            """
        )
        connection.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self._fts_table}
            USING fts5(
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                chunk_text,
                tokenize = 'unicode61'
            )
            """
        )
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._chunks_table}_document_id ON {self._chunks_table} (document_id)"
        )
        connection.commit()

    def upsert_documents(self, connection: sqlite3.Connection, documents: list[dict[str, Any]]) -> dict[str, Any]:
        indexed = 0
        with connection:
            for document in documents:
                chunk_id = str(document.get("chunk_id", "")).strip()
                chunk_text = str(document.get("chunk_text", "")).strip()
                if not chunk_id or not chunk_text:
                    continue
                metadata = dict(document.get("metadata") or {})
                connection.execute(
                    f"""
                    INSERT OR REPLACE INTO {self._chunks_table} (
                        chunk_id,
                        document_id,
                        chunk_text,
                        source_path,
                        page_number,
                        chunk_index,
                        page_type,
                        section,
                        chunk_char_count,
                        created_at,
                        raw_chunk_s3_uri,
                        entity_focus,
                        identity_hmacs_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        str(document.get("document_id", "")).strip(),
                        chunk_text,
                        str(document.get("source_path", "")).strip(),
                        int(document.get("page_number", 0) or 0),
                        int(document.get("chunk_index", 0) or 0),
                        str(document.get("page_type", "")).strip(),
                        str(document.get("section", "")).strip(),
                        int(document.get("chunk_char_count", 0) or 0),
                        str(document.get("created_at", "")).strip(),
                        str(metadata.get("raw_chunk_s3_uri", "")).strip(),
                        str(metadata.get("entity_focus", "")).strip(),
                        json.dumps(metadata.get("identity_hmacs") or {}, sort_keys=True),
                    ),
                )
                connection.execute(f"DELETE FROM {self._fts_table} WHERE chunk_id = ?", (chunk_id,))
                connection.execute(
                    f"INSERT INTO {self._fts_table} (chunk_id, document_id, chunk_text) VALUES (?, ?, ?)",
                    (chunk_id, str(document.get("document_id", "")).strip(), chunk_text),
                )
                indexed += 1
        return {"indexed": indexed, "table_name": self.settings.table_name}

    def keyword_search(
        self,
        connection: sqlite3.Connection,
        query_text: str,
        size: int,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        match_query = _to_fts_query(query_text)
        if not match_query:
            return []

        params: list[Any] = [match_query]
        where_clause = f"{self._fts_table} MATCH ?"
        if document_id:
            where_clause += f" AND c.document_id = ?"
            params.append(str(document_id).strip())
        params.append(int(size))

        rows = connection.execute(
            f"""
            SELECT
                f.chunk_id AS chunk_id,
                c.document_id AS document_id,
                c.chunk_text AS document,
                c.source_path AS source_path,
                c.page_number AS page_number,
                c.chunk_index AS chunk_index,
                c.page_type AS page_type,
                c.section AS section,
                c.chunk_char_count AS chunk_char_count,
                c.created_at AS created_at,
                c.raw_chunk_s3_uri AS raw_chunk_s3_uri,
                c.entity_focus AS entity_focus,
                c.identity_hmacs_json AS identity_hmacs_json,
                bm25({self._fts_table}) AS bm25_score
            FROM {self._fts_table} AS f
            JOIN {self._chunks_table} AS c
                ON c.chunk_id = f.chunk_id
            WHERE {where_clause}
            ORDER BY bm25_score ASC
            LIMIT ?
            """,
            params,
        ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            identity_hmacs_raw = str(row["identity_hmacs_json"] or "").strip()
            try:
                identity_hmacs = json.loads(identity_hmacs_raw) if identity_hmacs_raw else {}
            except json.JSONDecodeError:
                identity_hmacs = {}
            bm25_score = float(row["bm25_score"] if row["bm25_score"] is not None else 0.0)
            results.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "document": str(row["document"] or ""),
                    "metadata": {
                        "document_id": str(row["document_id"] or "").strip(),
                        "source_path": str(row["source_path"] or "").strip(),
                        "page_number": int(row["page_number"] or 0),
                        "chunk_index": int(row["chunk_index"] or 0),
                        "page_type": str(row["page_type"] or "").strip(),
                        "section": str(row["section"] or "").strip(),
                        "chunk_char_count": int(row["chunk_char_count"] or 0),
                        "created_at": str(row["created_at"] or "").strip(),
                        "raw_chunk_s3_uri": str(row["raw_chunk_s3_uri"] or "").strip(),
                        "entity_focus": str(row["entity_focus"] or "").strip().lower(),
                        "identity_hmacs": identity_hmacs if isinstance(identity_hmacs, dict) else {},
                    },
                    "bm25_score": bm25_score,
                    "keyword_score": _normalize_bm25_score(bm25_score),
                }
            )
        return results


def _to_fts_query(text: str) -> str:
    tokens = [token.strip().lower() for token in str(text or "").split() if token.strip()]
    cleaned_tokens = ["".join(character for character in token if character.isalnum()) for token in tokens]
    cleaned_tokens = [token for token in cleaned_tokens if token]
    if not cleaned_tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in cleaned_tokens)


def _normalize_bm25_score(score: float) -> float:
    if score < 0:
        return -score
    return 1.0 / (1.0 + score)
