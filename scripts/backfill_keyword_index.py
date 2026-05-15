"""Backfill the persistent SQLite keyword index from existing Chroma documents."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medical_extraction.core.config import load_runtime_config
from medical_extraction.storage.chroma_store import ChromaChunkIndexManager, ChromaSettings
from medical_extraction.storage.keyword_index import KeywordIndexSettings, SqliteKeywordIndexManager
from medical_extraction.utils.env import load_env_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill the persistent keyword index from Chroma.")
    parser.add_argument("--config", default=None, help="Optional YAML config override.")
    return parser


def main() -> None:
    load_env_file(PROJECT_ROOT / ".env")
    args = build_parser().parse_args()
    config = load_runtime_config(args.config)

    chroma_manager = ChromaChunkIndexManager(ChromaSettings.from_config(config.get("chroma", {})))
    keyword_manager = SqliteKeywordIndexManager(KeywordIndexSettings.from_config(config.get("keyword_index", {})))

    chroma_client = chroma_manager.create_client()
    keyword_connection = keyword_manager.create_connection()
    try:
        rows = chroma_manager.get_all_documents(chroma_client)
        documents = []
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            documents.append(
                {
                    "chunk_id": row.get("chunk_id"),
                    "document_id": metadata.get("document_id"),
                    "source_path": metadata.get("source_path"),
                    "page_number": metadata.get("page_number"),
                    "chunk_index": metadata.get("chunk_index"),
                    "page_type": metadata.get("page_type"),
                    "section": metadata.get("section"),
                    "chunk_char_count": metadata.get("chunk_char_count"),
                    "created_at": metadata.get("created_at"),
                    "chunk_text": row.get("document", ""),
                    "metadata": {
                        "raw_chunk_s3_uri": metadata.get("raw_chunk_s3_uri", ""),
                        "entity_focus": metadata.get("entity_focus", ""),
                        "identity_hmacs": dict(metadata.get("identity_hmacs") or {}),
                    },
                }
            )
        result = keyword_manager.upsert_documents(keyword_connection, documents)
    finally:
        keyword_connection.close()

    print(
        {
            "ok": True,
            "backfilled": int(result.get("indexed", 0)),
            "table_name": keyword_manager.settings.table_name,
            "persist_path": keyword_manager.settings.persist_path,
        }
    )


if __name__ == "__main__":
    main()
