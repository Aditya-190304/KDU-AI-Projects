"""CLI for vector, keyword, hybrid, and answer retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medical_extraction.core.config import load_runtime_config
from medical_extraction.retrieval import ChromaRetriever, role_is_authorized
from medical_extraction.utils.env import load_env_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run vector retrieval and prepare context chunks for answer generation.")
    parser.add_argument("--query", required=True, help="User query text.")
    parser.add_argument("--mode", choices=["vector", "keyword", "hybrid", "answer"], default="answer")
    parser.add_argument("--candidate-k", type=int, default=None, help="Override the initial vector retrieval depth.")
    parser.add_argument("--top-k", type=int, default=None, help="Override the final reranked context depth.")
    parser.add_argument("--document-id", default=None, help="Optional document id filter.")
    parser.add_argument("--role", choices=["doctor", "receptionist", "nurse"], default=None, help="Optional role to derive authorization.")
    parser.add_argument(
        "--authorized",
        action="store_true",
        help="Return raw hydrated chunks in context_chunks instead of redacted text.",
    )
    parser.add_argument("--config", default=None, help="Optional YAML config override.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Execution device hint.")
    return parser


def main() -> None:
    load_env_file(Path(".env"))
    args = build_parser().parse_args()
    config = load_runtime_config(args.config)
    retriever = ChromaRetriever(config=config, device=args.device)
    if args.mode == "vector":
        response = retriever.vector_search(args.query, k=args.candidate_k, document_id=args.document_id)
    elif args.mode == "keyword":
        response = retriever.keyword_search(args.query, k=args.candidate_k, document_id=args.document_id)
    elif args.mode == "hybrid":
        response = retriever.hybrid_search(args.query, k=args.candidate_k, document_id=args.document_id)
    else:
        authorized = role_is_authorized(args.role) if args.role else args.authorized
        response = retriever.retrieve_for_generation(
            query_text=args.query,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
            authorized=authorized,
            document_id=args.document_id,
        )
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
