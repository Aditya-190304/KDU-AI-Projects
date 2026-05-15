"""Persistent Chroma retrieval with BM25, cosine similarity, RRF, and post-retrieval redaction."""

from __future__ import annotations

import math
import re
from typing import Any

from medical_extraction.embeddings.openai_embedder import OpenAITextEmbedder
from medical_extraction.privacy.redaction import ChunkRedactor
from medical_extraction.retrieval.local_reranker import LocalCrossEncoderReranker
from medical_extraction.storage.chroma_store import ChromaChunkIndexManager, ChromaSettings
from medical_extraction.storage.keyword_index import KeywordIndexSettings, SqliteKeywordIndexManager


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


class ChromaRetriever:
    def __init__(self, config: dict[str, Any] | None = None, device: str = "cpu") -> None:
        self.config = config or {}
        self.device = device
        self.chroma_config = self.config.get("chroma", {})
        self.keyword_index_config = self.config.get("keyword_index", {})
        self.retrieval_config = self.config.get("retrieval", {})
        self.hybrid_config = self.retrieval_config.get("hybrid", {})
        self.rerank_config = self.retrieval_config.get("rerank", {})
        self.privacy_config = self.config.get("privacy", {})
        self.manager = ChromaChunkIndexManager(ChromaSettings.from_config(self.chroma_config))
        self.keyword_index_manager = SqliteKeywordIndexManager(KeywordIndexSettings.from_config(self.keyword_index_config))
        self.embedder = OpenAITextEmbedder(config=self.retrieval_config)
        self.redactor = ChunkRedactor(self.privacy_config)
        self.reranker = LocalCrossEncoderReranker(
            model_name=str(self.rerank_config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
            device=device,
            local_files_only=bool(self.rerank_config.get("local_files_only", False)),
            max_length=int(self.rerank_config.get("max_length", 512)),
            batch_size=int(self.rerank_config.get("batch_size", 8)),
        )

    def vector_search(self, query_text: str, k: int | None = None, document_id: str | None = None) -> dict[str, Any]:
        client = self.manager.create_client()
        query_embedding = self.embedder.encode_query(query_text)
        results = self.manager.semantic_search(client, query_embedding=query_embedding, size=self._resolve_candidate_k(query_text, k))
        if document_id:
            results = self._filter_candidates_by_document(results, document_id)
        return {"query": query_text, "candidate_count": len(results), "candidates": results}

    def keyword_search(self, query_text: str, k: int | None = None, document_id: str | None = None) -> dict[str, Any]:
        if not bool(self.keyword_index_config.get("enabled", False)):
            client = self.manager.create_client()
            corpus = self.manager.get_all_documents(client)
            if document_id:
                corpus = [row for row in corpus if str((row.get("metadata") or {}).get("document_id", "")).strip() == document_id]
            results = self._bm25_search(query_text, corpus, size=self._resolve_candidate_k(query_text, k))
            return {"query": query_text, "candidate_count": len(results), "candidates": results}

        connection = self.keyword_index_manager.create_connection()
        try:
            results = self.keyword_index_manager.keyword_search(
                connection,
                query_text=query_text,
                size=self._resolve_candidate_k(query_text, k),
                document_id=document_id,
            )
        finally:
            connection.close()
        return {"query": query_text, "candidate_count": len(results), "candidates": results}

    def hybrid_search(self, query_text: str, k: int | None = None, document_id: str | None = None) -> dict[str, Any]:
        semantic_response = self.vector_search(query_text, k=k, document_id=document_id)
        keyword_response = self.keyword_search(query_text, k=k, document_id=document_id)
        merged_candidates = self._merge_with_rrf(
            semantic_response.get("candidates", []),
            keyword_response.get("candidates", []),
        )
        self._apply_query_intent_boosts(merged_candidates, query_text)
        self._apply_document_cooccurrence_boost(merged_candidates, query_text)
        merged_candidates.sort(key=lambda candidate: candidate.get("hybrid_score", 0.0), reverse=True)
        return {
            "query": query_text,
            "candidate_count": len(merged_candidates),
            "candidates": merged_candidates,
            "semantic_response": semantic_response,
            "keyword_response": keyword_response,
        }

    def retrieve_for_generation(
        self,
        query_text: str,
        candidate_k: int | None = None,
        top_k: int | None = None,
        authorized: bool = False,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        hybrid_response = self.hybrid_search(
            query_text=query_text,
            k=self._resolve_candidate_k(query_text, candidate_k),
            document_id=document_id,
        )
        prepared_candidates = self._prepare_candidates(hybrid_response.get("candidates", []))
        reranked_candidates = self._rerank_candidates(
            query_text=query_text,
            prepared_candidates=prepared_candidates,
            top_k=self._resolve_rerank_k(
                query_text=query_text,
                requested_top_k=int(top_k or self.retrieval_config.get("top_k", 5)),
                available_candidates=len(prepared_candidates),
            ),
        )
        reranked_candidates = self._ensure_prescription_anchor_candidates(
            query_text=query_text,
            prepared_candidates=prepared_candidates,
            reranked_candidates=reranked_candidates,
        )
        reranked_candidates = self._select_final_candidates(
            query_text=query_text,
            candidates=reranked_candidates,
            top_k=int(top_k or self.retrieval_config.get("top_k", 5)),
        )
        context_chunks = [self._build_context_chunk(candidate, authorized=authorized) for candidate in reranked_candidates]
        return {
            "query": query_text,
            "document_id": document_id or "",
            "authorized": authorized,
            "candidate_count": len(prepared_candidates),
            "top_k": len(context_chunks),
            "candidates": reranked_candidates,
            "context_chunks": context_chunks,
            "hybrid_response": hybrid_response,
        }

    def _prepare_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for candidate in candidates:
            metadata = dict(candidate.get("metadata") or {})
            raw_text = str(candidate.get("document", "")).strip()
            if not raw_text:
                continue
            prepared.append(
                {
                    "chunk_id": candidate.get("chunk_id"),
                    "document_id": metadata.get("document_id"),
                    "page_number": metadata.get("page_number"),
                    "chunk_index": metadata.get("chunk_index"),
                    "page_type": metadata.get("page_type"),
                    "section": metadata.get("section"),
                    "chunk_char_count": metadata.get("chunk_char_count"),
                    "semantic_score": candidate.get("semantic_score"),
                    "keyword_score": candidate.get("keyword_score"),
                    "hybrid_score": candidate.get("hybrid_score"),
                    "raw_chunk_s3_uri": metadata.get("raw_chunk_s3_uri", ""),
                    "raw_text": raw_text,
                    "metadata": {
                        "identity_hmacs": dict(metadata.get("identity_hmacs") or {}),
                        "raw_chunk_s3_uri": str(metadata.get("raw_chunk_s3_uri", "")).strip(),
                        "entity_focus": str(metadata.get("entity_focus", "")).strip().lower(),
                    },
                }
            )
        return prepared

    def _filter_candidates_by_document(self, candidates: list[dict[str, Any]], document_id: str) -> list[dict[str, Any]]:
        target_document_id = str(document_id or "").strip()
        if not target_document_id:
            return candidates
        filtered: list[dict[str, Any]] = []
        for candidate in candidates:
            metadata = dict(candidate.get("metadata") or {})
            candidate_document_id = str(metadata.get("document_id", "")).strip()
            if candidate_document_id == target_document_id:
                filtered.append(candidate)
        return filtered

    def _resolve_candidate_k(self, query_text: str, requested_k: int | None) -> int:
        base_k = int(requested_k or self.retrieval_config.get("candidate_k", 20))
        preferred_sections = self._preferred_sections_for_query(query_text)
        if "prescription" in preferred_sections:
            return max(base_k, int(self.retrieval_config.get("prescription_candidate_k", 30)))
        return base_k

    def _resolve_rerank_k(self, query_text: str, requested_top_k: int, available_candidates: int) -> int:
        rerank_k = requested_top_k
        preferred_sections = self._preferred_sections_for_query(query_text)
        if "prescription" in preferred_sections:
            rerank_k = max(
                rerank_k,
                int(self.retrieval_config.get("prescription_candidate_floor", 8)),
                int(self.retrieval_config.get("prescription_rerank_k", 20)),
            )
        return min(max(rerank_k, requested_top_k), max(available_candidates, requested_top_k))

    def _bm25_search(self, query_text: str, corpus: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []
        documents = []
        for row in corpus:
            tokens = _tokenize(str(row.get("document", "")))
            if not tokens:
                continue
            documents.append({"row": row, "tokens": tokens})
        if not documents:
            return []

        avg_doc_len = sum(len(item["tokens"]) for item in documents) / max(len(documents), 1)
        doc_frequency: dict[str, int] = {}
        for item in documents:
            for token in set(item["tokens"]):
                doc_frequency[token] = doc_frequency.get(token, 0) + 1

        scored: list[dict[str, Any]] = []
        total_docs = len(documents)
        k1 = float(self.hybrid_config.get("bm25_k1", 1.5))
        b = float(self.hybrid_config.get("bm25_b", 0.75))
        for item in documents:
            term_counts: dict[str, int] = {}
            for token in item["tokens"]:
                term_counts[token] = term_counts.get(token, 0) + 1
            doc_len = len(item["tokens"])
            score = 0.0
            for token in query_tokens:
                freq = term_counts.get(token, 0)
                if freq <= 0:
                    continue
                idf = math.log(1 + ((total_docs - doc_frequency.get(token, 0) + 0.5) / (doc_frequency.get(token, 0) + 0.5)))
                denominator = freq + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1e-9)))
                score += idf * ((freq * (k1 + 1)) / denominator)
            if score <= 0.0:
                continue
            row = dict(item["row"])
            row["keyword_score"] = score
            scored.append(row)
        scored.sort(key=lambda candidate: candidate.get("keyword_score", 0.0), reverse=True)
        return scored[:size]

    def _merge_with_rrf(self, semantic_candidates: list[dict[str, Any]], keyword_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        rrf_k = float(self.hybrid_config.get("rrf_k", 60))
        semantic_weight = float(self.hybrid_config.get("semantic_weight", 1.0))
        keyword_weight = float(self.hybrid_config.get("keyword_weight", 1.0))

        for rank, candidate in enumerate(semantic_candidates, start=1):
            chunk_id = str(candidate.get("chunk_id", "")).strip()
            if not chunk_id:
                continue
            merged[chunk_id] = {
                "chunk_id": chunk_id,
                "document": candidate.get("document", ""),
                "metadata": dict(candidate.get("metadata") or {}),
                "semantic_score": float(candidate.get("semantic_score", 0.0) or 0.0),
                "keyword_score": 0.0,
                "hybrid_score": semantic_weight / (rrf_k + rank),
            }

        for rank, candidate in enumerate(keyword_candidates, start=1):
            chunk_id = str(candidate.get("chunk_id", "")).strip()
            if not chunk_id:
                continue
            merged_candidate = merged.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "document": candidate.get("document", ""),
                    "metadata": dict(candidate.get("metadata") or {}),
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "hybrid_score": 0.0,
                },
            )
            merged_candidate["keyword_score"] = float(candidate.get("keyword_score", 0.0) or 0.0)
            if not merged_candidate.get("document"):
                merged_candidate["document"] = candidate.get("document", "")
            if not merged_candidate.get("metadata"):
                merged_candidate["metadata"] = dict(candidate.get("metadata") or {})
            merged_candidate["hybrid_score"] = float(merged_candidate.get("hybrid_score", 0.0) or 0.0) + (
                keyword_weight / (rrf_k + rank)
            )
        return list(merged.values())

    def _apply_query_intent_boosts(self, candidates: list[dict[str, Any]], query_text: str) -> None:
        query_sections = self._preferred_sections_for_query(query_text)
        section_boost = float(self.hybrid_config.get("section_boost", 0.25))
        medication_boost = float(self.hybrid_config.get("medication_text_boost", 0.20))
        document_affinity_boost = float(self.hybrid_config.get("document_affinity_boost", 0.35))
        document_exact_phrase_boost = float(self.hybrid_config.get("document_exact_phrase_boost", 0.35))
        prescription_boilerplate_penalty = float(self.hybrid_config.get("prescription_boilerplate_penalty", 0.20))
        chunk_text_match_boost = float(self.hybrid_config.get("chunk_text_match_boost", 0.55))
        query_name_phrases = self._extract_query_name_phrases(query_text)
        for candidate in candidates:
            metadata = dict(candidate.get("metadata") or {})
            section = str(metadata.get("section", "")).strip().lower()
            raw_text = str(candidate.get("document", "")).strip().lower()
            entity_focus = str(metadata.get("entity_focus", "")).strip().lower()
            boost = 0.0
            if query_sections:
                if section and section in query_sections:
                    boost += section_boost
                if "prescription" in query_sections and entity_focus == "prescription_summary":
                    boost += medication_boost * 2.5
                elif "prescription" in query_sections and (entity_focus == "medication_order" or self._looks_like_medication_chunk(raw_text)):
                    boost += medication_boost
                if "prescription" in query_sections and self._looks_like_prescription_boilerplate(raw_text):
                    boost -= prescription_boilerplate_penalty
            boost += self._document_affinity_score(
                query_text=query_text,
                candidate=candidate,
                weight=document_affinity_boost,
                exact_phrase_weight=document_exact_phrase_boost,
            )
            boost += self._chunk_text_match_boost(
                raw_text=raw_text,
                query_name_phrases=query_name_phrases,
                weight=chunk_text_match_boost,
            )
            candidate["hybrid_score"] = float(candidate.get("hybrid_score", 0.0) or 0.0) + boost

    def _apply_document_cooccurrence_boost(self, candidates: list[dict[str, Any]], query_text: str) -> None:
        """Boost sibling chunks from documents where at least one chunk matched a query name phrase."""
        query_name_phrases = self._extract_query_name_phrases(query_text)
        if not query_name_phrases:
            return
        cooccurrence_boost = float(self.hybrid_config.get("document_cooccurrence_boost", 0.80))
        non_match_penalty = float(self.hybrid_config.get("document_non_match_penalty", 0.30))
        matched_doc_ids: set[str] = set()
        for candidate in candidates:
            raw_text = re.sub(r"[.'']", "", str(candidate.get("document", ""))).strip().lower()
            metadata = dict(candidate.get("metadata") or {})
            doc_id = str(metadata.get("document_id", "")).strip()
            if not doc_id:
                continue
            for phrase in query_name_phrases:
                normalized_phrase = re.sub(r"[.'']", "", phrase).lower()
                if normalized_phrase in raw_text:
                    matched_doc_ids.add(doc_id)
                    break
        if not matched_doc_ids:
            return
        for candidate in candidates:
            metadata = dict(candidate.get("metadata") or {})
            doc_id = str(metadata.get("document_id", "")).strip()
            if doc_id in matched_doc_ids:
                candidate["hybrid_score"] = float(candidate.get("hybrid_score", 0.0) or 0.0) + cooccurrence_boost
            else:
                candidate["hybrid_score"] = float(candidate.get("hybrid_score", 0.0) or 0.0) - non_match_penalty

    def _preferred_sections_for_query(self, query_text: str) -> set[str]:
        lowered = query_text.strip().lower()
        if not lowered:
            return set()
        sections: set[str] = set()
        if re.search(r"\b(prescription|medication|medications|medicine|medicines|dose|dosage|tablet|tablets|capsule|capsules|rx|drug|drugs|treatment)\b", lowered):
            sections.add("prescription")
            sections.add("medications")
        if re.search(r"\b(diagnosis|diagnoses|condition|disease|disorder|assessment|impression)\b", lowered):
            sections.add("diagnosis")
        if re.search(r"\b(lab|labs|investigation|investigations|result|results|test|tests)\b", lowered):
            sections.add("labs")
        return sections

    def _looks_like_medication_chunk(self, chunk_text: str) -> bool:
        return bool(
            re.search(
                r"\b(\d+mg|tablet|tablets|capsule|capsules|orally|once daily|twice daily|dispense|refills|therapy|cbt|amlodipine|losartan|hydrochlorothiazide|sertraline|bupropion)\b",
                chunk_text,
            )
            or re.search(r"\b\d+\.\s*[a-z][a-z0-9-]*\b", chunk_text)
        )

    def _looks_like_instruction_chunk(self, chunk_text: str) -> bool:
        return bool(
            re.search(
                r"\b(take|follow|instructions|daily|morning|evening|weekly|monitor|side effects|refills|dispense|schedule)\b",
                chunk_text,
            )
        )

    def _looks_like_prescription_boilerplate(self, chunk_text: str) -> bool:
        return bool(
            re.search(
                r"\b(patient instructions?|instructions carefully|take care and stay healthy|prescribing physician|signature|"
                r"page\s+\d+|\[your name\]|hello\b|watch out|follow the dosage|date:\s*\d{4}-\d{2}-\d{2}|patient:\s|mrn:|diagnosis:)\b",
                chunk_text,
            )
        )

    def _document_affinity_score(
        self,
        *,
        query_text: str,
        candidate: dict[str, Any],
        weight: float,
        exact_phrase_weight: float,
    ) -> float:
        metadata = dict(candidate.get("metadata") or {})
        document_id = str(metadata.get("document_id", "")).strip().lower()
        if not document_id:
            return 0.0
        query_tokens = set(_tokenize(query_text))
        if not query_tokens:
            return 0.0
        document_id_text = document_id.replace("-", " ").replace("_", " ")
        document_tokens = set(_tokenize(document_id_text))
        if not document_tokens:
            return 0.0
        overlap = query_tokens & document_tokens
        if not overlap:
            return 0.0
        score = min(len(overlap) / max(len(query_tokens), 1), 1.0) * weight
        query_terms = [token for token in _tokenize(query_text) if len(token) > 2]
        for first, second in zip(query_terms, query_terms[1:]):
            phrase = f"{first} {second}"
            if phrase in document_id_text:
                score += exact_phrase_weight
                break
        return score

    def _chunk_text_match_boost(
        self,
        raw_text: str,
        query_name_phrases: list[str],
        weight: float,
    ) -> float:
        """Boost chunks whose text contains name phrases extracted from the query."""
        if not query_name_phrases or not raw_text:
            return 0.0
        normalized_text = re.sub(r"[.'']", "", raw_text).lower()
        best_score = 0.0
        for phrase in query_name_phrases:
            normalized_phrase = re.sub(r"[.'']", "", phrase).lower()
            if normalized_phrase in normalized_text:
                token_count = len(normalized_phrase.split())
                match_score = weight * min(token_count, 3)
                best_score = max(best_score, match_score)
        return best_score

    def _extract_query_name_phrases(self, query_text: str) -> list[str]:
        """Extract potential person-name phrases from the query for chunk-text matching."""
        if not query_text or not query_text.strip():
            return []
        stop_words = {
            "doctor", "hospital", "clinic", "medical", "prescription", "medication",
            "medicine", "patient", "health", "what", "who", "where", "when", "how",
            "which", "the", "for", "and", "with", "from", "that", "this", "are",
            "was", "were", "been", "being", "have", "has", "had", "does", "did",
            "will", "would", "could", "should", "may", "might", "can", "shall",
            "not", "but", "about", "many", "much", "some", "any", "each", "every",
            "prescribed", "diagnosis", "treatment", "condition", "disease", "report",
            "test", "result", "days", "bed", "rest", "tablet", "dose", "daily",
            "to", "is", "of", "in", "on", "at", "by", "its", "his", "her", "a", "an",
        }
        phrases: list[str] = []
        title_matches = re.findall(
            r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+(?:[A-Z][A-Za-z.']+\s*)+",
            query_text,
        )
        for match in title_matches:
            clean = " ".join(match.split()).strip()
            tokens = clean.split()
            filtered = []
            for token in tokens:
                if token.lower().rstrip(".,?!'s") in stop_words:
                    break
                filtered.append(token)
            clean = re.sub(r"[''']s\b", "", " ".join(filtered).strip()).rstrip("?.,!'")
            if len(clean) > 3:
                phrases.append(clean)
                name_part = re.sub(r"^(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+", "", clean, flags=re.IGNORECASE).strip()
                if name_part and len(name_part) > 2:
                    phrases.append(name_part)
        capitalized_runs = re.findall(r"\b[A-Z][A-Za-z.']+(?:\s+[A-Z][A-Za-z.']+)+\b", query_text)
        for run in capitalized_runs:
            clean = run.strip()
            tokens = clean.split()
            filtered = []
            for token in tokens:
                if token.lower().rstrip(".,?!'s") in stop_words:
                    break
                filtered.append(token)
            clean = re.sub(r"[''']s\b", "", " ".join(filtered).strip()).rstrip("?.,!'")
            if clean and clean not in phrases and len(clean) > 3:
                phrases.append(clean)
        if not phrases:
            words = re.findall(r"\b[a-zA-Z][a-zA-Z.']+\b", query_text)
            candidate_tokens: list[str] = []
            for word in words:
                lower_word = word.lower().rstrip(".,?!'s")
                if lower_word in stop_words or len(lower_word) < 2:
                    if candidate_tokens and len(candidate_tokens) >= 2:
                        phrase = " ".join(candidate_tokens)
                        if phrase.lower() not in {p.lower() for p in phrases}:
                            phrases.append(phrase)
                    candidate_tokens = []
                else:
                    candidate_tokens.append(word)
            if len(candidate_tokens) >= 2:
                phrase = " ".join(candidate_tokens)
                if phrase.lower() not in {p.lower() for p in phrases}:
                    phrases.append(phrase)
        seen = set()
        unique: list[str] = []
        for phrase in phrases:
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                unique.append(phrase)
        return unique

    def _is_medication_order_candidate(self, candidate: dict[str, Any]) -> bool:
        raw_text = str(candidate.get("raw_text", "")).strip().lower()
        entity_focus = str((candidate.get("metadata") or {}).get("entity_focus", "")).strip().lower()
        if entity_focus in ("medication_order", "prescription_summary"):
            return True
        if self._looks_like_prescription_boilerplate(raw_text):
            return False
        return self._looks_like_medication_chunk(raw_text)

    def _rerank_candidates(self, query_text: str, prepared_candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not prepared_candidates:
            return []
        if not bool(self.rerank_config.get("enabled", True)):
            return prepared_candidates[:top_k]
        reranked = self.reranker.rerank(query_text, prepared_candidates, text_field="raw_text", top_k=top_k)
        hybrid_blend_weight = 0.35
        for candidate in reranked:
            hybrid_score = float(candidate.get("hybrid_score", 0.0) or 0.0)
            rerank_score = float(candidate.get("rerank_score", 0.0) or 0.0)
            candidate["combined_score"] = rerank_score + (hybrid_score * hybrid_blend_weight)
        reranked.sort(key=lambda c: c.get("combined_score", 0.0), reverse=True)
        return reranked

    def _select_final_candidates(self, query_text: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not candidates:
            return []

        query_name_phrases = self._extract_query_name_phrases(query_text)
        if query_name_phrases:
            matched_doc_ids: set[str] = set()
            for candidate in candidates:
                raw_text = re.sub(r"[.'']", "", str(candidate.get("raw_text", ""))).strip().lower()
                doc_id = str(candidate.get("document_id", "")).strip()
                if not doc_id:
                    continue
                for phrase in query_name_phrases:
                    normalized_phrase = re.sub(r"[.'']", "", phrase).lower()
                    if normalized_phrase in raw_text:
                        matched_doc_ids.add(doc_id)
                        break
            if matched_doc_ids:
                matched_chunks = [c for c in candidates if str(c.get("document_id", "")).strip() in matched_doc_ids]
                other_chunks = [c for c in candidates if str(c.get("document_id", "")).strip() not in matched_doc_ids]
                selected: list[dict[str, Any]] = []
                selected_ids: set[str] = set()
                for candidate in matched_chunks + other_chunks:
                    chunk_id = str(candidate.get("chunk_id", "")).strip()
                    if not chunk_id or chunk_id in selected_ids:
                        continue
                    selected.append(candidate)
                    selected_ids.add(chunk_id)
                    if len(selected) >= top_k:
                        return selected
                return selected[:top_k]

        preferred_sections = self._preferred_sections_for_query(query_text)
        if "prescription" not in preferred_sections:
            return candidates[:top_k]

        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        medication_like = [candidate for candidate in candidates if self._is_medication_order_candidate(candidate)]
        instruction_like = [
            candidate
            for candidate in candidates
            if self._looks_like_instruction_chunk(str(candidate.get("raw_text", "")).lower())
            and not self._looks_like_prescription_boilerplate(str(candidate.get("raw_text", "")).lower())
        ]

        for pool in (medication_like, instruction_like, candidates):
            for candidate in pool:
                chunk_id = str(candidate.get("chunk_id", "")).strip()
                if not chunk_id or chunk_id in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(chunk_id)
                if len(selected) >= top_k:
                    return selected
        return selected[:top_k]

    def _ensure_prescription_anchor_candidates(
        self,
        *,
        query_text: str,
        prepared_candidates: list[dict[str, Any]],
        reranked_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        preferred_sections = self._preferred_sections_for_query(query_text)
        if "prescription" not in preferred_sections:
            return reranked_candidates

        preserved: list[dict[str, Any]] = list(reranked_candidates)
        present_ids = {str(candidate.get("chunk_id", "")).strip() for candidate in preserved}
        medication_like = [
            candidate
            for candidate in prepared_candidates
            if self._is_medication_order_candidate(candidate)
        ]
        medication_like.sort(
            key=lambda candidate: (
                float(candidate.get("hybrid_score", 0.0) or 0.0),
                float(candidate.get("keyword_score", 0.0) or 0.0),
                float(candidate.get("semantic_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        anchor_limit = int(self.retrieval_config.get("prescription_anchor_k", 5))
        for candidate in medication_like[:anchor_limit]:
            chunk_id = str(candidate.get("chunk_id", "")).strip()
            if not chunk_id or chunk_id in present_ids:
                continue
            preserved.append({**candidate, "rerank_score": float(candidate.get("hybrid_score", 0.0) or 0.0)})
            present_ids.add(chunk_id)
        return preserved

    def _build_context_chunk(self, candidate: dict[str, Any], authorized: bool) -> dict[str, Any]:
        metadata = dict(candidate.get("metadata") or {})
        content = str(candidate.get("raw_text", "")).strip()
        chunk_id = candidate.get("chunk_id")
        document_id = candidate.get("document_id")
        if not authorized:
            redacted_chunk = self.redactor.redact_chunk(
                {
                    "chunk_id": candidate.get("chunk_id"),
                    "document_id": candidate.get("document_id"),
                    "chunk_text": content,
                    "metadata": metadata,
                }
            )
            content = str(redacted_chunk.get("chunk_text", "")).strip()
            redacted_metadata = dict(redacted_chunk.get("metadata") or {})
            metadata["identity_hmacs"] = dict(redacted_metadata.get("identity_hmacs") or {})
            metadata["redactions"] = list(redacted_metadata.get("redactions") or [])
            masked_doc_id = self._mask_document_id(str(document_id or ""))
            document_id = masked_doc_id
            raw_chunk_id = str(chunk_id or "")
            if ":" in raw_chunk_id:
                chunk_suffix = raw_chunk_id.rsplit(":", 1)[-1]
                chunk_id = f"{masked_doc_id}:{chunk_suffix}"
            else:
                chunk_id = raw_chunk_id
        return {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "content": content,
            "authorized": authorized,
            "page_number": candidate.get("page_number"),
            "section": candidate.get("section"),
            "metadata": metadata,
            "raw_chunk_s3_uri": candidate.get("raw_chunk_s3_uri") if authorized else None,
            "rerank_score": candidate.get("rerank_score"),
            "hybrid_score": candidate.get("hybrid_score"),
            "keyword_score": candidate.get("keyword_score"),
            "semantic_score": candidate.get("semantic_score"),
        }

    def _mask_document_id(self, document_id: str) -> str:
        """Replace document_id with a short HMAC hash to hide patient/doctor names."""
        raw = str(document_id or "").strip()
        if not raw:
            return raw
        digest = self.redactor._hmac(raw.lower())
        return f"doc_{digest[:12]}"


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(str(text or ""))]
