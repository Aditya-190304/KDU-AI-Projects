"""Serve the frontend and expose a minimal question-answer endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
from threading import Lock, Thread
from time import time
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from medical_extraction.answering import OpenAIAnswerer
from medical_extraction.benchmark import compute_ocr_accuracy, evaluate_answer, get_benchmark_summary, record_processing_time
from medical_extraction.core.config import load_runtime_config
from medical_extraction.core.pipeline import ExtractionPipeline
from medical_extraction.privacy.redaction import ChunkRedactor
from medical_extraction.retrieval import ChromaRetriever, role_is_admin, role_is_authorized
from medical_extraction.storage.audit_store import DynamoAuditSettings, DynamoAuditStore, NoOpAuditStore
from medical_extraction.utils.env import load_env_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_SOURCE_ROOT = PROJECT_ROOT / "frontend"
FRONTEND_DIST_ROOT = FRONTEND_SOURCE_ROOT / "dist"
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AppState:
    retriever: ChromaRetriever
    answerer: OpenAIAnswerer
    pipeline: ExtractionPipeline
    audit_store: Any
    audit_status: dict[str, Any]
    upload_jobs: dict[str, dict[str, Any]]
    upload_jobs_lock: Lock


class QaRequestHandler(SimpleHTTPRequestHandler):
    server_version = "MedicalRagFrontend/0.1"

    def __init__(self, *args, directory: str | None = None, app_state: AppState | None = None, **kwargs) -> None:
        self.app_state = app_state
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            payload = {"ok": True}
            if self.app_state is not None:
                payload["audit"] = dict(self.app_state.audit_status)
            self._write_json(HTTPStatus.OK, payload)
            return
        if parsed.path == "/api/upload/status":
            self._handle_upload_status(parsed)
            return
        if parsed.path == "/api/benchmark":
            self._handle_benchmark()
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/ask":
            self._handle_ask()
            return
        if parsed.path == "/api/search":
            self._handle_search()
            return
        if parsed.path == "/api/upload":
            self._handle_upload()
            return
        if parsed.path == "/api/admin/audit/logs":
            self._handle_audit_logs()
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def _handle_ask(self) -> None:
        if self.app_state is None:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Server state unavailable"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
            question = str(payload.get("question", "")).strip()
            role = str(payload.get("role", "")).strip().lower()
            actor_name = str(payload.get("actor_name", "")).strip() or "unknown"
            document_id = str(payload.get("document_id", "")).strip() or None
            if not question:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "question is required"})
                return
            authorized = role_is_authorized(role)
            retrieval = self.app_state.retriever.retrieve_for_generation(
                query_text=question,
                candidate_k=int(payload.get("candidate_k", 10)),
                top_k=int(payload.get("top_k", 3)),
                authorized=authorized,
                document_id=document_id,
            )
            answer = self.app_state.answerer.answer_question(
                question=question,
                context_chunks=list(retrieval.get("context_chunks") or []),
                authorized=authorized,
                role=role or "unknown",
            )
            answer_text = answer.get("answer", "")
            if not authorized:
                from medical_extraction.answering.prompting import normalize_generic_redactions
                answer_text = self.app_state.retriever.redactor.redact_text(answer_text)
                answer_text = normalize_generic_redactions(answer_text)
            self._log_audit_access(
                actor_name=actor_name,
                actor_role=role,
                query_text=question,
                authorized=authorized,
                requested_document_id=document_id,
                context_chunks=list(retrieval.get("context_chunks") or []),
                status="success",
            )
            # ── Benchmark: evaluate answer against answer key (per-question) ──
            answer_eval = {}
            try:
                answer_eval = evaluate_answer(question, answer_text, document_id)
            except Exception:
                pass
            self._write_json(
                HTTPStatus.OK,
                {
                    "role": role,
                    "authorized": authorized,
                    "answer": answer_text,
                    "model": answer.get("model", ""),
                    "context_chunks": retrieval.get("context_chunks", []),
                    "citations": [chunk.get("chunk_id") for chunk in retrieval.get("context_chunks", [])],
                    "document_id": document_id or "" if authorized else "",
                    "retrieval_accuracy": answer_eval.get("retrieval_accuracy"),
                    "answer_evaluation": answer_eval if answer_eval.get("matched") else None,
                },
            )
        except Exception as exc:  # pragma: no cover - thin server wrapper
            try:
                self._log_audit_access(
                    actor_name=str(locals().get("actor_name", "") or "unknown"),
                    actor_role=str(locals().get("role", "") or "unknown"),
                    query_text=str(locals().get("question", "") or ""),
                    authorized=bool(locals().get("authorized", False)),
                    requested_document_id=locals().get("document_id"),
                    context_chunks=[],
                    status="error",
                    error_message=str(exc),
                )
            except Exception:
                pass
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_search(self) -> None:
        if self.app_state is None:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Server state unavailable"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
            query = str(payload.get("query", "")).strip()
            role = str(payload.get("role", "")).strip().lower()
            document_id = str(payload.get("document_id", "")).strip() or None
            if not query:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "query is required"})
                return
            authorized = role_is_authorized(role)
            retrieval = self.app_state.retriever.retrieve_for_generation(
                query_text=query,
                candidate_k=int(payload.get("candidate_k", 10)),
                top_k=int(payload.get("top_k", 3)),
                authorized=authorized,
                document_id=document_id,
            )
            self._write_json(
                HTTPStatus.OK,
                {
                    "role": role,
                    "authorized": authorized,
                    "query": query,
                    "document_id": document_id or "" if authorized else "",
                    "context_chunks": retrieval.get("context_chunks", []),
                    "candidates": retrieval.get("candidates", []) if authorized else [],
                },
            )
        except Exception as exc:  # pragma: no cover
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_benchmark(self) -> None:
        if self.app_state is None:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Server state unavailable"})
            return
        try:
            summary = get_benchmark_summary()
            self._write_json(HTTPStatus.OK, summary)
        except Exception as exc:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_audit_logs(self) -> None:
        if self.app_state is None:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Server state unavailable"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
            role = str(payload.get("role", "")).strip().lower()
            if not role_is_admin(role):
                self._write_json(HTTPStatus.FORBIDDEN, {"error": "admin role is required"})
                return
            logs = self.app_state.audit_store.list_logs(
                date_from=str(payload.get("date_from", "")).strip() or None,
                date_to=str(payload.get("date_to", "")).strip() or None,
                page_size=int(payload.get("page_size", 10)),
                cursor=str(payload.get("cursor", "")).strip() or None,
            )
            self._write_json(
                HTTPStatus.OK,
                {
                    "items": logs.get("items", []),
                    "next_cursor": logs.get("next_cursor"),
                },
            )
        except Exception as exc:  # pragma: no cover
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_upload(self) -> None:
        if self.app_state is None:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Server state unavailable"})
            return
        try:
            filename = str(self.headers.get("X-Filename", "")).strip()
            if not filename:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "X-Filename header is required"})
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Uploaded file body is empty"})
                return
            safe_name = self._sanitize_filename(filename)
            uploads_dir = PROJECT_ROOT / "data" / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            input_path = uploads_dir / safe_name
            file_bytes = self.rfile.read(content_length)
            input_path.write_bytes(file_bytes)

            output_dir = PROJECT_ROOT / "output" / "uploads"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{input_path.stem}_result.json"
            job_id = uuid4().hex
            job_start_time = time()
            self._set_upload_job(
                job_id,
                {
                    "job_id": job_id,
                    "filename": safe_name,
                    "status": "queued",
                    "progress_percent": 0,
                    "stage": "queued",
                    "detail": "Upload received. Waiting to start processing.",
                    "document_id": "",
                    "summary": {},
                    "warnings": [],
                    "output_file": str(output_path),
                    "rag_text_file": "",
                    "chunk_file": "",
                    "error": "",
                    "updated_at": time(),
                    "started_at": job_start_time,
                    "elapsed_seconds": None,
                },
            )
            worker = Thread(
                target=self._run_upload_job,
                args=(job_id, str(input_path), str(output_path)),
                daemon=True,
            )
            worker.start()
            self._write_json(
                HTTPStatus.ACCEPTED,
                {
                    "ok": True,
                    "job_id": job_id,
                    "filename": safe_name,
                    "output_file": str(output_path),
                    "status": "queued",
                    "cache_hit": False,
                },
            )
        except Exception as exc:  # pragma: no cover
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_upload_status(self, parsed) -> None:
        if self.app_state is None:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Server state unavailable"})
            return
        query = parse_qs(parsed.query or "")
        job_id = str((query.get("job_id") or [""])[0]).strip()
        if not job_id:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "job_id is required"})
            return
        job = self._get_upload_job(job_id)
        if job is None:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "upload job not found"})
            return
        self._write_json(HTTPStatus.OK, dict(job))

    def _sanitize_filename(self, filename: str) -> str:
        candidate = Path(filename).name
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("._")
        return sanitized or "upload.bin"

    def _log_audit_access(
        self,
        *,
        actor_name: str,
        actor_role: str,
        query_text: str,
        authorized: bool,
        requested_document_id: str | None,
        context_chunks: list[dict[str, Any]],
        status: str,
        error_message: str | None = None,
    ) -> None:
        if self.app_state is None:
            return
        self.app_state.audit_store.log_access(
            actor_name=actor_name,
            actor_role=actor_role,
            query_text=query_text,
            authorized=authorized,
            requested_document_id=requested_document_id,
            context_chunks=context_chunks,
            status=status,
            error_message=error_message,
        )

    def _set_upload_job(self, job_id: str, payload: dict[str, Any]) -> None:
        if self.app_state is None:
            return
        with self.app_state.upload_jobs_lock:
            self.app_state.upload_jobs[job_id] = dict(payload)

    def _update_upload_job(self, job_id: str, **updates: Any) -> None:
        if self.app_state is None:
            return
        with self.app_state.upload_jobs_lock:
            current = dict(self.app_state.upload_jobs.get(job_id) or {})
            current.update(updates)
            current["updated_at"] = time()
            self.app_state.upload_jobs[job_id] = current

    def _get_upload_job(self, job_id: str) -> dict[str, Any] | None:
        if self.app_state is None:
            return None
        with self.app_state.upload_jobs_lock:
            job = self.app_state.upload_jobs.get(job_id)
            return dict(job) if isinstance(job, dict) else None

    def _run_upload_job(self, job_id: str, input_path: str, output_path: str) -> None:
        if self.app_state is None:
            return
        self._update_upload_job(
            job_id,
            status="running",
            progress_percent=2,
            stage="starting",
            detail="Starting extraction pipeline.",
        )
        try:
            payload = self.app_state.pipeline.run(
                input_path=input_path,
                output_path=output_path,
                text_only=False,
                progress_callback=lambda update: self._update_upload_job(
                    job_id,
                    status="running",
                    progress_percent=max(1, min(int(update.get("percent", 0) or 0), 100)),
                    stage=str(update.get("stage", "running")),
                    detail=str(update.get("detail", "Processing document.")),
                ),
            )
            debug_artifacts = payload.get("debug_artifacts") or {}
            job_data = self._get_upload_job(job_id) or {}
            job_started = float(job_data.get("started_at", 0) or 0)
            elapsed = round(time() - job_started, 1) if job_started > 0 else None
            self._update_upload_job(
                job_id,
                status="completed",
                progress_percent=100,
                stage="complete",
                detail="Document is ready for querying.",
                document_id=str(payload.get("document_id", "")).strip(),
                summary=payload.get("summary", {}),
                warnings=payload.get("warnings", []),
                output_file=str(output_path),
                rag_text_file=str(debug_artifacts.get("rag_text_file", "")),
                chunk_file=str(debug_artifacts.get("chunk_file", "")),
                cache_hit=bool(debug_artifacts.get("cache_hit", False)),
                file_hash=str(payload.get("file_hash", "")).strip(),
                error="",
                elapsed_seconds=elapsed,
            )
            # ── Benchmark: compute OCR accuracy and record processing time ──
            doc_id = str(payload.get("document_id", "")).strip()
            ocr_accuracy_pct = None
            ocr_detail = None
            rag_text_path = str(debug_artifacts.get("rag_text_file", "")).strip()
            if doc_id and rag_text_path:
                try:
                    rag_text = Path(rag_text_path).read_text(encoding="utf-8") if Path(rag_text_path).exists() else ""
                    if rag_text:
                        ocr_accuracy_pct = compute_ocr_accuracy(doc_id, rag_text)
                        from medical_extraction.benchmark.engine import _load_state
                        state = _load_state()
                        ocr_detail = state.get("ocr_scores", {}).get(doc_id)
                except Exception:
                    pass
            if elapsed is not None and doc_id:
                try:
                    record_processing_time(doc_id, elapsed)
                except Exception:
                    pass
            # Add OCR accuracy to job so frontend can display it
            self._update_upload_job(
                job_id,
                ocr_accuracy=ocr_accuracy_pct,
                ocr_detail=ocr_detail,
            )
        except Exception as exc:
            job_data = self._get_upload_job(job_id) or {}
            job_started = float(job_data.get("started_at", 0) or 0)
            elapsed = round(time() - job_started, 1) if job_started > 0 else None
            self._update_upload_job(
                job_id,
                status="error",
                progress_percent=100,
                stage="error",
                detail="Processing failed.",
                error=str(exc),
                elapsed_seconds=elapsed,
            )

    def log_message(self, format: str, *args) -> None:  # pragma: no cover
        return

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(config_path: str | None, host: str, port: int, device: str) -> None:
    load_env_file(PROJECT_ROOT / ".env")
    config = load_runtime_config(config_path)
    frontend_root = FRONTEND_DIST_ROOT if FRONTEND_DIST_ROOT.exists() else FRONTEND_SOURCE_ROOT
    privacy_config = config.get("privacy", {})
    audit_config = config.get("audit", {})
    audit_settings = DynamoAuditSettings.from_config(audit_config)
    audit_store = NoOpAuditStore()
    audit_status: dict[str, Any] = {
        "enabled": bool(audit_settings.enabled),
        "required": bool(audit_settings.required),
        "backend": str(audit_settings.backend),
        "table_name": str(audit_settings.table_name),
        "endpoint_url": str(audit_settings.endpoint_url),
        "ready": False,
        "error": "",
    }
    if audit_settings.enabled:
        try:
            audit_store = DynamoAuditStore(
                settings=audit_settings,
                redactor=ChunkRedactor(privacy_config),
            )
            audit_store.ensure_table()
            audit_status["ready"] = True
        except Exception as exc:
            audit_status["error"] = str(exc)
            LOGGER.warning("Audit backend initialization failed: %s", exc)
            if audit_settings.required:
                raise RuntimeError(f"Audit backend initialization failed: {exc}") from exc
            audit_store = NoOpAuditStore()
    app_state = AppState(
        retriever=ChromaRetriever(config=config, device=device),
        answerer=OpenAIAnswerer(),
        pipeline=ExtractionPipeline(config=config),
        audit_store=audit_store,
        audit_status=audit_status,
        upload_jobs={},
        upload_jobs_lock=Lock(),
    )

    def _handler(*args, **kwargs):
        return QaRequestHandler(*args, directory=str(frontend_root), app_state=app_state, **kwargs)

    server = ThreadingHTTPServer((host, port), _handler)
    try:
        server.serve_forever()
    finally:  # pragma: no cover
        server.server_close()
