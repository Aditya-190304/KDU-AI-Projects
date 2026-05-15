import { useEffect, useMemo, useState } from "react";

const AUTHORIZED_ROLES = new Set(["doctor"]);
const ADMIN_ROLES = new Set(["admin"]);
const MAX_BATCH_UPLOADS = 100;
const STORAGE_KEYS = {
  actorName: "medical_rag_actor_name",
  role: "medical_rag_role",
  documentId: "medical_rag_document_id",
  filename: "medical_rag_filename",
  uploadSummary: "medical_rag_upload_summary",
};

function loadStoredJson(key) {
  const raw = localStorage.getItem(key);
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw);
  } catch (_error) {
    return null;
  }
}

function labelForRole(role) {
  if (role === "doctor") return "Doctor";
  if (role === "receptionist") return "Receptionist";
  if (role === "admin") return "Administrator";
  return "Unknown";
}

function formatAuditTimestamp(value) {
  if (!value) return "Unknown time";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
    timeZoneName: "short",
  });
}

function labelForAccessScope(authorized) {
  return authorized ? "Authorized raw-access answer flow" : "Masked answer flow";
}

async function parseApiResponse(response, fallbackMessage) {
  const rawText = await response.text();
  if (!rawText.trim()) {
    throw new Error(`${fallbackMessage} (empty response from server)`);
  }

  let payload;
  try {
    payload = JSON.parse(rawText);
  } catch (_error) {
    throw new Error(`${fallbackMessage} (server returned non-JSON content)`);
  }

  if (!response.ok) {
    throw new Error(payload.error || fallbackMessage);
  }
  return payload;
}

export default function App() {
  const [actorName, setActorName] = useState(localStorage.getItem(STORAGE_KEYS.actorName) || "");
  const [currentRole, setCurrentRole] = useState(localStorage.getItem(STORAGE_KEYS.role) || "");
  const [currentDocumentId, setCurrentDocumentId] = useState(localStorage.getItem(STORAGE_KEYS.documentId) || "");
  const [currentFilename, setCurrentFilename] = useState(localStorage.getItem(STORAGE_KEYS.filename) || "");
  const [currentSummary, setCurrentSummary] = useState(loadStoredJson(STORAGE_KEYS.uploadSummary));
  const [authStatus, setAuthStatus] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [answerCitations, setAnswerCitations] = useState([]);
  const [answerEvaluation, setAnswerEvaluation] = useState(null);
  const [askStatus, setAskStatus] = useState("");
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadStatus, setUploadStatus] = useState("");
  const [progressPercent, setProgressPercent] = useState(0);
  const [progressLabel, setProgressLabel] = useState("Waiting for files...");
  const [showProgress, setShowProgress] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [batchStats, setBatchStats] = useState({
    total: 0,
    completed: 0,
    failed: 0,
    activeName: "",
  });
  const [lastProcessingTime, setLastProcessingTime] = useState(null);
  const [auditLogs, setAuditLogs] = useState([]);
  const [auditStatus, setAuditStatus] = useState("");
  const [auditDateFrom, setAuditDateFrom] = useState("");
  const [auditDateTo, setAuditDateTo] = useState("");
  const [auditPageSize, setAuditPageSize] = useState(10);
  const [auditNextCursor, setAuditNextCursor] = useState(null);
  const [auditCursorStack, setAuditCursorStack] = useState([null]);
  const [auditPageIndex, setAuditPageIndex] = useState(0);
  const [isAuditLoading, setIsAuditLoading] = useState(false);
  const [serverHealth, setServerHealth] = useState({ ok: false, audit: null });
  const [serverHealthStatus, setServerHealthStatus] = useState("");
  const [lastOcrAccuracy, setLastOcrAccuracy] = useState(null);

  const authorized = AUTHORIZED_ROLES.has(currentRole);
  const isAdmin = ADMIN_ROLES.has(currentRole);
  const hasRole = Boolean(currentRole);
  const hasDocument = Boolean(currentDocumentId);
  const selectedCount = selectedFiles.length;
  const progressRingStyle = useMemo(
    () => ({
      background: `conic-gradient(var(--doctor) 0deg ${Math.max(0, Math.min(progressPercent, 100)) * 3.6}deg, rgba(216, 204, 187, 0.6) ${Math.max(0, Math.min(progressPercent, 100)) * 3.6}deg 360deg)`,
    }),
    [progressPercent],
  );

  useEffect(() => {
    void loadServerHealth();
  }, []);

  async function loadServerHealth() {
    try {
      const response = await fetch("/api/health");
      const payload = await parseApiResponse(response, "Unable to load server health");
      setServerHealth(payload);
      const audit = payload.audit || {};
      if (audit.enabled && audit.ready) {
        setServerHealthStatus(`Audit store ready: ${audit.table_name}`);
      } else if (audit.enabled) {
        setServerHealthStatus(audit.error || "Audit store is not ready.");
      } else {
        setServerHealthStatus("Audit logging is disabled.");
      }
    } catch (error) {
      setServerHealth({ ok: false, audit: null });
      setServerHealthStatus(error.message || "Unable to load server health.");
    }
  }

  function chooseRole(role) {
    if (!actorName.trim()) {
      setAuthStatus("Enter your name or staff id before choosing a role.");
      return;
    }
    localStorage.setItem(STORAGE_KEYS.actorName, actorName.trim());
    setCurrentRole(role);
    localStorage.setItem(STORAGE_KEYS.role, role);
    setAuthStatus("");
    setAnswer("");
    setAnswerCitations([]);
    setAskStatus("");
    if (role === "admin") {
      setAuditLogs([]);
      setAuditStatus("Load audit logs to review recent access history.");
      setAuditNextCursor(null);
      setAuditCursorStack([null]);
      setAuditPageIndex(0);
    }
  }

  function switchRole() {
    setCurrentRole("");
    localStorage.removeItem(STORAGE_KEYS.role);
    setAnswer("");
    setAnswerCitations([]);
    setAskStatus("");
  }

  function clearDocumentScope() {
    setCurrentDocumentId("");
    setCurrentFilename("");
    setCurrentSummary(null);
    localStorage.removeItem(STORAGE_KEYS.documentId);
    localStorage.removeItem(STORAGE_KEYS.filename);
    localStorage.removeItem(STORAGE_KEYS.uploadSummary);
    setUploadStatus("Search scope cleared. Questions will now search the full persistent index.");
    setAnswer("");
    setAnswerCitations([]);
    setAskStatus("");
  }

  async function askQuestion() {
    if (!currentRole) {
      return;
    }
    if (!question.trim()) {
      setAskStatus("Enter a question first.");
      return;
    }

    setAskStatus("Asking model...");
    setAnswer("");
    setAnswerCitations([]);
    setAnswerEvaluation(null);

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          actor_name: actorName.trim(),
          role: currentRole,
          question: question.trim(),
          document_id: currentDocumentId || undefined,
          candidate_k: 10,
          top_k: 7,
        }),
      });
      const payload = await parseApiResponse(response, "Request failed");
      setAskStatus(`Answered with ${payload.model}.`);
      setAnswer(payload.answer || "");
      setAnswerCitations(payload.citations || []);
      setAnswerEvaluation(payload.answer_evaluation || null);
    } catch (error) {
      setAskStatus(error.message || "Unable to get answer.");
      setAnswer("");
      setAnswerCitations([]);
      setAnswerEvaluation(null);
    }
  }

  async function loadAuditLogs(cursor = null, nextPageIndex = 0, nextStack = [null]) {
    if (!isAdmin) {
      return;
    }
    setIsAuditLoading(true);
    setAuditStatus("Loading masked audit logs...");
    try {
      const response = await fetch("/api/admin/audit/logs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          role: currentRole,
          actor_name: actorName.trim(),
          date_from: auditDateFrom || undefined,
          date_to: auditDateTo || undefined,
          page_size: auditPageSize,
          cursor: cursor || undefined,
        }),
      });
      const payload = await parseApiResponse(response, "Audit log request failed");
      const items = payload.items || [];
      setAuditLogs(items);
      setAuditNextCursor(payload.next_cursor || null);
      setAuditCursorStack(nextStack);
      setAuditPageIndex(nextPageIndex);
      setAuditStatus(items.length ? `Showing ${items.length} masked audit log entries.` : "No audit logs found for the selected date range.");
    } catch (error) {
      setAuditLogs([]);
      setAuditStatus(error.message || "Unable to load audit logs.");
    } finally {
      setIsAuditLoading(false);
    }
  }

  function applyAuditFilters() {
    const resetStack = [null];
    void loadAuditLogs(null, 0, resetStack);
  }

  function nextAuditPage() {
    if (!auditNextCursor) {
      return;
    }
    const nextIndex = auditPageIndex + 1;
    const nextStack = [...auditCursorStack.slice(0, nextIndex), auditNextCursor];
    void loadAuditLogs(auditNextCursor, nextIndex, nextStack);
  }

  function previousAuditPage() {
    if (auditPageIndex <= 0) {
      return;
    }
    const previousIndex = auditPageIndex - 1;
    const previousCursor = auditCursorStack[previousIndex] || null;
    const previousStack = auditCursorStack.slice(0, previousIndex + 1);
    void loadAuditLogs(previousCursor, previousIndex, previousStack);
  }

  async function uploadDocument() {
    if (!selectedFiles.length) {
      setUploadStatus("Choose one or more files first.");
      return;
    }

    const uploadQueue = selectedFiles.slice(0, MAX_BATCH_UPLOADS);
    const skippedCount = Math.max(0, selectedFiles.length - uploadQueue.length);

    setIsUploading(true);
    setShowProgress(true);
    setProgressPercent(0);
    setProgressLabel("Starting upload...");
    setUploadStatus(skippedCount > 0 ? `Only the first ${MAX_BATCH_UPLOADS} files will be ingested in this batch.` : "");
    setAnswer("");
    setAnswerCitations([]);
    setAskStatus("");
    setLastProcessingTime(null);
    setLastOcrAccuracy(null);
    setAnswerEvaluation(null);
    setBatchStats({
      total: uploadQueue.length,
      completed: 0,
      failed: 0,
      activeName: uploadQueue[0]?.name || "",
    });

    let completed = 0;
    let failed = 0;
    const warningMessages = [];

    for (let index = 0; index < uploadQueue.length; index += 1) {
      const file = uploadQueue[index];
      setBatchStats({
        total: uploadQueue.length,
        completed,
        failed,
        activeName: file.name,
      });

      try {
        const payload = await uploadSingleFile({
          file,
          batchIndex: index,
          batchTotal: uploadQueue.length,
          onProgress: (percent) => {
            const overall = ((index + percent / 100) / uploadQueue.length) * 100;
            setProgressPercent(Math.round(overall));
            setProgressLabel(`Uploading ${file.name} (${index + 1}/${uploadQueue.length})`);
          },
          onProcessing: ({ percent, detail }) => {
            const overall = ((index + percent / 100) / uploadQueue.length) * 100;
            setProgressPercent(Math.round(overall));
            setProgressLabel(`${detail} (${index + 1}/${uploadQueue.length})`);
          },
        });

        const nextDocumentId = payload.document_id || "";
        const nextFilename = payload.filename || file.name;
        const nextSummary = payload.summary || {};
        const wasCached = Boolean(payload.cache_hit);

        setCurrentDocumentId(nextDocumentId);
        setCurrentFilename(nextFilename);
        setCurrentSummary(nextSummary);
        localStorage.setItem(STORAGE_KEYS.documentId, nextDocumentId);
        localStorage.setItem(STORAGE_KEYS.filename, nextFilename);
        localStorage.setItem(STORAGE_KEYS.uploadSummary, JSON.stringify(nextSummary));

        if (Array.isArray(payload.warnings) && payload.warnings.length) {
          warningMessages.push(`${nextFilename}: ${payload.warnings.join(" | ")}`);
        }
        if (wasCached) {
          warningMessages.push(`${nextFilename}: already processed earlier, reused cached extraction and embeddings.`);
        }
        if (payload.elapsed_seconds != null) {
          setLastProcessingTime(payload.elapsed_seconds);
        }
        if (payload.ocr_accuracy != null) {
          setLastOcrAccuracy({
            accuracy: payload.ocr_accuracy,
            detail: payload.ocr_detail || null,
            documentId: nextDocumentId,
          });
        }
        completed += 1;
      } catch (error) {
        failed += 1;
        warningMessages.push(`${file.name}: ${error.message || "Upload failed"}`);
      }

      setBatchStats({
        total: uploadQueue.length,
        completed,
        failed,
        activeName: file.name,
      });
    }

    setIsUploading(false);
    setProgressPercent(100);
    setProgressLabel("Batch ingestion complete.");
    setBatchStats({
      total: uploadQueue.length,
      completed,
      failed,
      activeName: "",
    });
    setUploadStatus(
      warningMessages.length
        ? `Processed ${completed}/${uploadQueue.length} files with ${failed} failures. ${warningMessages.join(" || ")}`
        : `Successfully ingested ${completed}/${uploadQueue.length} files.`,
    );
  }

  function uploadSingleFile({ file, batchIndex, batchTotal, onProgress, onProcessing }) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      let uploadComplete = false;
      xhr.open("POST", "/api/upload");
      xhr.setRequestHeader("X-Filename", file.name);

      xhr.upload.onprogress = (event) => {
        if (!event.lengthComputable) {
          return;
        }
        const nextPercent = Math.min(100, Math.round((event.loaded / event.total) * 100));
        uploadComplete = nextPercent >= 100;
        onProgress(nextPercent);
      };

      xhr.onloadstart = () => {
        onProgress(batchIndex === 0 && batchTotal > 0 ? 2 : 0);
      };

      xhr.onreadystatechange = () => {
        if (
          (xhr.readyState === XMLHttpRequest.HEADERS_RECEIVED || xhr.readyState === XMLHttpRequest.LOADING) &&
          uploadComplete
        ) {
          onProcessing({
            percent: 15,
            stage: "queued",
            detail: `Upload complete for ${file.name}. Waiting for backend processing.`,
          });
        }
      };

      xhr.onerror = () => {
        reject(new Error("Upload failed."));
      };

      xhr.onload = () => {
        if (xhr.status < 200 || xhr.status >= 300) {
          let message = "Upload failed.";
          try {
            const payload = JSON.parse(xhr.responseText || "{}");
            message = payload.error || message;
          } catch (_error) {
            // ignore parse failure
          }
          reject(new Error(message));
          return;
        }

        try {
          const payload = JSON.parse(xhr.responseText || "{}");
          const jobId = payload.job_id;
          if (!jobId) {
            resolve(payload);
            return;
          }
          onProcessing({
            percent: 18,
            stage: payload.status || "queued",
            detail: `Queued ${file.name} for extraction and indexing.`,
          });
          void pollUploadJob({
            jobId,
            onProcessing,
            resolve,
            reject,
          });
        } catch (error) {
          reject(new Error(error.message || "Unable to parse upload response."));
        }
      };

      xhr.send(file);
    });
  }

  async function pollUploadJob({ jobId, onProcessing, resolve, reject }) {
    const maxAttempts = 1200;
    let attempt = 0;

    while (attempt < maxAttempts) {
      attempt += 1;
      try {
        const response = await fetch(`/api/upload/status?job_id=${encodeURIComponent(jobId)}`);
        const payload = await parseApiResponse(response, "Upload status request failed");
        const backendPercent = Math.max(0, Math.min(Number(payload.progress_percent || 0), 100));
        const normalizedPercent = 15 + Math.round(backendPercent * 0.85);
        onProcessing({
          percent: Math.max(15, Math.min(normalizedPercent, 100)),
          stage: payload.stage || "processing",
          detail: payload.detail || "Processing document.",
        });

        if (payload.status === "completed") {
          resolve(payload);
          return;
        }
        if (payload.status === "error") {
          reject(new Error(payload.error || payload.detail || "Processing failed."));
          return;
        }
      } catch (error) {
        reject(error instanceof Error ? error : new Error("Unable to poll upload status."));
        return;
      }

      await new Promise((next) => window.setTimeout(next, 700));
    }

    reject(new Error("Upload processing timed out."));
  }

  const uploadStateText = hasDocument
    ? `Ready to query ${currentFilename || currentDocumentId}`
    : "No active document selected. Queries will search the full persistent Chroma index.";

  const summaryMarkup = currentSummary
    ? `Document ID: ${currentDocumentId} | Pages: ${Number(currentSummary.total_pages || 0)} | Blocks: ${Number(currentSummary.total_blocks || 0)} | Chunks: ${Number(currentSummary.total_chunks || 0)} | Indexed: ${Number(currentSummary.indexed_chunks || 0)}`
    : "";

  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">Local Medical RAG</p>
        <h1>Role-Based Clinical Search</h1>
        <p className="lede">
          Doctors receive authorized raw context. Reception staff see redacted context after
          reranking across the persistent document index. Records administrators review masked
          access logs.
        </p>
      </section>

      {!hasRole ? (
        <section className="card">
          <div className="card-head">
            <div>
              <p className="eyebrow">Access</p>
              <h2>Choose a role</h2>
            </div>
            <p className="caption">React frontend for role-based retrieval.</p>
          </div>

          <label className="field top-gap">
            <span>Your name or staff id</span>
            <input
              type="text"
              value={actorName}
              placeholder="Example: records-admin-01"
              onChange={(event) => setActorName(event.target.value)}
            />
          </label>

          {authStatus ? <p className="caption">{authStatus}</p> : null}

          <div className="role-grid">
            <button className="role-card doctor" onClick={() => chooseRole("doctor")}>
              <span className="role-kicker">Authorized</span>
              <strong>Doctor Login</strong>
              <span>Can receive raw top-k chunks after reranking.</span>
            </button>

            <button className="role-card receptionist" onClick={() => chooseRole("receptionist")}>
              <span className="role-kicker">Redacted</span>
              <strong>Receptionist Login</strong>
              <span>Sees masked patient context after reranking.</span>
            </button>

            <button className="role-card admin" onClick={() => chooseRole("admin")}>
              <span className="role-kicker">Audit</span>
              <strong>Admin Login</strong>
              <span>Reviews masked access logs with date filters and pagination.</span>
            </button>
          </div>
        </section>
      ) : (
        <section className="card">
          <div className="card-head">
            <div>
              <p className="eyebrow">Session</p>
              <h2>Search Workspace</h2>
            </div>
            <div className="session-meta">
              <span className={`badge ${currentRole}`}>{labelForRole(currentRole)} Session</span>
              <button className="ghost-button" onClick={switchRole}>
                Switch Role
              </button>
            </div>
          </div>

          <div className="panel-grid">
            <aside className="panel panel-side">
              <p className="panel-label">Progress</p>
              <div className="progress-circle-shell">
                <div className="progress-circle" style={progressRingStyle}>
                  <div className="progress-circle-inner">
                    <strong>{progressPercent}%</strong>
                    <span>{isUploading ? "Uploading" : "Ready"}</span>
                  </div>
                </div>
              </div>
              <p className="caption progress-caption">{progressLabel}</p>

              <div className="progress-stats">
                <div className="progress-stat">
                  <span className="progress-stat-label">Batch size</span>
                  <strong>{batchStats.total || selectedCount || 0}</strong>
                </div>
                <div className="progress-stat">
                  <span className="progress-stat-label">Completed</span>
                  <strong>{batchStats.completed}</strong>
                </div>
                <div className="progress-stat">
                  <span className="progress-stat-label">Failed</span>
                  <strong>{batchStats.failed}</strong>
                </div>
              </div>

              <div className="benchmark-section">
                <p className="panel-label">Performance</p>

                {(!lastOcrAccuracy && !lastProcessingTime && !answerEvaluation) ? (
                  <p className="caption">Upload a document to see OCR accuracy and processing time. Ask a question to see retrieval accuracy.</p>
                ) : null}

                {(lastOcrAccuracy || lastProcessingTime) ? (
                  <div className="benchmark-results">
                    <div className="benchmark-grid">
                      {lastOcrAccuracy ? (
                        <div className="benchmark-metric">
                          <span className={`benchmark-value ${
                            lastOcrAccuracy.accuracy >= 80 ? "accent-green" :
                            lastOcrAccuracy.accuracy >= 50 ? "accent-amber" : "accent-red"
                          }`}>
                            {lastOcrAccuracy.accuracy}%
                          </span>
                          <span className="benchmark-label">OCR Accuracy</span>
                        </div>
                      ) : null}
                      {lastOcrAccuracy?.detail ? (
                        <div className="benchmark-metric">
                          <span className="benchmark-value accent-blue">
                            {lastOcrAccuracy.detail.found}/{lastOcrAccuracy.detail.total}
                          </span>
                          <span className="benchmark-label">Phrases Found</span>
                        </div>
                      ) : null}
                      {lastProcessingTime != null ? (
                        <div className="benchmark-metric">
                          <span className="benchmark-value accent-amber">
                            {lastProcessingTime >= 60
                              ? `${Math.floor(lastProcessingTime / 60)}m ${Math.round(lastProcessingTime % 60)}s`
                              : `${lastProcessingTime}s`}
                          </span>
                          <span className="benchmark-label">Processing Time</span>
                        </div>
                      ) : null}
                      {answerEvaluation ? (
                        <div className="benchmark-metric">
                          <span className={`benchmark-value ${
                            answerEvaluation.retrieval_accuracy >= 80 ? "accent-green" :
                            answerEvaluation.retrieval_accuracy >= 50 ? "accent-amber" : "accent-red"
                          }`}>
                            {answerEvaluation.retrieval_accuracy}%
                          </span>
                          <span className="benchmark-label">Retrieval Accuracy</span>
                        </div>
                      ) : null}
                    </div>

                    {lastOcrAccuracy ? (
                      <div className="benchmark-probe">
                        <div className="probe-header">
                          <span className={`probe-dot ${lastOcrAccuracy.accuracy >= 70 ? "dot-pass" : "dot-fail"}`} />
                          <span className="probe-query" title={lastOcrAccuracy.documentId}>
                            OCR: {lastOcrAccuracy.documentId
                              ? (lastOcrAccuracy.documentId.length > 25
                                ? lastOcrAccuracy.documentId.slice(0, 25) + "..."
                                : lastOcrAccuracy.documentId)
                              : "last document"}
                          </span>
                        </div>
                      </div>
                    ) : null}

                    {answerEvaluation ? (
                      <div className="benchmark-probe">
                        <div className="probe-header">
                          <span className={`probe-dot ${answerEvaluation.retrieval_accuracy >= 70 ? "dot-pass" : "dot-fail"}`} />
                          <span className="probe-query">
                            Q: {answerEvaluation.matched_question
                              ? (answerEvaluation.matched_question.length > 30
                                ? answerEvaluation.matched_question.slice(0, 30) + "..."
                                : answerEvaluation.matched_question)
                              : "last question"}
                          </span>
                          <span className="probe-latency">{answerEvaluation.retrieval_accuracy}%</span>
                        </div>
                        <div className="probe-sections">
                          <span className="meta-chip">Expected: {
                            answerEvaluation.expected_answer?.length > 35
                              ? answerEvaluation.expected_answer.slice(0, 35) + "..."
                              : answerEvaluation.expected_answer
                          }</span>
                        </div>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </aside>

            <div className="panel panel-main">
              {isAdmin ? (
                <>
                  <section className="upload-panel">
                    <div className="upload-head">
                      <div>
                        <p className="panel-label">Audit</p>
                        <h3>Masked Access Logs</h3>
                      </div>
                      <span className="caption">Signed in as {actorName || "unknown"}</span>
                    </div>

                    <div className="audit-filters">
                      <label className="field">
                        <span>From date</span>
                        <input type="date" value={auditDateFrom} onChange={(event) => setAuditDateFrom(event.target.value)} />
                      </label>
                      <label className="field">
                        <span>To date</span>
                        <input type="date" value={auditDateTo} onChange={(event) => setAuditDateTo(event.target.value)} />
                      </label>
                      <label className="field">
                        <span>Page size</span>
                        <select value={auditPageSize} onChange={(event) => setAuditPageSize(Number(event.target.value))}>
                          <option value={10}>10</option>
                          <option value={20}>20</option>
                          <option value={50}>50</option>
                        </select>
                      </label>
                    </div>

                    <div className="actions">
                      <button className="primary-button" disabled={isAuditLoading} onClick={applyAuditFilters}>
                        {isAuditLoading ? "Loading..." : "Load Logs"}
                      </button>
                      <span className="caption">{auditStatus}</span>
                    </div>

                    <div className="actions">
                      <span className={`caption ${serverHealth?.audit?.ready ? "status-ok" : "status-warn"}`}>
                        {serverHealthStatus}
                      </span>
                      <button className="ghost-button" onClick={loadServerHealth}>
                        Refresh Status
                      </button>
                    </div>

                    <p className="caption">
                      All audit entries stay masked here. Queries, document references, and any stored
                      errors are shown only in redacted form.
                    </p>

                    <div className="actions">
                      <button className="ghost-button" disabled={auditPageIndex <= 0 || isAuditLoading} onClick={previousAuditPage}>
                        Previous Page
                      </button>
                      <span className="caption">Page {auditPageIndex + 1}</span>
                      <button className="ghost-button" disabled={!auditNextCursor || isAuditLoading} onClick={nextAuditPage}>
                        Next Page
                      </button>
                    </div>
                  </section>

                  <div className="results">
                    {auditLogs.map((log) => (
                      <article className="result-card" key={log.audit_id}>
                        <div className="result-top">
                          <div>
                            <span className="result-id">{log.actor_name}</span>
                            <div className="caption audit-subline">
                              Audit ID {log.audit_id?.slice(0, 12) || "unknown"}
                            </div>
                          </div>
                          <span className="result-score">{formatAuditTimestamp(log.accessed_at)}</span>
                        </div>
                        <div className="result-meta">
                          <span className="meta-chip">{labelForRole(log.actor_role)}</span>
                          <span className="meta-chip">{log.status}</span>
                          <span className="meta-chip">{labelForAccessScope(log.authorized)}</span>
                          <span className="meta-chip">{log.document_count} documents</span>
                          <span className="meta-chip">Day {log.accessed_day || "unknown"}</span>
                        </div>
                        <div className="audit-detail-grid">
                          <div className="audit-detail-block">
                            <p className="panel-label">Masked query</p>
                            <div className="answer-text">{log.query_masked || "No query stored."}</div>
                          </div>
                          <div className="audit-detail-block">
                            <p className="panel-label">Query fingerprint</p>
                            <div className="audit-hash">{log.query_hash || "Unavailable"}</div>
                          </div>
                        </div>
                        {Array.isArray(log.document_refs) && log.document_refs.length ? (
                          <div className="audit-detail-block top-gap">
                            <p className="panel-label">Masked document references</p>
                            <div className="result-meta">
                            {log.document_refs.map((documentRef, index) => (
                              <span className="meta-chip" key={`${log.audit_id}-doc-${index}`}>
                                {documentRef.document_label} | {documentRef.document_hash.slice(0, 12)}
                              </span>
                            ))}
                          </div>
                          </div>
                        ) : null}
                        {log.error_masked ? (
                          <div className="audit-detail-block top-gap">
                            <p className="panel-label">Masked error</p>
                            <div className="caption">{log.error_masked}</div>
                          </div>
                        ) : null}
                      </article>
                    ))}
                  </div>
                </>
              ) : (
                <>
              <section className="upload-panel">
                <div className="upload-head">
                  <div>
                    <p className="panel-label">Upload</p>
                    <h3>Ingest a document before querying</h3>
                  </div>
                  <span className="caption">{uploadStateText}</span>
                </div>

                <label className="field">
                  <span>Select up to {MAX_BATCH_UPLOADS} PDFs or images</span>
                  <input
                    type="file"
                    multiple
                    accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp"
                    onChange={(event) => {
                      const files = Array.from(event.target.files || []);
                      setSelectedFiles(files);
                    }}
                  />
                </label>

                <div className="actions">
                  <button className="primary-button" disabled={isUploading} onClick={uploadDocument}>
                    {isUploading ? "Uploading..." : "Upload & Index"}
                  </button>
                  {hasDocument ? (
                    <button className="ghost-button" disabled={isUploading} onClick={clearDocumentScope}>
                      Search All Documents
                    </button>
                  ) : null}
                  <span className="caption">{uploadStatus}</span>
                </div>

                <div className={`progress-shell ${showProgress ? "" : "hidden"}`}>
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
                  </div>
                  <div className="progress-meta">
                    <span className="caption">{progressLabel}</span>
                    <span className="caption">{progressPercent}%</span>
                  </div>
                </div>

                <div className={`uploaded-summary ${hasDocument ? "" : "hidden"}`}>
                  <strong>{currentFilename || currentDocumentId}</strong>
                  <br />
                  {summaryMarkup}
                </div>
              </section>

              <label className="field top-gap">
                <span>Ask a question</span>
                <textarea
                  rows="4"
                  value={question}
                  placeholder="Example: Summarize the appointment instructions and contact details."
                  onChange={(event) => setQuestion(event.target.value)}
                />
              </label>

              <div className="actions">
                <button className="primary-button" onClick={askQuestion}>
                  Ask GPT-4o mini
                </button>
                <span className={`mode-pill ${currentRole}`}>
                  {authorized ? "Authorized raw context" : "Redacted context"}
                </span>
                <span className="caption">{askStatus}</span>
              </div>

              <section className={`answer-panel ${answer ? "" : "hidden"}`}>
                <p className="panel-label">Answer</p>
                <div className="answer-text">{answer}</div>
                <div className="result-meta">
                  {answerCitations.map((citation) => (
                    <span className="meta-chip" key={citation}>
                      {citation}
                    </span>
                  ))}
                </div>
                {answerEvaluation ? (
                  <div className="eval-card">
                    <div className="eval-header">
                      <span className="eval-label">Retrieval Accuracy</span>
                      <span className={`eval-score ${
                        answerEvaluation.retrieval_accuracy >= 80 ? "eval-high" :
                        answerEvaluation.retrieval_accuracy >= 50 ? "eval-mid" : "eval-low"
                      }`}>
                        {answerEvaluation.retrieval_accuracy}%
                      </span>
                    </div>
                    <div className="eval-bar-track">
                      <div
                        className={`eval-bar-fill ${
                          answerEvaluation.retrieval_accuracy >= 80 ? "eval-fill-high" :
                          answerEvaluation.retrieval_accuracy >= 50 ? "eval-fill-mid" : "eval-fill-low"
                        }`}
                        style={{ width: `${Math.min(answerEvaluation.retrieval_accuracy, 100)}%` }}
                      />
                    </div>
                    <div className="eval-details">
                      <div className="eval-detail-row">
                        <span className="eval-detail-label">Matched Key</span>
                        <span className="eval-detail-value">{answerEvaluation.matched_question}</span>
                      </div>
                      <div className="eval-detail-row">
                        <span className="eval-detail-label">Expected Answer</span>
                        <span className="eval-detail-value">{answerEvaluation.expected_answer}</span>
                      </div>
                    </div>
                  </div>
                ) : null}
              </section>
                </>
              )}
            </div>
          </div>
        </section>
      )}
    </main>
  );
}
