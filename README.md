# Medical RAG System - Detailed Design Handoff

This document is a detailed implementation handoff for the medical document RAG system in this repository.

It is written to help another model or engineer produce a formal design document without having to reverse-engineer the codebase from scratch. It explains the major components, extraction paths, chunking behavior, retrieval stack, redaction behavior, audit pipeline, frontend flows, storage layers, and operational details.

This document intentionally does not include architecture diagram code, since the diagram can be supplied separately.

## 1. System Goal

The system ingests medical PDFs and images, extracts structured and unstructured clinical content, transforms the extracted content into retrieval-ready chunks, indexes those chunks into persistent retrieval stores, and serves role-aware question answering across the indexed document corpus.

The system supports:

- copyable digital PDFs
- mixed PDFs with selectable text plus embedded image regions
- full scanned pages
- handwritten prescriptions and handwritten page regions
- form-style documents with label-value pairs
- medical retrieval across many persisted documents
- role-based answer generation
- redaction for unauthorized users
- masked audit logging for administrator review

## 2. Core User Roles

The application currently exposes three user roles in the frontend.

### Doctor

- receives authorized raw retrieval context
- can see unmasked content in generated answers
- can query either the current uploaded document or the full persistent document index

### Receptionist

- receives redacted retrieval context
- sees masked placeholders in answers instead of raw PHI
- can still receive useful answers if the answer can be expressed with masked placeholders

### Admin

- does not use the QA flow as a clinical reader
- can view masked audit logs
- can filter audit logs by date and paginate through results

## 3. Main Runtime Components

### Backend API server

The backend is a custom Python HTTP server built on:

- `ThreadingHTTPServer`
- `SimpleHTTPRequestHandler`

It is not based on FastAPI, Flask, or aiohttp.

Main backend file:

- `src/medical_extraction/server/qa_server.py`

Core endpoints include:

- `POST /api/upload`
- `GET /api/upload/status`
- `POST /api/ask`
- `POST /api/search`
- `GET /api/health`
- `POST /api/admin/audit/logs`

### Frontend

The frontend is a Vite React application.

Main frontend files:

- `frontend/src/App.jsx`
- `frontend/src/main.jsx`
- `frontend/src/styles.css`

### Pipeline orchestrator

The main orchestration layer is:

- `src/medical_extraction/core/pipeline.py`

This layer coordinates:

- file validation
- dedup lookup
- extraction routing
- payload generation
- RAG text generation
- chunk generation
- embedding and indexing
- cache-hit refresh behavior

## 4. Model Inventory

This section lists the major models and model-backed components used across the system.

### 4.1 Model summary table

| Layer | Model / Engine | Type | Primary role in system |
| --- | --- | --- | --- |
| Page and crop classification | `microsoft/dit-base-finetuned-rvlcdip` | Document image classifier | Classifies image crops and supports routing of embedded image regions in mixed PDFs |
| Scanned OCR | `paddleocr_v5` | OCR engine | Extracts printed/scanned text and line-level OCR content |
| Layout and table extraction | `PPStructureV3` | Document structure engine | Detects layout regions and extracts table structure from scanned content |
| Handwritten and vision fallback OCR | `qwen2.5vl:3b` via Ollama | Vision-language model | Handles handwritten fallback OCR and difficult image-region OCR |
| Form understanding | `nielsr/layoutlmv3-finetuned-funsd` | Layout-aware token classification model | Reconstructs field-value pairs from OCR words and bounding boxes |
| Biomedical NER | `d4data/biomedical-ner-all` | Medical token classification model | Extracts clinical entities from normalized text blocks |
| Dense embeddings | `BAAI/bge-small-en-v1.5` | Local embedding model | Provides dense semantic representation for chunking-aware semantic grouping and semantic retrieval |
| Retrieval reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker | Reorders hybrid retrieval candidates by query relevance |

### DiT page and crop classifier

Purpose:

- crop and image-type classification
- embedded image routing support inside mixed PDFs

Model:

- `microsoft/dit-base-finetuned-rvlcdip`

Relevant files:

- `src/medical_extraction/classification/crop_classifier.py`
- `src/medical_extraction/models/dit_classifier.py`

### Paddle OCR and layout extraction

Purpose:

- scanned OCR
- line extraction
- layout-aware OCR
- table extraction
- handwritten line detection support

Relevant file:

- `src/medical_extraction/models/paddle_extractor.py`

Important note:

- the legacy `surya_extractor.py` name still exists in the repo as an alias
- operationally, this path is Paddle-based

### Qwen2.5-VL via Ollama

Purpose:

- handwritten OCR fallback
- difficult image-region OCR
- mixed-PDF embedded handwritten crop OCR

Relevant file:

- `src/medical_extraction/models/trocr_extractor.py`

Important note:

- the wrapper name still says `TrOcrExtractor`
- the actual vision-language model path is Qwen2.5-VL through Ollama

### LayoutLMv3 for form understanding

Purpose:

- question/answer labeling over OCR words and layout
- form-style field extraction
- label-value reconstruction

Model:

- `nielsr/layoutlmv3-finetuned-funsd`

Relevant file:

- `src/medical_extraction/models/layoutlmv3_form_extractor.py`

### Biomedical NER

Purpose:

- optional medical entity recognition on extracted text
- medical entity enrichment before chunk construction

Relevant files:

- `src/medical_extraction/models/biomedical_ner.py`
- `src/medical_extraction/models/model_registry.py`

### Dense embedding model

Purpose:

- local dense semantic representation for chunking-aware semantic grouping and local semantic retrieval preparation

Model:

- `BAAI/bge-small-en-v1.5`

Relevant file:

- `src/medical_extraction/embeddings/local_embedder.py`

Usage summary:

- used as the local embedding layer for dense semantic representation
- supports semantic chunk grouping
- supports local embedding-driven retrieval behavior over the persisted vector store

### Cross-encoder reranker

Purpose:

- post-retrieval reranking
- query-aware candidate refinement after hybrid retrieval

Model:

- `cross-encoder/ms-marco-MiniLM-L-6-v2`

Relevant file:

- `src/medical_extraction/retrieval/local_reranker.py`

## 4A. OCR and Extraction Models Evaluated During System Evolution

The repository shows a clear model evolution path. A few earlier OCR and extraction approaches were explored or retained in legacy naming, but the current production-oriented stack was chosen because it gave the most reliable results for the document mix this system targets.

### Legacy Surya-oriented OCR path

The codebase still contains:

- `surya-ocr` in project dependencies
- `src/medical_extraction/models/surya_extractor.py`
- legacy source labels such as `surya_line_ocr`, `surya_crop_ocr`, and `surya_column_ocr`

This indicates that a Surya-style OCR path was part of the system evolution.

However, the current implementation no longer keeps Surya as the primary active OCR engine. Instead:

- `SuryaExtractor` is now a compatibility alias to `PaddleExtractor`
- printed/scanned OCR, layout extraction, and table extraction are centered on the Paddle stack

This strongly suggests that the Paddle-based route delivered a more satisfactory practical result for the current workload, especially for:

- scanned documents
- layout-aware extraction
- table handling
- unified OCR plus structure extraction

### Earlier TrOCR-oriented handwriting wrapper

The repository still uses the wrapper name:

- `TrOcrExtractor`

but the active implementation is now:

- Qwen2.5-VL via Ollama

This is a meaningful signal about the system’s evolution. It indicates that an earlier TrOCR-oriented handwriting path was not retained as the final primary handwriting solution, and the project moved to a Qwen vision-language fallback instead.

For the current document mix, that evolution makes sense because the chosen Qwen path is better aligned with:

- difficult handwritten prescription lines
- image-region OCR fallback
- mixed handwriting plus printed-content interpretation
- robust full-page handwritten fallback behavior

### Why the current OCR stack is the one to document

The present architecture is built around the models that performed well enough to remain in the active code path:

- Paddle for scanned OCR, layout, and tables
- Qwen2.5-VL for handwritten and difficult image-region fallback OCR
- LayoutLMv3 for form understanding
- DiT for crop/image classification

That is the right stack to describe in a design document because it reflects the models the system actually settled on after earlier routes proved less satisfactory for the target medical-document workload.

## 5. High-Level End-to-End Flow

At a high level, the system behaves as follows:

1. A user uploads a PDF or image.
2. The backend computes a file-content hash and checks the processed-file registry.
3. If the file is new, the system runs extraction.
4. Extracted content is normalized into page blocks, OCR text, form fields, and tables.
5. Optional biomedical NER enriches the extracted text.
6. `LocalMedicalChunker` creates retrieval-ready medical chunks.
7. Chunks are embedded and indexed into persistent retrieval stores.
8. The user asks a question.
9. Hybrid retrieval gathers semantic and keyword candidates.
10. RRF and query-intent boosts merge and refine the candidate pool.
11. The cross-encoder reranker reorders candidates.
12. The final top-k context is assembled.
13. If the user is authorized, raw context is used.
14. If the user is unauthorized, context is redacted first.
15. The answer is generated from the resulting context.
16. A masked audit log entry is written after the query completes.

## 6. Ingestion and Extraction Paths

The system supports multiple extraction routes depending on document characteristics.

### 6.1 Page classification and routing

The first routing layer is a page classifier that decides whether a page behaves like:

- copyable
- mixed
- scanned
- handwritten

Relevant file:

- `src/medical_extraction/classification/page_classifier.py`

This routing decision is important because the downstream extraction strategy changes substantially by page type.

### 6.2 Copyable PDF extraction flow

Relevant file:

- `src/medical_extraction/extraction/copyable_pdf_extractor.py`

This flow is used for clean digital PDFs with selectable text.

Main operations:

- extract text blocks using PyMuPDF
- use pdfplumber when needed for text fallback
- detect and extract tables with Camelot
- normalize reading order across extracted blocks
- emit structured page payloads with text blocks, metadata, and coordinates

Best suited for:

- generated medical reports
- digital prescriptions
- typed discharge summaries
- typed multi-page hospital records

### 6.3 Mixed PDF extraction flow

Relevant file:

- `src/medical_extraction/extraction/mixed_pdf_extractor.py`

This flow handles PDFs that contain both:

- selectable text
- embedded image regions

Main operations:

- extract copyable text first
- detect embedded image blocks from the PDF page representation
- crop embedded image regions
- classify those image crops using DiT
- route image crops to the appropriate OCR path

Crop handling behavior includes:

- report-like printed region OCR
- form-like crop extraction
- handwritten-like crop OCR
- table-like crop interpretation

This path is especially useful for:

- documents that mix digital body text with scanned inserts
- PDFs that contain scanned signatures, stamps, small embedded scans, or image-based tables

#### Detailed mixed PDF execution flow

For a mixed PDF that contains normal copyable report text plus one or more embedded scanned image regions, the system behaves like this:

1. The page is routed into the mixed-PDF path by the page classifier.
2. `MixedPdfExtractor` extracts the copyable PDF text blocks first.
3. Those copyable blocks are preserved with page coordinates, source labels, and reading-order metadata.
4. PyMuPDF is used to inspect the page structure and detect embedded image blocks.
5. Each embedded image block is cropped out of the page.
6. Each crop is classified with the DiT crop/image classifier.
7. If the crop looks like printed clinical text, the crop is OCR’d with the Paddle path.
8. If the crop looks like a form region, OCR output from that crop is passed into LayoutLMv3 form extraction.
9. If the crop looks handwritten or difficult to interpret, the system uses the Qwen2.5-VL fallback OCR path.
10. If the crop appears table-like, Paddle table extraction is used to preserve row structure.
11. The extracted crop content is converted into additional structured blocks and merged with the original copyable-text blocks.
12. The merged page result is then passed forward into biomedical NER, chunking, indexing, and retrieval just like any other page.

This flow is important because it allows the system to treat a mixed PDF as a single coherent medical page rather than forcing it into either:

- copyable-only extraction
- or full-page scanned OCR

That gives better fidelity for documents where the clinically important information is split across:

- digital typed text
- embedded scanned inserts
- handwritten annotations inside image regions
- image-based medication tables or small form fragments

### 6.4 Scanned page extraction flow

Relevant file:

- `src/medical_extraction/extraction/scanned_page_extractor.py`

This flow is used when the full page is image-based and not copyable.

Main operations:

- render the page to an image
- preprocess the image
- run PaddleOCR for text extraction
- run layout-aware OCR and region structuring
- run table extraction when tabular structure is detected
- pass OCR words and boxes into LayoutLMv3 for field/value understanding

Best suited for:

- scanned reports
- scanned printed prescriptions
- scanned admission forms
- scanned hospital summaries

### 6.5 Handwritten extraction flow

Relevant file:

- `src/medical_extraction/extraction/handwritten_prescription_extractor.py`

This flow is used for handwritten clinical content, especially prescriptions.

Main operations:

- preprocess the image
- detect handwriting lines and regions
- segment header, body, and footer regions
- merge likely related handwritten lines using layout signals
- OCR with Paddle-based line detection and layout support
- use Qwen2.5-VL fallback when handwritten text requires stronger vision-language interpretation
- suppress likely footer noise where possible

Best suited for:

- handwritten prescriptions
- mixed handwritten prescription notes
- image uploads where handwriting dominates the page

### 6.6 Form extraction flow

Relevant file:

- `src/medical_extraction/models/layoutlmv3_form_extractor.py`

This is a structural extraction layer, not the primary OCR engine.

Main operations:

- take OCR words plus bounding boxes
- classify token roles such as field-like question text and answer text
- pair related labels and values
- reconstruct form structure from OCR text and layout
- fall back to rule-based pairing if model output is incomplete

Examples of outputs this flow helps reconstruct:

- `Patient Name: John Scott`
- `DOB: 12/26/1998`
- `MRN: MRN100008`
- `Diagnosis: Hypertension`

### 6.7 Biomedical NER flow

Relevant file:

- `src/medical_extraction/models/biomedical_ner.py`

This stage enriches extracted medical text with entity-level information.

Examples of entities it may surface:

- diagnoses
- medications
- clinical concepts
- findings
- lab-related medical phrases

It does not create chunks by itself. It enriches the extracted material so chunking and downstream retrieval can use stronger medical signals.

## 7. Extraction Output Structure

The ingestion pipeline emits structured payloads containing:

- document id
- input file metadata
- created timestamp
- page list
- page-level type
- extracted blocks
- bounding boxes
- OCR confidence
- source provenance
- table rows where available
- form fields where recoverable

This structured payload is more important than the plain rendered `.txt` view because chunking uses layout and metadata, not just flat text order.

## 8. Chunking

Relevant file:

- `src/medical_extraction/utils/chunking.py`

The chunking layer is implemented by `LocalMedicalChunker`.

This is not a naive fixed-length splitter. It is a medical-aware, layout-aware chunking layer that works over the structured extraction payload.

### 8.1 Core chunking objectives

The chunker is designed to:

- preserve clinically meaningful units
- keep label-value pairs intact
- keep medication rows intact
- keep medical entities together where possible
- create chunks that are retrievable by both semantic and keyword methods
- maintain enough context for downstream QA

### 8.2 Main chunking strategies

The current chunking logic combines several strategies.

#### Entity-preserving chunking

Narrative medical text is split in a way that tries to avoid breaking apart medically meaningful units.

Behavior includes:

- sentence-aware segmentation
- long-sentence fallback splitting
- bounded token packing
- overlap tokens between adjacent chunks

This is especially useful for:

- assessment sections
- findings
- impression text
- discharge summaries

#### Form-field pairing

The chunker reconstructs field-value pairs using:

- OCR block text
- bounding boxes
- reading order
- right-side value collection
- field-label alias normalization

This is what enables chunks such as:

- `Patient Name: Joseph McIntyre`
- `DOB: 12/26/1998`
- `Allergies: NKDA`
- `Weight: 65 kg`

This strategy is critical because OCR text can be flattened in a misleading order, especially in scanned forms where labels appear in one block and values appear in another.

#### Demographic profile chunking

When multiple demographic fields are recovered, the chunker emits:

- per-field chunks
- a bundled demographic profile chunk

Example bundled chunk behavior:

- patient name
- DOB
- MRN
- allergies
- weight

This improves retrieval for questions that ask about several patient identity or demographic fields at once.

#### Table-aware chunking

If a block is identified as a table, the chunker treats it differently from prose.

Behavior includes:

- row-aware extraction
- row text normalization
- preservation of cell relationships
- enrichment of row text with surrounding document context where useful

This is especially important for:

- medication tables
- dose/frequency tables
- lab result tables

#### Prescription row chunking

Prescription-like content is grouped into medication orders rather than split into arbitrary text fragments.

Behavior includes:

- detecting medication order starts
- detecting continuation lines
- skipping prescription boilerplate
- grouping multi-line orders into single medication-order chunks

This is how the system keeps units like:

- drug name
- strength
- route or daily usage
- dispense or refill lines

within coherent retrieval units.

#### Prescription summary chunking

When the system detects a medication table or medication order set, it can build a higher-level prescription summary chunk in addition to row-level chunks.

This summary layer can include:

- patient context
- physician context
- hospital context
- multiple medication rows in one bundled summary representation

This is useful for queries such as:

- "what medicines are prescribed?"
- "what medication is prescribed to John Scott?"

#### Lab-aware chunking

Lab parsing and lab chunking are supported so that structured lab results are not treated as ordinary prose.

This helps preserve:

- analyte
- value
- units
- interpretation-style structure

#### Section-aware chunking

The chunker tracks clinical section headings and section transitions such as:

- diagnosis
- prescription
- medications
- labs
- instructions
- follow up
- discharge summary

Section signals are used both during chunk creation and later during retrieval-time query-intent boosting.

#### Semantic grouping layer

The chunker also supports a semantic grouping layer using local dense embeddings.

This layer uses:

- `BAAI/bge-small-en-v1.5`

Behavior includes:

- sentence embeddings
- adjacent sentence similarity scoring
- semantic boundary detection
- fallback to token-budget packing if semantic grouping is unavailable

This is helpful for:

- longer narrative sections
- semantically coherent notes
- reducing bad chunk boundaries inside continuous clinical prose

### 8.3 Token budgeting behavior

Chunk construction is constrained by:

- target token budget
- hard max token budget
- minimum chunk token target
- overlap token count

This lets the system keep chunks:

- small enough for retrieval quality
- large enough to preserve clinical context

### 8.4 Chunk metadata

Each chunk carries metadata such as:

- chunk id
- document id
- page number
- section
- page type
- strategy
- source block ids
- OCR confidence
- block type
- entity types
- raw artifact linkage

This metadata is used later in:

- retrieval boosts
- redaction
- audit
- debugging

## 9. Chunking Failure Modes the System Is Designed to Address

The chunking layer explicitly tries to handle several common medical OCR failure cases:

- labels and values extracted into separate OCR blocks
- all labels appearing first and all values appearing later
- medication lines split across multiple OCR lines
- table rows flattened into prose-like text
- mixed form-plus-prescription layouts
- OCR regions that contain both demographics and medication information

The chunker uses layout and pairing heuristics so that retrieval is not forced to depend on the plain text file order alone.

## 10. Storage Layers

The system uses several persistent storage layers for different purposes.

### 10.1 ChromaDB

Purpose:

- persistent vector storage for locally generated BGE chunk embeddings

Stored content:

- chunk text
- chunk id
- document id
- page metadata
- section metadata
- embedding vector

Relevant files:

- `src/medical_extraction/storage/chroma_store.py`
- `src/medical_extraction/storage/rag_ingestion.py`

### 10.2 SQLite FTS keyword index

Purpose:

- persistent keyword retrieval
- lexical search across all indexed chunks

Stored content includes:

- chunk id
- document id
- chunk text
- retrieval metadata

Relevant files:

- `src/medical_extraction/storage/keyword_index.py`
- `src/medical_extraction/retrieval/chroma_retriever.py`

This avoids a brute-force full-text scan across all chunks in memory and makes keyword retrieval persistent and reusable across sessions.

### 10.3 S3 artifact storage

Purpose:

- storing raw generated artifacts for later access or debugging

Typical stored artifacts:

- rendered `rag.txt`
- raw chunk artifact text

Relevant file:

- `src/medical_extraction/storage/s3_storage.py`

### 10.4 Processed-file registry

Purpose:

- deduplication by exact file content
- avoiding unnecessary reprocessing

Relevant file:

- `src/medical_extraction/storage/processed_registry.py`

Behavior:

- compute SHA-256 over file bytes
- same bytes under a different filename still count as the same file
- different bytes under the same filename are treated as new content

### 10.5 DynamoDB audit storage

Purpose:

- persistent masked query audit logging

Relevant file:

- `src/medical_extraction/storage/audit_store.py`

This layer stores masked audit records, not raw PHI-bearing content.

## 11. Cache-Hit Reuse and Refresh Behavior

The system does not simply skip work on a cache hit.

Current behavior on an already-processed file:

- load cached payload
- load cached chunk output
- reuse cached extraction results
- refresh the live retrieval indexes from cached chunks
- avoid rerunning OCR and chunk reconstruction

This is important because it ensures:

- Chroma remains current
- SQLite FTS remains current
- reuploaded cached files do not leave the live index stale

## 12. Retrieval

Relevant file:

- `src/medical_extraction/retrieval/chroma_retriever.py`

The system uses hybrid retrieval rather than relying on only vector or only keyword search.

### 12.1 Semantic retrieval

The semantic side of retrieval is based on locally generated BGE dense embeddings and vector search through Chroma.

Purpose:

- retrieve meaning-aligned chunks
- support paraphrased medical questions
- recover semantically relevant content even when lexical wording differs
- keep the semantic representation local to the application stack

### 12.2 Keyword retrieval

The keyword side of retrieval uses SQLite FTS.

Purpose:

- match explicit names
- match MRNs and literal field text
- match exact medication strings
- recover table rows and direct lexical matches

### 12.3 Reciprocal Rank Fusion

The system merges semantic and keyword candidate lists with RRF.

Purpose:

- combine strengths of both retrieval styles
- avoid overcommitting to just one search method
- preserve strong lexical hits while still surfacing semantically relevant chunks

### 12.4 Query-intent boosting

After the hybrid candidate pool is formed, the system applies retrieval-time boosts and penalties.

Current boost categories include:

- section boosts
- medication-text boosts
- document-affinity boosts
- chunk-text match boosts
- exact-phrase document boosts
- document co-occurrence boosts
- boilerplate penalties

This layer helps questions such as:

- medication questions
- diagnosis questions
- lab questions
- name-targeted questions

### 12.5 Name and document affinity handling

The retrieval stack tries to align:

- names inside the query
- names inside chunk text
- names and entities encoded in document ids

This is useful for queries like:

- "what medicines are prescribed to John Scott?"

because the system can use both chunk text and document-id structure to keep the right document family near the top.

### 12.6 Prescription-aware retrieval behavior

Prescription-style questions receive special handling, including:

- wider candidate windows
- stronger preference for prescription sections
- stronger preference for medication-like rows
- penalties for header and boilerplate chunks

This is important because prescription documents often contain:

- headers
- patient information
- instructions
- physician signatures
- medication rows

and not all of those are equally useful for answering medication questions.

## 13. Reranker

Relevant file:

- `src/medical_extraction/retrieval/local_reranker.py`

The system reranks candidates locally after hybrid retrieval.

Model:

- `cross-encoder/ms-marco-MiniLM-L-6-v2`

Purpose:

- query-aware ordering of the retrieved candidate set
- stronger final relevance ordering before answer construction

Inputs to the reranker:

- user query
- candidate raw text
- hybrid candidate pool

Outputs:

- rerank score
- reordered candidate list

The reranker is especially important when several chunks all share similar lexical terms but differ in actual answer usefulness.

## 14. Final Context Construction

After reranking, the system constructs the final answer context.

This stage:

- selects final top-k chunks
- preserves role-specific behavior
- builds the context objects that the answer layer receives

The frontend currently uses a top-k large enough to allow several chunks to be included for a single answer while still keeping prompt size controlled.

## 15. Redaction and Privacy Enforcement

Relevant file:

- `src/medical_extraction/privacy/redaction.py`

The privacy layer is implemented through `ChunkRedactor`.

It is used both for:

- unauthorized answer context redaction
- audit log masking before persistence

### 15.1 PHI detection strategy

The redaction layer uses Presidio plus custom recognizers.

Custom recognized categories include:

- patient name
- generic person name
- MRN
- UHID
- IP number
- date of birth
- date/time
- phone number
- email address
- location
- age

### 15.2 Custom recognizer strategy

The redactor includes label-aware patterns for common medical and hospital document fields such as:

- `Patient Name:`
- `MRN:`
- `DOB:`
- `IP No:`
- `UHID:`

This is important because medical forms frequently express PHI as labeled fields rather than in free narrative text.

### 15.3 Overlap and priority handling

When multiple recognizers overlap, the redactor applies priority logic so that:

- stronger labeled entities win over weaker generic matches
- overlapping entities do not produce inconsistent replacements

### 15.4 Placeholder strategy

The redaction layer produces placeholders that are later normalized into generic answer-safe forms.

Typical final placeholders include:

- `[PERSON]`
- `[DATE]`
- `[CONTACT]`
- `[ID]`
- plus other uppercase bracketed placeholders where needed

### 15.5 Deterministic HMAC behavior

The redaction layer also creates deterministic HMAC values for matched identity-bearing fields.

Purpose:

- support audit-safe correlation
- support identity-consistent metadata without storing raw PHI

Important limitation:

- HMAC values are one-way
- they are not used for unmasking
- original PHI is not reconstructed from masked text

### 15.6 Unauthorized answer behavior

When the user is unauthorized:

- the final top-k retrieved chunks are redacted first
- the prompt explicitly instructs the model to preserve masking
- if the answer is inferable from masked context, the answer should use the masked placeholders rather than refusing

Example target behavior:

- if context says `DOB: [DATE]`
- the answer should say the DOB is `[DATE]`
- not say the DOB is unavailable

### 15.7 Authorized answer behavior

When the user is authorized:

- raw chunks are used directly
- no answer-time redaction is applied

## 16. Prompting and Summarization Behavior

The answer layer constructs a grounded response from the final retrieved context.

This layer does not summarize the full document blindly. It performs query-conditioned answer synthesis using:

- final selected chunks
- role-specific prompt rules
- citation-friendly chunk references

Summarization behavior is therefore:

- retrieval grounded
- role conditioned
- chunk bounded
- privacy aware

The answer stage should be treated as contextual synthesis over retrieved evidence, not unrestricted free-form summarization.

## 17. Audit Logging

Relevant file:

- `src/medical_extraction/storage/audit_store.py`

The system logs access after each question-answer interaction.

### 17.1 What gets logged

Audit entries include:

- actor name
- actor role
- access time
- authorized or masked path
- status
- masked query text
- query fingerprint hash
- masked document references
- masked error text if a failure occurred

### 17.2 How PHI is masked before saving

Before the audit entry is written:

- query text is passed through the redaction layer
- document labels are normalized and then redacted
- error text is redacted as well if present

The stored audit record therefore contains masked values only.

Important point:

- masking happens before persistence
- the database is not treated as a raw PHI sink

### 17.3 Query fingerprint

The system also stores a deterministic query hash.

Purpose:

- correlate repeated or similar queries safely
- give admins a stable audit identifier
- avoid storing raw query text as the only reference point

### 17.4 Document references

Document references are stored in masked form and also hashed.

This lets admins see which record family was accessed without exposing raw patient labels in audit storage.

## 18. HIPAA-Oriented Privacy Strengths

This system has a strong HIPAA-oriented design posture because the most sensitive stages of the document pipeline are handled locally and under explicit privacy controls.

### 18.1 Local-first clinical document processing

The document understanding stack is designed so that the core medical processing stages happen inside the local application environment.

This includes:

- page classification
- OCR
- form extraction
- biomedical NER
- medical chunking
- reranking
- redaction
- audit masking

This is an important design strength because raw medical documents are transformed, structured, and privacy-processed close to the application boundary rather than being broadly propagated through multiple external services.

### 18.2 Local model usage for sensitive preprocessing

The system uses local model-backed components for the high-sensitivity preprocessing stages that touch the raw document structure most directly.

This includes:

- DiT for crop and image classification
- Paddle-based OCR and layout extraction
- Qwen2.5-VL for handwritten fallback OCR
- LayoutLMv3 for form understanding
- Biomedical NER for medical entity enrichment
- BGE (`BAAI/bge-small-en-v1.5`) for dense semantic representation support
- local cross-encoder reranking

This is a major privacy advantage because the raw source material is first interpreted and normalized within the local stack before later downstream answer orchestration.

### 18.3 Data minimization before downstream use

The system is designed around data minimization.

That means:

- only retrieval-relevant chunks are carried forward
- unauthorized users never receive raw PHI-bearing context
- audit records are masked before storage
- identity-oriented metadata is transformed into HMAC-safe fingerprints rather than stored as plain identifiers

This reduces unnecessary exposure of raw clinical details and aligns well with least-privilege and minimum-necessary handling principles.

### 18.4 Role-based exposure control

The model behavior is not one-size-fits-all. It is constrained by role.

Current strengths include:

- doctors receive raw retrieval context when authorized
- reception staff receive redacted context instead of raw PHI
- admins view masked audit logs rather than raw query history

This is valuable from a HIPAA perspective because the system reduces overexposure of patient data at the answer layer, not just at storage time.

### 18.5 Redaction before persistence

One of the strongest privacy properties in the system is that audit records are masked before they are written to DynamoDB.

That means:

- masked query text is stored
- masked document labels are stored
- masked error text is stored
- query fingerprints are stored as deterministic hashes

This is a meaningful safeguard because operational observability is preserved without turning the audit database into a raw PHI repository.

### 18.6 Persistent local audit environment

The system uses DynamoDB Local through Docker for local audit storage workflows.

This gives the project a controlled local audit environment with:

- reproducible setup
- isolated local persistence
- role-aware masked log review
- support for pagination and date filtering

From a design-document perspective, this is a strong operational story because it shows that privacy controls were considered not only for answering, but also for monitoring and governance.

### 18.7 Privacy-preserving answer behavior

For unauthorized users, the answer pipeline does more than redact retrieved chunks once.

The system:

- redacts the final context chunks
- instructs the answer layer to preserve placeholders
- post-processes the answer to scrub any remaining PHI-like text
- normalizes identifiers into generic placeholders such as `[PERSON]`, `[DATE]`, `[CONTACT]`, and `[ID]`

This layered approach is a strong safety characteristic because it does not rely on a single privacy control point.

### 18.8 Why this design is strong

From a privacy and healthcare-systems perspective, the strongest qualities of the current design are:

- local-first handling of document understanding
- medical-aware chunking rather than blind text splitting
- role-conditioned answer exposure
- masked audit persistence
- deterministic identity-safe fingerprints
- persistent retrieval without requiring broad raw-document reuse

Overall, the model and pipeline design present a strong privacy-preserving architecture for medical document QA, especially because the clinically sensitive transformation steps are handled within the local system and because PHI exposure is constrained by both retrieval-time and post-generation controls.

## 19. DynamoDB Local and Docker Setup

The audit system is backed by DynamoDB Local during local development.

Relevant compose file:

- `docker-compose.dynamodb.yml`

Current Docker behavior:

- uses `amazon/dynamodb-local:latest`
- binds local port `8000`
- persists local data under `./data/dynamodb`
- runs in shared DB mode

Operational flow:

1. Start DynamoDB Local with Docker Compose.
2. Run the audit setup script.
3. Backend ensures the audit table exists.
4. Query activity is written into the table after masking.
5. Admin UI reads paginated masked logs back from that same store.

Because the backing store is Dockerized and volume-backed, audit records survive backend restarts as long as the local DynamoDB data volume remains intact.

## 20. Frontend Behavior

Relevant file:

- `frontend/src/App.jsx`

### 19.1 Main frontend capabilities

The frontend currently supports:

- role switching
- multiple file upload
- batch upload progress
- async upload job polling
- current-document querying
- full-index querying
- admin audit log browsing

### 19.2 Upload flow

The frontend can submit up to 100 files in a batch.

During upload and ingestion it shows:

- circular progress state
- stage text
- batch size
- completed count
- failed count

### 19.3 Query flow

The question input remains usable even if no new upload has just happened.

If a document is active:

- questions can be scoped to that document

If the user clears scope:

- questions run against the full persistent corpus

### 19.4 Search-all-documents mode

The frontend includes a control to clear the active document scope.

This is important because the retrieval system supports:

- document-scoped search
- whole-corpus search

and the UI needs to let the user switch between those two modes.

### 19.5 Admin page behavior

The admin page supports:

- masked audit log display
- date-based filtering
- pagination
- audit backend health display

## 21. End-to-End Operational Flows

### 20.1 Upload and first-time indexing flow

1. User uploads a file.
2. Backend writes raw upload data to disk.
3. Backend computes SHA-256 content hash.
4. Registry is checked for prior processing.
5. File is routed into the right extraction path.
6. OCR, form extraction, and optional biomedical NER run.
7. `LocalMedicalChunker` creates medical chunks.
8. Chunks are embedded and indexed.
9. Raw artifacts are written out.
10. Registry entry is stored.
11. Frontend receives indexed summary.

### 20.2 Cache-hit upload flow

1. User uploads a file whose bytes were seen before.
2. SHA-256 lookup hits the processed registry.
3. Cached payload and chunk output are loaded.
4. Live indexes are refreshed from the cached chunk data.
5. OCR and chunk reconstruction are skipped.
6. Frontend is told the file was already processed and reused.

### 20.3 Doctor question flow

1. User asks a question.
2. Hybrid retrieval runs.
3. RRF merges semantic and keyword candidates.
4. Query-intent boosts refine the pool.
5. Cross-encoder reranking reorders candidates.
6. Final top-k context is assembled.
7. Raw context is passed to answer generation.
8. Answer is returned.
9. Masked audit record is written.

### 20.4 Receptionist question flow

1. User asks a question.
2. Hybrid retrieval and reranking run the same way.
3. Final top-k context is selected.
4. Those context chunks are redacted.
5. The prompt instructs the answer layer to preserve placeholders.
6. The answer is returned with masked values where applicable.
7. Masked audit record is written.

### 20.5 Admin audit flow

1. Admin requests logs.
2. Backend queries DynamoDB Local or DynamoDB.
3. Results are returned in masked form.
4. Pagination cursor and date filters are applied.
5. Admin UI renders the log page.

## 22. Known Design Pressures and Tuning Areas

These are important for any formal design doc or follow-on architecture work.

### Medication row context linking

Medication rows can still become too decontextualized if retrieval relies only on row text and not enough shared prescription context.

This is why the system includes:

- prescription summaries
- section boosts
- medication boosts
- document-affinity boosts

### OCR flattening on scanned forms

Scanned forms often produce:

- labels first
- values later

The chunker works around this with layout-aware pairing, but this remains a core challenge in scanned document systems.

### Mixed PDF embedded image routing

Mixed PDFs require more than just text extraction because clinically useful content may sit inside embedded image blocks.

The system therefore uses:

- embedded image detection
- crop classification
- crop-specific OCR routing

### Redaction quality versus answer usefulness

The privacy layer must preserve enough structure to allow useful masked answers while still preventing PHI disclosure.

This is why the system redacts:

- final answer context for unauthorized roles
- audit text before persistence

while still preserving meaningful placeholders.

## 23. Repository Areas Most Important for a Formal Design Doc

If another model is preparing a polished architecture or design document, the most important implementation areas to study are:

- `src/medical_extraction/core/pipeline.py`
- `src/medical_extraction/server/qa_server.py`
- `src/medical_extraction/utils/chunking.py`
- `src/medical_extraction/retrieval/chroma_retriever.py`
- `src/medical_extraction/retrieval/local_reranker.py`
- `src/medical_extraction/privacy/redaction.py`
- `src/medical_extraction/storage/rag_ingestion.py`
- `src/medical_extraction/storage/audit_store.py`
- `frontend/src/App.jsx`

## 24. Repository File Structure

The repository is organized so that backend logic, frontend code, configs, generated data, and scripts are separated clearly.

### 24.1 Top-level structure

- `src/`
  - primary Python source tree
- `src/medical_extraction/`
  - main backend package
- `frontend/`
  - Vite React frontend
- `configs/`
  - runtime YAML configuration
- `data/`
  - local persistent runtime data such as uploads, Chroma, DynamoDB Local data, and registries
- `output/`
  - generated extraction outputs, RAG text files, and chunk JSON files
- `scripts/`
  - setup and maintenance scripts such as audit setup and keyword-index backfill
- `tests/`
  - unit and integration-style tests
- `.model_cache/`
  - local model cache area

### 24.2 Python package structure

Inside `src/medical_extraction/`, the major folders are:

- `answering/`
  - answer generation, prompting, and answer formatting
- `benchmark/`
  - OCR and system benchmarking utilities
- `classification/`
  - page and crop classification logic
- `cli/`
  - command-line entrypoints such as extraction and frontend runner commands
- `core/`
  - shared pipeline orchestration, config loading, constants, and core types
- `embeddings/`
  - embedding wrappers and local dense embedding helpers
- `extraction/`
  - copyable, mixed, scanned, and handwritten extraction flows
- `models/`
  - model wrappers for OCR, form extraction, NER, DiT, and handwriting vision models
- `parsing/`
  - structured parsing helpers such as medication and lab parsing
- `privacy/`
  - PHI redaction and masking logic
- `quality/`
  - quality-control helpers and validation-related logic
- `retrieval/`
  - Chroma retrieval, keyword retrieval, reranking, access rules, and hybrid retrieval logic
- `server/`
  - HTTP server and API handler
- `storage/`
  - Chroma storage, SQLite FTS, S3 artifact handling, audit store, and processed-file registry
- `utils/`
  - chunking, RAG text helpers, environment helpers, and general utility functions

### 24.3 Frontend structure

Inside `frontend/`, the important parts are:

- `src/`
  - React application source
- `dist/`
  - built frontend assets served by the backend when present
- `node_modules/`
  - frontend dependency installation directory

Frontend source is primarily centered on:

- `frontend/src/App.jsx`
- `frontend/src/main.jsx`
- `frontend/src/styles.css`

### 24.4 Data and generated artifact structure

The project writes several kinds of generated or persistent data:

- `data/uploads/`
  - uploaded source files received by the backend
- `data/chroma/`
  - persistent Chroma vector store
- `data/dynamodb/`
  - DynamoDB Local on-disk state when Docker volume mapping is used
- `data/processed_registry.json`
  - processed-file dedup registry keyed by SHA-256 content hash
- `output/uploads/`
  - extracted JSON, `rag.txt`, and chunk JSON outputs for uploaded files

### 24.5 Key configuration and environment files

- `.env`
  - local environment values
- `.env.example`
  - example environment template
- `configs/local.yaml`
  - main local runtime configuration
- `configs/default.yaml`
  - default configuration values
- `configs/models.yaml`
  - model-related configuration
- `configs/thresholds.yaml`
  - threshold and routing-related settings

### 24.6 Key operational support files

- `docker-compose.dynamodb.yml`
  - Docker Compose definition for DynamoDB Local
- `requirements.txt`
  - Python dependencies
- `pyproject.toml`
  - project metadata and dependency definition
- `README.md`
  - this implementation handoff document

## 25. Operational Commands

### Start backend

```powershell
cd "C:\Users\Dell\Desktop\rag project"
$env:PYTHONPATH="src"
python -m medical_extraction.cli.run_frontend --config configs/local.yaml
```

### Start frontend dev server

```powershell
cd "C:\Users\Dell\Desktop\rag project\frontend"
npm.cmd run dev
```

### Start DynamoDB Local

```powershell
cd "C:\Users\Dell\Desktop\rag project"
docker compose -f docker-compose.dynamodb.yml up -d
python scripts\setup_dynamodb_audit.py --config configs/local.yaml
```

### Rebuild frontend

```powershell
cd "C:\Users\Dell\Desktop\rag project\frontend"
npm.cmd run build
```

### Backfill keyword index if needed

```powershell
cd "C:\Users\Dell\Desktop\rag project"
$env:PYTHONPATH="src"
python scripts\backfill_keyword_index.py --config configs/local.yaml
```

