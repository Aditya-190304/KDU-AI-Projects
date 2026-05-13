# Medical Document Extraction MVP

This repository contains a local-first MVP for extracting structured data from medical PDFs and images.

The current implementation focuses on:

- page/image routing with `PyMuPDF` + image classifier
- OSS OCR stack with `PaddleOCR` for OCR/layout/table flows
- form extraction using `nielsr/layoutlmv3-finetuned-funsd`
- handwriting OCR using local `Ollama` vision model (default `qwen2.5vl:3b`) with broad handwritten-page OCR
- optional digital text/table fallbacks (`pdfplumber`, `Camelot`) for copyable PDFs
- medical entity, medication, and lab parsing
- plain-text output through the CLI (default)
- optional structured JSON output when `--text-only false` and JSON `--output` path are used

The architecture mirrors the long-term plan so local storage can later be swapped for S3/KMS adapters without rewriting the extraction pipeline.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m medical_extraction.cli.extract_document --input .\sample_docs\report.pdf
python -m medical_extraction.cli.extract_document --input .\sample_docs\report.pdf --output .\output\text\report_extraction.txt
python -m medical_extraction.cli.extract_document --input .\sample_docs\report.pdf --output .\output\report_extraction.json --text-only false
```

## Notes

- Heavy OCR/model integrations are wrapped behind model classes with graceful fallbacks.
- Reading order is deterministic: region split -> column clustering -> top-to-bottom line ordering.
- Mixed PDF ordering keeps selectable text and embedded-image OCR in one spatial reading order via block bounding boxes.
- Handwritten layout backend defaults to `paddle` (`MEDICAL_HANDWRITTEN_LAYOUT_BACKEND=paddle`) and falls back to full-page mode when needed.
- Set `MEDICAL_HANDWRITTEN_LAYOUT_BACKEND=rule` to skip region layout and force full-page OCR for handwritten pages.
- Paddle cache location can be pinned with `MEDICAL_PADDLE_CACHE_HOME` (default `.model_cache/paddlex`).
- Runtime defaults to `MEDICAL_PADDLE_OFFLINE_ONLY=true`; set it to `false` only when you explicitly want Paddle to auto-download missing models.
- Install OCR dependencies from `requirements.txt` before running handwritten/scanned routes.
- Debug images are optional because they may contain PHI.
