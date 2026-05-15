# Medical Document Extraction MVP Plan

## Purpose

This document describes the **extraction-only** phase for a medical document pipeline that will later feed into Secure RAG.

The current MVP should:

- Take a PDF from the local machine.
- Classify each page as copyable, mixed, scanned, or handwritten prescription.
- Extract useful text, tables, forms, and handwritten prescription content.
- Save the extraction result as a local structured JSON file.

This phase should **not** include:

- Presidio
- PHI/PII redaction
- HMAC placeholders
- chunking
- embeddings
- Elasticsearch/vector DB
- LLM calls
- RBAC
- S3/KMS integration

Those will be added later after the local extraction pipeline is validated.

---

# 1. MVP Scope

## Input

A local PDF file.

Example:

```bash
python -m medical_extraction.cli.extract_document \
  --input ./sample_docs/report.pdf \
  --output ./output/report_extraction.json \
  --debug-dir ./output/debug/report \
  --save-debug-images true \
  --enable-medical-ner true \
  --device cpu
```

## Output

A local JSON file containing the structured extraction output.

Example:

```text
./output/report_extraction.json
```

Optional debug output:

```text
./output/debug/report/page_1_render.png
./output/debug/report/page_1_crop_1.png
./output/debug/report/page_2_table_1.png
```

Debug images may contain PHI in real usage, so they should be disabled by default in production.

---

# 2. Chosen Model and Tool Stack

Use one chosen model/tool per responsibility.

| Responsibility | Chosen model/tool | Purpose |
|---|---|---|
| PDF inspection and page classification | `PyMuPDF / fitz` | Detect text layer, image blocks, image bounding boxes, page dimensions, and image coverage. |
| Copyable PDF text extraction | `pdfplumber` | Extract selectable text, words, text positions, paragraphs, and reading order from digital PDFs. |
| Digital/selectable PDF table extraction | `Camelot` | Extract tables from digital/selectable PDFs. |
| Crop classification | `microsoft/dit-base-finetuned-rvlcdip` | Classify cropped image regions into broad route categories such as form-like, handwritten-like, printed/report-like, or noisy/non-useful. |
| OCR, layout, reading order, scanned page processing, table recognition | `Surya` | Extract text, layout regions, reading order, and table structure from scanned pages/images. |
| Form understanding | `nielsr/layoutlmv3-finetuned-funsd` | Label OCR tokens as HEADER, QUESTION, ANSWER, OTHER for form field extraction. |
| Handwritten prescription OCR | `microsoft/trocr-base-handwritten` | OCR handwritten prescription regions. |
| Medical entity extraction after OCR | `d4data/biomedical-ner-all` | Extract medical entities such as symptoms, diseases, medications, procedures, anatomy, and clinical findings. |
| Medication parsing | Custom parser + RxNorm/local formulary | Extract and normalize medication name, dose, route, frequency, duration, quantity, and instructions. |
| Lab parsing | Custom rule-based parser | Extract and normalize lab test name, value, unit, reference range, and abnormal flag. |

## Important Responsibility Split

- **PyMuPDF** decides whether a PDF page is copyable, mixed, scanned, or mostly image-based.
- **DiT** classifies cropped image regions into broad image/document types.
- **Surya** extracts text, layout, reading order, and tables.
- **LayoutLMv3-FUNSD** understands forms after OCR tokens and bounding boxes are available.
- **TrOCR** reads handwritten prescription regions.
- **Biomedical NER** extracts clinical entities after text extraction.

---

# 3. Industry-Standard Project Structure

Use a layered project structure so local storage can later be replaced with S3/KMS without rewriting extraction logic.

```text
medical-document-extraction/
  README.md
  pyproject.toml
  requirements.txt
  .env.example
  .gitignore

  configs/
    default.yaml
    local.yaml
    models.yaml
    thresholds.yaml

  sample_docs/
    .gitkeep

  output/
    .gitkeep

  src/
    medical_extraction/
      __init__.py

      cli/
        __init__.py
        extract_document.py

      core/
        __init__.py
        pipeline.py
        types.py
        schemas.py
        exceptions.py
        constants.py

      storage/
        __init__.py
        base.py
        local_storage.py
        s3_storage.py

      classification/
        __init__.py
        page_classifier.py
        text_quality.py
        image_coverage.py
        crop_classifier.py

      extraction/
        __init__.py
        copyable_pdf_extractor.py
        mixed_pdf_extractor.py
        scanned_page_extractor.py
        handwritten_prescription_extractor.py
        merge_engine.py

      models/
        __init__.py
        model_registry.py
        dit_classifier.py
        surya_extractor.py
        layoutlmv3_form_extractor.py
        trocr_extractor.py
        biomedical_ner.py

      parsing/
        __init__.py
        medication_parser.py
        lab_parser.py
        clinical_validator.py

      quality/
        __init__.py
        confidence.py
        review_flags.py
        quality_checker.py

      utils/
        __init__.py
        pdf_utils.py
        image_utils.py
        json_utils.py
        logging_utils.py
        timing.py

  tests/
    unit/
      test_page_classifier.py
      test_text_quality.py
      test_image_coverage.py
      test_medication_parser.py
      test_lab_parser.py

    integration/
      test_extract_copyable_pdf.py
      test_extract_mixed_pdf.py
      test_extract_scanned_pdf.py
      test_extract_prescription.py

  scripts/
    download_models.py
    run_local_sample.sh
```

## Why this structure

- `storage/` separates local filesystem from future S3/KMS storage.
- `classification/` handles page/crop routing decisions.
- `extraction/` contains extraction paths by document/page type.
- `models/` wraps model-specific loading and inference.
- `parsing/` handles medical-specific parsing after OCR.
- `quality/` centralizes confidence and review logic.
- `core/pipeline.py` orchestrates the complete extraction flow.

---

# 4. High-Level Local Extraction Flow

```text
Local PDF
-> LocalInputAdapter reads PDF
-> PyMuPDF Page Classifier inspects each page
-> Page Router chooses extraction path
-> Extract text/tables/forms/handwriting
-> Run optional medical NER/parsers
-> Quality checker adds confidence and review flags
-> LocalOutputAdapter writes structured extraction JSON
```

---

# 5. Page Classification

## Component

```text
src/medical_extraction/classification/page_classifier.py
```

## Purpose

Classify each PDF page into one of these types:

1. `copyable_pdf`
2. `copyable_pdf_with_images`
3. `fully_scanned_report_form_table`
4. `handwritten_scanned_prescription`
5. `unknown_or_hybrid`

Classification must be **page-level**, not file-level.

A single PDF can contain both copyable pages and scanned pages.

## Page Classifier Responsibilities

The Page Classifier checks:

1. Does the page have selectable text?
2. How many text characters/words are present?
3. Is the selectable text meaningful or OCR garbage?
4. Does the page contain embedded images?
5. What are the bounding boxes of embedded images?
6. How much of the page area is covered by images?
7. Is the whole page basically one large image?
8. Are there digital table candidates?
9. Does the page appear scanned?
10. Does the page appear handwritten?

## PyMuPDF Checks

Use PyMuPDF to check:

- selectable text
- text blocks
- image blocks
- image bounding boxes
- page dimensions
- image coverage area

## Text Quality Checks

A PDF may have a bad text layer. Do not trust the existence of text alone.

Check:

- character count
- word count
- readable word ratio
- alphabetic character ratio
- noise/symbol ratio
- average word length
- duplicate text ratio
- mostly whitespace
- OCR garbage patterns

Good text example:

```text
Patient reports chest pain for two days.
```

Bad text example:

```text
□@# 11 lll ||| xzq ~~ 0O0O ///
```

## Image Threshold Rules

Recommended defaults:

- Ignore very small images such as logos if they are less than 2% of page area.
- Consider OCR for image regions larger than 10% of page area.
- If total image coverage is greater than 70% and selectable text is missing or poor, treat the page as scanned.
- If text layer exists but is low quality or garbage, treat the page as scanned or hybrid.
- If the page is mostly image and contains handwriting indicators, route to handwritten scanned prescription flow.

## Example Page Classifier Output

```json
{
  "document_id": "doc_123",
  "page_number": 2,
  "has_selectable_text": true,
  "selectable_text_chars": 1520,
  "text_quality": "good",
  "has_embedded_images": true,
  "image_count": 2,
  "image_coverage": 0.28,
  "is_mostly_image": false,
  "is_handwritten_candidate": false,
  "page_class": "copyable_pdf_with_images",
  "route": "pdf_text_plus_ocr_image_regions"
}
```

## Page Classification Pseudocode

```python
import fitz  # PyMuPDF


def classify_pdf_page(page):
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height

    text = page.get_text("text") or ""
    text_chars = len(text.strip())

    blocks = page.get_text("dict")["blocks"]

    image_area = 0
    image_count = 0

    for block in blocks:
        if block["type"] == 1:  # image block
            x0, y0, x1, y1 = block["bbox"]
            area = (x1 - x0) * (y1 - y0)

            # Ignore tiny images/logos
            if area / page_area < 0.02:
                continue

            image_area += area
            image_count += 1

    image_coverage = image_area / page_area if page_area else 0

    has_useful_text = text_chars > 100 and looks_readable(text)

    if is_likely_handwritten_prescription(page):
        return "handwritten_scanned_prescription"

    if has_useful_text and image_count == 0:
        return "copyable_pdf"

    if has_useful_text and image_count > 0:
        return "copyable_pdf_with_images"

    if not has_useful_text and image_coverage > 0.70:
        return "fully_scanned_report_form_table"

    return "unknown_or_hybrid"


def looks_readable(text):
    text = text.strip()

    if len(text) == 0:
        return False

    alpha_chars = sum(c.isalpha() for c in text)
    total_chars = len(text)
    alpha_ratio = alpha_chars / total_chars if total_chars else 0

    allowed_symbols = ".,:;-/()%"
    weird_chars = sum(
        1 for c in text
        if not c.isalnum() and not c.isspace() and c not in allowed_symbols
    )
    weird_ratio = weird_chars / total_chars if total_chars else 0

    return alpha_ratio > 0.45 and weird_ratio < 0.25
```

---

# 6. Flow 1: Copyable PDF Page

## Use when

- PDF has selectable text.
- Paragraphs are copyable.
- Tables may be digital/selectable.
- Page is not mainly scanned image.

## Extraction Flow

```text
PDF page
-> PyMuPDF page classification
-> pdfplumber text extraction
-> Camelot digital table extraction
-> structured JSON blocks
```

## Steps

1. Use PyMuPDF to classify the page as `copyable_pdf`.
2. Use `pdfplumber` to extract selectable text.
3. Preserve:
   - page number
   - headings
   - paragraphs
   - bullet points
   - text bounding boxes
   - reading order
4. If digital/selectable tables are detected, use `Camelot`.
5. Convert extracted content into structured JSON blocks.

## Example Paragraph Block

```json
{
  "block_id": "p1_b1",
  "type": "paragraph",
  "text": "Patient reports chest pain for two days.",
  "source": "pdf_text",
  "confidence": 1.0,
  "page_number": 1,
  "bbox": null,
  "needs_review": false
}
```

## Example Digital Table Block

```json
{
  "block_id": "p1_t1",
  "type": "table",
  "title": "Lab Results",
  "source": "digital_table_extraction",
  "structured_data": {
    "columns": ["Test", "Value", "Unit"],
    "rows": [
      {
        "Test": "HbA1c",
        "Value": "7.8",
        "Unit": "%"
      }
    ]
  },
  "text": "Lab Results: HbA1c value 7.8 percent.",
  "confidence": 1.0,
  "page_number": 1,
  "needs_review": false
}
```

---

# 7. Flow 2: Copyable PDF Page with Embedded Image Regions

## Use when

- PDF has useful selectable text.
- Page also contains embedded images.
- Images may contain scanned reports, screenshots, stamps, signatures, or small scanned sections.
- The system should not OCR the whole page.

## Extraction Flow

```text
PDF page
-> pdfplumber extracts selectable text
-> PyMuPDF detects embedded image bounding boxes
-> crop important image regions only
-> DiT classifies each crop
-> route crop to extractor
-> merge PDF text + image extraction by page position
-> structured JSON blocks
```

## Crop Classifier

Use:

```text
microsoft/dit-base-finetuned-rvlcdip
```

Purpose:

- classify cropped image regions into broad route categories.

Route categories:

- printed text image
- form-like image
- handwritten-like image
- report/letter/memo-like image
- logo/stamp/signature/noise
- unknown/review

## Crop Routing

```text
crop
-> DiT crop classifier
-> predicted class + confidence
-> Surya table check
-> route
```

Routing rules:

- Printed text -> Surya OCR
- Table -> Surya table recognition
- Form -> Surya OCR + LayoutLMv3-FUNSD
- Handwriting -> TrOCR handwritten
- Noise/logo/signature -> ignore, metadata, or review
- Low confidence -> fallback OCR or review

## Example Crop Block

```json
{
  "block_id": "p2_img1",
  "type": "image_ocr",
  "text": "Lab result: HbA1c 7.8 percent.",
  "source": "ocr_on_cropped_image_region",
  "bbox": [100, 300, 500, 650],
  "crop_classifier": {
    "model": "microsoft/dit-base-finetuned-rvlcdip",
    "predicted_class": "scientific_report",
    "confidence": 0.81
  },
  "confidence": 0.91,
  "page_number": 2,
  "needs_review": false
}
```

---

# 8. Flow 3: Fully Scanned Report, Form, or Table

## Use when

- Page has little or no selectable text.
- Page is mostly an image.
- It may contain printed paragraphs, tables, or forms.
- `pdfplumber` text extraction is empty or low quality.
- `Camelot` will not work because the table is not digital/selectable.

## Extraction Flow

```text
PDF/image page
-> render page as image
-> preprocess image
-> Surya layout/OCR/table recognition
-> crop detected regions
-> DiT classifier if region routing is needed
-> OCR/table/form extraction
-> reconstruct reading order
-> structured JSON blocks
```

## Steps

1. Use PyMuPDF to confirm the page is scanned/image-heavy.
2. Render the page as an image with PyMuPDF.
3. Preprocess image using OpenCV/Pillow:
   - deskew
   - denoise
   - rotate correction
   - contrast enhancement
   - border cleanup
   - orientation correction
4. Run Surya for layout/OCR/table recognition.
5. Route regions:
   - Text/paragraph -> Surya OCR
   - Table -> Surya table recognition
   - Form-like -> Surya OCR + LayoutLMv3-FUNSD
   - Signature/stamp/figure -> OCR only if useful; otherwise metadata/ignore
6. Reconstruct reading order.
7. Convert to structured JSON blocks.

## Example Scanned Text Block

```json
{
  "block_id": "p3_b1",
  "type": "paragraph",
  "text": "Patient reports fatigue and increased thirst.",
  "source": "surya_ocr",
  "confidence": 0.93,
  "page_number": 3,
  "bbox": [80, 120, 520, 220],
  "needs_review": false
}
```

## Example Scanned Table Block

```json
{
  "block_id": "p3_t1",
  "type": "table",
  "title": "Lab Results",
  "source": "surya_table_recognition",
  "structured_data": {
    "columns": ["Test", "Value", "Unit", "Reference Range"],
    "rows": [
      {
        "Test": "HbA1c",
        "Value": "7.8",
        "Unit": "%",
        "Reference Range": "4.0-5.6"
      }
    ]
  },
  "text": "Lab Results: HbA1c value 7.8 percent, reference range 4.0 to 5.6.",
  "confidence": 0.88,
  "page_number": 3,
  "needs_review": false
}
```

---

# 9. Flow 4: Handwritten Scanned Prescription

## Use when

- Page is handwritten.
- Prescription is scanned or photographed.
- It contains medicine name, dose, route, frequency, duration, patient name, doctor name, or date.
- Clinical risk is high if OCR is wrong.

## Extraction Flow

```text
Prescription PDF/image
-> render page as image
-> preprocess image
-> Surya layout detects prescription regions
-> crop medication/patient/date regions
-> DiT classifier confirms crop type if needed
-> TrOCR handwritten OCR
-> medication parser
-> dose/frequency validator
-> confidence checker
-> human review flag if needed
-> structured JSON blocks
```

## Steps

1. Render prescription page as image using PyMuPDF.
2. Preprocess with OpenCV/Pillow:
   - crop page
   - deskew
   - denoise
   - contrast enhancement
   - remove background noise
   - orientation correction
   - line segmentation
3. Use Surya layout to detect prescription regions:
   - patient info
   - date
   - medication lines
   - dosage/frequency instructions
   - doctor note
   - doctor signature
   - clinic stamp
4. Crop relevant handwritten regions.
5. Run `microsoft/trocr-base-handwritten`.
6. Run medication parser and validators.
7. Mark low-confidence fields for review.

## Example Prescription Block

```json
{
  "block_id": "p1_rx1",
  "type": "prescription_item",
  "source": "trocr_handwritten",
  "fields": {
    "medication": {
      "value": "Metformin",
      "confidence": 0.88
    },
    "dose": {
      "value": "500mg",
      "confidence": 0.76
    },
    "frequency": {
      "value": "twice daily",
      "confidence": 0.64
    }
  },
  "text": "Metformin 500mg twice daily.",
  "confidence": 0.64,
  "needs_review": true,
  "page_number": 1
}
```

## Prescription Review Policy

Recommended thresholds:

- `confidence >= 0.90` -> high confidence
- `confidence 0.70 to 0.90` -> usable but mark caution
- `confidence < 0.70` -> `needs_review = true`

For prescriptions, if medication, dose, route, or frequency is low confidence, always set:

```json
{
  "needs_review": true
}
```

---

# 10. Form Extraction

## Use when

A crop or page region is classified as form-like.

## Chosen model

```text
nielsr/layoutlmv3-finetuned-funsd
```

## Flow

```text
form-like crop
-> Surya OCR
-> OCR tokens + bounding boxes
-> LayoutLMv3-FUNSD
-> token labels: HEADER / QUESTION / ANSWER / OTHER
-> group question-answer pairs
-> structured form JSON
```

## Output Example

```json
{
  "block_id": "p3_f1",
  "type": "form",
  "source": "surya_ocr_plus_layoutlmv3_funsd",
  "fields": {
    "Name": "John Smith",
    "DOB": "12 Jan 1980",
    "MRN": "12345",
    "Symptoms": "chest pain"
  },
  "text": "Name: John Smith. DOB: 12 Jan 1980. MRN: 12345. Symptoms: chest pain.",
  "confidence": 0.86,
  "page_number": 3,
  "needs_review": false
}
```

## Important Note

LayoutLMv3-FUNSD is a better default than manual key-value rules, but it is trained on general forms, not specifically medical forms.

For production, fine-tune it on your own forms if accuracy is not enough.

---

# 11. Medical NER and Medical Parsers

## Medical NER

Run after OCR/text extraction.

Chosen model:

```text
d4data/biomedical-ner-all
```

Purpose:

- symptoms
- diseases
- diagnoses
- procedures
- medications
- lab tests
- anatomy
- clinical findings

## Medication Parser

MVP approach:

```text
custom medication parser + RxNorm/local formulary
```

Detect:

- drug name
- dose
- route
- frequency
- duration
- quantity
- instructions

Examples:

```text
Metformin 500mg twice daily
Warfarin 2.5mg orally once daily
Amoxicillin 500 mg TID for 7 days
```

Use:

- regex for dose
- frequency dictionary
- route dictionary
- local medication list if available
- RxNorm/local formulary for normalization

## Lab Parser

MVP approach:

```text
custom rule-based lab parser
```

Detect patterns such as:

```text
HbA1c 7.8 %
Glucose 180 mg/dL
LDL 140 mg/dL
Creatinine 1.2 mg/dL
```

Extract:

- test name
- value
- unit
- reference range
- abnormal flag

---

# 12. Final Extraction JSON Schema

The local output JSON should follow this structure:

```json
{
  "document_id": "report",
  "input_file": "./sample_docs/report.pdf",
  "extraction_version": "local_mvp_v1",
  "created_at": "2026-05-12T10:30:00Z",
  "summary": {
    "total_pages": 10,
    "copyable_pages": 4,
    "mixed_pages": 2,
    "scanned_pages": 3,
    "handwritten_pages": 1,
    "total_blocks": 84,
    "total_tables": 5,
    "total_forms": 2,
    "total_prescriptions": 1,
    "needs_review_blocks": 6,
    "processing_time_seconds": 182.4
  },
  "pages": [
    {
      "page_number": 1,
      "page_type": "copyable_pdf",
      "classification": {
        "has_selectable_text": true,
        "selectable_text_chars": 2450,
        "has_images": false,
        "image_coverage": 0.01,
        "route": "copyable_pdf"
      },
      "timing": {
        "classification_ms": 42,
        "extraction_ms": 870,
        "medical_ner_ms": 310
      },
      "blocks": [
        {
          "block_id": "p1_b1",
          "type": "paragraph",
          "text": "Patient reports chest pain for two days.",
          "source": "pdf_text",
          "confidence": 1.0,
          "bbox": null,
          "needs_review": false
        }
      ]
    }
  ],
  "medical_entities": [
    {
      "text": "chest pain",
      "type": "SYMPTOM",
      "page_number": 1,
      "block_id": "p1_b1",
      "confidence": 0.89
    }
  ],
  "medications": [],
  "labs": [],
  "warnings": [],
  "debug_artifacts": {
    "debug_folder": "./output/debug/report"
  }
}
```

---

# 13. Quality Checks and Review Flags

Set `needs_review = true` when:

- OCR confidence is low.
- Table confidence is low.
- Form extraction confidence is low.
- Handwritten OCR confidence is low.
- Medication name confidence is low.
- Dose/frequency confidence is low.
- Page classification is unknown.
- Extraction failed.
- OCR output is empty or mostly garbage.
- Table rows/columns appear broken.

Recommended confidence policy:

```text
confidence >= 0.90 -> high confidence
confidence 0.70 to 0.90 -> usable but mark caution
confidence < 0.70 -> needs_review = true
```

For handwritten prescriptions, be stricter. If medication, dose, route, or frequency is uncertain, mark the block for review.

---

# 14. CLI Requirements

## Required arguments

```text
--input
--output
```

## Optional arguments

```text
--debug-dir
--save-debug-images
--enable-medical-ner
--device cpu/cuda
--config configs/local.yaml
```

## Example command

```bash
python -m medical_extraction.cli.extract_document \
  --input ./sample_docs/report.pdf \
  --output ./output/report_extraction.json \
  --debug-dir ./output/debug/report \
  --save-debug-images true \
  --enable-medical-ner true \
  --device cpu
```

---

# 15. Error Handling

The pipeline should handle:

- missing file
- unsupported file type
- corrupted PDF
- empty PDF
- page render failure
- OCR model failure
- table extraction failure
- form extraction failure
- GPU unavailable
- model download failure

A single page failure should not fail the whole document.

Instead, output a page-level error:

```json
{
  "page_number": 4,
  "page_type": "error",
  "error": "OCR failed on page 4",
  "blocks": [],
  "needs_review": true
}
```

---

# 16. Acceptance Criteria

The MVP is complete when:

1. It accepts a local PDF path.
2. It classifies every page.
3. It extracts copyable text.
4. It extracts digital tables where possible.
5. It detects embedded image regions and OCRs only crops.
6. It classifies image crops using DiT.
7. It processes scanned pages through Surya.
8. It processes form-like regions with Surya OCR + LayoutLMv3-FUNSD.
9. It processes handwritten prescription regions with TrOCR.
10. It runs medical NER if enabled.
11. It runs medication/lab parsers if enabled.
12. It saves structured extracted JSON locally.
13. It saves confidence scores and `needs_review` flags.
14. It does not run Presidio, redaction, chunking, embedding, Elasticsearch, or LLM calls.

---

# 17. Future S3/KMS Upgrade

The current local MVP should be designed so S3 can be added later by swapping storage adapters.

## Current local adapters

```text
LocalInputAdapter
- read PDF from local path

LocalOutputAdapter
- write extraction JSON to local path
- write debug images to local folder
```

## Future S3 adapters

```text
S3InputAdapter
- read PDF from s3://raw-documents/tenant/doc.pdf
- KMS decryption handled by AWS permissions/client

S3OutputAdapter
- write extracted JSON to s3://extracted-json/tenant/doc/extraction_v1.json
- write debug artifacts if enabled and allowed
```

The extraction pipeline should stay the same:

```text
input_adapter.read()
-> extraction_pipeline.run()
-> output_adapter.write()
```

Only the storage adapter changes.

Recommended future S3 paths:

```text
s3://secure-bucket/raw-documents/tenant_a/doc_456/original.pdf
s3://secure-bucket/extracted-json/tenant_a/doc_456/extraction_v1.json
s3://secure-bucket/extraction-debug/tenant_a/doc_456/
```

In production, debug images should be disabled by default because they may contain PHI.

---

# 18. Final Simple Extraction Flow

```text
Local PDF
-> PyMuPDF page classification
-> route each page

Copyable PDF:
  -> pdfplumber + Camelot

Copyable PDF with images:
  -> pdfplumber text
  -> PyMuPDF image crops
  -> DiT crop classifier
  -> Surya OCR / Surya table / LayoutLMv3 form / TrOCR handwriting

Fully scanned report/form/table:
  -> render page
  -> OpenCV/Pillow preprocessing
  -> Surya layout/OCR/table extraction
  -> LayoutLMv3 for forms when needed

Handwritten prescription:
  -> render page
  -> Surya layout
  -> crop prescription regions
  -> TrOCR handwriting OCR
  -> medication parser + confidence checks

All paths:
  -> structured extracted JSON
  -> medical NER/parsers if enabled
  -> quality checker
  -> local JSON output
```
