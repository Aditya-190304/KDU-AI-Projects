"""Shared constants for the extraction pipeline."""

EXTRACTION_VERSION = "local_mvp_v1"
DEFAULT_ROUTE = "unknown_or_hybrid"
PAGE_CLASS_COPYABLE = "copyable_pdf"
PAGE_CLASS_MIXED = "copyable_pdf_with_images"
PAGE_CLASS_SCANNED = "fully_scanned_report_form_table"
PAGE_CLASS_HANDWRITTEN = "handwritten_scanned_prescription"
PAGE_CLASS_UNKNOWN = "unknown_or_hybrid"
PAGE_CLASS_ERROR = "error"

DEFAULT_THRESHOLDS = {
    "min_selectable_text_chars": 100,
    "min_alpha_ratio": 0.45,
    "max_weird_char_ratio": 0.25,
    "max_duplicate_line_ratio": 0.40,
    "tiny_image_area_ratio": 0.02,
    "ocr_candidate_image_area_ratio": 0.10,
    "scanned_image_coverage_ratio": 0.70,
    "handwritten_review_threshold": 0.70,
    "default_review_threshold": 0.70,
    "caution_threshold": 0.90,
}

PRESCRIPTION_HINTS = {
    "rx",
    "prescription",
    "tablet",
    "capsule",
    "syrup",
    "tab",
    "handwritten",
    "script",
}

FORM_FIELD_HINTS = {
    "name",
    "dob",
    "date",
    "mrn",
    "id",
    "symptoms",
    "diagnosis",
}
