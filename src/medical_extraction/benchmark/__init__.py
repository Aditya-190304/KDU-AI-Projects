"""Benchmark helpers for OCR accuracy, retrieval accuracy, and processing time."""

from medical_extraction.benchmark.engine import (
    compute_ocr_accuracy,
    evaluate_answer,
    get_average_ocr_accuracy,
    get_benchmark_summary,
    get_processing_times,
    record_processing_time,
)

__all__ = [
    "compute_ocr_accuracy",
    "evaluate_answer",
    "get_average_ocr_accuracy",
    "get_benchmark_summary",
    "get_processing_times",
    "record_processing_time",
]
