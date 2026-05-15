"""Local filesystem adapters."""

from __future__ import annotations

from pathlib import Path

from medical_extraction.core.exceptions import UnsupportedFileTypeError
from medical_extraction.storage.base import InputAdapter, OutputAdapter
from medical_extraction.utils.json_utils import write_json


class LocalInputAdapter(InputAdapter):
    SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

    def validate(self, input_path: str) -> Path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise UnsupportedFileTypeError(f"Unsupported file type: {path.suffix}")
        return path

    def document_id(self, input_path: str) -> str:
        return Path(input_path).stem


class LocalOutputAdapter(OutputAdapter):
    def write_result(self, output_path: str, payload: dict) -> None:
        write_json(output_path, payload)
