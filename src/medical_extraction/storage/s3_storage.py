"""Future S3 adapters."""

from __future__ import annotations

from medical_extraction.storage.base import InputAdapter, OutputAdapter


class S3InputAdapter(InputAdapter):
    def validate(self, input_path: str):
        raise NotImplementedError("S3 input support is planned but not implemented in the local MVP.")

    def document_id(self, input_path: str) -> str:
        raise NotImplementedError("S3 input support is planned but not implemented in the local MVP.")


class S3OutputAdapter(OutputAdapter):
    def write_result(self, output_path: str, payload: dict) -> None:
        raise NotImplementedError("S3 output support is planned but not implemented in the local MVP.")
