"""Abstract storage interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class InputAdapter(ABC):
    @abstractmethod
    def validate(self, input_path: str) -> Path:
        raise NotImplementedError

    @abstractmethod
    def document_id(self, input_path: str) -> str:
        raise NotImplementedError


class OutputAdapter(ABC):
    @abstractmethod
    def write_result(self, output_path: str, payload: dict) -> None:
        raise NotImplementedError
