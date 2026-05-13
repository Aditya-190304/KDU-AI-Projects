"""Runtime helpers for model-backed components."""

from __future__ import annotations

import os

import torch


def resolve_torch_device(device: str | None = None) -> str:
    requested = (device or os.environ.get("MEDICAL_EXTRACTION_DEVICE") or "cpu").lower()
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def inference_device_index(device: str) -> int:
    return 0 if device == "cuda" and torch.cuda.is_available() else -1
