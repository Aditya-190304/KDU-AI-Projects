"""Configuration loading helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency path
    yaml = None

from medical_extraction.core.constants import (
    DEFAULT_AUDIT_CONFIG,
    DEFAULT_CHROMA_CONFIG,
    DEFAULT_CHUNKING_CONFIG,
    DEFAULT_INDEXING_CONFIG,
    DEFAULT_KEYWORD_INDEX_CONFIG,
    DEFAULT_PRIVACY_CONFIG,
    DEFAULT_RETRIEVAL_CONFIG,
    DEFAULT_STORAGE_CONFIG,
    DEFAULT_THRESHOLDS,
)


def load_runtime_config(config_path: str | None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "pipeline": {},
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "chroma": deepcopy(DEFAULT_CHROMA_CONFIG),
        "keyword_index": deepcopy(DEFAULT_KEYWORD_INDEX_CONFIG),
        "chunking": deepcopy(DEFAULT_CHUNKING_CONFIG),
        "storage": deepcopy(DEFAULT_STORAGE_CONFIG),
        "privacy": deepcopy(DEFAULT_PRIVACY_CONFIG),
        "indexing": deepcopy(DEFAULT_INDEXING_CONFIG),
        "retrieval": deepcopy(DEFAULT_RETRIEVAL_CONFIG),
        "audit": deepcopy(DEFAULT_AUDIT_CONFIG),
    }
    default_path = Path("configs/default.yaml")
    if default_path.exists() and yaml is not None:
        _merge_loaded_config(config, _read_yaml(default_path))

    if config_path:
        if yaml is None:
            raise RuntimeError("PyYAML is required when using --config.")
        _merge_loaded_config(config, _read_yaml(Path(config_path)))
    return config


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Config file must contain a mapping: {path}")
    return loaded


def _merge_loaded_config(config: dict[str, Any], loaded: dict[str, Any]) -> None:
    for section in ("pipeline", "thresholds", "chroma", "keyword_index", "chunking", "storage", "privacy", "indexing", "retrieval", "audit"):
        section_payload = loaded.get(section)
        if isinstance(section_payload, dict):
            _deep_merge_dict(config[section], section_payload)


def _deep_merge_dict(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge_dict(target[key], value)
            continue
        target[key] = value
