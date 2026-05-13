"""Report local model cache status for the extraction pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path


HF_MODELS = {
    "dit": "models--microsoft--dit-base-finetuned-rvlcdip",
    "layoutlmv3_form": "models--nielsr--layoutlmv3-finetuned-funsd",
    "biomedical_ner": "models--d4data--biomedical-ner-all",
}

SURYA_CACHE_DIRS = {
    "foundation(shared)": Path("text_recognition"),
    "text_detection": Path("text_detection"),
    "text_recognition": Path("text_recognition"),
    "layout": Path("layout"),
    "table_recognition": Path("table_recognition"),
}


def folder_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        if path.is_file():
            return path.stat().st_size
    except Exception:
        return 0
    total = 0
    for file_path in path.rglob("*"):
        try:
            if file_path.is_file():
                total += file_path.stat().st_size
        except Exception:
            continue
    return total


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    if size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.2f} MB"
    return f"{size_bytes / (1024**3):.2f} GB"


def newest_modified(path: Path) -> str:
    if not path.exists():
        return "-"
    if path.is_file():
        try:
            return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
        except Exception:
            return "-"
    latest = None
    for file_path in path.rglob("*"):
        try:
            if file_path.is_file():
                modified = file_path.stat().st_mtime
                latest = modified if latest is None else max(latest, modified)
        except Exception:
            continue
    if latest is None:
        return "-"
    return datetime.fromtimestamp(latest).isoformat(timespec="seconds")


def print_section(title: str) -> None:
    print(title)
    print("-" * len(title))


def print_model_row(name: str, path: Path) -> None:
    exists = path.exists()
    size = format_size(folder_size_bytes(path))
    modified = newest_modified(path)
    print(f"{name:18} present={str(exists):5} size={size:>8} path={path}")
    print(f"{'':18} last_modified={modified}")


def _resolve_surya_root() -> Path:
    try:
        from surya.settings import settings

        return Path(settings.MODEL_CACHE_DIR)
    except Exception:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "datalab" / "datalab" / "Cache" / "models"
        return Path.home() / ".cache" / "surya"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check download/cache status for local extraction models.")
    parser.parse_args()

    hf_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    surya_root = _resolve_surya_root()

    print_section("Hugging Face Cache")
    print(f"root={hf_root}")
    for name, folder_name in HF_MODELS.items():
        print_model_row(name, hf_root / folder_name)

    print()
    print_section("Ollama Models")
    model_name = os.environ.get("MEDICAL_HANDWRITING_OLLAMA_MODEL", "qwen2.5vl:3b")
    present = _ollama_model_present(model_name)
    print(f"{model_name:18} present={str(present):5} note=Used for handwritten OCR")
    print(f"{'':18} last_modified=-")

    print()
    print_section("Surya Cache")
    print(f"root={surya_root}")
    for alias, relative_path in SURYA_CACHE_DIRS.items():
        target = surya_root / relative_path
        if alias == "layout" and not target.exists():
            print(f"{alias:18} present={str(False):5} note=May be shared in current Surya version.")
            print(f"{'':18} last_modified=-")
            continue
        print_model_row(alias, target)

    print()
    print_section("Download Logs")
    log_dir = Path.cwd() / "output" / "logs"
    print(f"root={log_dir}")
    for log_name in ("download_models.out.log", "download_models.err.log"):
        print_model_row(log_name, log_dir / log_name)


def _ollama_model_present(model_name: str) -> bool:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    return model_name.lower() in result.stdout.lower()


if __name__ == "__main__":
    main()
