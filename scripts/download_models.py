"""Prefetch model weights used by the local extraction pipeline."""

from __future__ import annotations

import argparse
import os

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForTokenClassification,
    AutoProcessor,
    AutoTokenizer,
)


HF_MODELS = {
    "dit": "microsoft/dit-base-finetuned-rvlcdip",
    "layoutlmv3_form": "nielsr/layoutlmv3-finetuned-funsd",
    "biomedical_ner": "d4data/biomedical-ner-all",
}


def _huggingface_model_cached(model_id: str) -> bool:
    from pathlib import Path

    hf_root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    folder = "models--" + model_id.replace("/", "--")
    return (hf_root / folder).exists()


def _load_hf(loader, model_id: str, prefer_cache: bool = False):
    if prefer_cache:
        return loader.from_pretrained(model_id, local_files_only=True)
    try:
        return loader.from_pretrained(model_id)
    except Exception:
        if _huggingface_model_cached(model_id):
            print(f"Network fetch failed for {model_id}; using local cache.")
            return loader.from_pretrained(model_id, local_files_only=True)
        raise


def download_huggingface_models(prefer_cache: bool = False) -> None:
    print("Downloading DiT...")
    _load_hf(AutoImageProcessor, HF_MODELS["dit"], prefer_cache=prefer_cache)
    _load_hf(AutoModelForImageClassification, HF_MODELS["dit"], prefer_cache=prefer_cache)

    print("Downloading LayoutLMv3 FUNSD...")
    _load_hf(AutoProcessor, HF_MODELS["layoutlmv3_form"], prefer_cache=prefer_cache)
    _load_hf(AutoModelForTokenClassification, HF_MODELS["layoutlmv3_form"], prefer_cache=prefer_cache)

    print("Downloading biomedical NER...")
    _load_hf(AutoTokenizer, HF_MODELS["biomedical_ner"], prefer_cache=prefer_cache)
    _load_hf(AutoModelForTokenClassification, HF_MODELS["biomedical_ner"], prefer_cache=prefer_cache)


def download_surya_models() -> None:
    print("Downloading Surya OCR/layout/table models...")
    try:
        from surya.detection import DetectionPredictor
        from surya.layout import LayoutPredictor
        from surya.recognition import RecognitionPredictor
        from surya.settings import settings
        from surya.table_rec import TableRecPredictor
    except Exception as exc:
        raise RuntimeError("surya-ocr is not installed") from exc

    sample = Image.new("RGB", (1200, 1600), color="white")
    det = DetectionPredictor()
    rec = RecognitionPredictor()
    layout = LayoutPredictor()
    table = TableRecPredictor()

    det([sample])
    rec([sample], det_predictor=det, return_words=True)
    layout([sample])
    table([sample])
    print(f"Surya cache root: {settings.MODEL_CACHE_DIR}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download model weights for local medical extraction.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Model download/init device.")
    parser.add_argument(
        "--disable-symlink-warning",
        action="store_true",
        help="Silence Hugging Face symlink warnings on Windows.",
    )
    parser.add_argument(
        "--prefer-cache",
        action="store_true",
        help="Use only local Hugging Face cache (no network checks).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.disable_symlink_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TORCH_DEVICE"] = "cuda" if args.device == "cuda" else "cpu"

    download_huggingface_models(prefer_cache=args.prefer_cache)
    download_surya_models()
    print("Model downloads initialized.")


if __name__ == "__main__":
    main()
