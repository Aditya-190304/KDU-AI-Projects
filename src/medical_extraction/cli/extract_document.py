"""CLI for the local medical extraction MVP."""

from __future__ import annotations

import argparse
from pathlib import Path

from medical_extraction.core.config import load_runtime_config
from medical_extraction.core.pipeline import ExtractionPipeline
from medical_extraction.utils.env import load_env_file


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract structured content from a medical PDF or image.")
    parser.add_argument("--input", required=True, help="Path to the input PDF/image file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. If omitted, writes plain text to --text-output-dir/<input_stem>.txt.",
    )
    parser.add_argument(
        "--text-only",
        type=parse_bool,
        default=True,
        help="Write only plain-text extraction (default: true). Set false to also write JSON when output is .json.",
    )
    parser.add_argument(
        "--text-output-dir",
        default="output/text",
        help="Directory used for text outputs when --output is not provided.",
    )
    parser.add_argument("--debug-dir", default=None, help="Directory for optional debug artifacts.")
    parser.add_argument("--save-debug-images", type=parse_bool, default=False, help="Save page/crop renders.")
    parser.add_argument("--enable-medical-ner", type=parse_bool, default=False, help="Enable medical NER.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Execution device hint.")
    parser.add_argument("--config", default=None, help="Optional YAML config override.")
    return parser


def main() -> None:
    load_env_file(Path(".env"))
    args = build_parser().parse_args()
    config = load_runtime_config(args.config)
    config["pipeline"]["device"] = args.device
    config["pipeline"]["save_debug_images"] = args.save_debug_images
    config["pipeline"]["enable_medical_ner"] = args.enable_medical_ner

    output_path = args.output
    if not output_path:
        output_path = str(Path(args.text_output_dir) / f"{Path(args.input).stem}.txt")
        args.text_only = True

    output_suffix = Path(output_path).suffix.lower()
    if output_suffix == ".txt":
        args.text_only = True

    pipeline = ExtractionPipeline(config=config)
    pipeline.run(
        input_path=args.input,
        output_path=output_path,
        debug_dir=args.debug_dir,
        save_debug_images=args.save_debug_images,
        enable_medical_ner=args.enable_medical_ner,
        text_only=args.text_only,
    )


if __name__ == "__main__":
    main()
