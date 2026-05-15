"""Run the local frontend + QA API server."""

from __future__ import annotations

import argparse

from medical_extraction.server.qa_server import run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local frontend and QA API server.")
    parser.add_argument("--config", default="configs/local.yaml", help="Runtime config path.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Execution device hint.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_server(config_path=args.config, host=args.host, port=args.port, device=args.device)


if __name__ == "__main__":
    main()
