"""Create or verify the local DynamoDB audit table used by the admin page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medical_extraction.core.config import load_runtime_config
from medical_extraction.privacy.redaction import ChunkRedactor
from medical_extraction.storage.audit_store import DynamoAuditSettings, DynamoAuditStore
from medical_extraction.utils.env import load_env_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or verify the DynamoDB audit table.")
    parser.add_argument("--config", default="configs/local.yaml", help="Runtime config path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_env_file(PROJECT_ROOT / ".env")
    config = load_runtime_config(args.config)
    audit_config = config.get("audit", {})
    privacy_config = config.get("privacy", {})
    settings = DynamoAuditSettings.from_config(audit_config)
    store = DynamoAuditStore(settings=settings, redactor=ChunkRedactor(privacy_config))
    store.ensure_table()

    status = {
        "ok": True,
        "backend": settings.backend,
        "table_name": settings.table_name,
        "endpoint_url": settings.endpoint_url,
        "region": settings.region,
        "enabled": settings.enabled,
        "required": settings.required,
    }
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
