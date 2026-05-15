"""Masked query audit logging backed by DynamoDB Local or DynamoDB."""

from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
import hashlib
import hmac
import json
from typing import Any
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key

from medical_extraction.answering.prompting import normalize_generic_redactions
from medical_extraction.privacy.redaction import ChunkRedactor


@dataclass(slots=True)
class DynamoAuditSettings:
    enabled: bool
    required: bool
    backend: str
    table_name: str
    region: str
    endpoint_url: str
    tenant_key: str
    page_size: int

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "DynamoAuditSettings":
        payload = config or {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            required=bool(payload.get("required", False)),
            backend=str(payload.get("backend", "dynamodb")).strip().lower() or "dynamodb",
            table_name=str(payload.get("table_name", "medical-rag-access-audit")).strip() or "medical-rag-access-audit",
            region=str(payload.get("region", "us-east-1")).strip() or "us-east-1",
            endpoint_url=str(payload.get("endpoint_url", "http://127.0.0.1:8000")).strip(),
            tenant_key=str(payload.get("tenant_key", "audit")).strip() or "audit",
            page_size=max(1, int(payload.get("page_size", 10))),
        )


class DynamoAuditStore:
    def __init__(
        self,
        settings: DynamoAuditSettings,
        redactor: ChunkRedactor,
        resource: Any | None = None,
    ) -> None:
        self.settings = settings
        self.redactor = redactor
        self._resource = resource

    def ensure_table(self) -> None:
        if not self.settings.enabled:
            return
        table = self._table()
        try:
            table.load()
            return
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code != "ResourceNotFoundException":
                raise

        resource = self._dynamodb()
        resource.create_table(
            TableName=self.settings.table_name,
            KeySchema=[
                {"AttributeName": "tenant", "KeyType": "HASH"},
                {"AttributeName": "accessed_at", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "tenant", "AttributeType": "S"},
                {"AttributeName": "accessed_at", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        resource.meta.client.get_waiter("table_exists").wait(TableName=self.settings.table_name)

    def log_access(
        self,
        *,
        actor_name: str,
        actor_role: str,
        query_text: str,
        authorized: bool,
        requested_document_id: str | None,
        context_chunks: list[dict[str, Any]] | None,
        status: str,
        error_message: str | None = None,
    ) -> None:
        if not self.settings.enabled:
            return

        now = datetime.now(UTC)
        audit_id = uuid4().hex
        accessed_sort_key = f"{now.isoformat()}#{audit_id}"
        document_refs = self._collect_document_refs(requested_document_id, context_chunks or [])
        item = {
            "tenant": self.settings.tenant_key,
            "accessed_at": accessed_sort_key,
            "accessed_day": now.date().isoformat(),
            "audit_id": audit_id,
            "actor_name": str(actor_name or "unknown").strip() or "unknown",
            "actor_role": str(actor_role or "unknown").strip().lower() or "unknown",
            "authorized": bool(authorized),
            "status": str(status or "unknown").strip().lower() or "unknown",
            "query_masked": self._mask_text(query_text),
            "query_hash": self._digest_value(query_text),
            "document_count": len(document_refs),
            "document_refs_json": json.dumps(document_refs, sort_keys=True),
            "error_masked": self._mask_text(error_message or ""),
        }
        self._table().put_item(Item=item)

    def list_logs(
        self,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        page_size: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        if not self.settings.enabled:
            return {"items": [], "next_cursor": None}

        limit = max(1, int(page_size or self.settings.page_size))
        table = self._table()
        key_condition = Key("tenant").eq(self.settings.tenant_key)
        range_start, range_end = self._build_date_range(date_from, date_to)
        if range_start and range_end:
            key_condition = key_condition & Key("accessed_at").between(range_start, range_end)

        query_kwargs: dict[str, Any] = {
            "KeyConditionExpression": key_condition,
            "ScanIndexForward": False,
            "Limit": limit,
        }
        decoded_cursor = self._decode_cursor(cursor)
        if decoded_cursor:
            query_kwargs["ExclusiveStartKey"] = decoded_cursor

        response = table.query(**query_kwargs)
        items = [self._to_public_record(item) for item in response.get("Items", [])]
        return {
            "items": items,
            "next_cursor": self._encode_cursor(response.get("LastEvaluatedKey")),
        }

    def _collect_document_refs(self, requested_document_id: str | None, context_chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
        seen: set[str] = set()
        refs: list[dict[str, str]] = []

        def _add(document_id: str) -> None:
            normalized = str(document_id or "").strip()
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            refs.append(
                {
                    "document_hash": self._digest_value(normalized),
                    "document_label": self._mask_document_label(normalized),
                }
            )

        if requested_document_id:
            _add(requested_document_id)
        for chunk in context_chunks:
            _add(str(chunk.get("document_id", "")).strip())
        return refs

    def _mask_document_label(self, document_id: str) -> str:
        normalized = reformat_identifier(document_id)
        return self._mask_text(normalized)

    def _mask_text(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        redacted = self.redactor.redact_chunk(
            {
                "chunk_id": "audit",
                "chunk_text": raw,
                "metadata": {},
            }
        )
        normalized = normalize_generic_redactions(str(redacted.get("chunk_text", "")).strip())
        normalized = normalized.replace("[PHONE]", "[CONTACT]").replace("[EMAIL]", "[CONTACT]")
        return normalized

    def _digest_value(self, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return ""
        digest = hmac.new(self.redactor.secret.encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256)
        return digest.hexdigest()

    def _build_date_range(self, date_from: str | None, date_to: str | None) -> tuple[str | None, str | None]:
        if not date_from and not date_to:
            return None, None
        lower_date = self._parse_date(date_from) if date_from else date.min
        upper_date = self._parse_date(date_to) if date_to else date.max
        lower = datetime.combine(lower_date, time.min, tzinfo=UTC).isoformat()
        upper = f"{datetime.combine(upper_date, time.max, tzinfo=UTC).isoformat()}\uffff"
        return lower, upper

    def _parse_date(self, value: str | None) -> date:
        parsed = date.fromisoformat(str(value or "").strip())
        return parsed

    def _encode_cursor(self, key: dict[str, Any] | None) -> str | None:
        if not key:
            return None
        payload = json.dumps(key, sort_keys=True).encode("utf-8")
        return urlsafe_b64encode(payload).decode("utf-8")

    def _decode_cursor(self, cursor: str | None) -> dict[str, Any] | None:
        if not cursor:
            return None
        try:
            payload = urlsafe_b64decode(cursor.encode("utf-8"))
            decoded = json.loads(payload.decode("utf-8"))
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            return None
        return None

    def _to_public_record(self, item: dict[str, Any]) -> dict[str, Any]:
        raw_refs = str(item.get("document_refs_json", "")).strip()
        try:
            document_refs = json.loads(raw_refs) if raw_refs else []
        except json.JSONDecodeError:
            document_refs = []
        return {
            "audit_id": str(item.get("audit_id", "")).strip(),
            "accessed_at": str(item.get("accessed_at", "")).split("#", 1)[0],
            "accessed_day": str(item.get("accessed_day", "")).strip(),
            "actor_name": str(item.get("actor_name", "")).strip(),
            "actor_role": str(item.get("actor_role", "")).strip(),
            "authorized": bool(item.get("authorized", False)),
            "status": str(item.get("status", "")).strip(),
            "query_masked": str(item.get("query_masked", "")).strip(),
            "query_hash": str(item.get("query_hash", "")).strip(),
            "document_count": int(item.get("document_count", 0) or 0),
            "document_refs": document_refs if isinstance(document_refs, list) else [],
            "error_masked": str(item.get("error_masked", "")).strip(),
        }

    def _dynamodb(self) -> Any:
        if self._resource is None:
            resource_kwargs: dict[str, Any] = {
                "service_name": "dynamodb",
                "region_name": self.settings.region,
                "aws_access_key_id": "local",
                "aws_secret_access_key": "local",
            }
            if self.settings.endpoint_url:
                resource_kwargs["endpoint_url"] = self.settings.endpoint_url
            self._resource = boto3.resource(**resource_kwargs)
        return self._resource

    def _table(self) -> Any:
        return self._dynamodb().Table(self.settings.table_name)


class NoOpAuditStore:
    def ensure_table(self) -> None:  # pragma: no cover - trivial
        return

    def log_access(self, **_: Any) -> None:  # pragma: no cover - trivial
        return

    def list_logs(self, **_: Any) -> dict[str, Any]:  # pragma: no cover - trivial
        return {"items": [], "next_cursor": None}


def reformat_identifier(value: str) -> str:
    return str(value or "").replace("_", " ").replace("-", " ")
