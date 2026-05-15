"""S3 helpers and adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import boto3

from medical_extraction.storage.base import InputAdapter, OutputAdapter


class S3InputAdapter(InputAdapter):
    def validate(self, input_path: str):
        raise NotImplementedError("S3 input support is planned but not implemented in the local MVP.")

    def document_id(self, input_path: str) -> str:
        raise NotImplementedError("S3 input support is planned but not implemented in the local MVP.")


class S3OutputAdapter(OutputAdapter):
    def write_result(self, output_path: str, payload: dict) -> None:
        raise NotImplementedError("S3 output support is planned but not implemented in the local MVP.")


@dataclass(slots=True)
class S3Settings:
    enabled: bool
    bucket: str
    region: str
    profile: str = ""
    kms_key_id: str = ""
    prefix: str = "documents"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "S3Settings":
        return cls(
            enabled=bool(config.get("enabled", False)),
            bucket=str(config.get("bucket", "")).strip(),
            region=str(config.get("region", "ap-southeast-1")).strip(),
            profile=str(config.get("profile", "")).strip(),
            kms_key_id=str(config.get("kms_key_id", "")).strip(),
            prefix=str(config.get("prefix", "documents")).strip().strip("/"),
        )


class S3ArtifactStore:
    def __init__(self, settings: S3Settings) -> None:
        self.settings = settings
        self._client = None

    def build_key(self, document_id: str, *parts: str) -> str:
        clean_parts = [self.settings.prefix, document_id.strip("/")]
        clean_parts.extend(part.strip("/") for part in parts if part and part.strip("/"))
        return "/".join(clean_parts)

    def put_json(self, key: str, payload: Any) -> str:
        body = json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8")
        self._put_object(key, body, content_type="application/json")
        return self._uri(key)

    def put_text(self, key: str, text: str) -> str:
        self._put_object(key, text.encode("utf-8"), content_type="text/plain; charset=utf-8")
        return self._uri(key)

    def get_text(self, key: str) -> str:
        if not self.settings.enabled:
            raise RuntimeError("S3 storage is disabled in configuration.")
        response = self._client_for_use().get_object(Bucket=self.settings.bucket, Key=key)
        body = response["Body"].read()
        return body.decode("utf-8")

    def get_text_from_uri(self, uri: str) -> str:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
            raise ValueError(f"Invalid S3 URI: {uri}")
        bucket = parsed.netloc
        if bucket != self.settings.bucket:
            raise ValueError(
                f"S3 URI bucket '{bucket}' does not match configured bucket '{self.settings.bucket}'."
            )
        key = parsed.path.lstrip("/")
        return self.get_text(key)

    def _put_object(self, key: str, body: bytes, content_type: str) -> None:
        if not self.settings.enabled:
            raise RuntimeError("S3 storage is disabled in configuration.")
        extra_args = {
            "Bucket": self.settings.bucket,
            "Key": key,
            "Body": body,
            "ContentType": content_type,
        }
        if self.settings.kms_key_id:
            extra_args["ServerSideEncryption"] = "aws:kms"
            extra_args["SSEKMSKeyId"] = self.settings.kms_key_id
        else:
            extra_args["ServerSideEncryption"] = "AES256"
        self._client_for_use().put_object(**extra_args)

    def _client_for_use(self):
        if self._client is not None:
            return self._client
        session = boto3.Session(
            profile_name=self.settings.profile or None,
            region_name=self.settings.region or None,
        )
        self._client = session.client("s3")
        return self._client

    def _uri(self, key: str) -> str:
        return f"s3://{self.settings.bucket}/{key}"
