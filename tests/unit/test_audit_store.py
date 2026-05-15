from medical_extraction.privacy.redaction import ChunkRedactor
from medical_extraction.storage.audit_store import DynamoAuditSettings, DynamoAuditStore


class _FakeTable:
    def __init__(self) -> None:
        self.items = []

    def load(self) -> None:
        return

    def put_item(self, Item):
        self.items.append(dict(Item))

    def query(self, **kwargs):
        limit = int(kwargs.get("Limit", 10))
        cursor = kwargs.get("ExclusiveStartKey")
        ordered = sorted(self.items, key=lambda item: item["accessed_at"], reverse=True)
        start_index = 0
        if cursor:
            for index, item in enumerate(ordered):
                if item["tenant"] == cursor.get("tenant") and item["accessed_at"] == cursor.get("accessed_at"):
                    start_index = index + 1
                    break
        page = ordered[start_index : start_index + limit]
        last_evaluated_key = None
        if start_index + limit < len(ordered) and page:
            last = page[-1]
            last_evaluated_key = {"tenant": last["tenant"], "accessed_at": last["accessed_at"]}
        return {"Items": page, "LastEvaluatedKey": last_evaluated_key}


class _FakeResource:
    def __init__(self, table: _FakeTable) -> None:
        self._table = table

    def Table(self, _name: str):
        return self._table


def _build_store() -> DynamoAuditStore:
    table = _FakeTable()
    resource = _FakeResource(table)
    redactor = ChunkRedactor(
        {
            "hmac_secret_env_var": "TEST_MEDICAL_RAG_HMAC_SECRET",
            "dev_fallback_secret": "unit-test-secret",
        }
    )
    return DynamoAuditStore(
        settings=DynamoAuditSettings(
            enabled=True,
            required=False,
            backend="dynamodb",
            table_name="audit",
            region="us-east-1",
            endpoint_url="http://127.0.0.1:8000",
            tenant_key="audit",
            page_size=1,
        ),
        redactor=redactor,
        resource=resource,
    )


def test_audit_store_masks_query_and_document_refs():
    store = _build_store()

    store.log_access(
        actor_name="records-admin-01",
        actor_role="doctor",
        query_text="What is Oliver Johnson's MRN and phone number?",
        authorized=True,
        requested_document_id="Oliver_Johnson-MRN100005",
        context_chunks=[{"document_id": "Oliver_Johnson-MRN100005"}],
        status="success",
    )

    record = store.list_logs()["items"][0]
    assert record["actor_name"] == "records-admin-01"
    assert "[PERSON]" in record["query_masked"]
    assert "[ID]" in record["query_masked"] or "[CONTACT]" in record["query_masked"]
    assert record["document_refs"][0]["document_hash"]
    assert record["document_refs"][0]["document_label"]


def test_audit_store_paginates_with_cursor():
    store = _build_store()
    for index in range(3):
        store.log_access(
            actor_name=f"user-{index}",
            actor_role="receptionist",
            query_text=f"query {index}",
            authorized=False,
            requested_document_id=f"doc-{index}",
            context_chunks=[{"document_id": f"doc-{index}"}],
            status="success",
        )

    first_page = store.list_logs(page_size=1)
    second_page = store.list_logs(page_size=1, cursor=first_page["next_cursor"])

    assert len(first_page["items"]) == 1
    assert len(second_page["items"]) == 1
    assert first_page["items"][0]["audit_id"] != second_page["items"][0]["audit_id"]
