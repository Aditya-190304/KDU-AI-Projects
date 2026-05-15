from medical_extraction.privacy.redaction import ChunkRedactor


def test_redaction_masks_labeled_phi_and_builds_hmacs():
    redactor = ChunkRedactor(
        {
            "hmac_secret_env_var": "TEST_MEDICAL_RAG_HMAC_SECRET",
            "dev_fallback_secret": "unit-test-secret",
        }
    )
    chunk = {
        "chunk_id": "c1",
        "chunk_text": "Patient Name: John Doe UHID: UH12345 Phone: +91 99999 88888 DOB: 02/01/1988",
        "metadata": {},
    }

    redacted = redactor.redact_chunk(chunk)

    assert "[PATIENT_NAME]" in redacted["chunk_text"]
    assert "[UHID]" in redacted["chunk_text"]
    assert "[PHONE]" in redacted["chunk_text"]
    assert "[DATE_OF_BIRTH]" in redacted["chunk_text"]
    assert "patient_name_hmac" in redacted["metadata"]["identity_hmacs"]
    assert "uhid_hmac" in redacted["metadata"]["identity_hmacs"]


def test_redaction_masks_generic_email_and_dates():
    redactor = ChunkRedactor(
        {
            "hmac_secret_env_var": "TEST_MEDICAL_RAG_HMAC_SECRET",
            "dev_fallback_secret": "unit-test-secret",
        }
    )
    chunk = {
        "chunk_id": "c2",
        "chunk_text": "Follow up on 2026-05-13. Contact test@example.com.",
        "metadata": {},
    }

    redacted = redactor.redact_chunk(chunk)

    assert "[DATE]" in redacted["chunk_text"]
    assert "[EMAIL]" in redacted["chunk_text"]
    assert "date_hmacs" in redacted["metadata"]["identity_hmacs"]
    assert "email_hmacs" in redacted["metadata"]["identity_hmacs"]


def test_redaction_masks_plain_person_names_detected_by_presidio():
    redactor = ChunkRedactor(
        {
            "hmac_secret_env_var": "TEST_MEDICAL_RAG_HMAC_SECRET",
            "dev_fallback_secret": "unit-test-secret",
        }
    )
    chunk = {
        "chunk_id": "c3",
        "chunk_text": "Oliver Johnson was seen for follow up with Dr. Jennifer Kim.",
        "metadata": {},
    }

    redacted = redactor.redact_chunk(chunk)

    assert "Oliver Johnson" not in redacted["chunk_text"]
    assert "[PATIENT_NAME]" in redacted["chunk_text"]
    assert "patient_name_hmac" in redacted["metadata"]["identity_hmacs"]
