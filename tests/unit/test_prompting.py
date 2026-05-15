from medical_extraction.answering.prompting import (
    build_authorized_system_prompt,
    build_question_instruction,
    build_unauthorized_system_prompt,
    normalize_generic_redactions,
    question_is_prescription_focused,
)


def test_unauthorized_prompt_mentions_generic_placeholders():
    prompt = build_unauthorized_system_prompt()
    assert "[PERSON]" in prompt
    assert "[DATE]" in prompt
    assert "[CONTACT]" in prompt
    assert "[ID]" in prompt
    assert "If any personal health information appears unmasked" in prompt
    assert "uppercase bracketed masking tags" in prompt
    assert "[ADDRESS]" in prompt
    assert "Use your best judgment" in prompt
    assert "available-but-redacted, not absent" in prompt
    assert "DOB is [DATE]" in prompt


def test_authorized_prompt_mentions_sources():
    prompt = build_authorized_system_prompt()
    assert "Sources:" in prompt


def test_normalize_generic_redactions_maps_specific_placeholders():
    text = "Patient Name: [PATIENT_NAME] on [DATE] called [PHONE] with MRN [MRN]"
    normalized = normalize_generic_redactions(text)
    assert "[PERSON]" in normalized
    assert "[DATE]" in normalized
    assert "[CONTACT]" in normalized
    assert "[ID]" in normalized


def test_prescription_question_detection_and_instruction():
    question = "What medications and prescription instructions were given?"
    assert question_is_prescription_focused(question) is True
    instruction = build_question_instruction(question, authorized=True)
    assert "every prescribed medication" in instruction
    assert "Do not stop after the first medication" in instruction
