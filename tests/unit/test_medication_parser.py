from medical_extraction.parsing.medication_parser import (
    clean_prescription_line,
    has_medication_anchor,
    is_footer_noise_line,
    looks_like_continuation_line,
    parse_medications,
)


def test_medication_parser_extracts_dose_and_frequency():
    text = "Metformin 500mg twice daily"
    items = parse_medications(text, page_number=1, block_id="p1_b1", confidence=0.85)
    assert len(items) == 1
    assert items[0]["medication"] == "Metformin"
    assert items[0]["dose"] == "500mg"
    assert items[0]["frequency"].lower() == "twice daily"
    assert items[0]["needs_review"] is False


def test_medication_parser_extracts_form_and_duration():
    text = "Tab. Augmentin 625 mg BD x 5 days"
    items = parse_medications(text, page_number=1, block_id="p1_b2", confidence=0.82)
    assert len(items) == 1
    assert items[0]["medication"] == "Augmentin"
    assert items[0]["form"] == "Tab"
    assert items[0]["duration"].lower() == "5 days"


def test_footer_noise_is_skipped():
    assert is_footer_noise_line("www.thewhitetusk.com Email info@thewhitetusk.com") is True
    assert parse_medications("www.thewhitetusk.com", page_number=1, block_id="p1_b3") == []


def test_cleanup_normalizes_common_ocr_noise():
    cleaned = clean_prescription_line('Adv: Hexigel grum paint<br>massage')
    assert "gum paint massage" in cleaned.lower()


def test_cleanup_normalizes_math_markup_and_spaced_frequency():
    cleaned = clean_prescription_line(r"Tab. Enzoflarn <math display=block>1 - 0 - I \times 5days</math>")
    assert "1-0-1" in cleaned
    assert "x 5 days" in cleaned.lower()


def test_anchor_and_continuation_detection():
    assert has_medication_anchor("Tab. Augmentin 625 mg") is True
    assert looks_like_continuation_line("x I week") is True
    assert looks_like_continuation_line("paint") is True


def test_parser_skips_narrative_certificate_text():
    text = "He is currently undergoing treatment in ICU for severe sepsis with MOBS"
    assert parse_medications(text, page_number=1, block_id="p1_b4") == []


def test_parser_skips_signature_and_hospital_lines():
    text = "NAVEEN POLAVARAPU MRCP (Lond), MRCP (Edin), Apollo Hospitals Jubilee Hills"
    assert parse_medications(text, page_number=1, block_id="p1_b5") == []
