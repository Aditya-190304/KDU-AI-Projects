from medical_extraction.parsing.lab_parser import parse_labs


def test_lab_parser_extracts_known_labs():
    text = "HbA1c 7.8 %\nGlucose 180 mg/dL"
    labs = parse_labs(text, page_number=1, block_id="p1_b1")
    assert len(labs) == 2
    assert labs[0]["test_name"].lower() == "hba1c"
    assert labs[1]["unit"] == "mg/dL"
