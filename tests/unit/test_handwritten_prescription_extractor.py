from medical_extraction.extraction.handwritten_prescription_extractor import HandwrittenPrescriptionExtractor


class _FakeScannedOcr:
    def ocr_image(self, _image):
        return {
            "lines": [
                {"text": "THE WHITE TUSK", "bbox": [10, 10, 200, 40], "confidence": 0.95, "words": []},
                {"text": "Tab. Remdesivir 100 mg", "bbox": [10, 120, 260, 160], "confidence": 0.9, "words": []},
                {"text": "BD x 4 days", "bbox": [10, 170, 180, 210], "confidence": 0.88, "words": []},
                {"text": "www.example.com", "bbox": [10, 360, 220, 390], "confidence": 0.99, "words": []},
            ]
        }


class _FakeHandwritingOcr:
    def __init__(self):
        self.calls = 0

    def extract_text(self, _image):
        self.calls += 1
        if self.calls == 1:
            return "Tab. Remdesivir 100 mg", 0.82
        return "BD x 4 days", 0.8


class _FakeRegistry:
    def __init__(self):
        self.scanned_ocr = _FakeScannedOcr()
        self.handwriting_ocr = _FakeHandwritingOcr()


class _FakeImage:
    width = 400
    height = 400

    def crop(self, _bbox):
        return self


def test_linewise_handwriting_fallback_uses_detected_lines():
    extractor = HandwrittenPrescriptionExtractor(_FakeRegistry())
    header, body, footer = extractor._segment_page_lines(_FakeScannedOcr().ocr_image(None)["lines"], 400)
    assert len(header) == 1
    assert len(body) == 2
    assert len(footer) == 1


def test_region_ocr_builds_prescription_or_paragraph_blocks():
    extractor = HandwrittenPrescriptionExtractor(_FakeRegistry())
    blocks = extractor._build_prescription_blocks(
        _FakeImage(),
        [
            {"text": "Tab. Remdesivir 100 mg", "bbox": [10, 120, 260, 160], "confidence": 0.9, "words": []},
            {"text": "BD x 4 days", "bbox": [10, 170, 180, 210], "confidence": 0.88, "words": []},
        ],
        page_number=1,
    )
    assert blocks
    assert blocks[0]["source"] in {"qwen_vision_region_ocr", "surya_column_ocr"}


def test_merge_body_lines_combines_short_continuations():
    extractor = HandwrittenPrescriptionExtractor(_FakeRegistry())
    merged = extractor._merge_body_lines(
        [
            {"text": "Tab. Remdesivir 100 mg", "bbox": [10, 120, 260, 160], "confidence": 0.9, "words": []},
            {"text": "BD x 4 days", "bbox": [12, 166, 180, 205], "confidence": 0.88, "words": []},
            {"text": "paint", "bbox": [16, 210, 100, 235], "confidence": 0.7, "words": []},
            {"text": "Adv: Hexigel gum massage", "bbox": [18, 238, 260, 275], "confidence": 0.92, "words": []},
        ],
        400,
        400,
    )
    assert len(merged) == 2
    assert "BD x 4 days" in merged[0]["text"]
    assert "paint Adv: Hexigel gum massage" in merged[1]["text"]


def test_merge_body_lines_does_not_cross_columns():
    extractor = HandwrittenPrescriptionExtractor(_FakeRegistry())
    merged = extractor._merge_body_lines(
        [
            {"text": "Tab. Cefixime 200 mg", "bbox": [10, 120, 220, 155], "confidence": 0.9, "words": []},
            {"text": "BD x 5 days", "bbox": [12, 160, 180, 190], "confidence": 0.88, "words": []},
            {"text": "Tab. Pantop 40 mg", "bbox": [280, 118, 390, 150], "confidence": 0.91, "words": []},
            {"text": "OD x 7 days", "bbox": [284, 158, 370, 188], "confidence": 0.89, "words": []},
        ],
        420,
        400,
    )
    assert len(merged) == 2
    assert "Cefixime" in merged[0]["text"]
    assert "Pantop" in merged[1]["text"]


def test_merge_body_lines_requires_strict_x_overlap():
    extractor = HandwrittenPrescriptionExtractor(_FakeRegistry())
    merged = extractor._merge_body_lines(
        [
            {"text": "Tab. Enzoflam 500 mg", "bbox": [10, 120, 220, 155], "confidence": 0.9, "words": []},
            {"text": "BD x 5 days", "bbox": [160, 158, 340, 188], "confidence": 0.88, "words": []},
        ],
        420,
        400,
    )
    assert len(merged) == 2


def test_column_region_bbox_expands_block():
    extractor = HandwrittenPrescriptionExtractor(_FakeRegistry())
    bbox = extractor._column_region_bbox(
        [{"bbox": [100, 120, 200, 150]}, {"bbox": [105, 160, 210, 190]}],
        400,
        400,
    )
    assert bbox is not None
    assert bbox[0] <= 100 and bbox[1] <= 120
    assert bbox[2] >= 210 and bbox[3] >= 190


class _NarrativeHandwritingOcr:
    def extract_text(self, _image):
        return "This is to inform that the patient is under my care and needs hospital stay", 0.84


class _NarrativeRegistry:
    def __init__(self):
        self.scanned_ocr = _FakeScannedOcr()
        self.handwriting_ocr = _NarrativeHandwritingOcr()


def test_region_ocr_keeps_narrative_columns_as_paragraphs():
    extractor = HandwrittenPrescriptionExtractor(_NarrativeRegistry())
    blocks = extractor._build_prescription_blocks(
        _FakeImage(),
        [
            {
                "text": "This is to inform that the patient is under my care",
                "bbox": [10, 120, 360, 150],
                "confidence": 0.84,
                "words": [],
            },
            {
                "text": "and needs hospital stay for complete recovery",
                "bbox": [10, 156, 360, 188],
                "confidence": 0.84,
                "words": [],
            }
        ],
        page_number=1,
    )
    assert blocks
    assert all(block["type"] == "paragraph" for block in blocks)
    assert "under my care\nand needs hospital stay" in blocks[0]["text"].lower()
