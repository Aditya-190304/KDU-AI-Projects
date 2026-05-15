from medical_extraction.extraction.copyable_pdf_extractor import CopyablePdfExtractor


class _FakeRect:
    height = 800.0


class _FakePage:
    rect = _FakeRect()

    def get_text(self, kind):
        if kind == "dict":
            return {
                "blocks": [
                    {
                        "type": 0,
                        "bbox": [20.0, 40.0, 200.0, 120.0],
                        "lines": [
                            {"spans": [{"text": "Patient Name:"}, {"text": "John Doe"}]},
                            {"spans": [{"text": "DOB: 02/01/1988"}]},
                        ],
                    },
                    {
                        "type": 1,
                        "bbox": [210.0, 40.0, 360.0, 180.0],
                    },
                    {
                        "type": 0,
                        "bbox": [20.0, 160.0, 360.0, 250.0],
                        "lines": [
                            {"spans": [{"text": "Assessment: Stable"}]},
                        ],
                    },
                ]
            }
        if kind == "text":
            return "Patient Name: John Doe\nDOB: 02/01/1988\n\nAssessment: Stable"
        return {}


def test_extract_text_blocks_uses_pdf_block_bboxes():
    extractor = CopyablePdfExtractor()
    blocks = extractor.extract_text_blocks(_FakePage(), "dummy.pdf", 1)
    assert len(blocks) == 2
    assert blocks[0]["bbox"] == [20.0, 40.0, 200.0, 120.0]
    assert blocks[1]["bbox"] == [20.0, 160.0, 360.0, 250.0]


def test_camelot_bbox_conversion_to_page_coordinates():
    extractor = CopyablePdfExtractor()
    bbox = extractor._camelot_bbox_to_page_bbox((100.0, 500.0, 300.0, 700.0), 800.0)
    assert bbox == [100.0, 100.0, 300.0, 300.0]
