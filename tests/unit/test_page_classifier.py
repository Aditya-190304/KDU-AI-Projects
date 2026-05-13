from medical_extraction.classification.page_classifier import PageClassifier


class _FakeRect:
    width = 100.0
    height = 100.0


class _FakePage:
    def __init__(self, text: str, blocks: list[dict]) -> None:
        self.number = 0
        self.rect = _FakeRect()
        self._text = text
        self._blocks = blocks

    def get_text(self, mode: str):
        if mode == "text":
            return self._text
        if mode == "dict":
            return {"blocks": self._blocks}
        raise ValueError(mode)


def test_handwritten_file_hint_routes_scanned_page_to_prescription():
    classifier = PageClassifier()
    page = _FakePage(
        text="",
        blocks=[{"type": 1, "bbox": [0, 0, 95, 95]}],
    )
    result = classifier.classify(page, input_path="C:\\docs\\HandWritten_D4-Blur.pdf")
    assert result.page_class == "handwritten_scanned_prescription"


def test_prescription_markers_route_image_page_to_prescription():
    classifier = PageClassifier()
    page = _FakePage(
        text="CC: fever\nAdv: remdec 2000 mg stat",
        blocks=[{"type": 1, "bbox": [0, 0, 95, 95]}],
    )
    result = classifier.classify(page, input_path="C:\\docs\\scan.pdf")
    assert result.page_class == "handwritten_scanned_prescription"
