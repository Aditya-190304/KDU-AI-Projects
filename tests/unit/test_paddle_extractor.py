from medical_extraction.models.paddle_extractor import PaddleExtractor


def test_merge_bbox_offsets_relative_line_bbox():
    extractor = PaddleExtractor.__new__(PaddleExtractor)
    merged = extractor._merge_bbox([0.0, 200.0, 100.0, 300.0], [10.0, 15.0, 80.0, 40.0])
    assert merged == [10.0, 215.0, 80.0, 240.0]
