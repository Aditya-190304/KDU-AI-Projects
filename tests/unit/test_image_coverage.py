from medical_extraction.classification.image_coverage import compute_image_coverage


def test_small_images_are_ignored():
    coverage, count = compute_image_coverage(
        page_area=1_000_000.0,
        image_bboxes=[[0, 0, 10, 10], [0, 0, 200, 200]],
        tiny_image_area_ratio=0.02,
    )
    assert count == 1
    assert coverage == 0.04
