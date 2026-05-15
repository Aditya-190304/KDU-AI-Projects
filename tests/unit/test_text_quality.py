from medical_extraction.classification.text_quality import analyze_text_quality, looks_readable


def test_good_text_is_readable():
    text = "Patient reports chest pain for two days.\nBlood pressure is stable."
    metrics = analyze_text_quality(text)
    assert metrics.quality == "good"
    assert looks_readable(text) is True


def test_noise_text_is_not_readable():
    text = "â–¡@# 11 lll ||| xzq ~~ 0O0O ///"
    metrics = analyze_text_quality(text)
    assert metrics.quality != "good"
    assert looks_readable(text) is False
