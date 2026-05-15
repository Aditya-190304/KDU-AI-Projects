from medical_extraction.benchmark.engine import evaluate_answer


def test_evaluate_answer_prefers_diagnosis_key_for_diagnosis_question() -> None:
    result = evaluate_answer(
        question='"What is CH Samuel\'s diagnosis?"',
        rag_answer="CH Samuel's diagnosis includes severe sepsis, likely liver abscess, and aspiration pneumonia.",
        document_id="10",
    )

    assert result["matched"] is True
    assert result["matched_question"] == "What is CH Samuel's diagnosis or condition?"
    assert result["retrieval_accuracy"] is not None
    assert float(result["retrieval_accuracy"]) >= 70.0


def test_evaluate_answer_prefers_doctor_key_for_doctor_question() -> None:
    result = evaluate_answer(
        question="Who is Mr. CH. Samuel's Doctor?",
        rag_answer="Dr. Naveen Polavarapu",
        document_id="10",
    )

    assert result["matched"] is True
    assert result["matched_question"] == "Who is Mr. CH. Samuel's Doctor?"
    assert float(result["retrieval_accuracy"]) == 100.0
