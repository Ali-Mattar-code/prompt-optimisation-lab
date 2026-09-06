from __future__ import annotations

from promptlab.graders import correct_output, grade, incorrect_output, token_f1
from promptlab.models import Task


def test_classification_grader() -> None:
    task = Task(
        "c", "classification", "classification", "Classify", "sales", labels=("sales", "billing")
    )
    assert grade(task, "sales") == (1.0, True)
    assert grade(task, "billing") == (0.0, False)
    assert correct_output(task) == "sales"
    assert incorrect_output(task) == "billing"


def test_extraction_grader_validates_contract_and_values() -> None:
    task = Task(
        "e",
        "extraction",
        "extraction",
        "Extract",
        {"company": "Acme", "seats": 10},
        constraints={"required_keys": ["company", "seats"]},
    )
    assert grade(task, '{"company":"Acme","seats":10}') == (1.0, True)
    score, passed = grade(task, '{"company":"Acme"}')
    assert 0 < score < 1 and not passed
    assert grade(task, "invalid") == (0.0, False)


def test_grounded_answer_combines_similarity_and_support() -> None:
    task = Task(
        "q",
        "grounded_qa",
        "reasoning",
        "When?",
        "Escalate immediately.",
        context=("Escalate immediately.",),
    )
    assert grade(task, "Escalate immediately.") == (1.0, True)
    assert grade(task, "Something unrelated.")[1] is False
    assert token_f1("same answer", "same answer") == 1.0
    assert token_f1("", "answer") == 0.0


def test_constrained_writing_checks_terms_and_length() -> None:
    task = Task(
        "w",
        "constrained_writing",
        "writing",
        "Write",
        "check",
        constraints={"required_terms": ["risk", "owner"], "max_words": 5},
    )
    assert grade(task, "risk owner assigned") == (1.0, True)
    assert grade(task, "risk only and many unnecessary extra words")[1] is False
    assert grade(task, correct_output(task))[1] is True
    assert grade(task, incorrect_output(task))[1] is False
