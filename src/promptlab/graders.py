from __future__ import annotations

import json
import re
from collections import Counter

from .models import Task

TOKEN = re.compile(r"[a-z0-9]+", re.I)
STOP = {"a", "an", "and", "are", "as", "at", "for", "from", "in", "is", "of", "on", "the", "to"}


def _tokens(value: str) -> list[str]:
    return [item.lower() for item in TOKEN.findall(value) if item.lower() not in STOP]


def token_f1(prediction: str, reference: str) -> float:
    left, right = Counter(_tokens(prediction)), Counter(_tokens(reference))
    overlap = sum((left & right).values())
    if not left or not right or overlap == 0:
        return 0.0
    precision, recall = overlap / sum(left.values()), overlap / sum(right.values())
    return 2 * precision * recall / (precision + recall)


def grade(task: Task, output: str) -> tuple[float, bool]:
    if task.kind == "classification":
        score = float(output.strip().lower() == str(task.expected).strip().lower())
        return score, score == 1.0
    if task.kind == "extraction":
        score = _grade_json(task, output)
        return score, score >= 0.99
    if task.kind == "grounded_qa":
        similarity = token_f1(output, str(task.expected))
        supported = _support_score(output, task.context)
        score = 0.65 * similarity + 0.35 * supported
        return score, score >= 0.70
    score = _grade_constraints(task, output)
    return score, score >= 0.99


def _grade_json(task: Task, output: str) -> float:
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(parsed, dict) or not isinstance(task.expected, dict):
        return 0.0
    required = set(task.constraints.get("required_keys", task.expected.keys()))
    contract = len(required & set(parsed)) / len(required) if required else 1.0
    correct = sum(parsed.get(key) == value for key, value in task.expected.items())
    accuracy = correct / len(task.expected) if task.expected else 1.0
    return 0.4 * contract + 0.6 * accuracy


def _support_score(output: str, context: tuple[str, ...]) -> float:
    prediction = _tokens(output)
    evidence = set(_tokens(" ".join(context)))
    return sum(token in evidence for token in prediction) / len(prediction) if prediction else 0.0


def _grade_constraints(task: Task, output: str) -> float:
    words = output.split()
    maximum = int(task.constraints.get("max_words", 10_000))
    required = [str(item).lower() for item in task.constraints.get("required_terms", [])]
    term_score = (
        sum(term in output.lower() for term in required) / len(required) if required else 1.0
    )
    length_score = float(len(words) <= maximum)
    return (term_score + length_score) / 2


def correct_output(task: Task) -> str:
    if task.kind == "classification":
        return str(task.expected)
    if task.kind == "extraction":
        return json.dumps(task.expected, sort_keys=True)
    if task.kind == "grounded_qa":
        return str(task.expected)
    terms = [str(item) for item in task.constraints.get("required_terms", [])]
    return " ".join(terms + ["delivered clearly."])


def incorrect_output(task: Task) -> str:
    if task.kind == "classification":
        alternatives = [label for label in task.labels if label != task.expected]
        return alternatives[0] if alternatives else "unknown"
    if task.kind == "extraction":
        return "{invalid-json"
    if task.kind == "grounded_qa":
        return "The evidence states a different unsupported answer."
    return "A generic response that ignores the requested constraints."
