from __future__ import annotations

import pytest

from promptlab.genome import PromptGenome, variant_library
from promptlab.models import Task
from promptlab.optimizer import mutation_neighbourhood


def extraction_task() -> Task:
    return Task(
        "x",
        "extraction",
        "extraction",
        "Extract a value.",
        {"value": 2},
        context=("The value is 2.",),
        examples=({"input": "value 1", "output": '{"value": 1}'},),
        constraints={"required_keys": ["value"]},
    )


def test_full_genome_renders_every_section() -> None:
    prompt = variant_library()["full_stack"].render(extraction_task())
    assert "SYSTEM ROLE" in prompt
    assert "<task>" in prompt
    assert "<examples>" in prompt
    assert "OUTPUT CONTRACT" not in prompt  # delimited tag uses lower-case contract name
    assert "<output_contract>" in prompt
    assert "VERIFICATION" in prompt
    assert prompt.index("<task>") < prompt.index("<evidence>")


def test_task_specific_output_contracts() -> None:
    genome = PromptGenome("schema", output_schema=True)
    classification = Task(
        "c", "classification", "classification", "Classify", "a", labels=("a", "b")
    )
    grounded = Task("q", "grounded_qa", "reasoning", "Answer", "yes")
    writing = Task(
        "w",
        "constrained_writing",
        "writing",
        "Write",
        "x",
        constraints={"required_terms": ["risk"], "max_words": 5},
    )
    assert "Return exactly one label" in genome.render(classification)
    assert "Answer only from the evidence" in genome.render(grounded)
    assert "at most 5 words" in genome.render(writing)


def test_mutations_are_explicit_and_validated() -> None:
    parent = PromptGenome("base")
    children = mutation_neighbourhood(parent)
    assert len(children) == 7
    assert all(child.complexity == 1 for child in children)
    assert parent.mutate("role").role is True
    assert parent.mutate("evidence_position").evidence_position == "after"
    with pytest.raises(ValueError, match="Unknown prompt component"):
        parent.mutate("magic")


def test_fingerprint_depends_on_components_not_name() -> None:
    assert PromptGenome("a", role=True).fingerprint() == PromptGenome("b", role=True).fingerprint()
    assert PromptGenome("a").fingerprint() != PromptGenome("a", role=True).fingerprint()
