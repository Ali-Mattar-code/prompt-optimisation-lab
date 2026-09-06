from __future__ import annotations

import pytest

from promptlab.experiment import ExperimentRunner, aggregate_trials
from promptlab.genome import PromptGenome
from promptlab.models import Task
from promptlab.providers import SyntheticLandscapeProvider


def tasks() -> list[Task]:
    return [
        Task(
            "dev",
            "classification",
            "classification",
            "Classify",
            "a",
            labels=("a", "b"),
            split="development",
            difficulty=0.2,
        ),
        Task(
            "hold",
            "classification",
            "classification",
            "Classify",
            "b",
            labels=("a", "b"),
            split="holdout",
            difficulty=0.8,
        ),
    ]


def test_synthetic_provider_is_deterministic() -> None:
    provider = SyntheticLandscapeProvider("atlas-sim", seed=42)
    genome = PromptGenome("minimal")
    task = tasks()[0]
    prompt = genome.render(task)
    first = provider.generate(task, genome, prompt, 0)
    second = provider.generate(task, genome, prompt, 0)
    assert first == second
    assert 0.03 <= provider.success_probability(task, genome) <= 0.97
    assert first.input_tokens > 0
    assert first.latency_ms > 0


def test_prompt_components_change_probability() -> None:
    provider = SyntheticLandscapeProvider("nova-sim", seed=1)
    task = tasks()[0]
    minimal = provider.success_probability(task, PromptGenome("minimal"))
    few_shot = provider.success_probability(task, PromptGenome("few", few_shot=True))
    assert few_shot > minimal


def test_runner_and_aggregates_cover_full_grid() -> None:
    variants = [PromptGenome("minimal"), PromptGenome("role", role=True)]
    providers = [SyntheticLandscapeProvider("atlas-sim", seed=1)]
    trials = ExperimentRunner().run(tasks(), variants, providers, repetitions=3)
    assert len(trials) == 12
    assert len(aggregate_trials(trials)) == 2
    assert all(item.trials == 3 for item in aggregate_trials(trials, split="holdout"))
    with pytest.raises(ValueError, match="positive"):
        ExperimentRunner().run(tasks(), variants, providers, repetitions=0)
