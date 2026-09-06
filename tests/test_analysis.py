from __future__ import annotations

import pytest

from promptlab.analysis import (
    component_effects,
    cross_model_transfer,
    paired_bootstrap_difference,
    pareto_frontier,
    robustness_retention,
    successive_halving,
)
from promptlab.genome import PromptGenome
from promptlab.models import Aggregate, Trial


def trial(model: str, variant: str, task: str, quality: float, split: str = "development") -> Trial:
    return Trial(
        task,
        "classification",
        "classification",
        split,
        "synthetic",
        model,
        variant,
        0,
        "hash",
        20 if variant == "minimal" else 40,
        quality,
        quality >= 0.7,
        100 if variant == "minimal" else 120,
        0.0,
    )


def sample_trials() -> list[Trial]:
    values = []
    for model, bonus in (("a", 0.0), ("b", 0.1)):
        values.extend(
            [
                trial(model, "minimal", "d1", 0.4 + bonus),
                trial(model, "role", "d1", 0.7 + bonus),
                trial(model, "minimal", "d2", 0.5 + bonus),
                trial(model, "role", "d2", 0.8 + bonus),
                trial(model, "minimal", "h1", 0.45 + bonus, "holdout"),
                trial(model, "role", "h1", 0.75 + bonus, "holdout"),
            ]
        )
    return values


def test_pareto_frontier_removes_dominated_points() -> None:
    values = [
        Aggregate("a", "bad", 1, 0.5, 0.5, 200, 0, 80),
        Aggregate("a", "fast", 1, 0.5, 0.5, 100, 0, 40),
        Aggregate("a", "quality", 1, 0.8, 0.8, 150, 0, 60),
    ]
    names = {item.variant for item in pareto_frontier(values)}
    assert names == {"fast", "quality"}


def test_bootstrap_and_component_effects() -> None:
    values = sample_trials()
    result = paired_bootstrap_difference(values, "role", samples=100, seed=1)
    assert result["mean_difference"] == pytest.approx(0.3)
    genomes = {"minimal": PromptGenome("minimal"), "role": PromptGenome("role", role=True)}
    effects = component_effects(values, genomes)
    role_effects = [row for row in effects if row["component"] == "role"]
    assert all(row["effect"] > 0 for row in role_effects)
    with pytest.raises(ValueError, match="No paired"):
        paired_bootstrap_difference(values, "missing")


def test_cross_model_transfer_and_halving() -> None:
    values = sample_trials()
    transfer = cross_model_transfer(values)
    assert len(transfer) == 4
    assert {row["selected_variant"] for row in transfer} == {"role"}
    history = successive_halving(values, initial_budget=1)
    assert history[-1]["leader"] == "role"
    assert history[-1]["candidates"] == 1


def test_robustness_retention() -> None:
    clean = sample_trials()
    stressed = [
        trial("a", "minimal", "stress", 0.3, "holdout"),
        trial("a", "role", "stress", 0.6, "holdout"),
        trial("b", "minimal", "stress", 0.4, "holdout"),
        trial("b", "role", "stress", 0.7, "holdout"),
    ]
    rows = robustness_retention(clean, stressed)
    assert len(rows) == 4
    assert all(row["retention"] > 0 for row in rows)
