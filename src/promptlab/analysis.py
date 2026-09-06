from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict
from typing import Any

from .genome import PromptGenome
from .models import Aggregate, Trial


def pareto_frontier(aggregates: list[Aggregate]) -> list[Aggregate]:
    """Quality is maximised; prompt tokens and latency are minimised."""
    frontier: list[Aggregate] = []
    for candidate in aggregates:
        dominated = any(
            other.mean_quality >= candidate.mean_quality
            and other.mean_prompt_tokens <= candidate.mean_prompt_tokens
            and other.mean_latency_ms <= candidate.mean_latency_ms
            and (
                other.mean_quality > candidate.mean_quality
                or other.mean_prompt_tokens < candidate.mean_prompt_tokens
                or other.mean_latency_ms < candidate.mean_latency_ms
            )
            for other in aggregates
            if other.provider == candidate.provider
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: (item.provider, -item.mean_quality))


def component_effects(
    trials: list[Trial], genomes: dict[str, PromptGenome]
) -> list[dict[str, Any]]:
    components = (
        "role",
        "delimiters",
        "few_shot",
        "output_schema",
        "decomposition",
        "verification",
        "evidence_after",
    )
    rows: list[dict[str, Any]] = []
    for model in sorted({trial.model for trial in trials}):
        model_trials = [trial for trial in trials if trial.model == model]
        for component in components:
            enabled: list[float] = []
            disabled: list[float] = []
            for trial in model_trials:
                genome = genomes[trial.variant]
                active = (
                    genome.evidence_position == "after"
                    if component == "evidence_after"
                    else bool(getattr(genome, component))
                )
                (enabled if active else disabled).append(trial.quality)
            if not enabled or not disabled:
                continue
            effect = _mean(enabled) - _mean(disabled)
            standard_error = math.sqrt(
                _variance(enabled) / len(enabled) + _variance(disabled) / len(disabled)
            )
            rows.append(
                {
                    "model": model,
                    "component": component,
                    "effect": effect,
                    "standard_error": standard_error,
                    "ci95_low": effect - 1.96 * standard_error,
                    "ci95_high": effect + 1.96 * standard_error,
                    "enabled_trials": len(enabled),
                    "disabled_trials": len(disabled),
                }
            )
    return rows


def paired_bootstrap_difference(
    trials: list[Trial],
    candidate: str,
    baseline: str = "minimal",
    *,
    samples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    paired: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for trial in trials:
        if trial.variant in {candidate, baseline}:
            paired[(trial.model, trial.task_id, trial.repetition)][trial.variant] = trial.quality
    differences = [
        values[candidate] - values[baseline]
        for values in paired.values()
        if candidate in values and baseline in values
    ]
    if not differences:
        raise ValueError("No paired observations were found.")
    rng = random.Random(seed)
    bootstrapped = sorted(
        _mean([rng.choice(differences) for _ in differences]) for _ in range(samples)
    )
    return {
        "mean_difference": _mean(differences),
        "ci95_low": _percentile(bootstrapped, 0.025),
        "ci95_high": _percentile(bootstrapped, 0.975),
        "paired_observations": float(len(differences)),
    }


def cross_model_transfer(trials: list[Trial]) -> list[dict[str, Any]]:
    models = sorted({trial.model for trial in trials})
    rows: list[dict[str, Any]] = []
    for source in models:
        source_dev = [
            trial for trial in trials if trial.model == source and trial.split == "development"
        ]
        by_variant: dict[str, list[float]] = defaultdict(list)
        for trial in source_dev:
            by_variant[trial.variant].append(trial.quality)
        selected = max(by_variant, key=lambda name: (_mean(by_variant[name]), -len(name)))
        for target in models:
            values = [
                trial.quality
                for trial in trials
                if trial.model == target and trial.variant == selected and trial.split == "holdout"
            ]
            oracle = max(
                (
                    _mean(
                        [
                            trial.quality
                            for trial in trials
                            if trial.model == target
                            and trial.variant == variant
                            and trial.split == "holdout"
                        ]
                    ),
                    variant,
                )
                for variant in {trial.variant for trial in trials}
            )
            rows.append(
                {
                    "source_model": source,
                    "selected_variant": selected,
                    "target_model": target,
                    "holdout_quality": _mean(values),
                    "target_oracle_quality": oracle[0],
                    "transfer_regret": oracle[0] - _mean(values),
                }
            )
    return rows


def successive_halving(trials: list[Trial], *, initial_budget: int = 3) -> list[dict[str, Any]]:
    variants = sorted({trial.variant for trial in trials})
    task_ids = sorted({trial.task_id for trial in trials if trial.split == "development"})
    history: list[dict[str, Any]] = []
    budget = min(initial_budget, len(task_ids))
    while len(variants) > 1:
        scores = {}
        allowed = set(task_ids[:budget])
        for variant in variants:
            values = [
                trial.quality
                for trial in trials
                if trial.variant == variant
                and trial.task_id in allowed
                and trial.split == "development"
            ]
            scores[variant] = _mean(values)
        ranked = sorted(variants, key=lambda item: (-scores[item], item))
        keep = max(1, math.ceil(len(ranked) / 2))
        history.append(
            {
                "round": len(history) + 1,
                "task_budget": budget,
                "candidates": len(variants),
                "leader": ranked[0],
                "leader_quality": scores[ranked[0]],
                "eliminated": ranked[keep:],
            }
        )
        variants = ranked[:keep]
        budget = min(len(task_ids), budget * 2)
    history.append(
        {
            "round": len(history) + 1,
            "task_budget": len(task_ids),
            "candidates": 1,
            "leader": variants[0],
            "leader_quality": _mean(
                [
                    trial.quality
                    for trial in trials
                    if trial.variant == variants[0] and trial.split == "development"
                ]
            ),
            "eliminated": [],
        }
    )
    return history


def robustness_retention(clean: list[Trial], stressed: list[Trial]) -> list[dict[str, Any]]:
    clean_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    stress_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for trial in clean:
        if trial.split == "holdout":
            clean_groups[(trial.model, trial.variant)].append(trial.quality)
    for trial in stressed:
        stress_groups[(trial.model, trial.variant)].append(trial.quality)
    rows = []
    for key in sorted(clean_groups):
        clean_mean = _mean(clean_groups[key])
        stress_mean = _mean(stress_groups.get(key, []))
        rows.append(
            {
                "model": key[0],
                "variant": key[1],
                "clean_holdout_quality": clean_mean,
                "stressed_quality": stress_mean,
                "absolute_change": stress_mean - clean_mean,
                "retention": stress_mean / clean_mean if clean_mean else 0.0,
            }
        )
    return rows


def aggregate_dicts(values: list[Aggregate]) -> list[dict[str, Any]]:
    return [asdict(value) for value in values]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[index]
