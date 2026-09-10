from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptlab.analysis import (  # noqa: E402
    aggregate_dicts,
    component_effects,
    cross_model_transfer,
    paired_bootstrap_difference,
    pareto_frontier,
    robustness_retention,
    selection_stability,
    successive_halving,
)
from promptlab.experiment import ExperimentRunner, aggregate_trials  # noqa: E402
from promptlab.genome import variant_library  # noqa: E402
from promptlab.io import load_tasks, load_yaml, write_csv, write_json  # noqa: E402
from promptlab.providers import SyntheticLandscapeProvider  # noqa: E402
from promptlab.reporting import write_quality_svg, write_report  # noqa: E402
from promptlab.robustness import robustness_suite  # noqa: E402


def build_payload() -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[object],
]:
    config = load_yaml(ROOT / "configs/experiment.yaml")
    tasks = load_tasks(ROOT / "datasets/task_suite.jsonl")
    genomes = variant_library()
    variants = [genomes[name] for name in config["variants"]]
    providers = [
        SyntheticLandscapeProvider(model=name, seed=int(config["seed"]))
        for name in config["providers"]
    ]
    trials = ExperimentRunner().run(
        tasks, variants, providers, repetitions=int(config["repetitions"])
    )
    development = aggregate_trials(trials, split="development")
    best_by_model: dict[str, dict[str, object]] = {}
    for model in config["providers"]:
        rows = [row for row in development if row.provider == model]
        best = max(rows, key=lambda row: (row.mean_quality, -row.mean_prompt_tokens))
        best_by_model[model] = best.to_dict()
    stability = selection_stability(
        trials,
        samples=int(config["bootstrap_samples"]),
        seed=int(config["seed"]),
    )
    winner_stability = {
        model: next(
            row["selection_frequency"]
            for row in stability
            if row["model"] == model and row["variant"] == best_by_model[model]["variant"]
        )
        for model in config["providers"]
    }
    bootstrap = {
        variant.name: paired_bootstrap_difference(
            trials,
            variant.name,
            samples=int(config["bootstrap_samples"]),
            seed=int(config["seed"]),
        )
        for variant in variants
        if variant.name != "minimal"
    }
    frontier = aggregate_dicts(pareto_frontier(aggregate_trials(trials)))
    effects = component_effects(trials, genomes)
    transfer = cross_model_transfer(trials)
    halving = successive_halving(trials)
    stressed_tasks = robustness_suite([task for task in tasks if task.split == "holdout"])
    stressed_trials = ExperimentRunner().run(stressed_tasks, variants, providers, repetitions=3)
    robustness = robustness_retention(trials, stressed_trials)
    payload: dict[str, object] = {
        "evidence_level": "deterministic-synthetic-response-surface",
        "evidence_boundary": (
            "Validates experiment, grading, optimisation, and analysis code. The named profiles "
            "are simulations and are not proxies for commercial models."
        ),
        "seed": config["seed"],
        "trials": len(trials),
        "robustness_trials": len(stressed_trials),
        "tasks": len(tasks),
        "variants": len(variants),
        "models": len(providers),
        "repetitions": config["repetitions"],
        "best_by_model": best_by_model,
        "development_winner_selection_frequency": winner_stability,
        "selection_stability_method": "paired task-cluster bootstrap on development tasks",
        "bootstrap_vs_minimal": bootstrap,
        "successive_halving": halving,
        "interpretation": (
            "The synthetic profiles select different prompt architectures and exhibit non-zero "
            "cross-model transfer regret. This demonstrates why prompts should be optimised "
            "against task-specific holdouts instead of treated as universal text recipes."
        ),
    }
    return payload, effects, frontier, transfer, robustness, stability, development


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic prompt-science benchmark.")
    parser.add_argument("--verify", action="store_true", help="Verify without rewriting artifacts.")
    args = parser.parse_args()
    payload, effects, frontier, transfer, robustness, stability, development = build_payload()
    if not args.verify:
        output = ROOT / "artifacts/demo"
        write_json(output / "summary.json", payload)
        write_csv(output / "component_effects.csv", effects)
        write_csv(output / "pareto_frontier.csv", frontier)
        write_csv(output / "cross_model_transfer.csv", transfer)
        write_csv(output / "robustness_retention.csv", robustness)
        write_csv(output / "selection_stability.csv", stability)
        write_report(output / "report.md", payload)
        write_quality_svg(output / "quality_by_model.svg", development)
    best = payload["best_by_model"]
    assert isinstance(best, dict)
    print(
        f"trials={payload['trials']} models={payload['models']} variants={payload['variants']} "
        f"winners={','.join(str(row['variant']) for row in best.values())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
