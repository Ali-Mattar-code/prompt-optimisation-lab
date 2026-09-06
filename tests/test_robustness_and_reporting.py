from __future__ import annotations

from pathlib import Path

import pytest

from promptlab.models import Aggregate, Task
from promptlab.reporting import markdown_report, write_quality_svg, write_report
from promptlab.robustness import perturb, robustness_suite


def base_task() -> Task:
    return Task(
        "x",
        "grounded_qa",
        "reasoning",
        "Return the answer.",
        "answer",
        context=("Evidence",),
        split="holdout",
    )


def test_robustness_perturbations_are_traceable() -> None:
    task = base_task()
    assert "teh" in perturb(task, "typos").instruction
    assert len(perturb(task, "distractor").context) == 2
    assert perturb(task, "instruction_order").instruction.startswith("Output only")
    assert len(robustness_suite([task])) == 3
    with pytest.raises(ValueError, match="Unknown perturbation"):
        perturb(task, "random")


def test_reports_and_svg_are_generated(tmp_path: Path) -> None:
    summary = {
        "trials": 10,
        "tasks": 2,
        "variants": 1,
        "models": 1,
        "repetitions": 5,
        "best_by_model": {
            "atlas-sim": {
                "variant": "minimal",
                "mean_quality": 0.5,
                "pass_rate": 0.4,
                "mean_prompt_tokens": 20,
            }
        },
        "interpretation": "Test interpretation.",
    }
    assert "Test interpretation" in markdown_report(summary)
    report = tmp_path / "report.md"
    write_report(report, summary)
    assert report.exists()
    svg = tmp_path / "quality.svg"
    write_quality_svg(
        svg,
        [
            Aggregate("atlas-sim", "minimal", 1, 0.5, 0.4, 100, 0, 20),
            Aggregate("nova-sim", "minimal", 1, 0.6, 0.5, 110, 0, 20),
            Aggregate("ember-sim", "minimal", 1, 0.7, 0.6, 120, 0, 20),
        ],
    )
    assert svg.read_text().startswith("<svg")
