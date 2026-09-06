from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from promptlab.cli import app

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_variants_command() -> None:
    result = runner.invoke(app, ["variants"])
    assert result.exit_code == 0
    assert "full_stack" in result.output


def test_render_command() -> None:
    result = runner.invoke(
        app,
        [
            "render",
            "ext-005",
            "--variant",
            "schema_first",
            "--dataset",
            str(ROOT / "datasets/task_suite.jsonl"),
        ],
    )
    assert result.exit_code == 0
    assert "output_contract" in result.output


def test_render_rejects_unknown_task_and_variant() -> None:
    dataset = str(ROOT / "datasets/task_suite.jsonl")
    assert runner.invoke(app, ["render", "missing", "--dataset", dataset]).exit_code != 0
    assert (
        runner.invoke(
            app, ["render", "ext-005", "--variant", "missing", "--dataset", dataset]
        ).exit_code
        != 0
    )


def test_live_command_validates_provider_before_calling_api() -> None:
    result = runner.invoke(
        app,
        [
            "run-live",
            "--provider",
            "unknown",
            "--model",
            "none",
            "--dataset",
            str(ROOT / "datasets/task_suite.jsonl"),
        ],
    )
    assert result.exit_code != 0
