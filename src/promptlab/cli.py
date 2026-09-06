from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .experiment import ExperimentRunner, aggregate_trials
from .genome import variant_library
from .io import load_tasks, write_json
from .providers import AnthropicProvider, GeminiProvider, OpenAIProvider, Provider

app = typer.Typer(help="Run controlled prompt experiments.", no_args_is_help=True)
console = Console()


@app.command("variants")
def list_variants() -> None:
    """List the committed prompt architectures and their components."""
    table = Table("Variant", "Complexity", "Fingerprint")
    for genome in variant_library().values():
        table.add_row(genome.name, str(genome.complexity), genome.fingerprint())
    console.print(table)


@app.command()
def render(
    task_id: Annotated[str, typer.Argument()],
    variant: Annotated[str, typer.Option()] = "full_stack",
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("datasets/task_suite.jsonl"),
) -> None:
    """Render one prompt genome against one task."""
    genomes = variant_library()
    if variant not in genomes:
        raise typer.BadParameter(f"Unknown variant: {variant}")
    tasks = {task.id: task for task in load_tasks(dataset)}
    if task_id not in tasks:
        raise typer.BadParameter(f"Unknown task: {task_id}")
    console.print(genomes[variant].render(tasks[task_id]))


@app.command("run-live")
def run_live(
    provider: Annotated[str, typer.Option(help="openai, anthropic, or gemini")],
    model: Annotated[str, typer.Option()],
    variant: Annotated[str, typer.Option()] = "full_stack",
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("datasets/task_suite.jsonl"),
    repetitions: Annotated[int, typer.Option(min=1, max=20)] = 3,
    output: Annotated[Path, typer.Option()] = Path("artifacts/live/run.json"),
) -> None:
    """Run an explicit paid-provider experiment and save its raw evidence."""
    adapter: Provider
    if provider == "openai":
        adapter = OpenAIProvider(model)
    elif provider == "anthropic":
        adapter = AnthropicProvider(model)
    elif provider == "gemini":
        adapter = GeminiProvider(model)
    else:
        raise typer.BadParameter("provider must be openai, anthropic, or gemini")
    genomes = variant_library()
    if variant not in genomes:
        raise typer.BadParameter(f"Unknown variant: {variant}")
    trials = ExperimentRunner().run(
        load_tasks(dataset), [genomes[variant]], [adapter], repetitions=repetitions
    )
    aggregates = aggregate_trials(trials)
    write_json(
        output,
        {
            "evidence_level": "live-provider-run",
            "provider": provider,
            "model": model,
            "variant": variant,
            "cost_warning": "Provider adapters record usage but do not infer changing list prices.",
            "trials": [trial.to_dict() for trial in trials],
            "aggregate": aggregates[0].to_dict(),
        },
    )
    console.print(f"Saved {len(trials)} trials to {output}")


if __name__ == "__main__":
    app()
