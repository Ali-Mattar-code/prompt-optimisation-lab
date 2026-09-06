from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import Aggregate


def markdown_report(summary: dict[str, Any]) -> str:
    best = summary["best_by_model"]
    lines = [
        "# Prompt optimisation research report",
        "",
        "> Evidence level: deterministic synthetic response-surface experiment. This validates the",
        "> optimisation machinery; it is not a performance claim about any commercial model.",
        "",
        "## Experiment scale",
        "",
        f"- Trials: **{summary['trials']:,}**",
        f"- Tasks: **{summary['tasks']}**",
        f"- Prompt variants: **{summary['variants']}**",
        f"- Simulated model profiles: **{summary['models']}**",
        f"- Repetitions per cell: **{summary['repetitions']}**",
        "",
        "## Best development prompt by model",
        "",
        "| Model profile | Variant | Mean quality | Pass rate | Prompt tokens |",
        "|---|---|---:|---:|---:|",
    ]
    for model, row in best.items():
        lines.append(
            f"| {model} | {row['variant']} | {row['mean_quality']:.3f} | "
            f"{row['pass_rate']:.1%} | {row['mean_prompt_tokens']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
            "",
            "The machine-readable result, component effects, Pareto frontier, transfer matrix,",
            "and successive-halving trace are stored beside this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(path: str | Path, summary: dict[str, Any]) -> None:
    Path(path).write_text(markdown_report(summary), encoding="utf-8")


def write_quality_svg(path: str | Path, aggregates: list[Aggregate]) -> None:
    """Render a dependency-free comparison chart from measured aggregates."""
    width, height = 1120, 610
    margin_left, chart_width = 175, 880
    models = sorted({item.provider for item in aggregates})
    variants = sorted({item.variant for item in aggregates})
    lookup = {(item.provider, item.variant): item.mean_quality for item in aggregates}
    colors = {"atlas-sim": "#4263eb", "nova-sim": "#0ca678", "ember-sim": "#f08c00"}
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#0d1117"/>',
        (
            "<style>text{font-family:Inter,Arial,sans-serif;fill:#e6edf3}"
            ".muted{fill:#8b949e}.grid{stroke:#30363d;stroke-width:1}</style>"
        ),
        (
            '<text x="40" y="42" font-size="24" font-weight="700">'
            "Development quality by prompt architecture</text>"
        ),
        (
            '<text x="40" y="69" font-size="14" class="muted">'
            "Deterministic synthetic response-surface experiment — "
            "not commercial-model results</text>"
        ),
    ]
    for index in range(6):
        x = margin_left + chart_width * index / 5
        lines.append(f'<line x1="{x:.1f}" y1="95" x2="{x:.1f}" y2="535" class="grid"/>')
        lines.append(
            f'<text x="{x:.1f}" y="560" text-anchor="middle" font-size="12" '
            f'class="muted">{index / 5:.1f}</text>'
        )
    row_height = 52
    bar_height = 11
    for row, variant in enumerate(variants):
        y_base = 110 + row * row_height
        lines.append(
            f'<text x="160" y="{y_base + 17}" text-anchor="end" font-size="13">{variant}</text>'
        )
        for model_index, model in enumerate(models):
            value = lookup[(model, variant)]
            y = y_base + model_index * 13
            bar_width = value * chart_width
            lines.append(
                f'<rect x="{margin_left}" y="{y}" width="{bar_width:.1f}" '
                f'height="{bar_height}" rx="3" fill="{colors[model]}"/>'
            )
    legend_x = 760
    for index, model in enumerate(models):
        x = legend_x + index * 120
        lines.extend(
            [
                f'<rect x="{x}" y="35" width="12" height="12" rx="2" fill="{colors[model]}"/>',
                f'<text x="{x + 18}" y="46" font-size="12">{model}</text>',
            ]
        )
    lines.append(
        '<text x="615" y="590" text-anchor="middle" font-size="13" '
        'class="muted">Mean quality score</text>'
    )
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
