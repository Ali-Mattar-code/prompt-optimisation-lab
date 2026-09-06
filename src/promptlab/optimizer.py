from __future__ import annotations

from .genome import PromptGenome

COMPONENTS = (
    "role",
    "delimiters",
    "few_shot",
    "output_schema",
    "decomposition",
    "verification",
    "evidence_position",
)


def mutation_neighbourhood(parent: PromptGenome) -> list[PromptGenome]:
    """Generate a transparent one-component evolutionary neighbourhood."""
    return [parent.mutate(component, name=f"mutation_{component}") for component in COMPONENTS]
