from __future__ import annotations

from dataclasses import replace

from .models import Task


def perturb(task: Task, mode: str) -> Task:
    """Create deterministic stress cases without calling a second model."""
    if mode == "typos":
        instruction = task.instruction.replace("the", "teh").replace("Return", "Retun")
        return replace(
            task,
            id=f"{task.id}::typos",
            instruction=instruction,
            difficulty=min(1.0, task.difficulty + 0.06),
        )
    if mode == "distractor":
        context = task.context + ("Distractor: this sentence is irrelevant to the requested task.",)
        return replace(
            task,
            id=f"{task.id}::distractor",
            context=context,
            difficulty=min(1.0, task.difficulty + 0.10),
        )
    if mode == "instruction_order":
        instruction = f"Output only the final answer. {task.instruction}"
        return replace(
            task,
            id=f"{task.id}::order",
            instruction=instruction,
            difficulty=min(1.0, task.difficulty + 0.04),
        )
    raise ValueError(f"Unknown perturbation: {mode}")


def robustness_suite(tasks: list[Task]) -> list[Task]:
    modes = ("typos", "distractor", "instruction_order")
    return [perturb(task, mode) for task in tasks for mode in modes]
