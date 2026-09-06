from __future__ import annotations

import hashlib
from collections import defaultdict

from .genome import PromptGenome
from .graders import grade
from .models import Aggregate, Task, Trial
from .providers import Provider


class ExperimentRunner:
    def run(
        self,
        tasks: list[Task],
        variants: list[PromptGenome],
        providers: list[Provider],
        *,
        repetitions: int,
    ) -> list[Trial]:
        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        trials: list[Trial] = []
        for provider in providers:
            for variant in variants:
                for task in tasks:
                    prompt = variant.render(task)
                    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
                    for repetition in range(repetitions):
                        output = provider.generate(task, variant, prompt, repetition)
                        quality, passed = grade(task, output.text)
                        trials.append(
                            Trial(
                                task_id=task.id,
                                task_kind=task.kind,
                                category=task.category,
                                split=task.split,
                                provider=output.provider,
                                model=output.model,
                                variant=variant.name,
                                repetition=repetition,
                                prompt_hash=prompt_hash,
                                prompt_tokens=output.input_tokens,
                                quality=quality,
                                passed=passed,
                                latency_ms=output.latency_ms,
                                cost_usd=output.cost_usd,
                            )
                        )
        return trials


def aggregate_trials(trials: list[Trial], *, split: str | None = None) -> list[Aggregate]:
    selected = [trial for trial in trials if split is None or trial.split == split]
    groups: dict[tuple[str, str], list[Trial]] = defaultdict(list)
    for trial in selected:
        groups[(trial.model, trial.variant)].append(trial)
    output: list[Aggregate] = []
    for (model, variant), values in sorted(groups.items()):
        total = len(values)
        output.append(
            Aggregate(
                provider=model,
                variant=variant,
                trials=total,
                mean_quality=sum(item.quality for item in values) / total,
                pass_rate=sum(item.passed for item in values) / total,
                mean_latency_ms=sum(item.latency_ms for item in values) / total,
                total_cost_usd=sum(item.cost_usd for item in values),
                mean_prompt_tokens=sum(item.prompt_tokens for item in values) / total,
            )
        )
    return output
