from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from .genome import PromptGenome
from .graders import correct_output, incorrect_output
from .models import ModelOutput, Task


class Provider(Protocol):
    name: str
    model: str

    def generate(
        self, task: Task, genome: PromptGenome, prompt: str, repetition: int
    ) -> ModelOutput: ...


MODEL_CATEGORY_SKILL = {
    "atlas-sim": {"reasoning": 0.12, "classification": 0.03, "extraction": 0.02, "writing": 0.04},
    "nova-sim": {"reasoning": 0.03, "classification": 0.12, "extraction": 0.08, "writing": 0.02},
    "ember-sim": {"reasoning": 0.02, "classification": 0.02, "extraction": 0.04, "writing": 0.13},
}


@dataclass
class SyntheticLandscapeProvider:
    """A deterministic response surface for testing experimental machinery."""

    model: str
    seed: int = 0
    name: str = "synthetic-landscape"

    def generate(
        self, task: Task, genome: PromptGenome, prompt: str, repetition: int
    ) -> ModelOutput:
        probability = self.success_probability(task, genome)
        sample = _stable_uniform(self.seed, self.model, task.id, genome.name, repetition)
        text = correct_output(task) if sample < probability else incorrect_output(task)
        prompt_tokens = _estimate_tokens(prompt)
        output_tokens = _estimate_tokens(text)
        jitter = _stable_uniform("latency", self.seed, self.model, task.id, genome.name, repetition)
        latency = 310 + 22 * genome.complexity + 4.5 * prompt_tokens + 90 * jitter
        cost = (prompt_tokens * 0.20 + output_tokens * 0.80) / 1_000_000
        return ModelOutput(text, self.name, self.model, latency, prompt_tokens, output_tokens, cost)

    def success_probability(self, task: Task, genome: PromptGenome) -> float:
        skill = MODEL_CATEGORY_SKILL.get(self.model, {}).get(task.category, 0.0)
        value = 0.52 - 0.32 * task.difficulty + skill
        if genome.role:
            value += 0.025
        if genome.delimiters and task.context:
            value += 0.05
        if genome.few_shot:
            value += 0.18 if task.kind in {"classification", "constrained_writing"} else 0.05
        if genome.output_schema:
            value += 0.24 if task.kind == "extraction" else 0.04
        if genome.decomposition:
            value += 0.16 if task.category == "reasoning" else -0.01
        if genome.verification:
            value += 0.10 if task.kind in {"grounded_qa", "extraction"} else 0.025
        if genome.evidence_position == "after" and task.context:
            value += 0.07
        if genome.complexity >= 6:
            # Synthetic profiles respond differently to over-specified prompts.
            # This creates a non-trivial transfer landscape for testing the optimiser.
            value -= {"atlas-sim": 0.20, "nova-sim": 0.24, "ember-sim": 0.20}.get(self.model, 0.18)
        if genome.few_shot and genome.output_schema and task.kind == "classification":
            value += 0.04
        return min(0.97, max(0.03, value))


def _stable_uniform(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / (2**64 - 1)


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


@dataclass
class OpenAIProvider:
    model: str
    name: str = "openai"

    def generate(
        self, task: Task, genome: PromptGenome, prompt: str, repetition: int
    ) -> ModelOutput:
        del task, genome, repetition
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install prompt-optimisation-lab[openai].") from exc
        client = OpenAI()
        started = perf_counter()
        result = client.responses.create(model=self.model, input=prompt)
        latency = (perf_counter() - started) * 1000
        usage = getattr(result, "usage", None)
        return ModelOutput(
            result.output_text,
            self.name,
            self.model,
            latency,
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
            0.0,
        )


@dataclass
class AnthropicProvider:
    model: str
    name: str = "anthropic"

    def generate(
        self, task: Task, genome: PromptGenome, prompt: str, repetition: int
    ) -> ModelOutput:
        del task, genome, repetition
        try:
            from anthropic import Anthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install prompt-optimisation-lab[anthropic].") from exc
        client = Anthropic()
        started = perf_counter()
        result = client.messages.create(
            model=self.model, max_tokens=800, messages=[{"role": "user", "content": prompt}]
        )
        latency = (perf_counter() - started) * 1000
        text = "".join(
            block.text for block in result.content if getattr(block, "type", "") == "text"
        )
        return ModelOutput(
            text,
            self.name,
            self.model,
            latency,
            int(result.usage.input_tokens),
            int(result.usage.output_tokens),
            0.0,
        )


@dataclass
class GeminiProvider:
    model: str
    name: str = "gemini"

    def generate(
        self, task: Task, genome: PromptGenome, prompt: str, repetition: int
    ) -> ModelOutput:
        del task, genome, repetition
        try:
            from google import genai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install prompt-optimisation-lab[gemini].") from exc
        client = genai.Client()
        started = perf_counter()
        result = client.models.generate_content(model=self.model, contents=prompt)
        latency = (perf_counter() - started) * 1000
        usage = getattr(result, "usage_metadata", None)
        return ModelOutput(
            str(result.text or ""),
            self.name,
            self.model,
            latency,
            int(getattr(usage, "prompt_token_count", 0) or 0),
            int(getattr(usage, "candidates_token_count", 0) or 0),
            0.0,
        )
