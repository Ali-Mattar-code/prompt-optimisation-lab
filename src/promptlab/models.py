from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TaskKind = Literal["classification", "extraction", "grounded_qa", "constrained_writing"]


@dataclass(frozen=True)
class Task:
    id: str
    kind: TaskKind
    category: str
    instruction: str
    expected: Any
    context: tuple[str, ...] = ()
    examples: tuple[dict[str, str], ...] = ()
    labels: tuple[str, ...] = ()
    constraints: dict[str, Any] = field(default_factory=dict)
    split: Literal["development", "holdout"] = "development"
    difficulty: float = 0.5

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Task:
        return cls(
            id=str(payload["id"]),
            kind=payload["kind"],
            category=str(payload["category"]),
            instruction=str(payload["instruction"]),
            expected=payload["expected"],
            context=tuple(payload.get("context", [])),
            examples=tuple(payload.get("examples", [])),
            labels=tuple(payload.get("labels", [])),
            constraints=dict(payload.get("constraints", {})),
            split=payload.get("split", "development"),
            difficulty=float(payload.get("difficulty", 0.5)),
        )


@dataclass(frozen=True)
class ModelOutput:
    text: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class Trial:
    task_id: str
    task_kind: str
    category: str
    split: str
    provider: str
    model: str
    variant: str
    repetition: int
    prompt_hash: str
    prompt_tokens: int
    quality: float
    passed: bool
    latency_ms: float
    cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Aggregate:
    provider: str
    variant: str
    trials: int
    mean_quality: float
    pass_rate: float
    mean_latency_ms: float
    total_cost_usd: float
    mean_prompt_tokens: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
