from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from .models import Task


@dataclass(frozen=True)
class PromptGenome:
    name: str
    role: bool = False
    delimiters: bool = False
    few_shot: bool = False
    output_schema: bool = False
    decomposition: bool = False
    verification: bool = False
    evidence_position: str = "before"

    @property
    def complexity(self) -> int:
        return sum(
            (
                self.role,
                self.delimiters,
                self.few_shot,
                self.output_schema,
                self.decomposition,
                self.verification,
                self.evidence_position == "after",
            )
        )

    def fingerprint(self) -> str:
        payload = "|".join(
            str(value)
            for value in (
                self.role,
                self.delimiters,
                self.few_shot,
                self.output_schema,
                self.decomposition,
                self.verification,
                self.evidence_position,
            )
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def mutate(self, component: str, *, name: str | None = None) -> PromptGenome:
        mutation_name = name or f"{self.name}+{component}"
        if component == "evidence_position":
            position = "after" if self.evidence_position == "before" else "before"
            return replace(self, name=mutation_name, evidence_position=position)
        if component == "role":
            return replace(self, name=mutation_name, role=not self.role)
        if component == "delimiters":
            return replace(self, name=mutation_name, delimiters=not self.delimiters)
        if component == "few_shot":
            return replace(self, name=mutation_name, few_shot=not self.few_shot)
        if component == "output_schema":
            return replace(self, name=mutation_name, output_schema=not self.output_schema)
        if component == "decomposition":
            return replace(self, name=mutation_name, decomposition=not self.decomposition)
        if component == "verification":
            return replace(self, name=mutation_name, verification=not self.verification)
        raise ValueError(f"Unknown prompt component: {component}")

    def render(self, task: Task) -> str:
        sections: list[str] = []
        if self.role:
            sections.append(
                "SYSTEM ROLE\nYou are a precise task specialist. "
                "Follow the supplied evidence and contract."
            )
        evidence = "\n".join(f"[{index}] {item}" for index, item in enumerate(task.context, 1))
        if evidence and self.evidence_position == "before":
            sections.append(self._block("EVIDENCE", evidence))
        sections.append(self._block("TASK", task.instruction))
        if self.few_shot and task.examples:
            examples = "\n\n".join(
                f"Input: {item['input']}\nOutput: {item['output']}" for item in task.examples
            )
            sections.append(self._block("EXAMPLES", examples))
        if self.decomposition:
            sections.append(
                "PROCESS\nDecompose the task internally into the minimum necessary "
                "steps before answering."
            )
        if self.output_schema:
            sections.append(self._block("OUTPUT CONTRACT", _output_contract(task)))
        if self.verification:
            sections.append(
                "VERIFICATION\nBefore returning, check evidence support, constraints, "
                "and output format."
            )
        if evidence and self.evidence_position == "after":
            sections.append(self._block("EVIDENCE", evidence))
        return "\n\n".join(sections)

    def _block(self, title: str, value: str) -> str:
        if self.delimiters:
            tag = title.lower().replace(" ", "_")
            return f"<{tag}>\n{value}\n</{tag}>"
        return f"{title}\n{value}"


def _output_contract(task: Task) -> str:
    if task.kind == "classification":
        return f"Return exactly one label: {', '.join(task.labels)}."
    if task.kind == "extraction":
        required = task.constraints.get("required_keys", [])
        return f"Return one valid JSON object with keys: {', '.join(required)}."
    if task.kind == "grounded_qa":
        return "Answer only from the evidence. If unsupported, say so."
    required = task.constraints.get("required_terms", [])
    maximum = task.constraints.get("max_words", 100)
    return f"Use at most {maximum} words and include: {', '.join(required)}."


def variant_library() -> dict[str, PromptGenome]:
    return {
        "minimal": PromptGenome("minimal"),
        "role_only": PromptGenome("role_only", role=True),
        "schema_first": PromptGenome("schema_first", delimiters=True, output_schema=True),
        "few_shot": PromptGenome("few_shot", delimiters=True, few_shot=True),
        "decompose": PromptGenome("decompose", role=True, decomposition=True),
        "verify": PromptGenome("verify", role=True, verification=True),
        "evidence_last": PromptGenome(
            "evidence_last", delimiters=True, output_schema=True, evidence_position="after"
        ),
        "full_stack": PromptGenome(
            "full_stack",
            role=True,
            delimiters=True,
            few_shot=True,
            output_schema=True,
            decomposition=True,
            verification=True,
            evidence_position="after",
        ),
    }
