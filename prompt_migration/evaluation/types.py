from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


JudgeKind = Literal["exact", "contains", "json_exact", "semantic_and_length"]


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str


@dataclass(frozen=True)
class GoldenCase:
    id: str
    conversation: list[ConversationMessage]
    expected: Any
    judge: JudgeKind = "exact"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def input(self) -> str:
        return self.conversation[-1].content if self.conversation else ""


@dataclass(frozen=True)
class ScoreResult:
    score: float
    feedback: str
    objective_scores: dict[str, float] | None = None


@dataclass(frozen=True)
class RunEvent:
    case_id: str
    model: str
    prompt_name: str
    input: str
    conversation: list[dict[str, str]]
    expected: Any
    output: str
    score: float
    feedback: str
    objective_scores: dict[str, float] | None = None
    error: str | None = None
    repetition: int = 0


@dataclass(frozen=True)
class EvalSummary:
    prompt_name: str
    model: str
    score: float
    events: list[RunEvent]
    run_scores: list[float] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for event in self.events if event.error)
