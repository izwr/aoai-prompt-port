from __future__ import annotations

from dataclasses import dataclass

from prompt_migration.core.extract import build_naive_port
from prompt_migration.evaluation.harness import evaluate_prompt
from prompt_migration.evaluation.metric import SemanticSimilarityJudge
from prompt_migration.evaluation.types import EvalSummary, GoldenCase
from prompt_migration.llm.model import ModelRunner


@dataclass(frozen=True)
class BaselineResult:
    source_on_source: EvalSummary
    naive_on_target: EvalSummary
    naive_prompt: str


def run_baselines(
    *,
    source_prompt: str,
    source_model: str,
    target_model: str,
    golden_set: list[GoldenCase],
    runner: ModelRunner,
    semantic_judge: SemanticSimilarityJudge | None = None,
    repetitions: int = 1,
) -> BaselineResult:
    naive_prompt = build_naive_port(source_prompt)
    return BaselineResult(
        source_on_source=evaluate_prompt(
            prompt_name="source_prompt",
            prompt=source_prompt,
            model=source_model,
            golden_set=golden_set,
            runner=runner,
            semantic_judge=semantic_judge,
            repetitions=repetitions,
            progress_label="source prompt on source model",
        ),
        naive_on_target=evaluate_prompt(
            prompt_name="naive_port",
            prompt=naive_prompt,
            model=target_model,
            golden_set=golden_set,
            runner=runner,
            semantic_judge=semantic_judge,
            repetitions=repetitions,
            progress_label="naive prompt on target model",
        ),
        naive_prompt=naive_prompt,
    )
