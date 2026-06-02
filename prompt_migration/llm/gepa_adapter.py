from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from gepa.core.adapter import EvaluationBatch

from prompt_migration.evaluation.harness import case_conversation_dicts
from prompt_migration.evaluation.metric import SemanticSimilarityJudge, score_output
from prompt_migration.evaluation.types import GoldenCase, ScoreResult
from prompt_migration.llm.model import ModelRunner


class PromptTrajectory(TypedDict):
    case_id: str
    input: str
    conversation: list[dict[str, str]]
    expected: Any
    output: str
    feedback: str
    score: float
    error: str | None


class PromptRolloutOutput(TypedDict):
    response: str


class PromptMigrationAdapter:
    propose_new_texts = None

    def __init__(
        self,
        *,
        target_model: str,
        runner: ModelRunner,
        semantic_judge: SemanticSimilarityJudge | None = None,
    ):
        self.target_model = target_model
        self.runner = runner
        self.semantic_judge = semantic_judge

    def evaluate(
        self,
        batch: list[GoldenCase],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[PromptTrajectory, PromptRolloutOutput]:
        system_prompt = candidate["system_prompt"]
        outputs: list[PromptRolloutOutput] = []
        scores: list[float] = []
        objective_scores: list[dict[str, float] | None] = []
        trajectories: list[PromptTrajectory] | None = [] if capture_traces else None

        for case in batch:
            conversation = case_conversation_dicts(case)
            error: str | None = None
            try:
                response = self.runner(self.target_model, system_prompt, conversation)
                scored = score_output(case, response, semantic_judge=self.semantic_judge)
            except Exception as exc:
                response = ""
                error = str(exc)
                scored = ScoreResult(
                    0.0,
                    f"Infrastructure error while evaluating case {case.id}: {exc}",
                    objective_scores={"score": 0.0},
                )

            outputs.append({"response": response})
            scores.append(scored.score)
            objective_scores.append(scored.objective_scores or {"score": scored.score})

            if trajectories is not None:
                trajectories.append(
                    {
                        "case_id": case.id,
                        "input": case.input,
                        "conversation": conversation,
                        "expected": case.expected,
                        "output": response,
                        "feedback": scored.feedback,
                        "score": scored.score,
                        "error": error,
                    }
                )

        objective_scores_arg = [{"score": 0.0} if score is None else score for score in objective_scores]

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores_arg,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[PromptTrajectory, PromptRolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if eval_batch.trajectories is None:
            raise ValueError("Trajectories are required for reflective prompt migration.")

        records = [_reflective_record(trace) for trace in eval_batch.trajectories if trace["score"] < 1.0]
        if not records:
            records = [_reflective_record(trace) for trace in eval_batch.trajectories]

        return {component: records for component in components_to_update}


def _reflective_record(trace: PromptTrajectory) -> dict[str, Any]:
    return {
        "Case ID": trace["case_id"],
        "Inputs": trace["input"],
        "Conversation": trace["conversation"],
        "Expected Output": trace["expected"],
        "Generated Output": trace["output"],
        "Score": trace["score"],
        "Feedback": trace["feedback"],
        "Error": trace["error"],
    }
