from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from gepa.core.adapter import EvaluationBatch

from prompt_migration.evaluation.harness import case_conversation_dicts, case_runner_messages
from prompt_migration.evaluation.metric import GuideAdherenceJudge, SemanticSimilarityJudge, score_output
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
        guide_text: str = "",
        guide_adherence_judge: GuideAdherenceJudge | None = None,
        guide_weight: float = 0.0,
    ):
        self.target_model = target_model
        self.runner = runner
        self.semantic_judge = semantic_judge
        self.guide_text = guide_text
        self.guide_adherence_judge = guide_adherence_judge
        self.guide_weight = guide_weight
        self._adherence_cache: dict[str, tuple[float, str]] = {}

    def guide_adherence(self, prompt: str) -> tuple[float, str] | None:
        """Cached guide-adherence (score, feedback) for a candidate prompt, or None when disabled."""
        if self.guide_weight <= 0.0 or not self.guide_text.strip() or self.guide_adherence_judge is None:
            return None
        if prompt not in self._adherence_cache:
            self._adherence_cache[prompt] = self.guide_adherence_judge(prompt=prompt, guide=self.guide_text)
        return self._adherence_cache[prompt]

    def evaluate(
        self,
        batch: list[GoldenCase],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[PromptTrajectory, PromptRolloutOutput]:
        system_prompt = candidate["system_prompt"]
        adherence = self.guide_adherence(system_prompt)
        outputs: list[PromptRolloutOutput] = []
        scores: list[float] = []
        objective_scores: list[dict[str, float] | None] = []
        trajectories: list[PromptTrajectory] | None = [] if capture_traces else None

        for case in batch:
            runner_messages = case_runner_messages(case)
            conversation = case_conversation_dicts(case)
            error: str | None = None
            try:
                response = self.runner(self.target_model, system_prompt, runner_messages)
                scored = score_output(case, response, semantic_judge=self.semantic_judge)
            except Exception as exc:
                response = ""
                error = str(exc)
                scored = ScoreResult(
                    0.0,
                    f"Infrastructure error while evaluating case {case.id}: {exc}",
                    objective_scores={"score": 0.0},
                )

            behavior_score = scored.score
            if adherence is not None:
                # Convex blend keeps behavior dominant (guide_weight is small) while letting a
                # restructured, guide-following prompt win the tie when behavior is preserved.
                final_score = (1.0 - self.guide_weight) * behavior_score + self.guide_weight * adherence[0]
            else:
                final_score = behavior_score

            case_objectives = dict(scored.objective_scores or {"score": behavior_score})
            if adherence is not None:
                case_objectives["guide_adherence"] = adherence[0]

            outputs.append({"response": response})
            scores.append(final_score)
            objective_scores.append(case_objectives)

            if trajectories is not None:
                feedback = scored.feedback
                if adherence is not None:
                    feedback = f"{feedback}\nGuide adherence={adherence[0]:.2f}: {adherence[1]}"
                trajectories.append(
                    {
                        "case_id": case.id,
                        "input": case.input,
                        "conversation": conversation,
                        "expected": case.expected,
                        "output": response,
                        "feedback": feedback,
                        "score": behavior_score,
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
