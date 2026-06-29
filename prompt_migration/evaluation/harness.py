from __future__ import annotations

from prompt_migration.evaluation.metric import SemanticSimilarityJudge, score_output
from prompt_migration.evaluation.multimodal import message_api_content, message_display_text
from prompt_migration.evaluation.types import EvalSummary, GoldenCase, RunEvent, ScoreResult
from prompt_migration.llm.model import ModelRunner
from prompt_migration.utils.logging import progress


def evaluate_prompt(
    *,
    prompt_name: str,
    prompt: str,
    model: str,
    golden_set: list[GoldenCase],
    runner: ModelRunner,
    semantic_judge: SemanticSimilarityJudge | None = None,
    repetitions: int = 1,
    progress_label: str | None = None,
) -> EvalSummary:
    events: list[RunEvent] = []
    run_scores: list[float] = []

    for repetition in range(repetitions):
        if progress_label:
            progress(f"[eval] {progress_label}: repetition {repetition + 1}/{repetitions}")
        start_idx = len(events)
        for case_index, case in enumerate(golden_set, start=1):
            runner_messages = case_runner_messages(case)
            conversation = case_conversation_dicts(case)
            error: str | None = None
            try:
                if progress_label:
                    progress(
                        f"[eval] {progress_label}: case {case_index}/{len(golden_set)} "
                        f"id={case.id} model={model}"
                    )
                output = runner(model, prompt, runner_messages)
                scored = score_output(case, output, semantic_judge=semantic_judge)
            except Exception as exc:
                output = ""
                error = str(exc)
                scored = ScoreResult(0.0, f"Infrastructure error while evaluating case {case.id}: {exc}")

            events.append(
                RunEvent(
                    case_id=case.id,
                    model=model,
                    prompt_name=prompt_name,
                    input=case.input,
                    conversation=conversation,
                    expected=case.expected,
                    output=output,
                    score=scored.score,
                    feedback=scored.feedback,
                    objective_scores=scored.objective_scores,
                    error=error,
                    repetition=repetition,
                )
            )
        run_events = events[start_idx:]
        run_scores.append(sum(event.score for event in run_events) / len(run_events) if run_events else 0.0)

    aggregate = sum(run_scores) / len(run_scores) if run_scores else 0.0
    return EvalSummary(prompt_name=prompt_name, model=model, score=aggregate, events=events, run_scores=run_scores)


def case_conversation_dicts(case: GoldenCase) -> list[dict[str, str]]:
    """Text-only conversation dicts for reporting, judging, and reflection."""
    return [
        {"role": message.role, "content": message_display_text(message)}
        for message in case.conversation
    ]


def case_runner_messages(case: GoldenCase) -> list[dict[str, object]]:
    """Chat-API messages for the model, including image and document content parts."""
    return [
        {"role": message.role, "content": message_api_content(message)}
        for message in case.conversation
    ]
