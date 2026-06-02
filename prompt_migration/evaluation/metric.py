from __future__ import annotations

import json
import re
from typing import Any, Protocol

from prompt_migration.evaluation.types import GoldenCase, ScoreResult

MIN_LENGTH_RATIO = 0.8
MAX_LENGTH_RATIO = 1.25


class SemanticSimilarityJudge(Protocol):
    def __call__(self, *, expected: str, actual: str, conversation: list[dict[str, str]]) -> tuple[float, str]: ...


def score_output(
    case: GoldenCase,
    output: str,
    *,
    semantic_judge: SemanticSimilarityJudge | None = None,
) -> ScoreResult:
    expected = case.expected
    actual = output.strip()

    if case.judge == "exact":
        expected_text = str(expected).strip()
        if actual == expected_text:
            return ScoreResult(1.0, "Correct: output exactly matches the expected label.")
        return ScoreResult(
            0.0,
            "Incorrect exact-match label. Preserve the required output format and return "
            f"exactly {expected_text!r}; got {actual!r}.",
        )

    if case.judge == "contains":
        expected_text = str(expected).strip()
        if expected_text in actual:
            return ScoreResult(1.0, "Correct: output contains the expected label.")
        return ScoreResult(
            0.0,
            f"Incorrect contains-match label. Include {expected_text!r} in the response without changing it.",
        )

    if case.judge == "json_exact":
        parsed = _parse_json(actual)
        if parsed == expected:
            return ScoreResult(1.0, "Correct: JSON output exactly matches the expected object.")
        if parsed is _JSON_PARSE_FAILED:
            return ScoreResult(
                0.0,
                "Invalid JSON. The target prompt should require only valid JSON with no prose or code fences.",
            )
        return ScoreResult(
            0.0,
            "Incorrect JSON label. Preserve the target schema and values exactly. "
            f"Expected {json.dumps(expected, sort_keys=True)}; got {json.dumps(parsed, sort_keys=True)}.",
        )

    if case.judge == "semantic_and_length":
        if semantic_judge is None:
            raise RuntimeError("semantic_and_length scoring requires a semantic similarity judge model.")
        conversation = [{"role": message.role, "content": message.content} for message in case.conversation]
        return _score_semantic_and_length(
            expected=str(expected),
            actual=actual,
            conversation=conversation,
            semantic_judge=semantic_judge,
        )

    raise ValueError(f"Unsupported judge: {case.judge}")


_JSON_PARSE_FAILED = object()


def _parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return _JSON_PARSE_FAILED


def _score_semantic_and_length(
    *,
    expected: str,
    actual: str,
    conversation: list[dict[str, str]],
    semantic_judge: SemanticSimilarityJudge,
) -> ScoreResult:
    semantic_score, judge_feedback = semantic_judge(expected=expected, actual=actual, conversation=conversation)
    length_score = _length_similarity(expected, actual)
    aggregate_score = min(semantic_score, length_score)
    feedback = (
        f"Metric semantic_similarity={semantic_score:.2f}. "
        f"Metric response_length={length_score:.2f}. "
        f"Expected about {_word_count(expected)} words; got {_word_count(actual)} words. "
        f"Semantic judge feedback: {judge_feedback} "
    )
    if semantic_score < 0.75:
        feedback += "Preserve the original bot turn's meaning, safety guidance, and next step. "
    if length_score < 0.75:
        feedback += "Keep the response length close to the original bot turn while preserving quality. "
    if semantic_score >= 0.85 and length_score >= 0.85:
        feedback += "The response is close to the original bot behavior on both metrics."

    return ScoreResult(
        score=aggregate_score,
        feedback=feedback.strip(),
        objective_scores={"semantic_similarity": semantic_score, "response_length": length_score},
    )


def _length_similarity(expected: str, actual: str) -> float:
    expected_words = _word_count(expected)
    actual_words = _word_count(actual)
    if expected_words == 0 and actual_words == 0:
        return 1.0
    if expected_words == 0 or actual_words == 0:
        return 0.0
    ratio = actual_words / expected_words
    if MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
        return 1.0
    if ratio < MIN_LENGTH_RATIO:
        return max(0.0, ratio / MIN_LENGTH_RATIO)
    return max(0.0, MAX_LENGTH_RATIO / ratio)


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _word_count(value: str) -> int:
    return len(_tokens(value))
