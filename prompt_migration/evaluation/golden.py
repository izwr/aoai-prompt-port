from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from prompt_migration.evaluation.multimodal import resolve_document_text, resolve_image_data_url
from prompt_migration.evaluation.types import ConversationMessage, GoldenCase

SUPPORTED_JUDGES = {"exact", "contains", "json_exact", "semantic_and_length"}
INPUT_FORMATS = {"chat", "qa", "qa-di"}


def load_golden_set(path: str | Path, input_format: str = "chat") -> list[GoldenCase]:
    if input_format not in INPUT_FORMATS:
        raise ValueError(f"input_format must be one of {sorted(INPUT_FORMATS)}; got {input_format!r}.")

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Golden set not found: {source}")

    raw_text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    else:
        raw = json.loads(raw_text)
        records = raw["cases"] if isinstance(raw, dict) and "cases" in raw else raw

    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list):
        raise ValueError("Golden set must be a JSON list, JSONL file, a transcript object, or an object with a 'cases' list.")

    cases: list[GoldenCase] = []
    for idx, record in enumerate(records):
        cases.extend(_parse_record(idx, record, input_format, base_dir=source.parent))
    if not cases:
        raise ValueError("Golden set must contain at least one case.")
    return cases


def _parse_record(idx: int, record: Any, input_format: str, *, base_dir: Path) -> list[GoldenCase]:
    if not isinstance(record, dict):
        raise ValueError(f"Golden case at index {idx} must be an object.")

    if input_format in {"qa", "qa-di"}:
        return [_parse_qa_case(idx, record, input_format, base_dir=base_dir)]

    if "expected" not in record and "answer" not in record:
        return _expand_transcript(idx, record)

    return [_parse_case(idx, record)]


def _parse_qa_case(idx: int, record: dict[str, Any], input_format: str, *, base_dir: Path) -> GoldenCase:
    case_id = str(record.get("id", idx))

    question = record.get("question", record.get("input", record.get("prompt")))
    if question is not None and not isinstance(question, str):
        raise ValueError(f"Golden case {case_id} 'question' must be a string.")

    images = _resolve_qa_images(case_id, record, base_dir=base_dir)

    document: str | None = None
    di_source = record.get("document_intelligence", record.get("document", record.get("di")))
    if input_format == "qa-di" and di_source is None:
        raise ValueError(f"Golden case {case_id} (qa-di) must include 'document_intelligence'.")
    if di_source is not None:
        document = resolve_document_text(di_source, base_dir=base_dir)

    if input_format == "qa" and not images:
        raise ValueError(f"Golden case {case_id} (qa) must include an 'image'.")
    if not images and not document and not question:
        raise ValueError(f"Golden case {case_id} must include an image, document, or question.")

    if "expected" in record:
        expected = record["expected"]
    elif "answer" in record:
        expected = record["answer"]
    else:
        raise ValueError(f"Golden case {case_id} must include 'expected' or 'answer'.")

    judge = record.get("judge", "json_exact")
    if judge not in SUPPORTED_JUDGES:
        raise ValueError(f"Golden case {case_id} has unsupported judge '{judge}'.")

    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Golden case {case_id} metadata must be an object.")

    message = ConversationMessage(
        role="user",
        content=question or "",
        images=tuple(images),
        document=document,
    )
    return GoldenCase(id=case_id, conversation=[message], expected=expected, judge=judge, metadata=metadata)


def _resolve_qa_images(case_id: str, record: dict[str, Any], *, base_dir: Path) -> list[str]:
    raw = record.get("image", record.get("images"))
    if raw is None:
        return []
    if isinstance(raw, str):
        raw_list = [raw]
    elif isinstance(raw, list):
        raw_list = raw
    else:
        raise ValueError(f"Golden case {case_id} 'image' must be a string or list of strings.")

    resolved: list[str] = []
    for item in raw_list:
        if not isinstance(item, str):
            raise ValueError(f"Golden case {case_id} 'image' entries must be strings.")
        resolved.append(resolve_image_data_url(item, base_dir=base_dir))
    return resolved


def _parse_case(idx: int, record: dict[str, Any]) -> GoldenCase:
    case_id = str(record.get("id", idx))
    conversation = _parse_conversation(case_id, record)

    if "expected" in record:
        expected = record["expected"]
    elif "answer" in record:
        expected = record["answer"]
    else:
        raise ValueError(f"Golden case {case_id} must include 'expected' or 'answer'.")

    judge = record.get("judge", "exact")
    if judge not in {"exact", "contains", "json_exact", "semantic_and_length"}:
        raise ValueError(f"Golden case {case_id} has unsupported judge '{judge}'.")

    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Golden case {case_id} metadata must be an object.")

    return GoldenCase(id=case_id, conversation=conversation, expected=expected, judge=judge, metadata=metadata)


def _expand_transcript(idx: int, record: dict[str, Any]) -> list[GoldenCase]:
    transcript_id = str(record.get("id", idx))
    transcript = _parse_conversation(transcript_id, record)
    judge = record.get("judge", "semantic_and_length")
    if judge not in {"exact", "contains", "json_exact", "semantic_and_length"}:
        raise ValueError(f"Transcript {transcript_id} has unsupported judge '{judge}'.")

    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Transcript {transcript_id} metadata must be an object.")

    cases: list[GoldenCase] = []
    for turn_index, message in enumerate(transcript):
        if message.role != "assistant":
            continue
        history = transcript[:turn_index]
        if not history:
            raise ValueError(f"Transcript {transcript_id} assistant turn {turn_index} has no prior conversation.")
        cases.append(
            GoldenCase(
                id=f"{transcript_id}:turn-{turn_index}",
                conversation=history,
                expected=message.content,
                judge=judge,
                metadata={**metadata, "transcript_id": transcript_id, "assistant_turn_index": turn_index},
            )
        )

    if not cases:
        raise ValueError(f"Transcript {transcript_id} must include at least one assistant turn.")
    return cases


def _parse_conversation(case_id: str, record: dict[str, Any]) -> list[ConversationMessage]:
    if "conversation" in record:
        raw_conversation = record["conversation"]
    elif "messages" in record:
        raw_conversation = record["messages"]
    elif "turns" in record:
        raw_conversation = record["turns"]
    elif "input" in record:
        input_text = record["input"]
        if not isinstance(input_text, str):
            raise ValueError(f"Golden case {case_id} field 'input' must be a string.")
        return [ConversationMessage(role="user", content=input_text)]
    else:
        raise ValueError(f"Golden case {case_id} must include 'input', 'conversation', or 'messages'.")

    if not isinstance(raw_conversation, list) or not raw_conversation:
        raise ValueError(f"Golden case {case_id} conversation must be a non-empty list.")

    conversation: list[ConversationMessage] = []
    for index, message in enumerate(raw_conversation):
        if not isinstance(message, dict):
            raise ValueError(f"Golden case {case_id} conversation item {index} must be an object.")
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"}:
            raise ValueError(
                f"Golden case {case_id} conversation item {index} role must be 'user' or 'assistant'."
            )
        if not isinstance(content, str):
            raise ValueError(f"Golden case {case_id} conversation item {index} content must be a string.")
        conversation.append(ConversationMessage(role=role, content=content))

    return conversation


def split_train_val(
    cases: list[GoldenCase],
    val_fraction: float = 0.4,
    *,
    shuffle: bool = True,
    seed: int = 0,
) -> tuple[list[GoldenCase], list[GoldenCase]]:
    if not cases:
        raise ValueError("Golden set must contain at least one case.")
    if len(cases) == 1:
        return cases, cases

    split_cases = list(cases)
    if shuffle:
        random.Random(seed).shuffle(split_cases)

    val_size = max(1, round(len(split_cases) * val_fraction))
    train_size = max(1, len(split_cases) - val_size)
    return split_cases[:train_size], split_cases[train_size:] or split_cases[:]
