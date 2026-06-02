from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptParts:
    system: str
    few_shot: str = ""
    schema: str = ""
    tools: str = ""


def extract_prompt_parts(prompt: str) -> PromptParts:
    """Lightweight placeholder parser for future role/few-shot/schema extraction."""
    return PromptParts(system=prompt.strip())


def build_naive_port(prompt: str) -> str:
    return extract_prompt_parts(prompt).system
