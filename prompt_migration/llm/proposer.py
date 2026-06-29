from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from prompt_migration.llm.model import ReflectionLM


GUIDED_PROPOSER_TEMPLATE = """You are migrating a prompt to a target model.

Target-model prompting guide:
```
{guide}
```

Current prompt component `{component}`:
```
{current_prompt}
```

Failure traces and metric feedback:
```json
{feedback}
```

Rewrite only what is needed to recover or improve the failing cases on the target model.
Follow the target-model guide. Preserve task intent, labels, schemas, tools, and few-shot examples unless the feedback shows a target-model convention problem.
If the guide calls for XML-style tags or tagged prompt blocks, use those tags in the migrated prompt. Do not replace a target-model tag convention with Markdown headings.

Return only the new prompt component inside one fenced block.
"""


RESTRUCTURE_TEMPLATE = """You are porting a prompt to a target model's prompt structure.

Target-model prompting guide:
```
{guide}
```

Current prompt:
```
{naive_prompt}
```

Rewrite the prompt so its organization and formatting follow the target-model guide
(for example: section ordering, headings vs. XML-style tags, and the guide's recommended
prompt blocks). This is a structural port only.

Preserve verbatim, do not reword or drop:
- the task intent and instructions
- output labels and exact wording of required outputs
- JSON/schema field names and value formats
- few-shot examples
- tool/function definitions

Do not add new capabilities, constraints, or content. If the guide calls for XML-style
tags or tagged blocks, use them instead of Markdown headings.

Return only the rewritten prompt inside one fenced block.
"""


def build_guided_port(
    *,
    naive_prompt: str,
    target_prompt_guide: str,
    reflection_model: str | None = None,
    lm: Any | None = None,
    completion_kwargs: dict | None = None,
) -> str:
    """One-shot structural rewrite of ``naive_prompt`` into the target guide's shape.

    Returns ``naive_prompt`` unchanged when no guide is available, when the rewrite is
    empty/whitespace, or when it is implausibly short relative to the input (a likely
    truncated or refused rewrite), so a bad pass degrades to the naive port.
    """
    if not target_prompt_guide.strip():
        return naive_prompt

    model = lm if lm is not None else ReflectionLM(
        reflection_model,
        completion_kwargs=completion_kwargs or {"temperature": 0, "drop_params": True},
    )
    prompt = RESTRUCTURE_TEMPLATE.format(
        guide=target_prompt_guide.strip(),
        naive_prompt=naive_prompt,
    )
    rewritten = _extract_fenced_block(model(prompt))
    if not rewritten.strip():
        return naive_prompt
    # A structural port should not lose most of the prompt; guard against truncation.
    if len(rewritten) < 0.5 * len(naive_prompt.strip()):
        return naive_prompt
    return rewritten


def make_guided_proposer(
    *,
    target_prompt_guide: str,
    reflection_model: str,
    completion_kwargs: dict | None = None,
):
    lm = ReflectionLM(reflection_model, completion_kwargs=completion_kwargs)

    def propose(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        proposals: dict[str, str] = {}
        for component in components_to_update:
            records = list(reflective_dataset.get(component, []))
            if not records:
                continue
            prompt = GUIDED_PROPOSER_TEMPLATE.format(
                guide=target_prompt_guide.strip(),
                component=component,
                current_prompt=candidate[component],
                feedback=json.dumps(records, indent=2, ensure_ascii=False),
            )
            proposals[component] = _extract_fenced_block(lm(prompt))
        return proposals

    return propose


def _extract_fenced_block(text: str) -> str:
    match = re.search(r"```(?:\w+)?\s*(.*?)```", text, flags=re.DOTALL)
    return (match.group(1) if match else text).strip()
