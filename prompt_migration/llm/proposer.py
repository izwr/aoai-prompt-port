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
