from __future__ import annotations

import difflib
import html
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from prompt_migration.evaluation.types import EvalSummary


def prompt_diff(before: str, after: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="naive_port",
            tofile="optimized",
            lineterm="",
        )
    )


def observed_noise(summary: EvalSummary) -> float:
    return statistics.pstdev(summary.run_scores) if len(summary.run_scores) > 1 else 0.0


def should_ship(
    *,
    naive_score: float,
    optimized_score: float,
    error_count: int = 0,
    required_margin: float = 0.0,
    prompts_equal: bool = False,
) -> bool:
    if error_count:
        return False
    if prompts_equal:
        return optimized_score >= naive_score
    return optimized_score > naive_score + required_margin


def build_report(
    *,
    source_on_source: EvalSummary,
    naive_on_target: EvalSummary,
    optimized_on_target: EvalSummary,
    naive_prompt: str,
    optimized_prompt: str,
    gepa_metadata: dict[str, Any] | None = None,
    ship_margin: float = 0.0,
) -> dict[str, Any]:
    error_count = source_on_source.error_count + naive_on_target.error_count + optimized_on_target.error_count
    naive_noise = observed_noise(naive_on_target)
    optimized_noise = observed_noise(optimized_on_target)
    required_margin = ship_margin + naive_noise + optimized_noise
    prompts_equal = naive_prompt == optimized_prompt
    ship = should_ship(
        naive_score=naive_on_target.score,
        optimized_score=optimized_on_target.score,
        error_count=error_count,
        required_margin=required_margin,
        prompts_equal=prompts_equal,
    )
    reason = ship_reason(
        ship=ship,
        error_count=error_count,
        prompts_equal=prompts_equal,
        required_margin=required_margin,
    )
    return {
        "scores": {
            "source_prompt_on_source_model": source_on_source.score,
            "source_prompt_on_target_model_naive_port": naive_on_target.score,
            "optimized_prompt_on_target_model": optimized_on_target.score,
        },
        "decision": "ship" if ship else "do_not_ship",
        "reason": reason,
        "error_count": error_count,
        "ship_gate": {
            "base_margin": ship_margin,
            "naive_noise": naive_noise,
            "optimized_noise": optimized_noise,
            "required_margin": required_margin,
            "prompts_equal": prompts_equal,
            "naive_run_scores": naive_on_target.run_scores,
            "optimized_run_scores": optimized_on_target.run_scores,
        },
        "prompts": {
            "naive_port": naive_prompt,
            "optimized": optimized_prompt,
        },
        "diff": prompt_diff(naive_prompt, optimized_prompt),
        "events": {
            "source_on_source": [asdict(event) for event in source_on_source.events],
            "naive_on_target": [asdict(event) for event in naive_on_target.events],
            "optimized_on_target": [asdict(event) for event in optimized_on_target.events],
        },
        "gepa": gepa_metadata or {},
    }


def ship_reason(*, ship: bool, error_count: int, prompts_equal: bool, required_margin: float) -> str:
    if error_count:
        return f"Run had {error_count} infrastructure/model-call error(s); do not ship from this run."
    if prompts_equal and ship:
        return "Naive port selected; no rewrite was required."
    if ship:
        return f"Optimized prompt clears naive by required margin {required_margin:.4f}."
    return f"Optimized prompt did not clear naive by required margin {required_margin:.4f}."


def write_outputs(*, output_dir: str | Path, migrated_prompt: str, report: dict[str, Any]) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "migrated_prompt.txt").write_text(migrated_prompt, encoding="utf-8")
    (destination / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (destination / "report.html").write_text(render_report_html(report), encoding="utf-8")


def render_report_html(report: dict[str, Any]) -> str:
    scores = report.get("scores", {})
    decision = str(report.get("decision", "unknown"))
    is_ship = decision == "ship"
    prompts = report.get("prompts", {})
    naive_prompt = str(prompts.get("naive_port", ""))
    optimized_prompt = str(prompts.get("optimized", ""))
    guide = report.get("gepa", {}).get("prompt_guide", {})
    ship_gate = report.get("ship_gate", {})
    events = report.get("events", {})
    optimized_events = events.get("optimized_on_target", [])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prompt Migration Report</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #657181;
      --line: #d9dee7;
      --accent: #1463ff;
      --good: #0f7a4f;
      --bad: #a93535;
      --warn: #8a5a00;
      --code: #0f1720;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      padding: 22px 28px;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px 28px 40px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 24px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; letter-spacing: 0; }}
    .subtle {{ color: var(--muted); }}
    .grid {{ display: grid; gap: 16px; }}
    .summary {{ grid-template-columns: 1.2fr repeat(3, 1fr); align-items: stretch; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }}
    .decision {{
      border-color: {"#9ed7bd" if is_ship else "#e2b0b0"};
      background: {"#f0fbf6" if is_ship else "#fff5f5"};
    }}
    .decision strong {{ color: {"var(--good)" if is_ship else "var(--bad)"}; }}
    .score {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: #fafbfc;
      font-size: 12px;
    }}
    .two-col {{ grid-template-columns: 1fr 1fr; }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .prompt-box {{
      min-height: 420px;
      max-height: 680px;
      overflow: auto;
      color: #e9eef5;
      background: var(--code);
      border-radius: 6px;
      padding: 14px;
      border: 1px solid #263445;
    }}
    .diff {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
    }}
    .diff-row {{
      display: grid;
      grid-template-columns: 44px 1fr;
      min-height: 22px;
      font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      border-bottom: 1px solid #edf0f5;
    }}
    .diff-row:last-child {{ border-bottom: 0; }}
    .ln {{ color: var(--muted); text-align: right; padding: 2px 8px; user-select: none; }}
    .txt {{ padding: 2px 10px; white-space: pre-wrap; overflow-wrap: anywhere; }}
    .add {{ background: #ecfdf3; }}
    .del {{ background: #fff0f0; }}
    .hunk {{ background: #eef4ff; color: #315481; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; border-bottom: 1px solid var(--line); padding: 9px 8px; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; font-size: 12px; }}
    td.score-cell {{ width: 88px; font-variant-numeric: tabular-nums; }}
    @media (max-width: 900px) {{
      main, header {{ padding-left: 16px; padding-right: 16px; }}
      .summary, .two-col {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Prompt Migration Report</h1>
    <div class="subtle">Azure OpenAI prompt migration evaluation</div>
  </header>
  <main class="grid">
    <section class="grid summary">
      <div class="panel decision">
        <h2>Decision</h2>
        <strong>{html.escape(decision.replace("_", " ").title())}</strong>
        <p>{html.escape(str(report.get("reason", "")))}</p>
        <div class="meta">
          <span class="pill">Errors: {int(report.get("error_count", 0))}</span>
          <span class="pill">Guide: {html.escape(str(guide.get("source", "unknown")))}</span>
          <span class="pill">Required margin: {float(ship_gate.get("required_margin") or 0.0):.4f}</span>
        </div>
      </div>
      {score_panel("Source / Source", scores.get("source_prompt_on_source_model"))}
      {score_panel("Naive / Target", scores.get("source_prompt_on_target_model_naive_port"))}
      {score_panel("Optimized / Target", scores.get("optimized_prompt_on_target_model"))}
    </section>

    <section class="panel">
      <h2>Guide</h2>
      <div class="subtle">{html.escape(str(guide.get("url") or "No guide URL recorded"))}</div>
    </section>

    <section class="grid two-col">
      <div class="panel">
        <h2>Old Prompt: Naive Port</h2>
        <div class="prompt-box"><pre>{html.escape(naive_prompt)}</pre></div>
      </div>
      <div class="panel">
        <h2>New Prompt: Optimized</h2>
        <div class="prompt-box"><pre>{html.escape(optimized_prompt)}</pre></div>
      </div>
    </section>

    <section class="panel">
      <h2>Prompt Diff</h2>
      <div class="diff">{render_diff(report.get("diff", ""))}</div>
    </section>

    <section class="panel">
      <h2>Optimized Target Cases</h2>
      {render_events_table(optimized_events)}
    </section>
  </main>
</body>
</html>
"""


def score_panel(label: str, value: Any) -> str:
    score = float(value or 0.0)
    return f"""
      <div class="panel">
        <h2>{html.escape(label)}</h2>
        <div class="score">{score:.3f}</div>
      </div>
    """


def render_diff(diff_text: Any) -> str:
    lines = str(diff_text or "No prompt changes.").splitlines()
    rows: list[str] = []
    for idx, line in enumerate(lines, start=1):
        cls = ""
        if line.startswith("+") and not line.startswith("+++"):
            cls = " add"
        elif line.startswith("-") and not line.startswith("---"):
            cls = " del"
        elif line.startswith("@@") or line.startswith("---") or line.startswith("+++"):
            cls = " hunk"
        rows.append(
            f'<div class="diff-row{cls}"><div class="ln">{idx}</div>'
            f'<div class="txt">{html.escape(line)}</div></div>'
        )
    return "".join(rows)


def render_events_table(events: list[dict[str, Any]]) -> str:
    if not events:
        return '<div class="subtle">No case events recorded.</div>'
    rows = []
    for event in events:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(event.get('case_id', '')))}</td>"
            f"<td class=\"score-cell\">{float(event.get('score') or 0.0):.3f}</td>"
            f"<td>{html.escape(str(event.get('feedback', '')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Case</th><th>Score</th><th>Feedback</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
