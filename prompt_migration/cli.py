from __future__ import annotations

import argparse
import json
import webbrowser
from argparse import Namespace
from pathlib import Path

from prompt_migration.core.migrate import MigrationConfig, migrate_prompt, read_text_arg
from prompt_migration.llm.model import EchoRunner
from prompt_migration.reporting.report import write_outputs
from prompt_migration.utils.logging import set_verbose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="migrate",
        description="Migrate a prompt between Azure OpenAI GPT deployments with GEPA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-prompt", required=True, help="Source prompt text or path to a text file.")
    parser.add_argument("--source-model", required=True, help="Azure GPT deployment, for example azure/gpt-4o.")
    parser.add_argument("--target-model", required=True, help="Azure GPT deployment, for example azure/gpt-5.")
    parser.add_argument("--golden", required=True, help="Path to JSON/JSONL golden set.")
    parser.add_argument(
        "--input-format",
        choices=["chat", "qa", "qa-di"],
        default="chat",
        help=(
            "Golden-set input shape. 'chat' replays conversations/transcripts; "
            "'qa' sends an image plus question for structured extraction; "
            "'qa-di' adds Azure Document Intelligence output text alongside the image."
        ),
    )
    parser.add_argument("--guide", default="", help="Target prompt guide text or path to a text file.")
    parser.add_argument("--guide-url", help="Explicit prompt guide URL. Defaults to a URL derived from --target-model.")
    parser.add_argument("--no-default-guide", action="store_true", help="Do not fetch the default OpenAI prompt guide.")
    parser.add_argument("--reflection-model", help="Azure GPT deployment for GEPA reflection. Defaults to --target-model.")
    parser.add_argument(
        "--semantic-judge",
        choices=["llm", "embeddings"],
        default="llm",
        help="Semantic similarity scorer for semantic_and_length golden cases.",
    )
    parser.add_argument("--judge-model", default="azure/gpt-4.1", help="Azure GPT deployment for LLM semantic judging.")
    parser.add_argument(
        "--embedding-model",
        default="azure/text-embedding-3-large",
        help="Azure text-embedding deployment for embedding semantic judging.",
    )
    parser.add_argument(
        "--guide-adherence-weight",
        type=float,
        default=0.2,
        help=(
            "Weight in [0, 1) for an LLM guide-adherence metric blended into GEPA's objective. "
            "Default 0.2; set 0 to disable. Lets a restructured, guide-following prompt win when "
            "it preserves behavior. Uses --judge-model and requires Azure calls."
        ),
    )
    parser.add_argument(
        "--restructure-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rewrite the naive port into the target guide's structure with one LLM call and use it "
            "as GEPA's seed (on by default). GEPA then spends its budget on behavior, not on "
            "discovering structure. Skipped without a guide, under --echo-runner, or with --no-optimize."
        ),
    )
    parser.add_argument("--eval-repetitions", type=int, default=3, help="Repeated final eval samples for ship gate.")
    parser.add_argument("--ship-margin", type=float, default=0.0, help="Base score margin optimized must clear beyond observed noise.")
    parser.add_argument("--timeout-seconds", type=int, default=30, help="Per Azure model-call timeout.")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Number of GEPA optimization steps (reflective proposals). Recommended knob: each step "
            "is one reflect-and-propose round, independent of golden-set size. When set, "
            "--max-metric-calls becomes an optional safety ceiling."
        ),
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help=(
            "Hard ceiling on total GEPA metric calls (case rollouts). Defaults to 60 when --steps "
            "is not set; when --steps is set, acts only as an optional safety ceiling (no limit if omitted)."
        ),
    )
    parser.add_argument("--val-fraction", type=float, default=0.4)
    parser.add_argument("--run-dir")
    parser.add_argument("--output-dir", default="migration_out")
    parser.add_argument("--no-open-report", action="store_true", help="Do not open report.html after writing outputs.")
    parser.add_argument("--quiet", action="store_true", help="Hide progress logs.")
    parser.add_argument("--no-optimize", action="store_true", help="Only run source/source and naive target baselines.")
    parser.add_argument("--echo-runner", action="store_true", help="Use deterministic local runner for harness tests.")
    return parser


def resolve_max_metric_calls(args: Namespace) -> int | None:
    """Resolve the GEPA metric-call ceiling.

    Explicit ``--max-metric-calls`` always wins. Otherwise default to 60 unless ``--steps``
    is given, in which case the step count is the binding stop condition and there is no
    metric-call ceiling.
    """
    if args.max_metric_calls is not None:
        return args.max_metric_calls
    return None if args.steps is not None else 60


def config_from_args(args: Namespace) -> MigrationConfig:
    return MigrationConfig(
        source_prompt=read_text_arg(args.source_prompt),
        source_model=args.source_model,
        target_model=args.target_model,
        golden_path=args.golden,
        input_format=args.input_format,
        target_prompt_guide=read_text_arg(args.guide) if args.guide else "",
        target_prompt_guide_url=None if args.guide else args.guide_url,
        use_default_prompt_guide=not args.no_default_guide and not args.guide_url,
        reflection_model=args.reflection_model,
        judge_model=args.judge_model,
        semantic_judge=args.semantic_judge,
        embedding_model=args.embedding_model,
        guide_adherence_weight=args.guide_adherence_weight,
        restructure_seed=args.restructure_seed,
        eval_repetitions=args.eval_repetitions,
        ship_margin=args.ship_margin,
        timeout_seconds=args.timeout_seconds,
        max_metric_calls=resolve_max_metric_calls(args),
        steps=args.steps,
        val_fraction=args.val_fraction,
        run_dir=args.run_dir,
        optimize=not args.no_optimize,
    )


def print_summary(result_report: dict, output_dir: str) -> None:
    print(json.dumps({"decision": result_report["decision"], "scores": result_report["scores"]}, indent=2))
    print(f"Wrote migrated prompt and report to {output_dir}")


def open_report(output_dir: str) -> None:
    report_path = Path(output_dir) / "report.html"
    opened = webbrowser.open(report_path.resolve().as_uri())
    if not opened:
        print(f"Could not open report automatically: {report_path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.guide and args.guide_url:
        parser.error("--guide and --guide-url are mutually exclusive.")
    set_verbose(not args.quiet)
    try:
        config = config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    result = migrate_prompt(config, runner=EchoRunner() if args.echo_runner else None)
    write_outputs(output_dir=args.output_dir, migrated_prompt=result.migrated_prompt, report=result.report)
    print_summary(result.report, args.output_dir)
    if not args.no_open_report:
        open_report(args.output_dir)
    return 0
