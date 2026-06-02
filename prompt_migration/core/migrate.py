from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gepa

from prompt_migration.core.baselines import run_baselines
from prompt_migration.evaluation.golden import load_golden_set, split_train_val
from prompt_migration.evaluation.harness import evaluate_prompt
from prompt_migration.evaluation.metric import SemanticSimilarityJudge
from prompt_migration.evaluation.types import GoldenCase
from prompt_migration.llm.gepa_adapter import PromptMigrationAdapter
from prompt_migration.llm.guide import build_prompt_guide_url, fetch_prompt_guide
from prompt_migration.llm.model import (
    AzureEmbeddingSimilarityJudge,
    AzureGPTLiteLLMRunner,
    AzureSemanticSimilarityJudge,
    ModelRunner,
    ReflectionLM,
    validate_azure_deployment_model,
    validate_azure_gpt_model,
)
from prompt_migration.llm.proposer import make_guided_proposer
from prompt_migration.reporting.report import build_report
from prompt_migration.utils.logging import progress

GEPA_COMPONENT = "system_prompt"


@dataclass(frozen=True)
class MigrationConfig:
    source_prompt: str
    source_model: str
    target_model: str
    golden_path: str
    target_prompt_guide: str = ""
    target_prompt_guide_url: str | None = None
    use_default_prompt_guide: bool = True
    reflection_model: str | None = None
    judge_model: str = "azure/gpt-4.1"
    semantic_judge: str = "llm"
    embedding_model: str = "azure/text-embedding-3-large"
    eval_repetitions: int = 3
    ship_margin: float = 0.0
    timeout_seconds: int = 30
    max_metric_calls: int = 60
    val_fraction: float = 0.4
    run_dir: str | None = None
    optimize: bool = True


@dataclass(frozen=True)
class MigrationResult:
    migrated_prompt: str
    report: dict[str, Any]


@dataclass(frozen=True)
class PromptGuideResult:
    text: str
    source: str
    url: str | None = None


@dataclass(frozen=True)
class OptimizationResult:
    prompt: str
    metadata: dict[str, Any]


def migrate_prompt(config: MigrationConfig, runner: ModelRunner | None = None) -> MigrationResult:
    validate_migration_config(config)
    progress("[migration] Loading golden set and preparing runners")
    active_runner = runner or AzureGPTLiteLLMRunner(timeout_seconds=config.timeout_seconds)
    golden_set = load_golden_set(config.golden_path)
    semantic_judge = resolve_semantic_judge(config, golden_set, runner=runner)

    progress("[migration] Running source and naive baselines")
    baselines = run_baselines(
        source_prompt=config.source_prompt,
        source_model=config.source_model,
        target_model=config.target_model,
        golden_set=golden_set,
        runner=active_runner,
        semantic_judge=semantic_judge,
        repetitions=config.eval_repetitions,
    )

    optimized_prompt = baselines.naive_prompt
    guide = resolve_target_prompt_guide(config) if config.optimize else PromptGuideResult(text="", source="not_used")
    gepa_metadata = build_gepa_metadata(guide=guide, ran=False)

    if config.optimize:
        optimization = optimize_prompt(
            config=config,
            seed_prompt=baselines.naive_prompt,
            golden_set=golden_set,
            runner=active_runner,
            semantic_judge=semantic_judge,
            guide=guide,
        )
        optimized_prompt = optimization.prompt
        gepa_metadata = optimization.metadata

    optimized_eval = evaluate_prompt(
        prompt_name="optimized_prompt",
        prompt=optimized_prompt,
        model=config.target_model,
        golden_set=golden_set,
        runner=active_runner,
        semantic_judge=semantic_judge,
        repetitions=config.eval_repetitions,
        progress_label="optimized prompt on target model",
    )

    report = build_report(
        source_on_source=baselines.source_on_source,
        naive_on_target=baselines.naive_on_target,
        optimized_on_target=optimized_eval,
        naive_prompt=baselines.naive_prompt,
        optimized_prompt=optimized_prompt,
        gepa_metadata=gepa_metadata,
        ship_margin=config.ship_margin,
    )

    if report["decision"] != "ship":
        optimized_prompt = baselines.naive_prompt

    return MigrationResult(migrated_prompt=optimized_prompt, report=report)


def optimize_prompt(
    *,
    config: MigrationConfig,
    seed_prompt: str,
    golden_set: list[GoldenCase],
    runner: ModelRunner,
    semantic_judge: SemanticSimilarityJudge | None,
    guide: PromptGuideResult,
) -> OptimizationResult:
    progress("[migration] Preparing GEPA optimization")
    trainset, valset = split_train_val(golden_set, val_fraction=config.val_fraction, seed=0)
    adapter = PromptMigrationAdapter(
        target_model=config.target_model,
        runner=runner,
        semantic_judge=semantic_judge,
    )
    proposer = make_prompt_proposer(config, guide)

    progress("[migration] Starting GEPA optimization")
    result = gepa.optimize(
        seed_candidate={GEPA_COMPONENT: seed_prompt},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=None if proposer else ReflectionLM(resolve_reflection_model(config)),
        custom_candidate_proposer=proposer,
        max_metric_calls=config.max_metric_calls,
        run_dir=config.run_dir,
        module_selector="round_robin",
        raise_on_exception=False,
        display_progress_bar=True,
    )
    progress("[migration] GEPA optimization complete")

    best_candidate = result.best_candidate
    if not isinstance(best_candidate, dict):
        raise TypeError("Expected GEPA best_candidate to be a dict.")
    prompt = best_candidate.get(GEPA_COMPONENT)
    if not isinstance(prompt, str):
        raise TypeError(f"Expected GEPA best_candidate[{GEPA_COMPONENT!r}] to be a string.")

    return OptimizationResult(
        prompt=prompt,
        metadata=build_gepa_metadata(
            guide=guide,
            ran=True,
            best_idx=result.best_idx,
            num_candidates=result.num_candidates,
            val_aggregate_scores=result.val_aggregate_scores,
            total_metric_calls=result.total_metric_calls,
        ),
    )


def make_prompt_proposer(config: MigrationConfig, guide: PromptGuideResult):
    if not guide.text.strip():
        return None
    return make_guided_proposer(
        target_prompt_guide=guide.text,
        reflection_model=resolve_reflection_model(config),
    )


def build_gepa_metadata(*, guide: PromptGuideResult, ran: bool, **extra: Any) -> dict[str, Any]:
    return {
        "ran": ran,
        "prompt_guide": {"source": guide.source, "url": guide.url},
        **extra,
    }


def validate_migration_config(config: MigrationConfig) -> None:
    validate_azure_gpt_model(config.source_model, field_name="source_model")
    validate_azure_gpt_model(config.target_model, field_name="target_model")
    if config.reflection_model is not None:
        validate_azure_gpt_model(config.reflection_model, field_name="reflection_model")
    if config.semantic_judge not in {"llm", "embeddings"}:
        raise ValueError("semantic_judge must be either 'llm' or 'embeddings'.")
    if config.semantic_judge == "llm":
        validate_azure_gpt_model(config.judge_model, field_name="judge_model")
    if config.semantic_judge == "embeddings":
        validate_azure_deployment_model(
            config.embedding_model,
            field_name="embedding_model",
            example="azure/text-embedding-3-large",
        )
    if config.eval_repetitions < 1:
        raise ValueError("eval_repetitions must be at least 1.")
    if config.timeout_seconds < 1:
        raise ValueError("timeout_seconds must be at least 1.")
    if config.max_metric_calls < 1:
        raise ValueError("max_metric_calls must be at least 1.")
    if not 0.0 < config.val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if config.ship_margin < 0.0:
        raise ValueError("ship_margin must be non-negative.")


def resolve_reflection_model(config: MigrationConfig) -> str:
    return config.reflection_model or config.target_model


def resolve_semantic_judge(
    config: MigrationConfig,
    golden_set: list[GoldenCase],
    *,
    runner: ModelRunner | None = None,
) -> SemanticSimilarityJudge | None:
    if not any(case.judge == "semantic_and_length" for case in golden_set):
        return None
    if runner is not None:
        raise RuntimeError("semantic_and_length scoring requires Azure model calls; do not use --echo-runner.")
    if config.semantic_judge == "embeddings":
        return AzureEmbeddingSimilarityJudge(config.embedding_model, timeout_seconds=config.timeout_seconds)
    return AzureSemanticSimilarityJudge(config.judge_model)


def resolve_target_prompt_guide(config: MigrationConfig) -> PromptGuideResult:
    if config.target_prompt_guide.strip():
        return PromptGuideResult(text=config.target_prompt_guide, source="inline")
    if config.target_prompt_guide_url:
        return PromptGuideResult(
            text=fetch_prompt_guide(config.target_prompt_guide_url),
            source="url",
            url=config.target_prompt_guide_url,
        )
    if config.use_default_prompt_guide:
        url = build_prompt_guide_url(config.target_model)
        return PromptGuideResult(text=fetch_prompt_guide(url), source="target_model_default", url=url)
    return PromptGuideResult(text="", source="disabled")


def read_text_arg(value_or_path: str) -> str:
    path = Path(value_or_path)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    if path.exists():
        raise ValueError(f"Expected a file path or literal text, but got non-file path: {path}")
    return value_or_path
