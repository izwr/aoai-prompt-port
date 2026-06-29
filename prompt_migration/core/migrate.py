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
    AzureGuideAdherenceJudge,
    AzureSemanticSimilarityJudge,
    ModelRunner,
    ReflectionLM,
    validate_azure_deployment_model,
    validate_azure_gpt_model,
)
from prompt_migration.evaluation.metric import GuideAdherenceJudge
from prompt_migration.llm.proposer import build_guided_port, make_guided_proposer
from prompt_migration.reporting.report import build_report
from prompt_migration.utils.logging import progress

GEPA_COMPONENT = "system_prompt"


@dataclass(frozen=True)
class MigrationConfig:
    source_prompt: str
    source_model: str
    target_model: str
    golden_path: str
    input_format: str = "chat"
    target_prompt_guide: str = ""
    target_prompt_guide_url: str | None = None
    use_default_prompt_guide: bool = True
    reflection_model: str | None = None
    judge_model: str = "azure/gpt-4.1"
    semantic_judge: str = "llm"
    embedding_model: str = "azure/text-embedding-3-large"
    guide_adherence_weight: float = 0.2
    restructure_seed: bool = True
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
    golden_set = load_golden_set(config.golden_path, input_format=config.input_format)
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
    guide_adherence_judge = (
        resolve_guide_adherence_judge(config, guide, runner=runner) if config.optimize else None
    )

    restructured_prompt = baselines.naive_prompt
    restructured_eval = None
    if config.optimize:
        restructured_prompt = resolve_restructure_seed(
            config, baselines.naive_prompt, guide, runner=runner
        )
        if restructured_prompt != baselines.naive_prompt:
            restructured_eval = evaluate_prompt(
                prompt_name="restructured_port",
                prompt=restructured_prompt,
                model=config.target_model,
                golden_set=golden_set,
                runner=active_runner,
                semantic_judge=semantic_judge,
                repetitions=config.eval_repetitions,
                progress_label="restructured prompt on target model",
            )

        optimization = optimize_prompt(
            config=config,
            seed_prompt=restructured_prompt,
            golden_set=golden_set,
            runner=active_runner,
            semantic_judge=semantic_judge,
            guide=guide,
            guide_adherence_judge=guide_adherence_judge,
        )
        optimized_prompt = optimization.prompt
        gepa_metadata = optimization.metadata

    naive_adherence, restructured_adherence, optimized_adherence = resolve_report_adherence(
        guide_adherence_judge=guide_adherence_judge,
        guide=guide,
        naive_prompt=baselines.naive_prompt,
        restructured_prompt=restructured_prompt,
        optimized_prompt=optimized_prompt,
    )

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
        naive_guide_adherence=naive_adherence,
        optimized_guide_adherence=optimized_adherence,
        restructured_on_target=restructured_eval,
        restructured_prompt=restructured_prompt if restructured_eval is not None else None,
        restructured_guide_adherence=restructured_adherence if restructured_eval is not None else None,
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
    guide_adherence_judge: GuideAdherenceJudge | None = None,
) -> OptimizationResult:
    progress("[migration] Preparing GEPA optimization")
    trainset, valset = split_train_val(golden_set, val_fraction=config.val_fraction, seed=0)
    adapter = PromptMigrationAdapter(
        target_model=config.target_model,
        runner=runner,
        semantic_judge=semantic_judge,
        guide_text=guide.text,
        guide_adherence_judge=guide_adherence_judge,
        guide_weight=config.guide_adherence_weight,
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


def resolve_restructure_seed(
    config: MigrationConfig,
    naive_prompt: str,
    guide: PromptGuideResult,
    *,
    runner: ModelRunner | None = None,
) -> str:
    """The GEPA seed: the naive port rewritten into the target guide's structure.

    Returns the naive port unchanged (no LLM call) when restructuring is disabled, no
    guide is available, or an offline/echo runner is in use.
    """
    if not config.restructure_seed or not guide.text.strip() or runner is not None:
        return naive_prompt
    progress("[migration] Restructuring naive port to the target prompt guide")
    return build_guided_port(
        naive_prompt=naive_prompt,
        target_prompt_guide=guide.text,
        reflection_model=resolve_reflection_model(config),
    )


def resolve_report_adherence(
    *,
    guide_adherence_judge: GuideAdherenceJudge | None,
    guide: PromptGuideResult,
    naive_prompt: str,
    restructured_prompt: str,
    optimized_prompt: str,
) -> tuple[float | None, float | None, float | None]:
    """Guide-adherence scores for the naive, restructured, and optimized prompts.

    Reuses scores across identical prompts so the judge is called at most once per
    distinct prompt.
    """
    if guide_adherence_judge is None or not guide.text.strip():
        return None, None, None

    cache: dict[str, float] = {}

    def adherence(prompt: str) -> float:
        if prompt not in cache:
            cache[prompt] = guide_adherence_judge(prompt=prompt, guide=guide.text)[0]
        return cache[prompt]

    return adherence(naive_prompt), adherence(restructured_prompt), adherence(optimized_prompt)


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
    if config.input_format not in {"chat", "qa", "qa-di"}:
        raise ValueError("input_format must be one of 'chat', 'qa', or 'qa-di'.")
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
    if not 0.0 <= config.guide_adherence_weight < 1.0:
        raise ValueError("guide_adherence_weight must be in [0, 1).")


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


def resolve_guide_adherence_judge(
    config: MigrationConfig,
    guide: PromptGuideResult,
    *,
    runner: ModelRunner | None = None,
) -> GuideAdherenceJudge | None:
    if config.guide_adherence_weight <= 0.0 or not guide.text.strip():
        return None
    if runner is not None:
        raise RuntimeError(
            "guide_adherence_weight scoring requires Azure model calls; do not use --echo-runner."
        )
    return AzureGuideAdherenceJudge(config.judge_model)


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
