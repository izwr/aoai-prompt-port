from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from prompt_migration.llm.guide import (
    DEFAULT_TARGET_PROMPT_GUIDE_URL,
    build_prompt_guide_url,
    fetch_prompt_guide,
    normalize_prompt_guide_model,
    prompt_convention_addendum,
)
from prompt_migration.evaluation.harness import evaluate_prompt
from prompt_migration.evaluation.golden import load_golden_set, split_train_val
from prompt_migration.evaluation.metric import score_output
from prompt_migration.evaluation.types import EvalSummary, GoldenCase, RunEvent
from prompt_migration.core.migrate import (
    MigrationConfig,
    migrate_prompt,
    read_text_arg,
    resolve_reflection_model,
    resolve_target_prompt_guide,
    validate_migration_config,
)
from prompt_migration.llm.gepa_adapter import PromptMigrationAdapter
from prompt_migration.llm.model import (
    AzureEmbeddingSimilarityJudge,
    EchoRunner,
    _cosine_similarity,
    _extract_embeddings,
    resolve_azure_auth_kwargs,
    resolve_azure_endpoint_kwargs,
    validate_azure_deployment_model,
    validate_azure_gpt_model,
)
from prompt_migration.reporting.report import build_report


class MetricTests(unittest.TestCase):
    def test_exact_feedback(self) -> None:
        case = GoldenCase(id="a", conversation=[], expected="yes")

        self.assertEqual(score_output(case, "yes").score, 1.0)
        failed = score_output(case, "no")
        self.assertEqual(failed.score, 0.0)
        self.assertIn("exactly", failed.feedback)

    def test_json_exact(self) -> None:
        case = GoldenCase(id="a", conversation=[], expected={"label": "yes"}, judge="json_exact")

        self.assertEqual(score_output(case, '{"label":"yes"}').score, 1.0)
        self.assertIn("Invalid JSON", score_output(case, "label: yes").feedback)

    def test_semantic_and_length_returns_two_metrics(self) -> None:
        def judge(*, expected, actual, conversation):
            return (1.0 if expected == actual else 0.5), "model-judged"

        case = GoldenCase(
            id="a",
            conversation=[],
            expected="Please check the official banking app and keep your payment reference.",
            judge="semantic_and_length",
        )

        close = score_output(
            case,
            "Please check the official banking app and keep your payment reference.",
            semantic_judge=judge,
        )
        too_short = score_output(case, "Check the app.", semantic_judge=judge)

        self.assertGreater(close.score, too_short.score)
        self.assertIn("semantic_similarity", close.objective_scores or {})
        self.assertIn("response_length", close.objective_scores or {})

    def test_semantic_and_length_requires_model_judge(self) -> None:
        case = GoldenCase(id="a", conversation=[], expected="x", judge="semantic_and_length")

        with self.assertRaisesRegex(RuntimeError, "semantic similarity judge model"):
            score_output(case, "x")

    def test_response_length_metric_has_full_credit_band(self) -> None:
        def judge(*, expected, actual, conversation):
            return 1.0, "same meaning"

        case = GoldenCase(
            id="a",
            conversation=[],
            expected="one two three four five six seven eight nine ten",
            judge="semantic_and_length",
        )

        in_band = score_output(case, "one two three four five six seven eight", semantic_judge=judge)
        too_long = score_output(
            case,
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen",
            semantic_judge=judge,
        )

        self.assertEqual((in_band.objective_scores or {})["response_length"], 1.0)
        self.assertLess((too_long.objective_scores or {})["response_length"], 1.0)


class MigrationTests(unittest.TestCase):
    def test_eval_summary_flags_infra_errors(self) -> None:
        def failing_runner(model, system_prompt, conversation):
            raise RuntimeError("rate limit")

        case = GoldenCase(id="case", conversation=[], expected="x")
        summary = evaluate_prompt(
            prompt_name="p",
            prompt="prompt",
            model="azure/gpt-4o",
            golden_set=[case],
            runner=failing_runner,
        )

        self.assertEqual(summary.error_count, 1)
        self.assertIn("Infrastructure error", summary.events[0].feedback)
        self.assertEqual(summary.events[0].error, "rate limit")

    def test_no_optimize_migration_uses_naive_port(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            golden = Path(temp_dir) / "golden.json"
            golden.write_text(
                json.dumps(
                    [
                        {"id": "1", "input": "task => A", "expected": "A"},
                        {"id": "2", "input": "task => B", "expected": "B"},
                    ]
                )
            )

            result = migrate_prompt(
                MigrationConfig(
                    source_prompt="Return only the label.",
                    source_model="azure/gpt-4o",
                    target_model="azure/gpt-5",
                    golden_path=str(golden),
                    optimize=False,
                ),
                runner=EchoRunner(),
            )

            self.assertEqual(result.migrated_prompt, "Return only the label.")
            self.assertEqual(result.report["decision"], "ship")
            self.assertEqual(result.report["scores"]["source_prompt_on_target_model_naive_port"], 1.0)

    def test_report_blocks_ship_when_eval_has_errors(self) -> None:
        def failing_runner(model, system_prompt, conversation):
            raise RuntimeError("network blip")

        with tempfile.TemporaryDirectory() as temp_dir:
            golden = Path(temp_dir) / "golden.json"
            golden.write_text(json.dumps([{"id": "1", "input": "task => A", "expected": "A"}]))

            result = migrate_prompt(
                MigrationConfig(
                    source_prompt="Return only the label.",
                    source_model="azure/gpt-4o",
                    target_model="azure/gpt-5",
                    golden_path=str(golden),
                    optimize=False,
                ),
                runner=failing_runner,
            )

        self.assertEqual(result.report["decision"], "do_not_ship")
        self.assertEqual(result.report["error_count"], 9)

    def test_ship_gate_requires_margin_for_changed_prompt(self) -> None:
        base_event = RunEvent(
            case_id="case",
            model="azure/gpt",
            prompt_name="p",
            input="",
            conversation=[],
            expected="",
            output="",
            score=0.0,
            feedback="",
        )
        source = EvalSummary("source", "azure/gpt", 1.0, [base_event], run_scores=[1.0, 1.0, 1.0])
        naive = EvalSummary("naive", "azure/gpt", 0.80, [base_event], run_scores=[0.78, 0.80, 0.82])
        optimized = EvalSummary("optimized", "azure/gpt", 0.81, [base_event], run_scores=[0.79, 0.81, 0.83])

        report = build_report(
            source_on_source=source,
            naive_on_target=naive,
            optimized_on_target=optimized,
            naive_prompt="old",
            optimized_prompt="new",
            ship_margin=0.0,
        )

        self.assertEqual(report["decision"], "do_not_ship")
        self.assertGreater(report["ship_gate"]["required_margin"], 0.0)


    def test_rejects_non_azure_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "Azure OpenAI GPT"):
            validate_azure_gpt_model("openai/gpt-4o")

    def test_rejects_non_gpt_azure_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "Azure GPT"):
            validate_azure_gpt_model("azure/claude-sonnet")

    def test_accepts_azure_embedding_deployment_model(self) -> None:
        self.assertEqual(
            validate_azure_deployment_model("azure/text-embedding-3-large"),
            "azure/text-embedding-3-large",
        )

    def test_embedding_cosine_similarity(self) -> None:
        self.assertEqual(_cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertEqual(_cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_extracts_embedding_response_vectors(self) -> None:
        response = {"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]}

        self.assertEqual(_extract_embeddings(response), [[1.0, 0.0], [0.0, 1.0]])

    def test_embedding_similarity_judge_scores_with_litellm_embedding(self) -> None:
        judge = AzureEmbeddingSimilarityJudge.__new__(AzureEmbeddingSimilarityJudge)
        judge.model = "azure/text-embedding-3-large"
        judge.azure_kwargs = {}
        judge.embedding_kwargs = {}
        judge.max_retries = 0
        judge.retry_backoff_seconds = 0.0

        with patch.object(
            judge,
            "_embed",
            return_value=[[1.0, 0.0], [1.0, 0.0]],
        ):
            score, feedback = judge(expected="same", actual="same", conversation=[])

        self.assertEqual(score, 1.0)
        self.assertIn("Embedding cosine similarity", feedback)

    def test_azure_openai_token_takes_precedence(self) -> None:
        with patch.dict(os.environ, {"AZURE_OPENAI_TOKEN": "env-token"}, clear=False):
            auth_kwargs = resolve_azure_auth_kwargs()

        self.assertEqual(auth_kwargs, {"azure_ad_token": "env-token"})

    def test_azure_cli_credential_is_fallback(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("azure.identity.AzureCliCredential") as credential_cls:
                with patch("azure.identity.get_bearer_token_provider") as provider_fn:
                    provider = MagicMock(name="token_provider")
                    provider_fn.return_value = provider

                    auth_kwargs = resolve_azure_auth_kwargs()

        credential_cls.assert_called_once_with()
        provider_fn.assert_called_once()
        self.assertEqual(auth_kwargs, {"azure_ad_token_provider": provider})

    def test_azure_openai_url_sets_api_base(self) -> None:
        with patch.dict(
            os.environ,
            {"AZURE_OPENAI_URL": "https://example.openai.azure.com/", "AZURE_API_VERSION": "2024-10-21"},
            clear=True,
        ):
            endpoint_kwargs = resolve_azure_endpoint_kwargs()

        self.assertEqual(
            endpoint_kwargs,
            {"api_base": "https://example.openai.azure.com", "api_version": "2024-10-21"},
        )

    def test_azure_openai_url_precedes_azure_api_base(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_URL": "https://primary.openai.azure.com",
                "AZURE_API_BASE": "https://fallback.openai.azure.com",
            },
            clear=True,
        ):
            endpoint_kwargs = resolve_azure_endpoint_kwargs()

        self.assertEqual(endpoint_kwargs, {"api_base": "https://primary.openai.azure.com"})

    def test_azure_api_base_is_fallback_endpoint(self) -> None:
        with patch.dict(os.environ, {"AZURE_API_BASE": "https://fallback.openai.azure.com/"}, clear=True):
            endpoint_kwargs = resolve_azure_endpoint_kwargs()

        self.assertEqual(endpoint_kwargs, {"api_base": "https://fallback.openai.azure.com"})

    def test_required_endpoint_fails_before_litellm(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "AZURE_OPENAI_URL"):
                resolve_azure_endpoint_kwargs(required=True)

    def test_fetch_prompt_guide_extracts_gpt_41_section(self) -> None:
        html = """
        <html><body>
        <nav>Ignore navigation</nav>
        <h1>GPT-4.1 prompting guide</h1>
        <p>Use precise instructions.</p>
        <h1>Was this page useful?</h1>
        <p>Footer</p>
        </body></html>
        """

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return html.encode("utf-8")

        with patch("prompt_migration.llm.guide.urlopen", return_value=Response()):
            guide = fetch_prompt_guide()

        self.assertIn(DEFAULT_TARGET_PROMPT_GUIDE_URL, guide)
        self.assertIn("GPT-4.1 prompting guide", guide)
        self.assertIn("Use precise instructions.", guide)
        self.assertNotIn("Footer", guide)

    def test_fetch_prompt_guide_extracts_requested_model_section(self) -> None:
        html = """
        <html><body>
        <section data-prompting-guide-model="gpt-5.5">
          <h1>GPT-5.5 prompting guide</h1>
          <p>Wrong section.</p>
        </section>
        <section data-prompting-guide-model="gpt-5.4">
          <h1>GPT-5.4 prompting guide</h1>
          <p>Use XML-style tags.</p>
        </section>
        <section data-prompting-guide-model="gpt-5.3">
          <h1>GPT-5.3 prompting guide</h1>
          <p>Next section.</p>
        </section>
        </body></html>
        """

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return html.encode("utf-8")

        with patch("prompt_migration.llm.guide.urlopen", return_value=Response()):
            guide = fetch_prompt_guide("https://developers.openai.com/api/docs/guides/prompt-guidance?model=gpt-5.4")

        self.assertIn("GPT-5.4 prompting guide", guide)
        self.assertIn("Use XML-style tags.", guide)
        self.assertIn("Target convention addendum for GPT-5.4", guide)
        self.assertNotIn("Wrong section.", guide)
        self.assertNotIn("Next section.", guide)

    def test_gpt_54_addendum_prefers_tags(self) -> None:
        addendum = prompt_convention_addendum("gpt-5.4")

        self.assertIn("<output_contract>", addendum)
        self.assertIn("XML-style section tags", addendum)

    def test_inline_guide_overrides_default_guide_url(self) -> None:
        config = MigrationConfig(
            source_prompt="p",
            source_model="azure/gpt-4o",
            target_model="azure/gpt-5",
            golden_path="unused.json",
            target_prompt_guide="custom guide",
        )

        with patch("prompt_migration.core.migrate.fetch_prompt_guide") as fetch:
            guide = resolve_target_prompt_guide(config)

        fetch.assert_not_called()
        self.assertEqual(guide.text, "custom guide")
        self.assertEqual(guide.source, "inline")

    def test_explicit_guide_url_still_used_when_default_disabled(self) -> None:
        config = MigrationConfig(
            source_prompt="p",
            source_model="azure/gpt-4o",
            target_model="azure/gpt-5",
            golden_path="unused.json",
            target_prompt_guide_url="https://example.test/guide",
            use_default_prompt_guide=False,
        )

        with patch("prompt_migration.core.migrate.fetch_prompt_guide", return_value="guide") as fetch:
            guide = resolve_target_prompt_guide(config)

        fetch.assert_called_once_with("https://example.test/guide")
        self.assertEqual(guide.text, "guide")
        self.assertEqual(guide.source, "url")

    def test_prompt_guide_model_normalization(self) -> None:
        cases = {
            "azure/gpt-4o": "gpt-4.1",
            "azure/gpt-4": "gpt-4.1",
            "azure/gpt-3.5-turbo": "gpt-4.1",
            "azure/gpt-4.1-mini": "gpt-4.1",
            "azure/gpt-4.1-2025-04-14": "gpt-4.1",
            "azure/gpt-5-mini": "gpt-5",
            "azure/gpt-5.5-mini": "gpt-5.5",
            "azure/prod-gpt-5.5-mini-eastus": "gpt-5.5",
        }

        for model, expected in cases.items():
            with self.subTest(model=model):
                self.assertEqual(normalize_prompt_guide_model(model), expected)

    def test_dynamic_default_guide_url_uses_target_model_family(self) -> None:
        config = MigrationConfig(
            source_prompt="p",
            source_model="azure/gpt-4o",
            target_model="azure/gpt-5.5-mini",
            golden_path="unused.json",
        )

        with patch("prompt_migration.core.migrate.fetch_prompt_guide", return_value="guide") as fetch:
            guide = resolve_target_prompt_guide(config)

        fetch.assert_called_once_with(build_prompt_guide_url("azure/gpt-5.5-mini"))
        self.assertIn("model=gpt-5.5", fetch.call_args.args[0])
        self.assertEqual(guide.text, "guide")
        self.assertEqual(guide.source, "target_model_default")
        self.assertIn("model=gpt-5.5", guide.url or "")

    def test_no_default_guide_disables_dynamic_fetch(self) -> None:
        config = MigrationConfig(
            source_prompt="p",
            source_model="azure/gpt-4o",
            target_model="azure/gpt-5.5-mini",
            golden_path="unused.json",
            use_default_prompt_guide=False,
        )

        with patch("prompt_migration.core.migrate.fetch_prompt_guide") as fetch:
            guide = resolve_target_prompt_guide(config)

        fetch.assert_not_called()
        self.assertEqual(guide.text, "")
        self.assertEqual(guide.source, "disabled")

    def test_reflection_model_defaults_to_target_model(self) -> None:
        config = MigrationConfig(
            source_prompt="p",
            source_model="azure/gpt-4o-mini",
            target_model="azure/gpt-5.4-mini",
            golden_path="unused.json",
        )

        self.assertEqual(resolve_reflection_model(config), "azure/gpt-5.4-mini")

    def test_rejects_invalid_numeric_config(self) -> None:
        config = MigrationConfig(
            source_prompt="p",
            source_model="azure/gpt-4o",
            target_model="azure/gpt-5",
            golden_path="unused.json",
            val_fraction=1.0,
        )

        with self.assertRaisesRegex(ValueError, "val_fraction"):
            validate_migration_config(config)

    def test_embedding_semantic_judge_allows_non_gpt_embedding_deployment(self) -> None:
        config = MigrationConfig(
            source_prompt="p",
            source_model="azure/gpt-4o",
            target_model="azure/gpt-5",
            golden_path="unused.json",
            semantic_judge="embeddings",
            embedding_model="azure/text-embedding-3-large",
            judge_model="azure/not-a-gpt-judge",
        )

        validate_migration_config(config)

    def test_read_text_arg_rejects_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "non-file path"):
                read_text_arg(temp_dir)

    def test_rejects_empty_golden_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            golden = Path(temp_dir) / "golden.json"
            golden.write_text("[]")

            with self.assertRaisesRegex(ValueError, "at least one case"):
                load_golden_set(golden)

    def test_loads_conversation_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            golden = Path(temp_dir) / "golden.json"
            golden.write_text(
                json.dumps(
                    [
                        {
                            "id": "conversation",
                            "conversation": [
                                {"role": "user", "content": "first"},
                                {"role": "assistant", "content": "second"},
                                {"role": "user", "content": "task => done"},
                            ],
                            "expected": "done",
                        }
                    ]
                )
            )

            cases = load_golden_set(golden)

        self.assertEqual(cases[0].input, "task => done")
        self.assertEqual(len(cases[0].conversation), 3)
        self.assertEqual(cases[0].conversation[1].role, "assistant")

    def test_expands_original_bot_transcript_into_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            golden = Path(temp_dir) / "golden.json"
            golden.write_text(
                json.dumps(
                    {
                        "id": "transcript",
                        "turns": [
                            {"role": "user", "content": "first question"},
                            {"role": "assistant", "content": "first answer"},
                            {"role": "user", "content": "follow up"},
                            {"role": "assistant", "content": "second answer"},
                        ],
                    }
                )
            )

            cases = load_golden_set(golden)

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].id, "transcript:turn-1")
        self.assertEqual(cases[0].input, "first question")
        self.assertEqual(cases[0].expected, "first answer")
        self.assertEqual(cases[0].judge, "semantic_and_length")
        self.assertEqual([message.content for message in cases[1].conversation], ["first question", "first answer", "follow up"])
        self.assertEqual(cases[1].expected, "second answer")

    def test_train_val_split_is_seeded_shuffle(self) -> None:
        cases = [GoldenCase(id=str(idx), conversation=[], expected=str(idx)) for idx in range(8)]

        train_a, val_a = split_train_val(cases, val_fraction=0.25, seed=7)
        train_b, val_b = split_train_val(cases, val_fraction=0.25, seed=7)

        self.assertEqual([case.id for case in train_a], [case.id for case in train_b])
        self.assertEqual([case.id for case in val_a], [case.id for case in val_b])
        self.assertNotEqual([case.id for case in train_a + val_a], [case.id for case in cases])

    def test_gepa_adapter_keeps_objectives_for_mixed_judges(self) -> None:
        def judge(*, expected, actual, conversation):
            return 1.0, "same enough"

        cases = [
            GoldenCase(id="semantic", conversation=[], expected="semantic reply", judge="semantic_and_length"),
            GoldenCase(id="exact", conversation=[], expected="", judge="exact"),
        ]
        adapter = PromptMigrationAdapter(
            target_model="azure/gpt-5",
            runner=EchoRunner(),
            semantic_judge=judge,
        )

        batch = adapter.evaluate(cases, {"system_prompt": ""})

        self.assertIsNotNone(batch.objective_scores)
        assert batch.objective_scores is not None
        self.assertEqual(batch.objective_scores[0].keys(), {"semantic_similarity", "response_length"})
        self.assertEqual(batch.objective_scores[1], {"score": 1.0})

    def test_gepa_adapter_uses_scalar_objective_for_exact_sets(self) -> None:
        cases = [GoldenCase(id="exact", conversation=[], expected="", judge="exact")]
        adapter = PromptMigrationAdapter(target_model="azure/gpt-5", runner=EchoRunner())

        batch = adapter.evaluate(cases, {"system_prompt": ""})

        self.assertEqual(batch.scores, [1.0])
        self.assertEqual(batch.objective_scores, [{"score": 1.0}])

    def test_rejects_invalid_conversation_role(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            golden = Path(temp_dir) / "golden.json"
            golden.write_text(
                json.dumps(
                    [
                        {
                            "conversation": [{"role": "system", "content": "nope"}],
                            "expected": "x",
                        }
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "role must be 'user' or 'assistant'"):
                load_golden_set(golden)


if __name__ == "__main__":
    unittest.main()
