from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Protocol, Sequence, TypedDict

AZURE_OPENAI_SCOPE = "https://cognitiveservices.azure.com/.default"


class ChatMessage(TypedDict):
    role: str
    content: "str | list[dict[str, Any]]"


class ModelRunner(Protocol):
    def __call__(self, model: str, system_prompt: str, conversation: Sequence[ChatMessage]) -> str: ...


def validate_azure_deployment_model(
    model: str,
    *,
    field_name: str = "model",
    example: str = "azure/gpt-4o",
) -> str:
    if not model.startswith("azure/"):
        raise ValueError(
            f"{field_name} must be an Azure OpenAI deployment in LiteLLM format, "
            f"for example {example!r}. Got {model!r}."
        )

    deployment = model.removeprefix("azure/")
    if not deployment:
        raise ValueError(f"{field_name} must include a deployment name after 'azure/'. Got {model!r}.")
    return model


def validate_azure_gpt_model(model: str, *, field_name: str = "model") -> str:
    if not model.startswith("azure/"):
        raise ValueError(
            f"{field_name} must be an Azure OpenAI GPT deployment in LiteLLM format, "
            f"for example 'azure/gpt-4o'. Got {model!r}."
        )
    validate_azure_deployment_model(model, field_name=field_name, example="azure/gpt-4o")
    deployment = model.removeprefix("azure/")
    if not deployment or "gpt" not in deployment.lower():
        raise ValueError(
            f"{field_name} must reference an Azure GPT deployment. "
            f"Use a deployment name containing 'gpt', for example 'azure/gpt-4o'. Got {model!r}."
        )

    return model


def resolve_azure_auth_kwargs() -> dict:
    env_token = os.environ.get("AZURE_OPENAI_TOKEN")
    if env_token:
        return {"azure_ad_token": env_token}

    try:
        from azure.identity import AzureCliCredential, get_bearer_token_provider
    except ImportError as exc:
        raise RuntimeError(
            "Azure CLI authentication requires the 'azure-identity' package. "
            "Install project dependencies with 'uv sync'."
        ) from exc

    credential = AzureCliCredential()
    return {"azure_ad_token_provider": get_bearer_token_provider(credential, AZURE_OPENAI_SCOPE)}


def resolve_azure_endpoint_kwargs(*, required: bool = False) -> dict:
    endpoint = (os.environ.get("AZURE_OPENAI_URL") or os.environ.get("AZURE_API_BASE") or "").strip()
    api_version = (os.environ.get("AZURE_API_VERSION") or "").strip()
    if required and not endpoint:
        raise RuntimeError(
            "Azure OpenAI endpoint is required. Set AZURE_OPENAI_URL, for example "
            "'https://YOUR-RESOURCE.openai.azure.com'. AZURE_API_BASE is also supported as a fallback."
        )
    if not endpoint:
        return {}

    kwargs = {"api_base": endpoint.rstrip("/")}
    if api_version:
        kwargs["api_version"] = api_version
    return kwargs


def azure_call_context(model: str) -> str:
    endpoint = resolve_azure_endpoint_kwargs(required=False).get("api_base") or "<not set>"
    api_version = os.environ.get("AZURE_API_VERSION") or "<not set>"
    deployment = model.removeprefix("azure/")
    return f"model={model!r}, deployment={deployment!r}, api_base={endpoint!r}, api_version={api_version!r}"


DEFAULT_TIMEOUT_SECONDS = 30


class AzureGPTLiteLLMRunner:
    def __init__(
        self,
        completion_kwargs: dict | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.azure_kwargs = {**resolve_azure_endpoint_kwargs(required=True), **resolve_azure_auth_kwargs()}
        self.completion_kwargs = completion_kwargs or {}
        self.completion_kwargs.setdefault("timeout", timeout_seconds)
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def __call__(self, model: str, system_prompt: str, conversation: Sequence[ChatMessage]) -> str:
        import litellm

        validate_azure_gpt_model(model)
        response = _completion_with_retries(
            litellm=litellm,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                *conversation,
            ],
            azure_kwargs=self.azure_kwargs,
            completion_kwargs=self.completion_kwargs,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            failure_message=(
                "Azure OpenAI task-model call failed. Check that AZURE_OPENAI_URL points to the correct "
                "resource and that the deployment after 'azure/' exists in that resource."
            ),
        )
        content = response.choices[0].message.content
        return (content or "").strip()


class EchoRunner:
    """Deterministic test runner: returns the input, or text after '=> ' if present."""

    def __call__(self, model: str, system_prompt: str, conversation: Sequence[ChatMessage]) -> str:
        from prompt_migration.evaluation.multimodal import message_text

        user_input = message_text(conversation[-1]["content"]) if conversation else ""
        if "=>" in user_input:
            return user_input.rsplit("=>", 1)[1].strip()
        return user_input.strip()


class ReflectionLM:
    def __init__(
        self,
        model: str,
        completion_kwargs: dict | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.model = validate_azure_gpt_model(model, field_name="reflection_model")
        self.azure_kwargs = {**resolve_azure_endpoint_kwargs(required=True), **resolve_azure_auth_kwargs()}
        self.completion_kwargs = completion_kwargs or {}
        self.completion_kwargs.setdefault("timeout", timeout_seconds)
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def __call__(self, prompt: str | list[dict]) -> str:
        import litellm

        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        response = _completion_with_retries(
            litellm=litellm,
            model=self.model,
            messages=messages,
            azure_kwargs=self.azure_kwargs,
            completion_kwargs=self.completion_kwargs,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            failure_message=(
                "Azure OpenAI reflection-model call failed. Check that --reflection-model names an existing "
                "deployment in AZURE_OPENAI_URL. If omitted, reflection uses --target-model."
            ),
        )
        content = response.choices[0].message.content
        return (content or "").strip()


class AzureSemanticSimilarityJudge:
    def __init__(self, model: str):
        self.lm = ReflectionLM(model, completion_kwargs={"temperature": 0, "drop_params": True})

    def __call__(self, *, expected: str, actual: str, conversation: list[dict[str, str]]) -> tuple[float, str]:
        prompt = (
            "You are judging whether a migrated support-bot response preserves the meaning of an "
            "original bot response.\n\n"
            "Score semantic similarity only. Ignore response length; another metric handles length.\n"
            "Consider: safety guidance, policy caveats, next step, tone intent, and factual constraints.\n"
            "Return strict JSON only: {\"score\": number between 0 and 1, \"feedback\": string}.\n\n"
            f"Conversation before assistant turn:\n{json.dumps(conversation, indent=2)}\n\n"
            f"Original assistant response:\n{expected}\n\n"
            f"Migrated assistant response:\n{actual}\n"
        )
        raw = self.lm(prompt)
        data = _parse_judge_json(raw)
        score = data.get("score")
        feedback = data.get("feedback", "")
        if not isinstance(score, int | float):
            raise RuntimeError(f"Semantic judge returned invalid score: {raw!r}")
        if not isinstance(feedback, str):
            feedback = str(feedback)
        return max(0.0, min(1.0, float(score))), feedback.strip()


class AzureGuideAdherenceJudge:
    """LLM judge scoring how well a prompt follows a target-model prompting guide.

    Scores only structural/stylistic adherence to the guide (section and tag
    conventions, instruction placement, formatting, recommended patterns), not the
    task content or expected behavior. Returns a (score, feedback) pair in [0, 1].
    """

    def __init__(self, model: str):
        self.lm = ReflectionLM(model, completion_kwargs={"temperature": 0, "drop_params": True})

    def __call__(self, *, prompt: str, guide: str) -> tuple[float, str]:
        judge_prompt = (
            "You are auditing whether a prompt follows a target model's prompting guide.\n\n"
            "Score only structural and stylistic adherence to the guide: section/tag conventions, "
            "instruction placement, formatting, and recommended prompt patterns. Do NOT reward or "
            "penalize the task content, domain, or expected behavior itself.\n"
            "Return strict JSON only: {\"score\": number between 0 and 1, \"feedback\": string}.\n\n"
            f"Target-model prompting guide:\n{guide}\n\n"
            f"Prompt under review:\n{prompt}\n"
        )
        raw = self.lm(judge_prompt)
        data = _parse_judge_json(raw)
        score = data.get("score")
        feedback = data.get("feedback", "")
        if not isinstance(score, int | float):
            raise RuntimeError(f"Guide-adherence judge returned invalid score: {raw!r}")
        if not isinstance(feedback, str):
            feedback = str(feedback)
        return max(0.0, min(1.0, float(score))), feedback.strip()


class AzureEmbeddingSimilarityJudge:
    def __init__(
        self,
        model: str,
        embedding_kwargs: dict | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.model = validate_azure_deployment_model(
            model,
            field_name="embedding_model",
            example="azure/text-embedding-3-large",
        )
        self.azure_kwargs = {**resolve_azure_endpoint_kwargs(required=True), **resolve_azure_auth_kwargs()}
        self.embedding_kwargs = embedding_kwargs or {}
        self.embedding_kwargs.setdefault("timeout", timeout_seconds)
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def __call__(self, *, expected: str, actual: str, conversation: list[dict[str, str]]) -> tuple[float, str]:
        expected_embedding, actual_embedding = self._embed([expected, actual])
        similarity = _cosine_similarity(expected_embedding, actual_embedding)
        score = max(0.0, min(1.0, similarity))
        return (
            score,
            f"Embedding cosine similarity via {self.model}: {similarity:.4f}. "
            "This compares response meaning directly and does not use conversation context.",
        )

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        import litellm

        response = _embedding_with_retries(
            litellm=litellm,
            model=self.model,
            inputs=inputs,
            azure_kwargs=self.azure_kwargs,
            embedding_kwargs=self.embedding_kwargs,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            failure_message=(
                "Azure OpenAI embedding-model call failed. Check that --embedding-model names an existing "
                "embedding deployment in AZURE_OPENAI_URL."
            ),
        )
        return _extract_embeddings(response)


def _parse_judge_json(raw: str) -> dict:
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Semantic judge returned non-JSON response: {raw!r}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Semantic judge returned JSON that is not an object: {raw!r}")
    return data


def _completion_with_retries(
    *,
    litellm,
    model: str,
    messages: list[dict],
    azure_kwargs: dict,
    completion_kwargs: dict,
    max_retries: int,
    retry_backoff_seconds: float,
    failure_message: str,
):
    last_exc: Exception | None = None
    attempts = max_retries + 1
    safe_completion_kwargs = dict(completion_kwargs)
    safe_completion_kwargs.setdefault("drop_params", True)
    for attempt in range(attempts):
        try:
            return litellm.completion(
                model=model,
                messages=messages,
                **azure_kwargs,
                **safe_completion_kwargs,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _is_retryable_exception(exc):
                break
            time.sleep(retry_backoff_seconds * (2**attempt))

    raise RuntimeError(
        f"{failure_message} Context: {azure_call_context(model)}. "
        f"Underlying error: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def _embedding_with_retries(
    *,
    litellm,
    model: str,
    inputs: list[str],
    azure_kwargs: dict,
    embedding_kwargs: dict,
    max_retries: int,
    retry_backoff_seconds: float,
    failure_message: str,
):
    last_exc: Exception | None = None
    attempts = max_retries + 1
    safe_embedding_kwargs = dict(embedding_kwargs)
    for attempt in range(attempts):
        try:
            return litellm.embedding(
                model=model,
                input=inputs,
                **azure_kwargs,
                **safe_embedding_kwargs,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _is_retryable_exception(exc):
                break
            time.sleep(retry_backoff_seconds * (2**attempt))

    raise RuntimeError(
        f"{failure_message} Context: {azure_call_context(model)}. "
        f"Underlying error: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def _extract_embeddings(response: Any) -> list[list[float]]:
    data = response.get("data") if isinstance(response, dict) else getattr(response, "data", None)
    if not isinstance(data, list) or len(data) < 2:
        raise RuntimeError(f"Embedding response did not include two vectors: {response!r}")

    embeddings = [_extract_embedding(item) for item in data[:2]]
    if any(not embedding for embedding in embeddings):
        raise RuntimeError(f"Embedding response included an empty vector: {response!r}")
    return embeddings


def _extract_embedding(item: Any) -> list[float]:
    raw_embedding = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
    if not isinstance(raw_embedding, list):
        raise RuntimeError(f"Embedding response item did not include a vector: {item!r}")
    return [float(value) for value in raw_embedding]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError(f"Embedding vectors must have the same length; got {len(left)} and {len(right)}.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _is_retryable_exception(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in ("rate limit", "timeout", "temporarily", "connection", "server error"))
