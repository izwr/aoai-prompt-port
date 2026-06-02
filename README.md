# aoai-prompt-port

A CLI that migrates an existing prompt from one Azure OpenAI GPT deployment to
another. Pointing a prompt that was tuned for, say, `gpt-4o-mini` at a newer
deployment such as `gpt-5.x` often shifts tone, safety behavior, and answer
quality. This tool ports the prompt, optimizes it for the target model with
[GEPA](https://github.com/gepa-ai/gepa) (a reflective prompt optimizer), and
evaluates the result against a golden set of expected behavior before telling
you whether the new prompt is safe to ship.

It reports:

- source prompt on the source model (the original baseline)
- source prompt on the target model (the naive port)
- optimized prompt on the target model
- a prompt diff, per-case events, and a ship / no-ship decision

The ship gate never returns an optimized prompt that scores below the naive
target-model port, so a migration can only keep quality the same or improve it.

## Requirements

- Python 3.10+
- An [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/) resource
  with the GPT deployments you want to migrate between
- Authentication via either the `AZURE_OPENAI_TOKEN` environment variable or the
  [Azure CLI](https://learn.microsoft.com/cli/azure/) (`az login`)

## Installation

Install from source with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/izwr/aoai-prompt-port.git
cd aoai-prompt-port
uv sync
```

Or install into the current environment with pip:

```bash
pip install .
```

This installs two equivalent console scripts: `aoai-prompt-port` and the shorter
`migrate`.

## Configuration

Set the Azure OpenAI endpoint and (optionally) API version before running.

Bash / Zsh:

```bash
export AZURE_OPENAI_URL="https://YOUR-RESOURCE.openai.azure.com"
export AZURE_API_VERSION="2024-10-21"
# Either provide a token directly...
export AZURE_OPENAI_TOKEN="<aad-access-token>"
# ...or rely on the Azure CLI credential (run `az login` first).
```

PowerShell:

```powershell
$env:AZURE_OPENAI_URL = "https://YOUR-RESOURCE.openai.azure.com"
$env:AZURE_API_VERSION = "2024-10-21"
# Either provide a token directly...
$env:AZURE_OPENAI_TOKEN = "<aad-access-token>"
# ...or rely on the Azure CLI credential (run `az login` first).
```

`AZURE_API_BASE` is accepted as a compatibility fallback for `AZURE_OPENAI_URL`.

## CLI

Bash / Zsh:

```bash
uv run aoai-prompt-port \
  --source-prompt ./examples/prompt.gpt-4o-mini.txt \
  --source-model azure/gpt-4o-mini \
  --target-model azure/gpt-5.4-mini \
  --golden ./examples/golden.bfsi-debt-support.json \
  --output-dir migration_out
```

PowerShell:

```powershell
uv run aoai-prompt-port `
  --source-prompt ./examples/prompt.gpt-4o-mini.txt `
  --source-model azure/gpt-4o-mini `
  --target-model azure/gpt-5.4-mini `
  --golden ./examples/golden.bfsi-debt-support.json `
  --output-dir migration_out
```

`uv run` executes the CLI in the project environment without a separate
activation step. The shorter `migrate` command is also available, so
`uv run migrate ...` is equivalent. If you installed the package into your own
environment (`pip install .`), drop the `uv run` prefix and call
`aoai-prompt-port` directly.

The only difference between the shells is the line-continuation character:
backslash (`\`) in Bash/Zsh and backtick (`` ` ``) in PowerShell. The remaining
examples below show the Bash form; substitute the backtick to run them in
PowerShell.

By default, optimized migrations fetch and use OpenAI prompt guidance derived from the target model family. Models older than GPT-4.1, including GPT-4o, use GPT-4.1 guidance. Variant deployments use their base family, so `azure/gpt-5.5-mini` fetches:

```text
https://developers.openai.com/api/docs/guides/prompt-guidance?model=gpt-5.5
```

Use `--guide ./target-gpt-guide.txt` to override it, `--guide-url <url>` to fetch a different guide, or `--no-default-guide` to run GEPA without guide injection.

Model IDs must use LiteLLM's Azure format, `azure/<deployment-name>`, and the deployment name must reference a GPT model. Configure the endpoint with `AZURE_OPENAI_URL`; `AZURE_API_BASE` is also supported as a compatibility fallback. Set `AZURE_API_VERSION` for the Azure OpenAI API version.

Authentication is resolved in this order:

1. `AZURE_OPENAI_TOKEN`
2. `AzureCliCredential`

If `--reflection-model` is omitted, GEPA uses `--target-model` for reflection. Pass `--reflection-model azure/<deployment-name>` only when that deployment exists in the same `AZURE_OPENAI_URL` resource.

Semantic judging defaults to `--semantic-judge llm`, using `--judge-model azure/gpt-4.1` with `temperature=0`. You can use Azure OpenAI text embeddings instead:

```bash
uv run aoai-prompt-port \
  --source-prompt ./examples/prompt.gpt-4o-mini.txt \
  --source-model azure/gpt-4o-mini \
  --target-model azure/gpt-5.4-mini \
  --golden ./examples/golden.bfsi-debt-support.json \
  --semantic-judge embeddings \
  --embedding-model azure/text-embedding-3-large
```

Embeddings are faster and cheaper, but they compare response text directly and are less reliable than an LLM judge for policy caveats, safety guidance, and next-step quality.

Final ship/no-ship uses repeated evaluation (`--eval-repetitions`, default `3`) and requires a changed optimized prompt to beat the naive target-model prompt by `--ship-margin` plus observed run-to-run noise. Unsupported model parameters are dropped via LiteLLM's `drop_params=True` safety net.

For local harness testing without model calls:

```bash
uv run aoai-prompt-port \
  --source-prompt "Return only the label." \
  --source-model azure/gpt-4o \
  --target-model azure/gpt-5 \
  --golden examples/golden.echo.json \
  --no-optimize \
  --echo-runner
```

`--echo-runner` is only useful for deterministic fixtures such as `examples/golden.echo.json`. Transcript goldens use model-judged semantic similarity and require Azure model calls.

Golden sets can be JSON arrays, JSONL, a single transcript object, or `{ "cases": [...] }`.

For prompt migration, the preferred golden-set shape is a transcript from the original bot. Each assistant turn becomes an eval case automatically: the conversation before that assistant turn is replayed, and the original assistant turn is used as the expected behavior. Transcript cases emit two metrics by default: `semantic_similarity` and `response_length`. `semantic_similarity` is judged by either an Azure GPT model or Azure text embeddings, depending on `--semantic-judge`; `response_length` is computed locally.

```json
{
  "id": "payment-posting",
  "turns": [
    { "role": "user", "content": "I paid yesterday but still got a reminder." },
    { "role": "assistant", "content": "Payment reminders can continue until the payment is posted. Please check the official app and keep your transaction reference." },
    { "role": "user", "content": "Should I pay again?" },
    { "role": "assistant", "content": "Do not make a duplicate payment unless the official app or support team confirms it is needed." }
  ]
}
```

The older explicit case shape is still supported:

```json
{
  "id": "case-1",
  "input": "user input",
  "expected": "gold label",
  "judge": "exact"
}
```

Cases can also provide a multi-turn `conversation` or `messages` array:

```json
{
  "id": "case-2",
  "conversation": [
    { "role": "user", "content": "We are classifying sentiment." },
    { "role": "assistant", "content": "Understood." },
    { "role": "user", "content": "classify sentiment => negative" }
  ],
  "expected": "negative",
  "judge": "exact"
}
```

Supported judges: `semantic_and_length`, `exact`, `contains`, and `json_exact`.

For support-bot migrations, use original bot turns or complete assistant responses, not intent labels. The example BFSI debt-management set is a set of original-bot transcripts so migration is evaluated on tone, safety, and next-step quality.

## Development

Run the test suite with explicit discovery:

```bash
uv run python -m unittest discover -s tests
```

The test suite uses the deterministic `--echo-runner` path and does not make any
Azure model calls, so it runs offline.

## License

Licensed under the [Apache License 2.0](LICENSE).
