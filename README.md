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

### Guide adherence metric

GEPA's behavior metric measures only similarity to the golden set, which has no notion of whether the prompt follows the target-model prompt guide. A restructured, guide-following prompt that merely *preserves* behavior is a tie on that objective, so GEPA would keep the original. `--guide-adherence-weight` (a value in `[0, 1)`, default `0.2`) blends an LLM guide-adherence score into the objective; set it to `0` to disable:

```bash
uv run aoai-prompt-port \
  --source-prompt ./examples/prompt.gpt-4o-mini.txt \
  --source-model azure/gpt-4o-mini \
  --target-model azure/gpt-5.4-mini \
  --golden ./examples/golden.bfsi-debt-support.json \
  --guide-adherence-weight 0.2
```

The objective becomes `(1 - weight) * behavior + weight * guide_adherence`, so behavior stays dominant (GEPA will not trade meaningful behavior for structure) but a guide-following rewrite wins when behavior is preserved. An LLM judge (`--judge-model`, `temperature=0`) scores how well the *prompt* follows the guide's structural conventions; it does not judge task content. When the weight is active, the ship gate also ships a restructured prompt that holds behavior within observed noise and improves guide adherence, and the report records naive→optimized adherence. The metric is on by default at `0.2`; pass `--guide-adherence-weight 0` to disable it and make no extra model calls. Because it requires Azure calls, it is automatically skipped under `--echo-runner`.

### Restructure-then-optimize

Rather than asking GEPA to *discover* the guide-conformant structure and preserve behavior at the same time (which burns its limited `--max-metric-calls` budget on a large monolithic prompt and often returns the original unchanged), the migration first rewrites the naive port into the target guide's structure with a single LLM call, then seeds GEPA with that restructured prompt. GEPA then spends its whole budget on behavior preservation/tuning, and the guide-adherence metric acts as a *guard* against structural drift rather than the driver.

This runs by default whenever a guide is available and you are optimizing. The restructure uses the reflection model (`--reflection-model`, defaulting to `--target-model`). The report shows three target-model rows — naive → restructured → optimized — with their guide-adherence scores, so you can see where structure and behavior each moved. If the rewrite comes back empty or implausibly short, it falls back to the naive port. Disable it with `--no-restructure-seed`. It is skipped under `--echo-runner` and with `--no-optimize`.

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

## Input formats

`--input-format` controls how each golden record is turned into a model request.
It defaults to `chat`, so existing golden sets are unaffected.

- `chat` (default): replays conversations and original-bot transcripts, as
  described above.
- `qa`: structured data extraction from an **image** plus a **question**. The
  prompt being migrated is the extraction instruction (the system prompt); the
  image and question are sent as the user turn.
- `qa-di`: the same as `qa`, but also injects **Azure Document Intelligence**
  output text alongside the image, so the model can read both the layout image
  and the OCR/structure extraction.

QA records use these fields:

| Field | `qa` | `qa-di` | Notes |
| --- | --- | --- | --- |
| `image` | required | optional | File path, `http(s)` URL, or `data:` URL. A list of images is allowed. File paths resolve relative to the golden file. |
| `document_intelligence` | optional | required | Path to a Document Intelligence JSON/text file, an inline string, or an inline object. JSON is reduced to its extracted `content` text. Aliases: `document`, `di`. |
| `question` | optional | optional | The instruction/question for the user turn. Aliases: `input`, `prompt`. |
| `expected` / `answer` | required | required | The gold extraction. |
| `judge` | optional | optional | Defaults to `json_exact` for structured extraction. |

```bash
uv run aoai-prompt-port \
  --source-prompt ./examples/prompt.extraction.txt \
  --source-model azure/gpt-4o \
  --target-model azure/gpt-5.4 \
  --golden ./examples/golden.invoice-qa-di.json \
  --input-format qa-di \
  --output-dir migration_out
```

Example QA golden record (`qa-di`):

```json
{
  "id": "invoice-total",
  "image": "invoices/invoice-1.png",
  "document_intelligence": "di/invoice-1.json",
  "question": "Extract invoice_number, invoice_date (YYYY-MM-DD), and total. Return JSON.",
  "expected": { "invoice_number": "INV-1001", "invoice_date": "2026-01-14", "total": "1240.00" },
  "judge": "json_exact"
}
```

Images are inlined to the model as base64 `data:` URLs, but reports and the GEPA
reflection dataset show a compact `[image N]` placeholder plus the Document
Intelligence text, so report files stay readable. QA migrations need real Azure
model calls (`--echo-runner` cannot extract data); see `examples/golden.invoice-qa.json`
and `examples/golden.invoice-qa-di.json` for self-contained samples.

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
