# Hallucination Evaluator

A harness for measuring **where LLM faithfulness collapses as a function of context length** — the gap between a model's *claimed* context window and its *effective faithful* one.

It runs in two modes from one entry point (`main.py`):

1. **Local leaderboard mode** — the original tool: loads open-weight models via HuggingFace `transformers`, truncates `tau/scrolls` government reports to exact token boundaries, generates summaries, and fact-checks them with a local judge. No API keys, no network beyond the model/dataset download.
2. **API research pipeline** — compares frontier API models (e.g. Claude, GPT, Kimi) against open-weight baselines on contamination-safe documents, with a claim-level judging pipeline, persistence/resume, cost tracking, and statistical analysis (bootstrap CIs, Effective Faithful Context, cross-judge agreement).

Try it with zero setup:

```bash
python main.py --dry-run      # full generate → judge → analyze, offline, on mock data
```

## AI Security & Red Teaming

Understanding the operational boundaries of an LLM is a **vulnerability-assessment** task: when a model is fed context beyond its effective capacity it degrades unpredictably — hallucinating facts, leaking data, or dropping safety guardrails. Mapping the breaking point yields quantitative metrics for model comparison before deployment.

The pipeline is aligned with the [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/): the summarizer and judge treat all document/summary/claim text as **untrusted data** and resist embedded prompt-injection (LLM01) via delimiter-neutralized tags and instruction re-assertion (see `src/prompts.py`); the supply chain is pinned via versioned dependencies, optional Hub revision pinning, and `trust_remote_code=False` (LLM03/LLM04); API keys are read only from environment variables and never logged (LLM06); spreadsheet-formula injection in CSV output is neutralized (LLM05).

## Architecture

```
main.py  (dispatch)
 ├── --dry-run ......... src.pipeline.run_dry_run       mock backends + bundled fixtures → analyze
 ├── --model-spec ...... src.pipeline.run_backend_pipeline   API research pipeline (persist/resume)
 └── (default) ......... legacy local-HF leaderboard

src/backends/            ModelBackend abstraction
  base.py                ABC + GenerationResult; truncate(), calibrate()
  factory.py             spec strings: hf:… / openai:… / anthropic:… / mock:…
  hf_local.py            local transformers (4-/8-bit, Hub revision pinning)
  openai_compat.py       OpenAI-compatible (OpenAI, xAI, Moonshot, DeepSeek, OpenRouter, …)
  anthropic_client.py    Anthropic Messages API
  mock.py                deterministic offline backend (dry-run / tests)
  retry.py               single retry layer (429/5xx/529→6, timeout→2, 4xx→fail-fast)
  pricing.py             pricing.yaml → cost_estimate_usd (tiered thresholds)

src/pipeline.py          generate → judge → leaderboard; per-provider concurrency; resume
src/judging/             claims.py (decompose + verify), retrieval.py (self-contained BM25)
src/persistence.py       RunStore: append-only JSONL, idempotency keys, resume
src/corpus.py            Corpus + ScrollsSource document sources
src/prompts.py           versioned prompts, OWASP LLM01 hardened

scripts/build_corpus.py             folder of .txt/.md/.html → corpus/{name}.jsonl (+ bundles)
scripts/analyze.py                  rate-vs-length + bootstrap CIs, EFC, contamination gap, kappa, curves.png
scripts/export_validation_sample.py stratified claim sample → CSV for blind human labeling
```

## Installation

Python 3.10+ recommended. A CUDA GPU is required **only** for local HF mode; the API pipeline and `--dry-run` need neither GPU nor network (beyond the API calls themselves).

```bash
git clone https://github.com/DroidPrezzo/hallucination-evaluator.git
cd hallucination-evaluator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables (API keys)

Keys are read **only** from environment variables and are never hardcoded or logged. Copy the template and fill in what you need:

```bash
cp .env.example .env
# then export them into your shell, e.g.:
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export MOONSHOT_API_KEY=...
export XAI_API_KEY=...
```

Which variable a model uses is controlled by the spec string's `key_env` parameter (default `OPENAI_API_KEY` for `openai:`, `ANTHROPIC_API_KEY` for `anthropic:`), so one OpenAI-compatible client class covers many providers.

## Usage

### (a) Local HF leaderboard mode

Unchanged from the original tool. Loads models locally and writes `results/leaderboard.md`:

```bash
python main.py \
  --models Qwen/Qwen2.5-1.5B-Instruct microsoft/Phi-3.5-mini-instruct \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --context-lengths 1000 2000 4000 8000 \
  --samples 10 \
  --use-4bit --judge-4bit
```

### (b) Full API comparison run

Each `--model-spec` is a backend spec; `--judge-spec` is repeatable. `--judge-mode claims` (the default for API runs) decomposes each summary into atomic claims and verifies each against the source.

```bash
python main.py \
  --model-spec openai:gpt-5.5 \
  --model-spec anthropic:claude-opus-4-8 \
  --model-spec "openai:kimi-k3?base_url=https://api.moonshot.ai/v1&key_env=MOONSHOT_API_KEY" \
  --model-spec hf:Qwen/Qwen2.5-7B-Instruct \
  --judge-spec anthropic:claude-opus-4-8 \
  --judge-mode claims \
  --context-lengths 1000 4000 16000 64000 256000 \
  --samples 20 \
  --corpus corpus/fresh2026.jsonl \
  --max-concurrency 2 \
  --run-id fresh-compare-01
```

Every generation and judgment is persisted to `results/runs/fresh-compare-01/` before anything else touches it. Re-running the same command **resumes** — completed cells are skipped, so paid API calls are never repeated.

### (c) Re-judging cached generations (free)

Add a second judge and re-run with the **same `--run-id`**. The generation phase finds every cell already on disk and skips it (no generation cost); the shared claim decomposition is reused; only the new judge runs verification:

```bash
python main.py \
  --model-spec openai:gpt-5.5 --model-spec anthropic:claude-opus-4-8 \
  --judge-spec anthropic:claude-opus-4-8 \
  --judge-spec openai:gpt-5.5 \
  --judge-mode claims \
  --corpus corpus/fresh2026.jsonl \
  --run-id fresh-compare-01
```

(Pass the same `--model-spec` / `--corpus` / `--context-lengths` so the generation idempotency keys match what's on disk.)

### (d) Analysis

```bash
python scripts/analyze.py results/runs/fresh-compare-01 \
  --baseline-length 1000 --delta 0.10 --bootstrap 1000
```

Writes into the run directory:

- **`analysis.md`** — per-model hallucination rate vs context length with **bootstrap 95% CIs** (resampled over documents); **Effective Faithful Context (EFC)** = largest tested length `L` where `rate(L) ≤ rate(baseline) + delta` (baseline/delta echoed for the record); fresh-vs-legacy **contamination gap**; **cost per model**; and a **cross-judge Cohen's kappa** table when ≥2 judges ran.
- **`curves.png`** — rate vs length, one line per model, with CI bands.
- **`raw_summary.csv`** — per (model, contamination, length) rate + CI.

### Building a contamination-safe corpus

Download GAO / SEC / Federal Register documents yourself (post-cutoff for `fresh2026` safety) into a folder, then:

```bash
python scripts/build_corpus.py --input-dir downloads/fresh2026 \
  --name fresh2026 --contamination-risk fresh2026 --metadata meta.json

# very-long-context bundles (~256k tokens each):
python scripts/build_corpus.py --input-dir downloads/fresh2026 \
  --name fresh2026_bundles --contamination-risk fresh2026 --bundle-target-tokens 256000
```

`--metadata` is an optional JSON keyed by filename supplying real `title` / `source_url` / `pub_date` / `doc_id`. The contamination gap in analysis requires **both** a `fresh2026` corpus and the `public_legacy` `tau/scrolls` data present under one run id.

### Exporting a blind human-validation sample

```bash
python scripts/export_validation_sample.py \
  --run-dir results/runs/fresh-compare-01 --n 200 --output validation_sample.csv
```

Produces a stratified (model × length) claim sample with source excerpts and a blank `human_label` column; the model's `judge_verdict` is the trailing column so it can be hidden during labeling.

## Run directory layout

```
results/runs/{run_id}/
  config.json            resolved config: git commit, prompt versions, generation/judge params,
                         calibrations, retrieval params, document source, cost totals
  generations.jsonl      one record per generation (dual token counts, usage, latency, cost)
  contexts.jsonl         truncated source per generation (enables free re-judging)
  decompositions.jsonl   shared claim sets            (claims mode)
  verifications.jsonl    per-claim verdicts per judge (claims mode)
  judgments.jsonl        holistic YES/NO verdicts     (holistic mode)
  analysis.md · curves.png · raw_summary.csv          (written by scripts/analyze.py)
```

## Reproducibility & cost controls

- **Determinism:** temperature 0 and fixed seed everywhere; prompts are versioned (`PROMPT_VERSION`, `JUDGE_PROMPT_VERSION`, `DECOMPOSE_PROMPT_VERSION`, `VERIFY_PROMPT_VERSION`); every output-affecting parameter is logged in `config.json`.
- **Never pay twice:** each generation/judgment is flushed to disk before use; idempotency key = hash of `(model_spec, doc_id, requested_context_tokens, prompt_version)`; re-runs resume.
- **Retries (single layer, ours):** 429/5xx/529 → up to 6 tries with exponential backoff + jitter (honoring `Retry-After`); timeouts → 2 tries; 4xx → fail fast and record the cell as failed. Official SDKs run with `max_retries=0` so our layer owns all retries. Connect timeout 30 s, read timeout 900 s.
- **Concurrency:** `--max-concurrency` (default 2) per provider; local HF runs sequentially; any request at/above 256k tokens is forced to concurrency 1.
- **Tokenization:** truncation targets the test model's own tokenizer (HF tokenizer, or `tiktoken`, or a per-model chars-per-token ratio calibrated from a couple of live calls and recorded in `config.json`). Both our token count and the provider-reported count are stored so drift is visible.
- **Cost:** `pricing.yaml` (USD per 1M tokens, with tiered long-context thresholds) drives `cost_estimate_usd` on every record; run-level totals land in `config.json`.

## Testing

```bash
pip install -r requirements.txt
pytest                    # unit + end-to-end dry-run tests
python main.py --dry-run  # full pipeline offline → results/runs/dry-run/analysis.md + curves.png
```

The suite covers truncation exactness at token boundaries, spec-string parsing, resume/idempotency, malformed judge-JSON handling, EFC computation on synthetic curves, and a full offline generate → judge → analyze dry run.

## Output

Local mode writes `results/leaderboard.md`; the API pipeline writes per-run artifacts under `results/runs/{run_id}/`. A `0.00` rate means every summary at that length was fully faithful; higher is worse.

## Acknowledgments

Co-authored by **Claude Code** (Anthropic's agentic coding assistant), which assisted in the API-backend refactor, the claim-level judging and analysis pipeline, security and OWASP LLM Top 10 hardening, and debugging.
