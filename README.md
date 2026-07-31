# Hallucination Evaluator

A harness for measuring **where LLM faithfulness collapses as a function of context length** — the gap between a model's *claimed* context window and its *effective faithful* one.

It runs in two modes from one entry point (`main.py`):

1. **Local leaderboard mode** — the original tool: loads open-weight models via HuggingFace `transformers`, truncates `tau/scrolls` government reports to exact token boundaries, generates summaries, and fact-checks them with a local judge. No API keys, no network beyond the model/dataset download.
2. **API research pipeline** — compares frontier API models (e.g. Claude, GPT, Kimi, DeepSeek, GLM, Gemini) against open-weight baselines on contamination-safe documents, with a claim-level judging pipeline, persistence/resume, cost tracking, and statistical analysis (bootstrap CIs, Effective Faithful Context, cross-judge agreement).

Try it with zero setup:

```bash
python main.py --dry-run      # full generate → judge → analyze, offline, on mock data
```

## What this is (and isn't)

This measures **generative faithfulness degradation** — how much a model's summaries stop being supported by the source as input length grows — and reports an **Effective Faithful Context (EFC)** per model: the largest input length at which claim-level hallucination stays within a set margin of the model's own short-context baseline.

Known, and built on rather than claimed: that faithfulness degrades with context (Roig 2026, on synthetic document Q&A to 200K; Chroma's "context rot" across 18 models), that advertised context exceeds usable context (RULER's "effective context length"), and that post-cutoff documents defend against training contamination (AntiLeakBench and related).

**What this adds:** a contamination-controlled, claim-level faithfulness degradation curve for free-form summarization extending toward the ~1M-token windows today's newest models advertise, with day-one coverage of recently released models. Prior summarization-faithfulness work largely stops at ≤200K; the nearest study (Roig 2026) tests synthetic Q&A to 200K and names extension past it toward 1M as future work. We take that step, on real post-cutoff government/financial documents rather than synthetic text.

This is **not** a safety-guardrail or jailbreak red-teaming tool. It measures faithfulness on benign documents; it does not test refusal behavior, guardrail circumvention, or adversarial prompting.

**Rigor:** analysis pre-registered before data collection (`ANALYSIS_PLAN.md`), an independent judge held outside the test set, ~400 human-validated claims as ground truth, full public release of code, corpus, generations, and judgments. Prior art tracked in `relatedwork.md`.

## Input Handling & Security

The harness treats all document, summary, and claim text as **untrusted data** — because in a faithfulness pipeline it is: the summarizer reads arbitrary source documents, and the judge reads model-generated summaries. Handling that text safely is an engineering requirement, independent of what the study measures.

The pipeline follows relevant items from the [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/): the summarizer and judge resist embedded prompt-injection (LLM01) via delimiter-neutralized tags and instruction re-assertion (see `src/prompts.py`); the supply chain is pinned via versioned dependencies, optional Hub revision pinning, and `trust_remote_code=False` (LLM03/LLM04); API keys are read only from environment variables and never logged (LLM06); spreadsheet-formula injection in CSV output is neutralized (LLM05).

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
  openai_compat.py       OpenAI-compatible (OpenAI, xAI, Moonshot, DeepSeek, Z.ai, Gemini, OpenRouter, …)
  anthropic_client.py    Anthropic Messages API
  mock.py                deterministic offline backend (dry-run / tests)
  retry.py               single retry layer (429/5xx/529→6, timeout→2, 4xx→fail-fast)
  pricing.py             pricing.yaml → cost_estimate_usd (tiered thresholds)

src/pipeline.py          generate → judge → leaderboard; per-provider concurrency; resume
src/judging/             claims.py (decompose + verify), retrieval.py (self-contained BM25)
src/persistence.py       RunStore: append-only JSONL, idempotency keys, resume
src/corpus.py            Corpus + ScrollsSource document sources
src/prompts.py           versioned prompts, OWASP LLM01 hardened

scripts/build_corpus.py                folder + JSON metadata → corpus/{name}.jsonl (+ bundles)
scripts/build_corpus_from_manifest.py  manifest.csv → corpus/{name}.jsonl (paired with fetchers)
scripts/fetch_federal_register.py      Federal Register API → raw docs + manifest rows
scripts/fetch_sec_edgar.py             SEC EDGAR FTS → raw docs + manifest rows (needs User-Agent)
scripts/gao_manifest_helper.py         scan manually-saved GAO reports → manifest rows
scripts/analyze.py                     rate-vs-length + bootstrap CIs, EFC, contamination gap, kappa, curves.png
scripts/export_validation_sample.py    stratified claim sample → CSV for blind human labeling
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
export DEEPSEEK_API_KEY=...
export ZAI_API_KEY=...
export GEMINI_API_KEY=...
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

Each `--model-spec` is a backend spec; `--judge-spec` is repeatable. `--judge-mode claims` (the default for API runs) decomposes each summary into atomic claims and verifies each against the source. The judge is held outside the test set to avoid self-evaluation.

```bash
python main.py \
  --model-spec openai:gpt-5.5 \
  --model-spec "openai:kimi-k3?base_url=https://api.moonshot.ai/v1&key_env=MOONSHOT_API_KEY" \
  --model-spec "openai:deepseek-v4-pro?base_url=https://api.deepseek.com/v1&key_env=DEEPSEEK_API_KEY" \
  --model-spec "openai:glm-5.2?base_url=https://api.z.ai/api/paas/v4&key_env=ZAI_API_KEY" \
  --model-spec "openai:gemini-3.1-pro-preview?base_url=https://generativelanguage.googleapis.com/v1beta/openai/&key_env=GEMINI_API_KEY" \
  --judge-spec anthropic:claude-opus-4-8 \
  --judge-mode claims \
  --context-lengths 1000 4000 16000 64000 128000 256000 480000 900000 \
  --samples 30 \
  --corpus corpus/fresh2026.jsonl \
  --max-concurrency 2 \
  --run-id fresh-compare-01
```

Every generation and judgment is persisted to `results/runs/fresh-compare-01/` before anything else touches it. Re-running the same command **resumes** — completed cells are skipped, so paid API calls are never repeated.

**Note on the long-context cells.** The 480k and 900k lengths cross provider long-context surcharge thresholds (e.g. GPT-5.5 above 272k, Gemini above 200k) and dominate cost; the study tapers to fewer documents at those two lengths. Not all test models are run at 900k — size the top cells to each model's usable window.

### (c) Re-judging cached generations (free)

Add a second judge and re-run with the **same `--run-id`**. The generation phase finds every cell already on disk and skips it (no generation cost); the shared claim decomposition is reused; only the new judge runs verification. This is how the pilot measures Opus↔Grok agreement (kappa) before deciding single- vs. dual-judge for the full run:

```bash
python main.py \
  --model-spec openai:gpt-5.5 \
  --judge-spec anthropic:claude-opus-4-8 \
  --judge-spec "openai:grok-4.5?base_url=https://api.x.ai/v1&key_env=XAI_API_KEY" \
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

The Federal Register and SEC EDGAR fetchers pull post-cutoff documents automatically and append `manifest.csv` rows; GAO reports are saved manually and picked up by the helper. All fetchers save into a per-corpus subdirectory (e.g. `scripts/raw/fresh2026/`) so the folder stays homogeneous for bundling.

```bash
# scripted sources (post-cutoff for fresh2026 safety):
python scripts/fetch_federal_register.py --start-date 2026-06-01 --end-date 2026-07-31 \
  --doc-types RULE --max-docs 40 --out-dir scripts/raw/fresh2026
python scripts/fetch_sec_edgar.py --start-date 2026-06-01 --end-date 2026-07-31 \
  --forms 10-K,10-Q --max-docs 20 --out-dir scripts/raw/fresh2026   # needs SEC_EDGAR_USER_AGENT

# manually-saved GAO reports → manifest rows:
python scripts/gao_manifest_helper.py --raw-dir scripts/raw/fresh2026

# manifest → corpus JSONL:
python scripts/build_corpus_from_manifest.py \
  --raw-dir scripts/raw/fresh2026 --manifest scripts/raw/manifest.csv --out-dir corpus

# very-long-context bundles (built with the folder+JSON tool; longest length first):
python scripts/build_corpus.py --input-dir scripts/raw/fresh2026 \
  --name fresh2026_bundles --contamination-risk fresh2026 --bundle-target-tokens 256000
```

The manifest single-doc corpus (`fresh2026.jsonl`) and the bundle corpus (`fresh2026_bundles.jsonl`) are distinct files with disjoint `doc_id` namespaces, so the two builders never collide. The contamination gap in analysis requires **both** a `fresh2026` corpus and the `public_legacy` `tau/scrolls` data present under one run id.

### Exporting a blind human-validation sample

```bash
python scripts/export_validation_sample.py \
  --run-dir results/runs/fresh-compare-01 --n 400 --output validation_sample.csv
```

Produces a stratified (model × length) claim sample with source excerpts and a blank `human_label` column; the model's `judge_verdict` is the trailing column so it can be hidden during labeling. The default of 400 is sized to be the study's sole ground-truth anchor under a single judge.

## Run directory layout

```
results/runs/{run_id}/
  config.json            resolved config: git commit, prompt versions, generation/judge params,
                         calibrations, retrieval params, document source, cost totals,
                         judge_decision {pilot_kappa, threshold, branch}
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
- **Cost:** `pricing.yaml` (USD per 1M tokens, with tiered long-context thresholds) drives `cost_estimate_usd` on every record; run-level totals land in `config.json`. Batch-API and off-peak scheduling paths are available for the full run; the pilot runs synchronously so same-day turnaround isn't blocked by batch queues.

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
