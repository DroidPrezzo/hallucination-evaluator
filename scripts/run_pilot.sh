#!/usr/bin/env bash
#
# run_pilot.sh — the dual-judge PILOT run (Opus 4.8 + Grok 4.5) that feeds the
# kappa >= 0.75 decision gate for the full run.
#
# Runs SYNCHRONOUSLY on purpose: the Batch API's async queue (up to 24h) is
# useless for a same-day pilot. Batch is the standing rule for the FULL run
# only (see scripts/run_full.sh, added after the pilot launches).
#
# Every model/judge spec is single-quoted: the '&' in the query strings would
# otherwise background the shell process.
#
# Prereqs (env vars):
#   ANTHROPIC_API_KEY   (Opus judge — draws against account credit balance)
#   XAI_API_KEY         (Grok secondary judge)
#   OPENAI_API_KEY      (gpt-5.5)
#   MOONSHOT_API_KEY    (kimi-k3)
#   DEEPSEEK_API_KEY    (deepseek-v4-pro)
#   ZAI_API_KEY         (glm-5.2)
#   GEMINI_API_KEY      (gemini-3.1-pro-preview)
set -euo pipefail
cd "$(dirname "$0")/.."

# Use the venv's `python` if active, else fall back to python3.
PY="${PYTHON:-$(command -v python || command -v python3)}"

# ---- Pilot grid — CONFIRM before launch (affects kappa reliability) ----------
# Smaller than the full run: enough model x length spread to estimate cross-judge
# kappa cheaply, not the full 20-sample / 5-length sweep.
RUN_ID="${RUN_ID:-pilot-01}"
# Dedicated pilot corpus of LONG fresh SEC 10-Ks (each >= 128k tokens) so all
# three lengths below are valid via truncation. Build it with:
#   export SEC_EDGAR_USER_AGENT="Name email"
#   python scripts/fetch_sec_edgar.py --start-date <d> --end-date <d> --forms 10-K \
#       --max-docs 8 --corpus fresh2026_pilot --out-dir scripts/raw/fresh2026_pilot
#   python scripts/build_corpus_from_manifest.py   # -> corpus/fresh2026_pilot.jsonl
# then confirm every doc is >= 128k tokens (drop any that aren't).
CORPUS="${CORPUS:-corpus/fresh2026_pilot.jsonl}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-1000 16000 128000}"  # high end stress-tests kappa near the hard range
SAMPLES="${SAMPLES:-5}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-2}"

# ---- Test models (5) — Grok is deliberately NOT here; it's a judge -----------
MODELS=(
  --model-spec 'openai:kimi-k3?base_url=https://api.moonshot.ai/v1&key_env=MOONSHOT_API_KEY'
  --model-spec 'openai:gpt-5.5'
  --model-spec 'openai:deepseek-v4-pro?base_url=https://api.deepseek.com/v1&key_env=DEEPSEEK_API_KEY'
  --model-spec 'openai:glm-5.2?base_url=https://api.z.ai/api/paas/v4/&key_env=ZAI_API_KEY'
  --model-spec 'openai:gemini-3.1-pro-preview?base_url=https://generativelanguage.googleapis.com/v1beta/openai/&key_env=GEMINI_API_KEY'
)

# ---- Dual judge: Opus 4.8 (primary) + Grok 4.5 (secondary) ------------------
# Gemini is permanently excluded from any judge role (it is a test subject).
JUDGES=(
  --judge-spec 'anthropic:claude-opus-4-8'
  --judge-spec 'openai:grok-4.5?base_url=https://api.x.ai/v1&key_env=XAI_API_KEY'
)

"$PY" main.py \
  "${MODELS[@]}" \
  "${JUDGES[@]}" \
  --judge-mode claims \
  --corpus "$CORPUS" \
  --context-lengths $CONTEXT_LENGTHS \
  --samples "$SAMPLES" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --run-id "$RUN_ID"

# Analysis computes cross-judge Cohen's kappa and applies the pre-registered
# >= 0.75 gate, writing judge_decision to config.json + analysis.md.
"$PY" scripts/analyze.py "results/runs/$RUN_ID" \
  --baseline-length 1000 --delta 0.10 --bootstrap 1000

echo
echo "Pilot complete. Judge decision (single vs dual for the full run):"
"$PY" - "results/runs/$RUN_ID/config.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
d = cfg.get("judge_decision")
print(json.dumps(d, indent=2) if d else "  (judge_decision not found — is analyze.py's kappa gate implemented?)")
PY
