from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from tqdm import tqdm

from src.backends import ModelBackend, create_backend
from src.context_builder import ContextBuilder
from src.leaderboard import LeaderboardGenerator
from src.persistence import RunStore
from src.prompts import (
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM,
    PROMPT_VERSION,
    SUMMARIZE_SYSTEM,
    build_judge_prompt,
    build_summarize_prompt,
)

logger = logging.getLogger(__name__)

# Parameters that affect output — logged into config.json for reproducibility.
GEN_MAX_TOKENS = 512
GEN_TEMPERATURE = 0.0
GEN_SEED = 0  # applied by OpenAI-compatible backends; ignored where unsupported
JUDGE_MAX_TOKENS = 10
JUDGE_TEMPERATURE = 0.0

# Above this requested context length, force concurrency to 1 (per user decision).
LARGE_CONTEXT_THRESHOLD = 256_000


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _generation_key(
    model_spec: str, doc_id: str, context_tokens: int, prompt_version: str
) -> str:
    raw = f"{model_spec}|{doc_id}|{context_tokens}|{prompt_version}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _judgment_key(
    generation_key: str, judge_spec: str, judge_prompt_version: str
) -> str:
    raw = f"{generation_key}|{judge_spec}|{judge_prompt_version}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _effective_concurrency(provider: str, length: int, max_concurrency: int) -> int:
    """Per-provider concurrency, with two hard overrides:

    - Local HF runs sequentially (single GPU, backend not thread-safe).
    - Any request at/above LARGE_CONTEXT_THRESHOLD runs alone.
    """
    if provider == "hf":
        return 1
    if length >= LARGE_CONTEXT_THRESHOLD:
        return 1
    return max(1, max_concurrency)


def _process_concurrently(
    tasks: list,
    worker: Callable[[Any], Any],
    persist: Callable[[Any], None],
    max_workers: int,
    pbar: Optional[tqdm] = None,
) -> None:
    """Run `worker` over `tasks`; `persist` each result in the CALLING thread.

    Persistence stays single-threaded so JSONL appends and the in-memory key
    sets need no locking. Workers only do the (network-bound) API call.
    """
    if max_workers <= 1:
        for task in tasks:
            persist(worker(task))
            if pbar is not None:
                pbar.update()
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            persist(future.result())
            if pbar is not None:
                pbar.update()


def _make_generation_worker(
    backend: ModelBackend, spec: str, context_builder: ContextBuilder
) -> Callable[[tuple[int, int]], tuple[str, str, dict]]:
    """Build a worker that produces (generation_key, context, record) for a doc.

    The worker never raises: any failure is captured as a failed-cell record so
    one bad document cannot abort a long, paid run.
    """

    def worker(task: tuple[int, int]) -> tuple[str, str, dict]:
        doc_idx, length = task
        doc_id = f"scrolls:{doc_idx}"
        key = _generation_key(spec, doc_id, length, PROMPT_VERSION)

        context = ""
        source_sha = ""
        local_token_count = 0
        try:
            doc = context_builder.get_document(doc_idx)
            context = backend.truncate(doc, length)
            source_sha = hashlib.sha256(context.encode()).hexdigest()
            prompt = build_summarize_prompt(context)
            local_token_count = backend.count_tokens(prompt)

            result = backend.generate(
                prompt,
                max_tokens=GEN_MAX_TOKENS,
                temperature=GEN_TEMPERATURE,
                system_prompt=SUMMARIZE_SYSTEM,
            )
            record = {
                "idempotency_key": key,
                "doc_id": doc_id,
                "source_sha256": source_sha,
                "corpus": "scrolls",
                "model_spec": spec,
                "requested_context_tokens": length,
                "actual_input_tokens_model_tokenizer": local_token_count,
                "provider_reported_input_tokens": result.input_tokens,
                "prompt_version": PROMPT_VERSION,
                "output_text": result.text,
                "usage": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                },
                "latency_s": result.latency_s,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cost_estimate_usd": result.cost_estimate_usd,
                "finish_reason": result.finish_reason,
            }
        except Exception as exc:
            logger.error("Generation failed doc=%s len=%d: %s", doc_id, length, exc)
            record = {
                "idempotency_key": key,
                "doc_id": doc_id,
                "source_sha256": source_sha,
                "corpus": "scrolls",
                "model_spec": spec,
                "requested_context_tokens": length,
                "actual_input_tokens_model_tokenizer": local_token_count,
                "provider_reported_input_tokens": None,
                "prompt_version": PROMPT_VERSION,
                "output_text": None,
                "error": str(exc),
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "latency_s": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cost_estimate_usd": None,
                "finish_reason": "error",
            }
        return key, context, record

    return worker


def _make_judge_worker(
    judge_backend: ModelBackend, judge_spec: str, contexts: dict[str, str]
) -> Callable[[dict], dict]:
    """Build a worker that judges one generation and returns a judgment record.

    Source text always comes from the persisted context cache. If it is missing
    (only possible after a corrupted partial write), the record is marked
    judge_error rather than silently re-truncating with the wrong tokenizer.
    """

    def worker(gen: dict) -> dict:
        gen_key = gen["idempotency_key"]
        jkey = _judgment_key(gen_key, judge_spec, JUDGE_PROMPT_VERSION)
        base = {
            "idempotency_key": jkey,
            "generation_key": gen_key,
            "judge_spec": judge_spec,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judge_mode": "holistic",
        }

        source_text = contexts.get(gen_key)
        if source_text is None:
            logger.error("Context missing for %s; marking judge_error", gen_key[:12])
            return {
                **base,
                "verdict": None,
                "error": "context_missing",
                "raw_response": None,
                "usage": {"input_tokens": 0, "provider_reported_input_tokens": None, "output_tokens": 0},
                "latency_s": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cost_estimate_usd": None,
            }

        prompt = build_judge_prompt(source_text, gen["output_text"])
        local_judge_tokens = judge_backend.count_tokens(prompt)

        try:
            result = judge_backend.generate(
                prompt,
                max_tokens=JUDGE_MAX_TOKENS,
                temperature=JUDGE_TEMPERATURE,
                system_prompt=JUDGE_SYSTEM,
            )
            resp = result.text.upper().strip()
            if "YES" in resp:
                verdict: Optional[bool] = True
            elif "NO" in resp:
                verdict = False
            else:
                logger.warning(
                    "Ambiguous judge response: %s — defaulting to hallucination", result.text
                )
                verdict = True

            return {
                **base,
                "verdict": verdict,
                "raw_response": result.text,
                "usage": {
                    "input_tokens": local_judge_tokens,
                    "provider_reported_input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                },
                "latency_s": result.latency_s,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cost_estimate_usd": result.cost_estimate_usd,
            }
        except Exception as exc:
            logger.error("Judging failed for %s: %s", gen_key[:12], exc)
            return {
                **base,
                "verdict": None,
                "error": str(exc),
                "raw_response": None,
                "usage": {"input_tokens": local_judge_tokens, "provider_reported_input_tokens": None, "output_tokens": 0},
                "latency_s": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cost_estimate_usd": None,
            }

    return worker


def _apply_calibration(
    backend: ModelBackend,
    spec: str,
    calibrations: dict[str, float],
    config: dict,
    store: RunStore,
) -> None:
    """Calibrate a backend if needed, reusing a persisted ratio when present."""
    if not backend.needs_calibration():
        return
    if spec in calibrations:
        backend._chars_per_token = calibrations[spec]
        logger.info("Loaded calibration for %s: %.2f chars/tok", spec, calibrations[spec])
    else:
        ratio = backend.calibrate()
        calibrations[spec] = round(ratio, 4)
        config["calibrations"] = calibrations
        store.save_config(config)


def run_backend_pipeline(args: argparse.Namespace) -> None:
    run_id = getattr(args, "run_id", None) or (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:6]
    )
    store = RunStore(run_id, args.output_dir)

    dataset_revision = getattr(args, "dataset_revision", None)
    context_builder = ContextBuilder(
        dataset_name="tau/scrolls",
        subset="gov_report",
        split="validation",
        revision=dataset_revision,
    )

    existing_config = store.load_config()
    calibrations: dict[str, float] = (
        existing_config.get("calibrations", {}) if existing_config else {}
    )

    config = {
        "run_id": run_id,
        "model_specs": args.model_spec,
        "judge_spec": args.judge_spec,
        "context_lengths": args.context_lengths,
        "samples": args.samples,
        "max_concurrency": args.max_concurrency,
        "large_context_threshold": LARGE_CONTEXT_THRESHOLD,
        "prompt_version": PROMPT_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "generation_params": {
            "max_tokens": GEN_MAX_TOKENS,
            "temperature": GEN_TEMPERATURE,
            "seed": GEN_SEED,
        },
        "judge_params": {
            "max_tokens": JUDGE_MAX_TOKENS,
            "temperature": JUDGE_TEMPERATURE,
            "seed": GEN_SEED,
        },
        "git_commit": _git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "calibrations": calibrations,
        "dataset": {
            "name": "tau/scrolls",
            "subset": "gov_report",
            "split": "validation",
            "revision": dataset_revision,
        },
    }
    store.save_config(config)

    dataset_len = context_builder.get_document_length()
    actual_samples = min(args.samples, dataset_len)

    # ------------------------------------------------------------------
    # Phase 1  —  Generation
    # ------------------------------------------------------------------
    logger.info("PHASE 1: Generating summaries with API backends")

    def _persist_generation(result: tuple[str, str, dict]) -> None:
        # Context first so the idempotency-gating generation record is written
        # last: a crash between the two re-generates (key absent) rather than
        # leaving a persisted generation with no cached source.
        key, context, record = result
        store.save_context(key, context)
        store.append_generation(record)

    for spec in args.model_spec:
        logger.info("Model: %s", spec)
        backend = create_backend(spec, cache_dir=getattr(args, "cache_dir", None))
        _apply_calibration(backend, spec, calibrations, config, store)

        worker = _make_generation_worker(backend, spec, context_builder)
        total = len(args.context_lengths) * actual_samples
        with tqdm(total=total, desc=spec, unit="gen") as pbar:
            for length in args.context_lengths:
                pending: list[tuple[int, int]] = []
                for doc_idx in range(actual_samples):
                    key = _generation_key(
                        spec, f"scrolls:{doc_idx}", length, PROMPT_VERSION
                    )
                    if store.has_generation(key):
                        pbar.update()
                    else:
                        pending.append((doc_idx, length))

                conc = _effective_concurrency(backend.provider, length, args.max_concurrency)
                _process_concurrently(pending, worker, _persist_generation, conc, pbar)

        if hasattr(backend, "release"):
            backend.release()
            logger.info("Released resources for %s", spec)

    # ------------------------------------------------------------------
    # Phase 2  —  Judging (separate phase: reads generations, writes judgments)
    # ------------------------------------------------------------------
    judge_spec = args.judge_spec
    if not judge_spec:
        logger.warning("No --judge-spec provided; skipping judgment phase.")
    else:
        logger.info("PHASE 2: Judging with %s", judge_spec)
        judge_backend = create_backend(judge_spec)
        _apply_calibration(judge_backend, judge_spec, calibrations, config, store)

        generations = store.load_generations()
        contexts = store.load_contexts()
        worker = _make_judge_worker(judge_backend, judge_spec, contexts)

        pending_judgments: list[dict] = []
        for gen in generations:
            if gen.get("output_text") is None:
                continue
            jkey = _judgment_key(gen["idempotency_key"], judge_spec, JUDGE_PROMPT_VERSION)
            if store.has_judgment(jkey):
                continue
            pending_judgments.append(gen)

        # Large-context judgments (or an HF judge) must run alone; everything
        # else runs concurrently. Run the concurrent batch first, then the
        # serial batch, so per-provider in-flight requests never exceed the cap.
        serial = [
            g for g in pending_judgments
            if _effective_concurrency(
                judge_backend.provider, g["requested_context_tokens"], args.max_concurrency
            ) == 1
        ]
        concurrent = [g for g in pending_judgments if g not in serial]

        with tqdm(total=len(pending_judgments), desc=f"judge ({judge_spec})", unit="jdg") as pbar:
            _process_concurrently(
                concurrent, worker, store.append_judgment, args.max_concurrency, pbar
            )
            _process_concurrently(serial, worker, store.append_judgment, 1, pbar)

        if hasattr(judge_backend, "release"):
            judge_backend.release()

    # ------------------------------------------------------------------
    # Phase 3  —  Leaderboard
    # ------------------------------------------------------------------
    logger.info("PHASE 3: Generating leaderboard")
    rates = store.compute_hallucination_rates()
    if rates:
        LeaderboardGenerator(output_dir=args.output_dir).generate_leaderboard(rates)

    totals = store.compute_cost_totals()
    if totals:
        logger.info("Cost summary:")
        for label, cost in totals.items():
            logger.info("  %s: $%.4f", label, cost)
        logger.info("  Total: $%.4f", sum(totals.values()))
        config["cost_totals"] = {k: round(v, 6) for k, v in totals.items()}
        store.save_config(config)

    logger.info("Run complete — id: %s  dir: %s", run_id, store.run_dir)
