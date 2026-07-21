from __future__ import annotations

import hashlib
import logging
import subprocess
import uuid
from datetime import datetime, timezone

from tqdm import tqdm

from src.backends import create_backend
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


def run_backend_pipeline(args) -> None:
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
        "prompt_version": PROMPT_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
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

    for spec in args.model_spec:
        logger.info("Model: %s", spec)
        backend = create_backend(
            spec, cache_dir=getattr(args, "cache_dir", None)
        )

        if backend.needs_calibration():
            if spec in calibrations:
                backend._chars_per_token = calibrations[spec]
                logger.info(
                    "Loaded calibration for %s: %.2f chars/tok", spec, calibrations[spec]
                )
            else:
                ratio = backend.calibrate()
                calibrations[spec] = round(ratio, 4)
                config["calibrations"] = calibrations
                store.save_config(config)

        total = len(args.context_lengths) * actual_samples
        with tqdm(total=total, desc=spec, unit="gen") as pbar:
            for length in args.context_lengths:
                for doc_idx in range(actual_samples):
                    doc_id = f"scrolls:{doc_idx}"
                    key = _generation_key(spec, doc_id, length, PROMPT_VERSION)

                    if store.has_generation(key):
                        pbar.update()
                        continue

                    doc = context_builder.get_document(doc_idx)
                    context = backend.truncate(doc, length)
                    source_sha = hashlib.sha256(context.encode()).hexdigest()
                    prompt = build_summarize_prompt(context)

                    local_token_count = backend.count_tokens(prompt)

                    try:
                        result = backend.generate(
                            prompt,
                            max_tokens=512,
                            temperature=0.0,
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

                    store.append_generation(record)
                    store.save_context(key, context)
                    pbar.update()

        if hasattr(backend, "release"):
            backend.release()
            logger.info("Released resources for %s", spec)

    # ------------------------------------------------------------------
    # Phase 2  —  Judging
    # ------------------------------------------------------------------
    judge_spec = args.judge_spec
    if not judge_spec:
        logger.warning("No --judge-spec provided; skipping judgment phase.")
    else:
        logger.info("PHASE 2: Judging with %s", judge_spec)
        judge_backend = create_backend(judge_spec)

        if judge_backend.needs_calibration():
            if judge_spec in calibrations:
                judge_backend._chars_per_token = calibrations[judge_spec]
                logger.info(
                    "Loaded calibration for judge %s: %.2f chars/tok",
                    judge_spec, calibrations[judge_spec],
                )
            else:
                ratio = judge_backend.calibrate()
                calibrations[judge_spec] = round(ratio, 4)
                config["calibrations"] = calibrations
                store.save_config(config)

        generations = store.load_generations()
        contexts = store.load_contexts()

        judgeable = [g for g in generations if g.get("output_text") is not None]
        for gen in tqdm(judgeable, desc=f"judge ({judge_spec})", unit="jdg"):
            gen_key = gen["idempotency_key"]
            jkey = _judgment_key(gen_key, judge_spec, JUDGE_PROMPT_VERSION)

            if store.has_judgment(jkey):
                continue

            source_text = contexts.get(gen_key)
            if source_text is None:
                doc_idx = int(gen["doc_id"].split(":")[1])
                source_text = backend.truncate(
                    context_builder.get_document(doc_idx),
                    gen["requested_context_tokens"],
                )
                logger.warning(
                    "Context for %s not in cache; re-truncated from dataset "
                    "(tokenizer may differ)",
                    gen_key[:12],
                )

            prompt = build_judge_prompt(source_text, gen["output_text"])
            local_judge_tokens = judge_backend.count_tokens(prompt)

            try:
                result = judge_backend.generate(
                    prompt,
                    max_tokens=10,
                    temperature=0.0,
                    system_prompt=JUDGE_SYSTEM,
                )
                resp = result.text.upper().strip()
                if "YES" in resp:
                    verdict: bool | None = True
                elif "NO" in resp:
                    verdict = False
                else:
                    logger.warning("Ambiguous judge response: %s — defaulting to hallucination", result.text)
                    verdict = True

                judgment = {
                    "idempotency_key": jkey,
                    "generation_key": gen_key,
                    "judge_spec": judge_spec,
                    "judge_prompt_version": JUDGE_PROMPT_VERSION,
                    "judge_mode": "holistic",
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
                judgment = {
                    "idempotency_key": jkey,
                    "generation_key": gen_key,
                    "judge_spec": judge_spec,
                    "judge_prompt_version": JUDGE_PROMPT_VERSION,
                    "judge_mode": "holistic",
                    "verdict": None,
                    "error": str(exc),
                    "raw_response": None,
                    "usage": {
                        "input_tokens": local_judge_tokens,
                        "provider_reported_input_tokens": None,
                        "output_tokens": 0,
                    },
                    "latency_s": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "cost_estimate_usd": None,
                }

            store.append_judgment(judgment)

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
