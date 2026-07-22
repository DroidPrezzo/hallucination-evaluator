#!/usr/bin/env python3
"""Export a stratified random sample of claim verdicts for blind human labeling.

Reads a completed run directory and writes a CSV where each row is one verified
claim with the source excerpt the labeler needs, a blank `human_label` column,
and the model's verdict in a separate trailing `judge_verdict` column that can be
hidden (delete/collapse the column) during blind annotation.

Usage:
    python scripts/export_validation_sample.py \
        --run-dir results/runs/<run_id> \
        --n 200 \
        --output validation_sample.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judging.retrieval import bm25_topk, build_chunks  # noqa: E402

MAX_EXCERPT_CHARS = 3000


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _excerpt_for_claim(
    source: str, claim: str, chunk_ids: list[int] | None, retrieval: dict
) -> str:
    """Build a human-readable source excerpt for a claim.

    Uses the exact chunks the judge saw when available (retrieved case);
    otherwise (full-source case) shows the top BM25 chunks so the labeler sees
    the most relevant passages without the entire document.
    """
    chunks = build_chunks(source, retrieval["chunk_size"], retrieval["overlap"])
    by_index = {c.index: c.text for c in chunks}

    if chunk_ids:
        texts = [by_index[i] for i in chunk_ids if i in by_index]
    else:
        top = bm25_topk(chunks, claim, min(3, retrieval["topk"]))
        texts = [t for _, t in top]

    excerpt = "\n[...]\n".join(texts) if texts else source
    if len(excerpt) > MAX_EXCERPT_CHARS:
        excerpt = excerpt[:MAX_EXCERPT_CHARS] + " […truncated]"
    return excerpt


def _stratified_sample(
    candidates: list[dict], n: int, seed: int
) -> list[dict]:
    """Sample n rows spread across (model_spec, length) strata, seed-reproducible."""
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in candidates:
        strata[(row["model_spec"], row["requested_context_tokens"])].append(row)

    if n >= len(candidates):
        out = list(candidates)
        rng.shuffle(out)
        return out

    per = max(1, n // len(strata))
    selected: list[dict] = []
    selected_uids: set[int] = set()
    for rows in strata.values():
        rng.shuffle(rows)
        for row in rows[:per]:
            selected.append(row)
            selected_uids.add(row["_uid"])

    if len(selected) > n:
        rng.shuffle(selected)
        selected = selected[:n]
    elif len(selected) < n:
        remaining = [r for r in candidates if r["_uid"] not in selected_uids]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Path to results/runs/<run_id>")
    parser.add_argument("--n", type=int, default=200, help="Number of claims to sample")
    parser.add_argument("--judge-spec", default=None,
                        help="Which judge's verdicts to sample (default: first in config)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--output", default="validation_sample.csv", help="Output CSV path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    retrieval = config.get(
        "retrieval",
        {"topk": 6, "chunk_size": 400, "overlap": 80, "full_source_threshold": 8000},
    )

    judge_spec = args.judge_spec or (config.get("judge_specs") or [None])[0]
    if judge_spec is None:
        sys.exit("No judge spec found in config; pass --judge-spec.")

    generations = {g["idempotency_key"]: g for g in _load_jsonl(run_dir / "generations.jsonl")}
    contexts = {c["generation_key"]: c["source_text"]
                for c in _load_jsonl(run_dir / "contexts.jsonl")}
    verifications = _load_jsonl(run_dir / "verifications.jsonl")

    candidates: list[dict] = []
    uid = 0
    for v in verifications:
        if v["judge_spec"] != judge_spec:
            continue
        if v.get("verdict") not in ("supported", "unsupported", "ambiguous"):
            continue
        gen = generations.get(v["generation_key"])
        if gen is None:
            continue
        candidates.append({
            "_uid": uid,
            "generation_key": v["generation_key"],
            "claim_id": v["claim_id"],
            "claim_text": v["claim_text"],
            "model_spec": gen["model_spec"],
            "requested_context_tokens": gen["requested_context_tokens"],
            "retrieved_chunk_ids": v.get("retrieved_chunk_ids"),
            "judge_verdict": v["verdict"],
        })
        uid += 1

    if not candidates:
        sys.exit(f"No verified claims found for judge '{judge_spec}' in {run_dir}.")

    sample = _stratified_sample(candidates, args.n, args.seed)

    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "model_spec", "requested_context_tokens",
            "generation_key", "claim_id", "claim_text", "source_excerpt",
            "human_label", "judge_spec", "judge_verdict",
        ])
        for i, row in enumerate(sample):
            source = contexts.get(row["generation_key"], "")
            excerpt = _excerpt_for_claim(
                source, row["claim_text"], row["retrieved_chunk_ids"], retrieval
            )
            writer.writerow([
                i, row["model_spec"], row["requested_context_tokens"],
                row["generation_key"], row["claim_id"], row["claim_text"], excerpt,
                "", judge_spec, row["judge_verdict"],
            ])

    print(f"Wrote {len(sample)} claims (of {len(candidates)} candidates) to {out_path}")
    print(f"Judge: {judge_spec}  |  seed: {args.seed}")
    print("Blind labeling: hide the 'judge_verdict' column, fill 'human_label' with "
          "supported / unsupported / ambiguous.")


if __name__ == "__main__":
    main()
