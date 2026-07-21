from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class RunStore:

    def __init__(self, run_id: str, output_dir: str = "results"):
        self.run_id = run_id
        self.run_dir = Path(output_dir) / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._gen_path = self.run_dir / "generations.jsonl"
        self._judge_path = self.run_dir / "judgments.jsonl"
        self._config_path = self.run_dir / "config.json"
        self._contexts_path = self.run_dir / "contexts.jsonl"
        self._decomp_path = self.run_dir / "decompositions.jsonl"
        self._verify_path = self.run_dir / "verifications.jsonl"

        self._gen_keys: set[str] = set()
        self._judge_keys: set[str] = set()
        self._decomp_keys: set[str] = set()
        self._verify_keys: set[str] = set()
        self._load_existing_keys()

    def _load_existing_keys(self) -> None:
        for path, key_set in [
            (self._gen_path, self._gen_keys),
            (self._judge_path, self._judge_keys),
            (self._decomp_path, self._decomp_keys),
            (self._verify_path, self._verify_keys),
        ]:
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        key_set.add(json.loads(line)["idempotency_key"])
        if self._gen_keys:
            logger.info(
                "Resumed run %s: %d generations, %d judgments, "
                "%d decompositions, %d verifications already on disk",
                self.run_id, len(self._gen_keys), len(self._judge_keys),
                len(self._decomp_keys), len(self._verify_keys),
            )

    # ---- config ----

    def save_config(self, config: dict) -> None:
        with open(self._config_path, "w") as f:
            json.dump(config, f, indent=2)

    def load_config(self) -> Optional[dict]:
        if not self._config_path.exists():
            return None
        with open(self._config_path) as f:
            return json.load(f)

    # ---- generations ----

    def has_generation(self, key: str) -> bool:
        return key in self._gen_keys

    def append_generation(self, record: dict) -> None:
        with open(self._gen_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        self._gen_keys.add(record["idempotency_key"])

    def load_generations(self) -> list[dict]:
        if not self._gen_path.exists():
            return []
        with open(self._gen_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    # ---- contexts (source text keyed by generation key) ----

    def save_context(self, generation_key: str, source_text: str) -> None:
        record = {"generation_key": generation_key, "source_text": source_text}
        with open(self._contexts_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    def load_contexts(self) -> dict[str, str]:
        if not self._contexts_path.exists():
            return {}
        result: dict[str, str] = {}
        with open(self._contexts_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    result[rec["generation_key"]] = rec["source_text"]
        return result

    # ---- judgments ----

    def has_judgment(self, key: str) -> bool:
        return key in self._judge_keys

    def append_judgment(self, record: dict) -> None:
        with open(self._judge_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        self._judge_keys.add(record["idempotency_key"])

    def load_judgments(self) -> list[dict]:
        if not self._judge_path.exists():
            return []
        with open(self._judge_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    # ---- decompositions (claims mode) ----

    def has_decomposition(self, key: str) -> bool:
        return key in self._decomp_keys

    def append_decomposition(self, record: dict) -> None:
        with open(self._decomp_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        self._decomp_keys.add(record["idempotency_key"])

    def load_decompositions(self) -> list[dict]:
        if not self._decomp_path.exists():
            return []
        with open(self._decomp_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    # ---- verifications (claims mode) ----

    def has_verification(self, key: str) -> bool:
        return key in self._verify_keys

    def append_verification(self, record: dict) -> None:
        with open(self._verify_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        self._verify_keys.add(record["idempotency_key"])

    def load_verifications(self) -> list[dict]:
        if not self._verify_path.exists():
            return []
        with open(self._verify_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    # ---- aggregation helpers ----

    def compute_hallucination_rates(self) -> dict[str, dict[int, float]]:
        generations = {g["idempotency_key"]: g for g in self.load_generations()}
        judgments = self.load_judgments()

        groups: dict[str, dict[int, list[bool]]] = {}
        for j in judgments:
            if j.get("verdict") is None:
                continue
            gen = generations.get(j["generation_key"])
            if gen is None:
                continue
            model = gen["model_spec"]
            length = gen["requested_context_tokens"]
            groups.setdefault(model, {}).setdefault(length, []).append(j["verdict"])

        results: dict[str, dict[int, float]] = {}
        for model, lengths in groups.items():
            results[model] = {}
            for length, verdicts in lengths.items():
                results[model][length] = sum(verdicts) / len(verdicts) if verdicts else 0.0
        return results

    def compute_claim_hallucination_rates(
        self, judge_spec: str
    ) -> dict[str, dict[int, float]]:
        """Per (model_spec, length), mean over summaries of the per-summary
        claim-level hallucination rate (unsupported / verified claims) for one
        judge. Summaries with no successfully-verified claims are skipped.
        """
        generations = {g["idempotency_key"]: g for g in self.load_generations()}

        per_gen: dict[str, dict[str, int]] = defaultdict(
            lambda: {"supported": 0, "unsupported": 0, "ambiguous": 0}
        )
        for v in self.load_verifications():
            if v["judge_spec"] != judge_spec:
                continue
            verdict = v.get("verdict")
            if verdict in ("supported", "unsupported", "ambiguous"):
                per_gen[v["generation_key"]][verdict] += 1

        groups: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for gen_key, counts in per_gen.items():
            total = counts["supported"] + counts["unsupported"] + counts["ambiguous"]
            if total == 0:
                continue
            gen = generations.get(gen_key)
            if gen is None:
                continue
            rate = counts["unsupported"] / total
            groups[gen["model_spec"]][gen["requested_context_tokens"]].append(rate)

        results: dict[str, dict[int, float]] = {}
        for model, lengths in groups.items():
            results[model] = {
                length: sum(rates) / len(rates) for length, rates in lengths.items()
            }
        return results

    def compute_cost_totals(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for gen in self.load_generations():
            cost = gen.get("cost_estimate_usd")
            if cost is not None:
                totals[gen["model_spec"]] = totals.get(gen["model_spec"], 0.0) + cost
        for j in self.load_judgments():
            cost = j.get("cost_estimate_usd")
            if cost is not None:
                key = f"{j['judge_spec']} (judge)"
                totals[key] = totals.get(key, 0.0) + cost
        for d in self.load_decompositions():
            cost = d.get("cost_estimate_usd")
            if cost is not None:
                key = f"{d['decomposer_spec']} (decompose)"
                totals[key] = totals.get(key, 0.0) + cost
        for v in self.load_verifications():
            cost = v.get("cost_estimate_usd")
            if cost is not None:
                key = f"{v['judge_spec']} (verify)"
                totals[key] = totals.get(key, 0.0) + cost
        return totals
