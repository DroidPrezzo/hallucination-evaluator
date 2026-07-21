from __future__ import annotations

import json
import logging
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

        self._gen_keys: set[str] = set()
        self._judge_keys: set[str] = set()
        self._load_existing_keys()

    def _load_existing_keys(self) -> None:
        for path, key_set in [
            (self._gen_path, self._gen_keys),
            (self._judge_path, self._judge_keys),
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
                "Resumed run %s: %d generations, %d judgments already on disk",
                self.run_id, len(self._gen_keys), len(self._judge_keys),
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
        return totals
