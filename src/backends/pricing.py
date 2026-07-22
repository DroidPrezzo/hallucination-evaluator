from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_PRICING_PATH = Path(__file__).resolve().parent.parent.parent / "pricing.yaml"

_pricing_cache: Optional[dict] = None


def _load_pricing(path: Path = _DEFAULT_PRICING_PATH) -> dict:
    global _pricing_cache
    if _pricing_cache is not None:
        return _pricing_cache
    if not path.exists():
        logger.warning("pricing.yaml not found at %s; cost estimates disabled", path)
        _pricing_cache = {}
        return _pricing_cache
    with open(path) as f:
        _pricing_cache = yaml.safe_load(f) or {}
    return _pricing_cache


def estimate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    pricing_path: Path = _DEFAULT_PRICING_PATH,
) -> Optional[float]:
    data = _load_pricing(pricing_path)
    models = data.get("models", {})

    entry = models.get(model_id)
    if entry is None:
        for key, val in models.items():
            if model_id.startswith(key):
                entry = val
                break
    if entry is None:
        return None

    input_rate = entry["input_per_1m"]
    output_rate = entry["output_per_1m"]

    for tier in sorted(entry.get("tiers", []), key=lambda t: t["above_input_tokens"]):
        if input_tokens > tier["above_input_tokens"]:
            input_rate = tier.get("input_per_1m", input_rate)
            output_rate = tier.get("output_per_1m", output_rate)

    cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    return round(cost, 6)
