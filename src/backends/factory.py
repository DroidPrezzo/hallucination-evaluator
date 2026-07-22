from __future__ import annotations

import logging
from typing import Any
from urllib.parse import parse_qs

from .base import ModelBackend

logger = logging.getLogger(__name__)


def _to_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("1", "true", "yes")


def create_backend(spec: str, **overrides: Any) -> ModelBackend:
    """Parse a model spec string and return a configured backend.

    Formats:
        hf:Qwen/Qwen2.5-7B-Instruct
        hf:Qwen/Qwen2.5-7B-Instruct?load_in_4bit=1&revision=abc123
        openai:gpt-5.5
        openai:grok-4.5?base_url=https://api.x.ai/v1&key_env=XAI_API_KEY
        openai:kimi-k3?base_url=https://api.moonshot.ai/v1&key_env=MOONSHOT_API_KEY
        anthropic:claude-opus-4-8
    """
    if ":" not in spec:
        raise ValueError(
            f"Invalid model spec '{spec}': expected 'provider:model_id[?params]'. "
            "Examples: hf:Qwen/Qwen2.5-7B-Instruct, openai:gpt-5.5, "
            "anthropic:claude-opus-4-8"
        )

    provider, rest = spec.split(":", 1)

    if "?" in rest:
        model_id, query = rest.split("?", 1)
        params: dict[str, Any] = {k: v[0] for k, v in parse_qs(query).items()}
    else:
        model_id = rest
        params = {}

    params.update(overrides)

    if provider == "hf":
        from .hf_local import HFLocalBackend

        return HFLocalBackend(
            model_name=model_id,
            cache_dir=params.get("cache_dir"),
            load_in_4bit=_to_bool(params.get("load_in_4bit", False)),
            load_in_8bit=_to_bool(params.get("load_in_8bit", False)),
            revision=params.get("revision"),
        )

    if provider == "openai":
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(
            model_id=model_id,
            base_url=params.get("base_url"),
            key_env=params.get("key_env", "OPENAI_API_KEY"),
            max_context=int(params["max_context"]) if "max_context" in params else None,
        )

    if provider == "anthropic":
        from .anthropic_client import AnthropicBackend

        return AnthropicBackend(
            model_id=model_id,
            key_env=params.get("key_env", "ANTHROPIC_API_KEY"),
            max_context=int(params["max_context"]) if "max_context" in params else None,
        )

    raise ValueError(f"Unknown provider '{provider}' in spec '{spec}'")
