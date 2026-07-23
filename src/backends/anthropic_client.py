from __future__ import annotations

import logging
import os
import time
from typing import Optional, Tuple

import anthropic
import httpx

from .base import GenerationResult, ModelBackend
from .pricing import estimate_cost
from .retry import RetryConfig, retry_api_call

logger = logging.getLogger(__name__)


def _classify_anthropic_error(exc: Exception) -> Tuple[str, Optional[float]]:
    if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return "timeout", None
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code in (429, 529) or exc.status_code >= 500:
            retry_after = None
            if exc.response is not None:
                raw = exc.response.headers.get("retry-after")
                if raw:
                    try:
                        retry_after = float(raw)
                    except ValueError:
                        pass
            return "server", retry_after
        return "client", None
    return "client", None


class AnthropicBackend(ModelBackend):

    def __init__(
        self,
        model_id: str,
        *,
        key_env: str = "ANTHROPIC_API_KEY",
        max_context: Optional[int] = None,
    ):
        self._model_id = model_id
        self._max_context = max_context

        api_key = os.environ.get(key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable {key_env} is not set. "
                f"Required for model {model_id}."
            )

        self._client = anthropic.Anthropic(
            api_key=api_key,
            max_retries=0,
            timeout=httpx.Timeout(
                connect=30.0, read=900.0, write=30.0, pool=30.0
            ),
        )
        self._retry_config = RetryConfig()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def max_context(self) -> Optional[int]:
        return self._max_context

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        kwargs: dict = {
            "model": self._model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            # Prompt caching (#2): the system prompt/rubric is byte-identical
            # across every call, so mark it as a cache breakpoint. Anthropic only
            # caches prefixes at/above its minimum cacheable length, so for the
            # current short rubrics this is a harmless no-op that starts paying
            # off once the cached prefix grows (e.g. nested-prefix bundles).
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        def _call() -> anthropic.types.Message:
            return self._client.messages.create(**kwargs)

        start = time.perf_counter()
        response = retry_api_call(
            _call, _classify_anthropic_error, self._retry_config
        )
        elapsed = time.perf_counter() - start

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=round(elapsed, 3),
            finish_reason=response.stop_reason or "unknown",
            raw={"id": response.id, "model": response.model},
            cost_estimate_usd=estimate_cost(
                self._model_id, input_tokens, output_tokens
            ),
        )

    def needs_calibration(self) -> bool:
        return self._chars_per_token is None

    def count_tokens(self, text: str) -> int:
        ratio = self._chars_per_token or 4.0
        return max(1, int(len(text) / ratio))
