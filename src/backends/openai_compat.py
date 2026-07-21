from __future__ import annotations

import logging
import os
import time
from typing import Optional, Tuple

import httpx
import openai

from .base import GenerationResult, ModelBackend
from .pricing import estimate_cost
from .retry import RetryConfig, retry_api_call

logger = logging.getLogger(__name__)


def _classify_openai_error(exc: Exception) -> Tuple[str, Optional[float]]:
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return "timeout", None
    if isinstance(exc, openai.APIStatusError):
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


class OpenAICompatBackend(ModelBackend):

    def __init__(
        self,
        model_id: str,
        *,
        base_url: Optional[str] = None,
        key_env: str = "OPENAI_API_KEY",
        max_context: Optional[int] = None,
    ):
        self._model_id = model_id
        self._max_context = max_context
        self._key_env = key_env

        api_key = os.environ.get(key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable {key_env} is not set. "
                f"Required for model {model_id}."
            )

        kwargs: dict = {
            "api_key": api_key,
            "max_retries": 0,
            "timeout": httpx.Timeout(
                connect=30.0, read=900.0, write=30.0, pool=30.0
            ),
        }
        if base_url:
            kwargs["base_url"] = base_url

        self._client = openai.OpenAI(**kwargs)
        self._retry_config = RetryConfig()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "openai"

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
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _call() -> openai.types.chat.ChatCompletion:
            return self._client.chat.completions.create(
                model=self._model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=0,
            )

        start = time.perf_counter()
        response = retry_api_call(_call, _classify_openai_error, self._retry_config)
        elapsed = time.perf_counter() - start

        choice = response.choices[0]
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return GenerationResult(
            text=choice.message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=round(elapsed, 3),
            finish_reason=choice.finish_reason or "unknown",
            raw={
                "id": response.id,
                "model": response.model,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
            },
            cost_estimate_usd=estimate_cost(
                self._model_id, input_tokens, output_tokens
            ),
        )

    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self._model_id)
            return len(enc.encode(text))
        except (KeyError, ImportError):
            return max(1, len(text) // 4)

    def truncate(self, text: str, target_length: int) -> str:
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self._model_id)
            max_chars = target_length * 8
            if len(text) > max_chars:
                text = text[:max_chars]
            tokens = enc.encode(text)
            if len(tokens) <= target_length:
                return text
            return enc.decode(tokens[:target_length])
        except (KeyError, ImportError):
            return super().truncate(text, target_length)
