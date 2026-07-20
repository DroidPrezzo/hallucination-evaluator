from __future__ import annotations

import logging
import time
from typing import Optional

from .base import GenerationResult, ModelBackend

logger = logging.getLogger(__name__)


class HFLocalBackend(ModelBackend):

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        revision: Optional[str] = None,
    ):
        from src.model_runner import ModelRunner

        self._model_name = model_name
        self._runner = ModelRunner(
            model_name=model_name,
            cache_dir=cache_dir,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            revision=revision,
        )

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "hf"

    @property
    def max_context(self) -> Optional[int]:
        cfg = getattr(self._runner.model, "config", None)
        if cfg is None:
            return None
        return getattr(cfg, "max_position_embeddings", None)

    @property
    def tokenizer(self):
        return self._runner.tokenizer

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        input_token_count = len(
            self._runner.tokenizer.encode(prompt, add_special_tokens=False)
        )

        start = time.perf_counter()
        text = self._runner.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.perf_counter() - start

        output_token_count = len(
            self._runner.tokenizer.encode(text, add_special_tokens=False)
        )

        return GenerationResult(
            text=text,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            latency_s=round(elapsed, 3),
            finish_reason="stop",
            raw={},
        )

    def count_tokens(self, text: str) -> int:
        return len(self._runner.tokenizer.encode(text, add_special_tokens=False))

    def release(self) -> None:
        import gc

        import torch

        del self._runner.model
        del self._runner.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
