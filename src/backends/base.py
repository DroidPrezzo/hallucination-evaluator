from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    finish_reason: str
    raw: dict = field(default_factory=dict)
    cost_estimate_usd: Optional[float] = None


class ModelBackend(ABC):

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def provider(self) -> str: ...

    @property
    @abstractmethod
    def max_context(self) -> Optional[int]: ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult: ...

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...
