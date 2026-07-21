from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


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

    _chars_per_token: Optional[float] = None

    def truncate(self, text: str, target_length: int) -> str:
        max_chars = target_length * 8
        if len(text) > max_chars:
            text = text[:max_chars]
        current = self.count_tokens(text)
        if current <= target_length:
            return text
        keep_ratio = target_length / current
        return text[: max(1, int(len(text) * keep_ratio))]

    def needs_calibration(self) -> bool:
        return False

    def calibrate(self) -> float:
        """Calibrate chars-per-token from two cheap API calls.

        Uses the delta between a short and a long probe to cancel out
        fixed message-framing overhead.
        """
        short = "The quick brown fox jumps over the lazy dog. " * 5
        long = "The quick brown fox jumps over the lazy dog. " * 100
        try:
            short_r = self.generate(short, max_tokens=1, temperature=0.0)
            long_r = self.generate(long, max_tokens=1, temperature=0.0)
            delta_chars = len(long) - len(short)
            delta_tokens = long_r.input_tokens - short_r.input_tokens
            if delta_tokens > 0:
                self._chars_per_token = delta_chars / delta_tokens
            else:
                self._chars_per_token = 4.0
                logger.warning("Calibration delta was 0; using default 4.0")
        except Exception as exc:
            self._chars_per_token = 4.0
            logger.warning("Calibration failed (%s); using default 4.0", exc)
        logger.info(
            "Calibrated %s: %.2f chars/token", self.model_id, self._chars_per_token
        )
        return self._chars_per_token
