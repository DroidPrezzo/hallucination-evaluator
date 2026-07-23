from __future__ import annotations

import hashlib
import json
from typing import Optional

from .base import GenerationResult, ModelBackend


def _h(s: str) -> int:
    """Stable non-cryptographic hash -> int, for deterministic pseudo-variation."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


class MockBackend(ModelBackend):
    """Deterministic, offline backend for --dry-run and tests.

    Uses a whitespace word tokenizer (exact and round-trip stable, so truncation
    lands on exact token boundaries) and returns canned outputs chosen by the
    system-prompt role. Outputs vary by a hash of the prompt so summaries, claim
    sets, and verdicts differ across documents; verdicts share a base signal
    across judges (correlated) with a small per-judge flip so cross-judge kappa
    is non-trivial. No network, keys, or GPU.
    """

    def __init__(self, model_id: str = "mock", provider: str = "mock", **_ignored):
        self._model_id = model_id
        self._provider = provider
        self._chars_per_token: Optional[float] = 4.0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def max_context(self) -> Optional[int]:
        return None

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def truncate(self, text: str, target_length: int) -> str:
        words = text.split()
        if len(words) <= target_length:
            return text
        return " ".join(words[:target_length])

    def needs_calibration(self) -> bool:
        return False

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        sysp = system_prompt or ""
        if "claim-extraction" in sysp:
            text = self._decompose(prompt)
            out_tokens = self.count_tokens(text)
        elif "fact-checker" in sysp and "one word" in sysp:
            text = self._verify(prompt)
            out_tokens = 1
        elif "fact-checker" in sysp:  # holistic YES/NO judge
            text = "YES" if _h("holistic|" + prompt) % 2 == 0 else "NO"
            out_tokens = 1
        else:  # summarizer
            text = self._summary(prompt)
            out_tokens = self.count_tokens(text)

        return GenerationResult(
            text=text,
            input_tokens=self.count_tokens(prompt),
            output_tokens=out_tokens,
            latency_s=0.0,
            finish_reason="stop",
            raw={"mock": True},
            cost_estimate_usd=None,
        )

    def _summary(self, prompt: str) -> str:
        # Vary by model_id so different test models yield different summaries
        # (and therefore different claims and hallucination rates downstream).
        tag = _h("sum|" + self._model_id + "|" + prompt) % 1000
        return (f"Mock summary {tag} ({self._model_id}): the source reports "
                f"several figures and describes agency activities for the period.")

    def _decompose(self, prompt: str) -> str:
        # Claim text carries a per-summary tag so verification verdicts (which
        # hash the claim) differ across the summaries being judged.
        tag = _h("dc|" + prompt) % 10000
        claims = [f"The source asserts point {i} under reference {tag}." for i in range(3)]
        return json.dumps(claims)

    def _verify(self, prompt: str) -> str:
        # Base signal is shared across judges (same prompt) so verdicts correlate;
        # a small per-judge flip keyed on model_id keeps kappa below 1.
        base = _h("verify|" + prompt) % 100
        verdict = "unsupported" if base < 35 else "supported"
        if _h(self._model_id + "|flip|" + prompt) % 100 < 10:
            verdict = "supported" if verdict == "unsupported" else "unsupported"
        return verdict
