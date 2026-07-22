"""Malformed judge-JSON handling and verdict parsing."""
from typing import Optional

from src.backends.base import GenerationResult, ModelBackend
from src.judging.claims import decompose_summary, parse_claims, parse_verdict


def test_parse_claims_valid():
    assert parse_claims('["a", "b"]') == ["a", "b"]


def test_parse_claims_empty_is_valid_not_none():
    assert parse_claims("[]") == []  # a valid "no claims" result, distinct from None


def test_parse_claims_strips_code_fence():
    assert parse_claims('```json\n["x"]\n```') == ["x"]


def test_parse_claims_extracts_embedded_array():
    assert parse_claims('Sure, here you go: ["a", "b"] done') == ["a", "b"]


def test_parse_claims_malformed_returns_none():
    assert parse_claims("not json at all") is None
    assert parse_claims('{"not": "a list"}') is None


def test_parse_claims_filters_non_strings_and_blanks():
    assert parse_claims('["a", 5, "", "  ", "b"]') == ["a", "b"]


def test_parse_verdict_checks_unsupported_before_supported():
    assert parse_verdict("unsupported") == "unsupported"
    assert parse_verdict("supported") == "supported"
    assert parse_verdict("The claim is UNSUPPORTED by the source.") == "unsupported"
    assert parse_verdict("ambiguous") == "ambiguous"
    assert parse_verdict("no idea") is None


class _ScriptedBackend(ModelBackend):
    """Returns a fixed list of outputs, one per generate() call."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self._i = 0

    @property
    def model_id(self) -> str:
        return "scripted"

    @property
    def provider(self) -> str:
        return "scripted"

    @property
    def max_context(self) -> Optional[int]:
        return None

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt, *, max_tokens, temperature=0.0, system_prompt=None):
        out = self._outputs[min(self._i, len(self._outputs) - 1)]
        self._i += 1
        return GenerationResult(text=out, input_tokens=1, output_tokens=1,
                                latency_s=0.0, finish_reason="stop")


def test_decompose_retries_once_then_succeeds():
    backend = _ScriptedBackend(["garbage, not json", '["a", "b"]'])
    claims, results = decompose_summary(backend, "some summary")
    assert claims == ["a", "b"]
    assert len(results) == 2  # first attempt failed, retried once


def test_decompose_malformed_twice_returns_none():
    backend = _ScriptedBackend(["garbage", "still not json"])
    claims, results = decompose_summary(backend, "some summary")
    assert claims is None          # caller marks judge_error
    assert len(results) == 2       # retried exactly once, did not loop forever
