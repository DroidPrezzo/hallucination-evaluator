from __future__ import annotations

import json
import logging
import re
from typing import Optional

from src.backends.base import GenerationResult, ModelBackend
from src.prompts import (
    DECOMPOSE_SYSTEM,
    VERIFY_SYSTEM,
    build_decompose_prompt,
    build_verify_prompt,
)

logger = logging.getLogger(__name__)

DECOMPOSE_MAX_TOKENS = 2048
VERIFY_MAX_TOKENS = 10

VALID_VERDICTS = ("supported", "unsupported", "ambiguous")

_FENCE_RE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)


def parse_claims(raw: str) -> Optional[list[str]]:
    """Parse a JSON array of claim strings from a model response.

    Returns the list (possibly empty, a valid 'no claims' result) or None when
    the response is not parseable as a JSON string array.
    """
    text = _FENCE_RE.sub("", raw).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    return [c.strip() for c in data if isinstance(c, str) and c.strip()]


def parse_verdict(raw: str) -> Optional[str]:
    """Extract a verdict from a verification response.

    Checks 'unsupported' before 'supported' because the latter is a substring of
    the former. Returns None when no verdict word is present.
    """
    text = raw.lower()
    if "unsupported" in text:
        return "unsupported"
    if "supported" in text:
        return "supported"
    if "ambiguous" in text:
        return "ambiguous"
    return None


def decompose_summary(
    backend: ModelBackend, summary: str
) -> tuple[Optional[list[str]], list[GenerationResult]]:
    """Extract atomic claims from a summary.

    Retries once on malformed JSON. Returns (claims, results); claims is None
    when both attempts fail to parse (caller marks judge_error). results holds
    every GenerationResult so cost/usage across retries can be summed.
    """
    prompt = build_decompose_prompt(summary)
    results: list[GenerationResult] = []
    for attempt in range(2):
        result = backend.generate(
            prompt,
            max_tokens=DECOMPOSE_MAX_TOKENS,
            temperature=0.0,
            system_prompt=DECOMPOSE_SYSTEM,
        )
        results.append(result)
        claims = parse_claims(result.text)
        if claims is not None:
            return claims, results
        logger.warning("Malformed decomposition JSON (attempt %d/2)", attempt + 1)
    return None, results


def verify_claim(
    backend: ModelBackend, source: str, claim: str
) -> tuple[Optional[str], list[GenerationResult]]:
    """Verify a single claim against a source passage.

    Retries once on an unparseable verdict. Returns (verdict, results); verdict
    is None when both attempts fail to parse (caller records a verify error).
    """
    prompt = build_verify_prompt(source, claim)
    results: list[GenerationResult] = []
    for attempt in range(2):
        result = backend.generate(
            prompt,
            max_tokens=VERIFY_MAX_TOKENS,
            temperature=0.0,
            system_prompt=VERIFY_SYSTEM,
        )
        results.append(result)
        verdict = parse_verdict(result.text)
        if verdict is not None:
            return verdict, results
        logger.warning("Unparseable verdict '%s' (attempt %d/2)", result.text, attempt + 1)
    return None, results
