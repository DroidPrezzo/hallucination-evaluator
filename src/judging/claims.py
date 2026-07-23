from __future__ import annotations

import json
import logging
import re
from typing import Optional

from src.backends.base import GenerationResult, ModelBackend
from src.prompts import (
    DECOMPOSE_SYSTEM,
    VERIFY_SYSTEM,
    build_batch_verify_prompt,
    build_decompose_prompt,
    build_verify_prompt,
)

logger = logging.getLogger(__name__)

DECOMPOSE_MAX_TOKENS = 2048
VERIFY_MAX_TOKENS = 10
# Batched verification returns a JSON array of {id, verdict}. Budget ~24 tokens
# per claim plus slack; summaries decompose to at most a few dozen claims.
VERIFY_BATCH_MAX_TOKENS = 1536

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


def parse_verdict_array(raw: str, n_claims: int) -> Optional[list[Optional[str]]]:
    """Parse a batched verification response into per-claim verdicts.

    Expects a JSON array of {"id": int, "verdict": str}. Returns a list of length
    n_claims positioned by id (entries left None when missing/invalid), or None
    when the response is not a JSON array at all (caller retries).
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

    verdicts: list[Optional[str]] = [None] * n_claims
    for item in data:
        if not isinstance(item, dict):
            continue
        cid, verdict = item.get("id"), item.get("verdict")
        if not isinstance(cid, int) or not (0 <= cid < n_claims):
            continue
        if isinstance(verdict, str) and verdict.strip().lower() in VALID_VERDICTS:
            verdicts[cid] = verdict.strip().lower()
    return verdicts


def verify_claims_batch(
    backend: ModelBackend, source: str, claims: list[str]
) -> tuple[list[Optional[str]], list[GenerationResult]]:
    """Verify ALL of a summary's claims against the source in one API call.

    Retries once if the array is unparseable or any claim is left unverified.
    Returns (verdicts, results): verdicts is length-len(claims), positioned by
    claim id, with None for any claim the judge failed to resolve (caller records
    those as verify errors). results holds every GenerationResult for cost/usage.
    """
    if not claims:
        return [], []
    prompt = build_batch_verify_prompt(source, claims)
    results: list[GenerationResult] = []
    best: Optional[list[Optional[str]]] = None
    for attempt in range(2):
        result = backend.generate(
            prompt,
            max_tokens=VERIFY_BATCH_MAX_TOKENS,
            temperature=0.0,
            system_prompt=VERIFY_SYSTEM,
        )
        results.append(result)
        parsed = parse_verdict_array(result.text, len(claims))
        if parsed is not None:
            best = parsed
            if all(v is not None for v in parsed):
                return parsed, results
        logger.warning(
            "Batched verify incomplete/unparseable (attempt %d/2)", attempt + 1
        )
    return (best if best is not None else [None] * len(claims)), results


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
