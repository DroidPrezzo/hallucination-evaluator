from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Optional

_WORD_RE = re.compile(r"\w+")

# Standard Okapi BM25 free parameters.
_BM25_K1 = 1.5
_BM25_B = 0.75

# Separator inserted between non-adjacent retrieved chunks so the judge sees a
# clear discontinuity rather than a false-contiguous passage.
_CHUNK_SEPARATOR = "\n[...]\n"


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class Chunk:
    index: int
    text: str
    tokens: list[str]


def build_chunks(text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Split text into overlapping word-based chunks, preserving original characters.

    Chunk boundaries fall on word edges; the chunk text is the original
    substring spanning those words (whitespace/punctuation retained).
    """
    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return []

    step = max(1, chunk_size - overlap)
    chunks: list[Chunk] = []
    i = 0
    n = len(matches)
    while i < n:
        window = matches[i : i + chunk_size]
        start_char = window[0].start()
        end_char = window[-1].end()
        chunks.append(
            Chunk(
                index=len(chunks),
                text=text[start_char:end_char],
                tokens=[m.group().lower() for m in window],
            )
        )
        if i + chunk_size >= n:
            break
        i += step
    return chunks


def bm25_topk(
    chunks: list[Chunk], query: str, k: int
) -> list[tuple[int, str]]:
    """Rank chunks by BM25 against the query; return top-k as (index, text).

    Results are returned in original document order (ascending index), not by
    score, so the reassembled passage reads top-to-bottom.
    """
    n = len(chunks)
    if n == 0:
        return []

    query_terms = set(_tokenize(query))
    if not query_terms:
        return [(c.index, c.text) for c in chunks[:k]]

    df: dict[str, int] = {}
    for c in chunks:
        for t in set(c.tokens):
            df[t] = df.get(t, 0) + 1

    avgdl = sum(len(c.tokens) for c in chunks) / n
    idf = {
        t: math.log(1 + (n - df_t + 0.5) / (df_t + 0.5)) for t, df_t in df.items()
    }

    scored: list[tuple[float, int]] = []
    for c in chunks:
        tf: dict[str, int] = {}
        for t in c.tokens:
            tf[t] = tf.get(t, 0) + 1
        dl = len(c.tokens)
        norm = _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avgdl) if avgdl > 0 else _BM25_K1
        score = 0.0
        for t in query_terms:
            f = tf.get(t, 0)
            if f == 0:
                continue
            score += idf.get(t, 0.0) * (f * (_BM25_K1 + 1)) / (f + norm)
        scored.append((score, c.index))

    # Highest score first; ties broken by earlier document position.
    top = sorted(scored, key=lambda s: (-s[0], s[1]))[:k]
    top_indices = sorted(idx for _, idx in top)
    by_index = {c.index: c.text for c in chunks}
    return [(idx, by_index[idx]) for idx in top_indices]


def select_source(
    source_text: str,
    claim: str,
    *,
    count_tokens: Callable[[str], int],
    topk: int,
    chunk_size: int,
    overlap: int,
    full_source_threshold: int,
) -> tuple[str, Optional[list[int]]]:
    """Choose the source passage to verify a claim against.

    Short sources (<= full_source_threshold tokens) are returned whole to avoid
    retrieval misses. Longer sources are reduced to the top-k BM25 chunks so the
    full long document is never re-sent per claim.

    Returns (passage, chunk_ids). chunk_ids is None when the full source is used.
    """
    if count_tokens(source_text) <= full_source_threshold:
        return source_text, None

    chunks = build_chunks(source_text, chunk_size, overlap)
    top = bm25_topk(chunks, claim, topk)
    chunk_ids = [idx for idx, _ in top]
    passage = _CHUNK_SEPARATOR.join(text for _, text in top)
    return passage, chunk_ids
