#!/usr/bin/env python3
"""Shared helpers for the corpus download scripts (Federal Register, SEC EDGAR,
GAO). Reuses the HTML/token helpers from build_corpus_from_manifest.py so text
extraction and token estimation stay identical to the corpus builder.

The manifest these scripts write is the same one build_corpus_from_manifest.py
reads. Rows are appended in the corpus-relative-subpath convention, e.g.
    fresh2026/fedreg_2026-06-15_2026-XXXXX.html
so a single --raw-dir root (scripts/raw) plus a per-corpus subdir stays
consistent for both ingest and long-context bundling.

Ambiguous rows (a field the fetcher/helper could not confidently fill) are
written as `# REVIEW(<reason>): <row>` comment lines: they are skipped by the
builder until a human promotes them to a normal row.
"""
from __future__ import annotations

import csv
import io
import re
import time
from pathlib import Path
from typing import Iterator, Optional

# Reuse the builder's extraction/estimation so every path produces identical text.
from build_corpus_from_manifest import (  # noqa: F401
    estimate_tokens,
    load_text,
    strip_html,
)

MANIFEST_COLUMNS = ["filename", "title", "source_url", "pub_date", "corpus"]

_REVIEW_RE = re.compile(r"^#\s*REVIEW\(([^)]*)\):\s*(.*)$")


def _csv_row(fields: list[str]) -> str:
    buf = io.StringIO()
    csv.writer(buf).writerow(fields)
    return buf.getvalue().rstrip("\r\n")


def iter_manifest_records(manifest_path: Path) -> Iterator[dict]:
    """Yield every manifest record as a dict, active or REVIEW-commented.

    Each dict has the MANIFEST_COLUMNS keys plus `_review` (the reason string, or
    None for active rows). Used for de-duplication so re-running a fetcher never
    appends the same file twice, whether or not it was flagged for review.
    """
    if not manifest_path.exists():
        return
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        m = _REVIEW_RE.match(s)
        if m:
            reason, row_text = m.group(1), m.group(2)
        elif s.startswith("#"):
            continue  # some other comment
        else:
            reason, row_text = None, s
        row = next(csv.reader([row_text]), None)
        if not row or row[0].strip() == "filename":
            continue
        rec = dict(zip(MANIFEST_COLUMNS, row + [""] * (len(MANIFEST_COLUMNS) - len(row))))
        rec["_review"] = reason
        yield rec


def manifest_filenames(manifest_path: Path) -> set[str]:
    """Set of `filename` values already present (active OR review-flagged)."""
    return {r["filename"].strip() for r in iter_manifest_records(manifest_path) if r.get("filename")}


def append_manifest_row(
    manifest_path: Path,
    *,
    filename: str,
    title: str,
    source_url: str,
    pub_date: str,
    corpus: str,
    review_reason: Optional[str] = None,
) -> None:
    """Append one row to the manifest, creating the header if the file is new.

    If review_reason is set, the row is written as a `# REVIEW(<reason>): <row>`
    comment so the builder skips it until a human fixes and un-comments it.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not manifest_path.exists()
    row_text = _csv_row([filename, title or "", source_url or "", pub_date or "", corpus])
    with manifest_path.open("a", encoding="utf-8", newline="") as f:
        if header_needed:
            f.write(",".join(MANIFEST_COLUMNS) + "\n")
        if review_reason:
            f.write(f"# REVIEW({review_reason}): {row_text}\n")
        else:
            f.write(row_text + "\n")


class RateLimiter:
    """Minimal monotonic-clock rate limiter (min seconds between calls)."""

    def __init__(self, min_interval: float):
        self._min = min_interval
        self._last = 0.0

    def wait(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self._min:
            time.sleep(self._min - dt)
        self._last = time.monotonic()


def polite_get(session, url: str, *, headers: dict, limiter: RateLimiter,
               params: Optional[dict] = None, timeout: float = 60.0):
    """A rate-limited GET that raises for HTTP errors."""
    limiter.wait()
    resp = session.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


def rel_to_root(path: Path, raw_root: Path) -> str:
    """Manifest `filename` value: the file path relative to the raw-dir root, so
    build_corpus_from_manifest.py resolves raw_root / filename correctly."""
    return str(path.resolve().relative_to(raw_root.resolve()))
