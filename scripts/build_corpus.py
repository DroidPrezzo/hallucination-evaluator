#!/usr/bin/env python3
"""Ingest a folder of .txt/.md/.html documents into a corpus JSONL.

Each source file becomes one record with metadata (doc_id, title, source_url,
pub_date, sha256, token_count_estimate, contamination_risk) plus the extracted
text. Optionally packs documents into bundles for very-long-context tests.

Download the source documents yourself (e.g. GAO / SEC / Federal Register) into
a folder, then:

    python scripts/build_corpus.py \
        --input-dir ./downloads/fresh2026 \
        --name fresh2026 \
        --contamination-risk fresh2026

    # very-long-context bundles (~256k tokens each):
    python scripts/build_corpus.py \
        --input-dir ./downloads/fresh2026 \
        --name fresh2026_bundles \
        --contamination-risk fresh2026 \
        --bundle-target-tokens 256000

Optional --metadata is a JSON file keyed by source filename, e.g.:
    {"gao-25-107085.html": {"title": "...", "source_url": "https://...",
                            "pub_date": "2026-03-15", "doc_id": "gao-25-107085"}}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

_TEXT_SUFFIXES = {".txt"}
_MD_SUFFIXES = {".md", ".markdown"}
_HTML_SUFFIXES = {".html", ".htm"}

# Tags whose text content is not document body text.
_SKIP_TAGS = {"script", "style", "noscript"}
# Block-level tags that should introduce a line break in the extracted text.
_BLOCK_TAGS = {
    "p", "br", "div", "li", "tr", "section", "article", "blockquote", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "table", "header", "footer",
}


def estimate_tokens(text: str) -> int:
    """Model-agnostic token estimate. Uses cl100k_base if tiktoken is present,
    otherwise a ~4 chars/token heuristic. This is only a planning estimate;
    the pipeline records exact per-model counts at run time."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


class _HTMLToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        self.text_parts.append(data)


def _clean_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    collapsed = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", collapsed).strip()


def extract_document(path: Path) -> tuple[str, Optional[str]]:
    """Return (text, auto_title) for a source file."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()

    if suffix in _HTML_SUFFIXES:
        parser = _HTMLToText()
        parser.feed(raw)
        parser.close()
        text = _clean_whitespace("".join(parser.text_parts))
        title = _clean_whitespace("".join(parser.title_parts)) or None
        return text, title

    if suffix in _MD_SUFFIXES:
        m = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
        return raw.strip(), (m.group(1).strip() if m else None)

    return raw.strip(), None


def _sanitize_id(stem: str) -> str:
    cleaned = re.sub(r"[^a-z0-9._-]+", "-", stem.lower()).strip("-._")
    return cleaned or "doc"


def _unique_id(base: str, seen: set[str]) -> str:
    doc_id = base
    i = 1
    while doc_id in seen:
        i += 1
        doc_id = f"{base}-{i}"
    seen.add(doc_id)
    return doc_id


def build_doc_record(path: Path, metadata: dict, contamination_risk: str, seen: set[str]) -> Optional[dict]:
    text, auto_title = extract_document(path)
    if not text.strip():
        print(f"  skipping empty document: {path.name}", file=sys.stderr)
        return None

    meta = metadata.get(path.name, {})
    doc_id = _unique_id(meta.get("doc_id") or _sanitize_id(path.stem), seen)
    return {
        "doc_id": doc_id,
        "title": meta.get("title") or auto_title or path.stem,
        "source_url": meta.get("source_url"),
        "pub_date": meta.get("pub_date"),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_count_estimate": estimate_tokens(text),
        "contamination_risk": contamination_risk,
        "is_bundle": False,
        "text": text,
    }


def _make_bundle(members: list[dict], idx: int, name: str, contamination_risk: str) -> dict:
    parts = []
    for i, d in enumerate(members, 1):
        parts.append(f"===== DOCUMENT {i}: {d['title']} [{d['doc_id']}] =====\n{d['text']}")
    text = "\n\n".join(parts)
    return {
        "doc_id": f"{name}-bundle-{idx:04d}",
        "title": f"Bundle of {len(members)} documents",
        "source_url": None,
        "pub_date": None,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_count_estimate": estimate_tokens(text),
        "contamination_risk": contamination_risk,
        "is_bundle": True,
        "bundle": [
            {
                "doc_id": d["doc_id"],
                "title": d["title"],
                "token_count_estimate": d["token_count_estimate"],
            }
            for d in members
        ],
        "text": text,
    }


def build_bundles(docs: list[dict], target_tokens: int, name: str, contamination_risk: str) -> list[dict]:
    bundles: list[dict] = []
    current: list[dict] = []
    current_tokens = 0
    for d in docs:
        if current and current_tokens + d["token_count_estimate"] > target_tokens:
            bundles.append(_make_bundle(current, len(bundles), name, contamination_risk))
            current, current_tokens = [], 0
        current.append(d)
        current_tokens += d["token_count_estimate"]
    if current:
        bundles.append(_make_bundle(current, len(bundles), name, contamination_risk))
    return bundles


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", required=True, help="Folder of .txt/.md/.html files")
    parser.add_argument("--name", required=True, help="Corpus name (output file stem)")
    parser.add_argument("--contamination-risk", default="fresh2026",
                        help="fresh2026 (post-cutoff) or public_legacy (default: fresh2026)")
    parser.add_argument("--output-dir", default="corpus", help="Output directory")
    parser.add_argument("--metadata", default=None,
                        help="Optional JSON file of per-filename metadata overrides")
    parser.add_argument("--bundle-target-tokens", type=int, default=None,
                        help="If set, pack documents into bundles up to this token count")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")

    metadata: dict = {}
    if args.metadata:
        metadata = json.loads(Path(args.metadata).read_text())

    suffixes = _TEXT_SUFFIXES | _MD_SUFFIXES | _HTML_SUFFIXES
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in suffixes)
    if not files:
        sys.exit(f"No .txt/.md/.html files found in {input_dir}")

    seen_ids: set[str] = set()
    docs: list[dict] = []
    for path in files:
        record = build_doc_record(path, metadata, args.contamination_risk, seen_ids)
        if record is not None:
            docs.append(record)

    if not docs:
        sys.exit("No non-empty documents ingested.")

    if args.bundle_target_tokens:
        records = build_bundles(docs, args.bundle_target_tokens, args.name, args.contamination_risk)
        kind = f"{len(records)} bundle(s) from {len(docs)} documents"
    else:
        records = docs
        kind = f"{len(records)} document(s)"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.name}.jsonl"
    with open(out_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_tokens = sum(r["token_count_estimate"] for r in records)
    print(f"Wrote {kind} to {out_path}")
    print(f"contamination_risk={args.contamination_risk}  "
          f"total_est_tokens={total_tokens:,}  "
          f"mean_est_tokens={total_tokens // len(records):,}")


if __name__ == "__main__":
    main()
