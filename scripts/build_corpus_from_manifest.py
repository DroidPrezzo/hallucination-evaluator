#!/usr/bin/env python3
"""
build_corpus_from_manifest.py

Turns a folder of downloaded documents (HTML/TXT/MD) plus a manifest.csv
into corpus/{corpus_name}.jsonl files for the hallucination-evaluator harness.

Each output record:
    doc_id, title, source_url, pub_date, sha256, token_count_estimate,
    contamination_risk, char_count, text

Usage:
    python build_corpus_from_manifest.py \
        --raw-dir scripts/raw \
        --manifest scripts/raw/manifest.csv \
        --out-dir corpus

Manifest columns (required): filename, title, source_url, pub_date, corpus
  - filename   : path of the file relative to --raw-dir (e.g. fresh2026/fedreg_0001.html)
  - title      : document title
  - source_url : canonical URL you downloaded it from
  - pub_date   : ISO date, e.g. 2026-06-15
  - corpus     : a short label -- this becomes the output filename AND the
                 contamination_risk value. Use "fresh2026" for post-cutoff
                 documents and "public_legacy" for tau/scrolls-era public data.

doc_ids are namespaced "mf:{corpus}-{NNNN}" so they can never collide with the
folder-based scripts/build_corpus.py ids (which are colon-free) or scrolls ids.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    HAVE_BS4 = True
except ImportError:
    HAVE_BS4 = False

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    HAVE_TIKTOKEN = True
except ImportError:
    HAVE_TIKTOKEN = False

REQUIRED_COLUMNS = {"filename", "title", "source_url", "pub_date", "corpus"}


def strip_html(raw: str) -> str:
    """Basic HTML -> text. Uses bs4 if available, else a crude regex fallback."""
    if HAVE_BS4:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    else:
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        import html as html_module
        text = html_module.unescape(text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def load_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in (".html", ".htm"):
        return strip_html(raw)
    return raw.strip()


def estimate_tokens(text: str) -> int:
    """Rough estimate only -- the harness recounts with each model's own
    tokenizer before truncation (see Phase 3 of the playbook)."""
    if HAVE_TIKTOKEN:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="scripts/raw", help="Folder containing downloaded files")
    ap.add_argument("--manifest", default="scripts/raw/manifest.csv", help="CSV manifest path")
    ap.add_argument("--out-dir", default="corpus", help="Output folder for corpus/{name}.jsonl")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}")

    if not HAVE_BS4:
        print("WARNING: bs4 not installed -- using a crude regex HTML stripper.\n"
              "         Run: pip install beautifulsoup4 --break-system-packages", file=sys.stderr)
    if not HAVE_TIKTOKEN:
        print("NOTE: tiktoken not installed -- token_count_estimate uses a chars/4 heuristic.\n"
              "      This is only a planning estimate; each model backend recounts tokens\n"
              "      with its own tokenizer before truncation. For a closer estimate, run:\n"
              "      pip install tiktoken --break-system-packages", file=sys.stderr)

    by_corpus = defaultdict(list)
    seen_hashes = {}
    counts = defaultdict(int)
    skipped = 0

    with manifest_path.open(newline="", encoding="utf-8") as f:
        # Skip "# REVIEW(...)" comment rows (and any other #-comment) so the
        # fetchers/GAO helper can park ambiguous rows for human review without
        # them entering the corpus.
        rows = (ln for ln in f if not ln.lstrip().startswith("#"))
        reader = csv.DictReader(rows)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"Manifest is missing columns: {sorted(missing)}")

        for row in reader:
            fname = row["filename"].strip()
            fpath = raw_dir / fname
            if not fpath.exists():
                print(f"SKIP: '{fname}' listed in manifest but not found in {raw_dir}/", file=sys.stderr)
                skipped += 1
                continue

            text = load_text(fpath)
            if len(text) < 200:
                print(f"WARNING: '{fname}' extracted to only {len(text)} chars -- "
                      f"check for a failed download or bad HTML extraction", file=sys.stderr)

            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if sha in seen_hashes:
                print(f"SKIP: '{fname}' is a duplicate of '{seen_hashes[sha]}' (identical sha256)", file=sys.stderr)
                skipped += 1
                continue
            seen_hashes[sha] = fname

            corpus_name = row["corpus"].strip()
            counts[corpus_name] += 1
            doc_id = f"mf:{corpus_name}-{counts[corpus_name]:04d}"

            record = {
                "doc_id": doc_id,
                "title": row["title"].strip(),
                "source_url": row["source_url"].strip(),
                "pub_date": row["pub_date"].strip(),
                "sha256": sha,
                "token_count_estimate": estimate_tokens(text),
                "contamination_risk": corpus_name,
                "char_count": len(text),
                "text": text,
            }
            by_corpus[corpus_name].append(record)

    for corpus_name, records in by_corpus.items():
        out_path = out_dir / f"{corpus_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        total_tokens = sum(r["token_count_estimate"] for r in records)
        print(f"{out_path}: {len(records)} docs, ~{total_tokens:,} estimated tokens")

    if skipped:
        print(f"\n{skipped} row(s) skipped -- see warnings above.", file=sys.stderr)


if __name__ == "__main__":
    main()
