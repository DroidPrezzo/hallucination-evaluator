#!/usr/bin/env python3
"""fetch_federal_register.py

Download Federal Register documents in a date range via the public FR API
(no auth) and append manifest rows for the corpus builder.

    GET https://www.federalregister.gov/api/v1/documents.json
        ?conditions[publication_date][gte]=YYYY-MM-DD
        &conditions[publication_date][lte]=YYYY-MM-DD
        &conditions[type][]=RULE (etc.)
        &fields[]=document_number&fields[]=title&fields[]=publication_date
        &fields[]=html_url&fields[]=raw_text_url

Each document is saved to <out-dir> (prefer raw_text_url, else html_url), and a
manifest row is appended: filename (corpus-relative subpath), title,
source_url=html_url, pub_date=publication_date, corpus=fresh2026.

Resumable: documents whose output file already exists (or that are already in
the manifest) are skipped.

Usage:
    python fetch_federal_register.py \
        --start-date 2026-06-01 --end-date 2026-06-30 \
        --doc-types RULE PRORULE NOTICE \
        --max-docs 40 \
        --out-dir scripts/raw/fresh2026
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_fetch_common import (  # noqa: E402
    RateLimiter,
    append_manifest_row,
    manifest_filenames,
    polite_get,
    rel_to_root,
)

API = "https://www.federalregister.gov/api/v1/documents.json"
CORPUS = "fresh2026"
UA = "hallucination-evaluator/1.0 (corpus builder; contact via repo)"


def _params(start, end, doc_types, per_page, page):
    p = [
        ("conditions[publication_date][gte]", start),
        ("conditions[publication_date][lte]", end),
        ("per_page", str(per_page)),
        ("page", str(page)),
        ("order", "oldest"),
    ]
    for field in ("document_number", "title", "publication_date", "html_url", "raw_text_url"):
        p.append(("fields[]", field))
    for t in doc_types or []:
        p.append(("conditions[type][]", t))
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-date", required=True, help="ISO date, inclusive (YYYY-MM-DD)")
    ap.add_argument("--end-date", required=True, help="ISO date, inclusive (YYYY-MM-DD)")
    ap.add_argument("--doc-types", nargs="*", default=None,
                    help="FR types, e.g. RULE PRORULE NOTICE PRESDOCU (default: all)")
    ap.add_argument("--max-docs", type=int, default=40, help="Cap on documents downloaded")
    ap.add_argument("--out-dir", default="scripts/raw/fresh2026", help="Where to save files")
    ap.add_argument("--raw-root", default="scripts/raw", help="Raw-dir root for manifest paths")
    ap.add_argument("--manifest", default="scripts/raw/manifest.csv", help="Manifest CSV")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_root = Path(args.raw_root)
    manifest = Path(args.manifest)
    already = manifest_filenames(manifest)

    session = requests.Session()
    limiter = RateLimiter(0.2)  # be polite; FR has no documented hard limit
    headers = {"User-Agent": UA}

    added = skipped = 0
    page = 1
    while added < args.max_docs:
        resp = polite_get(session, API, headers=headers, limiter=limiter,
                          params=_params(args.start_date, args.end_date,
                                         args.doc_types, 100, page))
        results = resp.json().get("results", [])
        if not results:
            break
        for doc in results:
            if added >= args.max_docs:
                break
            docnum = doc.get("document_number") or "unknown"
            pub_date = doc.get("publication_date") or ""
            html_url = doc.get("html_url") or ""
            raw_url = doc.get("raw_text_url")
            src_url, ext = (raw_url, ".txt") if raw_url else (html_url, ".html")
            if not src_url:
                continue

            fname = f"fedreg_{pub_date}_{docnum}{ext}"
            fpath = out_dir / fname
            rel = rel_to_root(fpath, raw_root)
            if fpath.exists() or rel in already:
                skipped += 1
                continue

            try:
                body = polite_get(session, src_url, headers=headers, limiter=limiter).text
            except requests.RequestException as exc:
                print(f"WARN: failed {docnum}: {exc}", file=sys.stderr)
                continue
            fpath.write_text(body, encoding="utf-8")
            append_manifest_row(manifest, filename=rel, title=doc.get("title") or docnum,
                                source_url=html_url, pub_date=pub_date, corpus=CORPUS)
            already.add(rel)
            added += 1
            print(f"added: {rel}")
        page += 1

    print(f"\n{added} document(s) added, {skipped} already present.")


if __name__ == "__main__":
    main()
