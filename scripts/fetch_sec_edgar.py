#!/usr/bin/env python3
"""fetch_sec_edgar.py

Download SEC EDGAR filings in a date range via the full-text search API and
append manifest rows. 10-K/10-Q filings are long (often >128k tokens), so this
is the source that reaches the hard end of the context-length grid.

    GET https://efts.sec.gov/LATEST/search-index
        ?q=&forms=10-K&startdt=YYYY-MM-DD&enddt=YYYY-MM-DD&from=<offset>

SEC REQUIRES a descriptive User-Agent with a real contact (name + email), or it
returns 403 and may temporarily block your IP. Set it via the environment:

    export SEC_EDGAR_USER_AGENT="Your Name your.email@example.com"

Stays under SEC's 10 req/s guidance. Each hit is resolved to its primary
document under www.sec.gov/Archives/edgar/data/... and downloaded.

Resumable: filings whose output file already exists (or are already in the
manifest) are skipped.

Usage:
    export SEC_EDGAR_USER_AGENT="Jane Doe jane@example.com"
    python fetch_sec_edgar.py \
        --start-date 2026-05-01 --end-date 2026-06-30 \
        --forms 10-K 10-Q \
        --max-docs 20 \
        --out-dir scripts/raw/fresh2026
"""
from __future__ import annotations

import argparse
import os
import re
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

FTS_API = "https://efts.sec.gov/LATEST/search-index"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
CORPUS = "fresh2026"


def _document_url(hit: dict) -> tuple[str, str, str]:
    """Resolve a full-text-search hit to (url, accession, filename).

    hit["_id"] is "<accession-with-dashes>:<filename>"; _source["ciks"][0] gives
    the CIK. The archive path uses the CIK without leading zeros and the
    accession number with dashes stripped.
    """
    _id = hit["_id"]
    accession, _, filename = _id.partition(":")
    source = hit.get("_source", {})
    ciks = source.get("ciks") or []
    if not ciks or not filename:
        raise ValueError(f"unexpected hit shape: {_id}")
    cik = int(str(ciks[0]).lstrip("0") or "0")
    acc_nodash = accession.replace("-", "")
    return f"{ARCHIVES}/{cik}/{acc_nodash}/{filename}", accession, filename


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-date", required=True, help="ISO date, inclusive (YYYY-MM-DD)")
    ap.add_argument("--end-date", required=True, help="ISO date, inclusive (YYYY-MM-DD)")
    ap.add_argument("--forms", nargs="*", default=["10-K"], help="Form types (default: 10-K)")
    ap.add_argument("--max-docs", type=int, default=20, help="Cap on filings downloaded")
    ap.add_argument("--out-dir", default="scripts/raw/fresh2026", help="Where to save files")
    ap.add_argument("--raw-root", default="scripts/raw", help="Raw-dir root for manifest paths")
    ap.add_argument("--manifest", default="scripts/raw/manifest.csv", help="Manifest CSV")
    ap.add_argument("--corpus", default=CORPUS,
                    help="Corpus label = output jsonl name + contamination_risk "
                         "(e.g. fresh2026_pilot for the long-doc pilot corpus)")
    args = ap.parse_args()

    ua = os.environ.get("SEC_EDGAR_USER_AGENT")
    if not ua:
        sys.exit("SEC_EDGAR_USER_AGENT is not set. Export a real 'Name email' or SEC returns 403.\n"
                 '  export SEC_EDGAR_USER_AGENT="Jane Doe jane@example.com"')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_root = Path(args.raw_root)
    manifest = Path(args.manifest)
    already = manifest_filenames(manifest)

    session = requests.Session()
    limiter = RateLimiter(0.2)  # ~5 req/s, well under SEC's 10 req/s guidance
    headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}

    added = skipped = 0
    offset = 0
    while added < args.max_docs:
        params = {
            "q": "",
            "forms": ",".join(args.forms),
            "startdt": args.start_date,
            "enddt": args.end_date,
            "from": offset,
        }
        resp = polite_get(session, FTS_API, headers=headers, limiter=limiter, params=params)
        hits = resp.json().get("hits", {}).get("hits", [])
        if not hits:
            break
        for hit in hits:
            if added >= args.max_docs:
                break
            try:
                url, accession, filename = _document_url(hit)
            except (KeyError, ValueError) as exc:
                print(f"WARN: skipping malformed hit: {exc}", file=sys.stderr)
                continue

            source = hit.get("_source", {})
            pub_date = source.get("file_date") or ""
            names = source.get("display_names") or []
            title = f"{names[0] if names else 'SEC filing'} — {accession}"
            safe_fn = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
            fname = f"sec_{pub_date}_{accession}_{safe_fn}"
            if not fname.lower().endswith((".htm", ".html", ".txt")):
                fname += ".html"
            fpath = out_dir / fname
            rel = rel_to_root(fpath, raw_root)
            if fpath.exists() or rel in already:
                skipped += 1
                continue

            try:
                body = polite_get(session, url, headers=headers, limiter=limiter).text
            except requests.RequestException as exc:
                print(f"WARN: failed {accession}: {exc}", file=sys.stderr)
                continue
            fpath.write_text(body, encoding="utf-8")
            append_manifest_row(manifest, filename=rel, title=title,
                                source_url=url, pub_date=pub_date, corpus=args.corpus)
            already.add(rel)
            added += 1
            print(f"added: {rel}")
        offset += len(hits)

    print(f"\n{added} filing(s) added, {skipped} already present.")


if __name__ == "__main__":
    main()
