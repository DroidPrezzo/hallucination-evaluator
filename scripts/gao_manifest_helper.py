#!/usr/bin/env python3
"""gao_manifest_helper.py

GAO has no public download API, so GAO report pages are saved manually (e.g.
"Save Page As" from gao.gov) into a folder. This helper scans that folder for
report HTML files not yet in the manifest and, for each, fills what it can:

  - title      : <title>, falling back to og:title
  - source_url : og:url, falling back to a canonical URL built from the
                 GAO-YY-NNNNNN product number in the filename/page
  - pub_date   : a "Published:"/"Publicly Released:" date on the page (ISO-normalized)

Confident rows are appended normally; anything ambiguous (missing/unparseable
field) is appended as a `# REVIEW(<reason>): <row>` line so you can spot and fix
it fast instead of blocking. Idempotent: files already in the manifest (active
or review-flagged) are skipped.

Usage:
    python gao_manifest_helper.py \
        --dir scripts/raw/fresh2026 \
        --raw-root scripts/raw \
        --manifest scripts/raw/manifest.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_fetch_common import (  # noqa: E402
    append_manifest_row,
    manifest_filenames,
    rel_to_root,
)

CORPUS = "fresh2026"

_GAO_NUM_RE = re.compile(r"GAO-\d{2}-\d{4,6}", re.IGNORECASE)
_META_RE = re.compile(
    r'<meta[^>]+(?:property|name)=["\'](?P<key>og:title|og:url)["\'][^>]*'
    r'content=["\'](?P<val>[^"\']*)["\']',
    re.IGNORECASE,
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
# GAO pages render e.g. "Published: Jun 18, 2026" / "Publicly Released: Jul 01, 2026".
_DATE_RE = re.compile(
    r"(?:Published|Publicly Released)\s*:?\s*"
    r"([A-Z][a-z]{2,9}\.?\s+\d{1,2},\s+\d{4})",
    re.IGNORECASE,
)
_DATE_FORMATS = ("%b %d, %Y", "%B %d, %Y", "%b. %d, %Y")


def _meta(html: str) -> dict:
    out = {}
    for m in _META_RE.finditer(html):
        out.setdefault(m.group("key").lower(), m.group("val").strip())
    return out


def _title(html: str, meta: dict) -> Optional[str]:
    t = _TITLE_RE.search(html)
    title = (t.group(1) if t else "").strip()
    title = re.sub(r"\s+", " ", title)
    return title or meta.get("og:title")


def _pub_date(html: str) -> Optional[str]:
    m = _DATE_RE.search(html)
    if not m:
        return None
    raw = re.sub(r"\s+", " ", m.group(1).strip())
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _source_url(filename: str, html: str, meta: dict) -> Optional[str]:
    if meta.get("og:url"):
        return meta["og:url"]
    m = _GAO_NUM_RE.search(filename) or _GAO_NUM_RE.search(html)
    if m:
        return f"https://www.gao.gov/products/{m.group(0).upper()}"
    return None


def _iter_report_html(scan_dir: Path):
    """Top-level .html/.htm report files only; skip the '*_files/' asset dirs
    that 'Save Page As' produces."""
    for p in sorted(scan_dir.rglob("*.htm*")):
        if any(part.endswith("_files") for part in p.parts):
            continue
        if p.suffix.lower() in (".html", ".htm"):
            yield p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="scripts/raw/fresh2026", help="Folder of saved GAO report HTML")
    ap.add_argument("--raw-root", default="scripts/raw", help="Raw-dir root the manifest filenames are relative to")
    ap.add_argument("--manifest", default="scripts/raw/manifest.csv", help="Manifest CSV to append to")
    args = ap.parse_args()

    scan_dir = Path(args.dir)
    raw_root = Path(args.raw_root)
    manifest = Path(args.manifest)
    if not scan_dir.is_dir():
        sys.exit(f"Scan dir not found: {scan_dir}")

    already = manifest_filenames(manifest)
    added = review = skipped = 0

    for path in _iter_report_html(scan_dir):
        rel = rel_to_root(path, raw_root)
        if rel in already:
            skipped += 1
            continue

        html = path.read_text(encoding="utf-8", errors="replace")
        meta = _meta(html)
        title = _title(html, meta)
        source_url = _source_url(path.name, html, meta)
        pub_date = _pub_date(html)

        missing = [n for n, v in (("title", title), ("source_url", source_url),
                                  ("pub_date", pub_date)) if not v]
        reason = ("could not fill " + ", ".join(missing)) if missing else None

        append_manifest_row(
            manifest, filename=rel, title=title or "", source_url=source_url or "",
            pub_date=pub_date or "", corpus=CORPUS, review_reason=reason,
        )
        if reason:
            review += 1
            print(f"REVIEW ({reason}): {path.name}", file=sys.stderr)
        else:
            added += 1
            print(f"added: {rel}")

    print(f"\n{added} row(s) added, {review} flagged for review, {skipped} already present.")
    if review:
        print("Fix the '# REVIEW(...)' lines in the manifest, then remove the '# REVIEW(...): ' "
              "prefix to activate them.", file=sys.stderr)


if __name__ == "__main__":
    main()
