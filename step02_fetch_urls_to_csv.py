"""
Fetch Hemnet listing URLs for locations in locations.csv and write urls.csv.

Incremental features:
  • --append (default): read existing urls.csv and skip URLs already present
  • --stop-when-old N (default 3): if N consecutive pages yield 0 NEW URLs for a
    location, stop paging that location (useful for daily runs)

Other features (from before):
  • hemnet.get_urls() with fallback JSON-LD parser
  • GLOBAL de-duplication by URL across locations (first seen wins)
  • Verbose logging by default

Usage:
  python step02_fetch_urls_to_csv.py --input locations.csv --output urls.csv
  python step02_fetch_urls_to_csv.py --no-append                  # overwrite, full run
  python step02_fetch_urls_to_csv.py --stop-when-old 2            # stop earlier
  python step02_fetch_urls_to_csv.py --allow-duplicates           # disable global dedupe
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

import requests
from hemnet import get_urls as hemnet_get_urls

# -------- Defaults --------
INPUT_CSV_DEFAULT = "locations.csv"
OUTPUT_CSV_DEFAULT = "urls.csv"
REQUEST_THROTTLE_SECONDS_DEFAULT = 0.8
MAX_EMPTY_PAGES_DEFAULT = 2          # hard stop when even the site is empty
MAX_PAGES_DEFAULT: Optional[int] = None
STOP_WHEN_OLD_DEFAULT = 3            # early-stop after N pages with 0 NEW URLs (incremental)

TYPE_PRIORITY = {
    "DISTRICT": 0,
    "NEIGHBORHOOD": 0,
    "CITY": 1,
    "MUNICIPALITY": 1,
    "COUNTY": 2,
    "REGION": 2,
}

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "sv-SE,sv;q=0.9,en-US;q=0.8,en;q=0.7",
})
LD_JSON_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)

OUT_FIELDS = [
    "first_seen_at",        # NEW: timestamp when this URL first appeared in your file
    "location_term",
    "location_name",
    "location_type",
    "location_id",
    "url",
    "page",
]


# ---------------- Utility / Debug ----------------
def dprint(verbose: bool, *args, **kwargs) -> None:
    if verbose:
        print(*args, **kwargs)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------- Fallback scraping ----------------
def _extract_listing_urls_from_html(html: str, verbose: bool = False) -> List[str]:
    urls: List[str] = []
    blocks = list(LD_JSON_RE.finditer(html))
    dprint(verbose, f"    [fallback] Found {len(blocks)} <script type='application/ld+json'> blocks")
    for idx, m in enumerate(blocks, start=1):
        block = m.group(1).strip()
        try:
            data = json.loads(block)
        except Exception as e:
            dprint(verbose, f"    [fallback] JSON-LD block {idx}: failed to parse ({e})")
            continue
        objs = data if isinstance(data, list) else [data]
        for obj in objs:
            if not isinstance(obj, dict) or obj.get("@type") != "ItemList":
                continue
            elements = obj.get("itemListElement") or []
            dprint(verbose, f"    [fallback] ItemList with {len(elements)} elements")
            for el in elements:
                if isinstance(el, dict) and "url" in el:
                    urls.append(el["url"])
                elif isinstance(el, dict) and isinstance(el.get("item"), dict) and "url" in el["item"]:
                    urls.append(el["item"]["url"])
    # stable de-dup in page
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _fallback_fetch_page_urls(location_id: str, page: int, verbose: bool = False) -> List[str]:
    params = [("location_ids[]", location_id), ("page", str(page))]
    url = "https://www.hemnet.se/bostader"
    try:
        resp = SESSION.get(url, params=params, timeout=20)
    except requests.RequestException as e:
        dprint(verbose, f"    [fallback] GET {url} failed: {e}")
        return []
    dprint(verbose, f"    [fallback] GET {resp.url} → {resp.status_code}, {len(resp.text)} bytes")
    if resp.status_code != 200:
        return []
    return _extract_listing_urls_from_html(resp.text, verbose=verbose)


# ---------------- CSV I/O ----------------
def read_locations_csv(path: str, verbose: bool = False) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dprint(verbose, f"[input] Loaded {len(rows)} locations from {path}")
    if verbose:
        for r in rows:
            dprint(True, f"  - {r.get('term')} | id={r.get('id')} | name={r.get('name')} | type={r.get('type')}")
    return rows


def load_existing_urls(path: str, verbose: bool = False) -> Set[str]:
    """Return set of URLs already present in an existing urls.csv (any schema, only 'url' column used)."""
    urls: Set[str] = set()
    if not os.path.exists(path):
        return urls
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                u = row.get("url")
                if u:
                    urls.add(u)
        dprint(verbose, f"[resume] Found {len(urls)} existing URLs in {path}")
    except Exception as e:
        dprint(verbose, f"[resume] Could not read existing URLs from {path}: {e}")
    return urls


def write_urls_csv(rows: Iterable[Dict], path: str, append: bool, verbose: bool = False) -> int:
    mode = "a" if append and os.path.exists(path) else "w"
    need_header = (mode == "w") or not os.path.exists(path) or os.path.getsize(path) == 0
    count = 0
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        if need_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    dprint(verbose, f"[output] {'Appended' if mode=='a' else 'Wrote'} {count} rows → {path}")
    return count


# ---------------- Orchestration ----------------
def _try_hemnet_get_urls(location_id: str, page: int, verbose: bool = False) -> List[str]:
    try:
        urls = hemnet_get_urls(location_id, page, live=False) or []
        dprint(verbose, f"    [hemnet] get_urls(id={location_id}, page={page}) → {len(urls)} urls")
        return urls
    except Exception as e:
        dprint(verbose, f"    [hemnet] get_urls raised {type(e).__name__}: {e}")
        return []


def iter_location_urls(
    location: Dict,
    global_seen: Set[str],
    verbose: bool = True,
    max_empty_pages: int = MAX_EMPTY_PAGES_DEFAULT,
    throttle_s: float = REQUEST_THROTTLE_SECONDS_DEFAULT,
    max_pages: Optional[int] = MAX_PAGES_DEFAULT,
    stop_when_old: int = STOP_WHEN_OLD_DEFAULT,
) -> Iterator[Tuple[str, Dict, bool]]:
    """
    Yield (url, row, is_new) for one location.
    Early-stop when we hit 'stop_when_old' consecutive pages with 0 NEW URLs.
    """
    lid = location["id"]
    lname = location.get("name", "")
    ltype = location["type"]
    lterm = location["term"]

    dprint(verbose, f"\n[location] {lterm} | name={lname} | type={ltype} | id={lid}")

    backend = "hemnet"
    seen_here: Set[str] = set()
    empty_streak = 0
    old_streak = 0
    total_urls = 0
    page = 1

    # Decide backend on page 1
    urls = _try_hemnet_get_urls(lid, page, verbose=verbose)
    if not urls:
        dprint(verbose, "  [switch] hemnet.get_urls yielded 0 on page 1 → switching to fallback parser")
        backend = "fallback"
        urls = _fallback_fetch_page_urls(lid, page, verbose=verbose)

    while True:
        new_this_page = 0

        if not urls:
            empty_streak += 1
            dprint(verbose, f"  [page {page}] 0 urls (empty_streak={empty_streak})")
            if empty_streak >= max_empty_pages:
                dprint(verbose, f"  [stop] Reached MAX_EMPTY_PAGES={max_empty_pages}")
                break
        else:
            empty_streak = 0
            dprint(verbose, f"  [page {page}] {len(urls)} urls via {backend}")

            for u in urls:
                if u in seen_here:
                    continue
                seen_here.add(u)
                is_new_global = u not in global_seen
                if is_new_global:
                    new_this_page += 1
                total_urls += 1
                row = {
                    "first_seen_at": utcnow_iso() if is_new_global else "",
                    "location_term": lterm,
                    "location_name": lname,
                    "location_type": ltype,
                    "location_id": lid,
                    "url": u,
                    "page": page,
                }
                yield u, row, is_new_global

            dprint(verbose, f"    [delta] new_in_global={new_this_page}, already_had={len(urls)-new_this_page}")

        # early-stop on old pages with no new URLs
        if stop_when_old is not None and stop_when_old >= 0:
            if new_this_page == 0:
                old_streak += 1
                if old_streak >= stop_when_old:
                    dprint(verbose, f"  [stop] Early-stop: {old_streak} pages in a row had 0 NEW URLs")
                    break
            else:
                old_streak = 0

        page += 1
        if max_pages is not None and page > max_pages:
            dprint(verbose, f"  [stop] Reached MAX_PAGES={max_pages}")
            break

        time.sleep(throttle_s)

        if backend == "fallback":
            urls = _fallback_fetch_page_urls(lid, page, verbose=verbose)
        else:
            urls = _try_hemnet_get_urls(lid, page, verbose=verbose)

    dprint(verbose, f"[summary] {lterm} ({lid}) → pages_old_streak_max={old_streak}, total_seen_here={total_urls}")


def _location_priority(row: Dict) -> int:
    return TYPE_PRIORITY.get(row.get("type", ""), 99)


def build_url_rows(
    locations: List[Dict],
    existing_global: Set[str],
    unique_by_url: bool = True,
    verbose: bool = True,
    max_empty_pages: int = MAX_EMPTY_PAGES_DEFAULT,
    throttle_s: float = REQUEST_THROTTLE_SECONDS_DEFAULT,
    max_pages: Optional[int] = MAX_PAGES_DEFAULT,
    stop_when_old: int = STOP_WHEN_OLD_DEFAULT,
    stats: Optional[Dict[str, int]] = None,
) -> Iterable[Dict]:
    """
    Yield rows for all locations. Respects an existing global set (from previous runs)
    to avoid re-emitting known URLs. If unique_by_url, also de-duplicates within this run.
    """
    locations = sorted(locations, key=_location_priority)

    global_seen = set(existing_global)  # copy so we can add during this run
    if stats is None:
        stats = {}
    stats.setdefault("skipped_global_duplicates", 0)
    stats.setdefault("already_in_file", len(existing_global))

    for loc in locations:
        for url, row, is_new in iter_location_urls(
            loc,
            global_seen,
            verbose=verbose,
            max_empty_pages=max_empty_pages,
            throttle_s=throttle_s,
            max_pages=max_pages,
            stop_when_old=stop_when_old,
        ):
            if unique_by_url:
                if url in global_seen:
                    # If it’s “new in this location” but already known globally (from prior rows this run),
                    # treat as duplicate.
                    if is_new:
                        # should not happen because we add immediately; still guard
                        stats["skipped_global_duplicates"] += 1
                    # If it’s from existing file, we simply don’t yield it.
                    continue
                global_seen.add(url)
            else:
                # Even with duplicates allowed, avoid emitting rows that existed from a previous file when appending
                if url in existing_global:
                    continue
            yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Hemnet listing URLs to urls.csv (incremental).")
    parser.add_argument("--input", default=INPUT_CSV_DEFAULT, help="Path to locations.csv")
    parser.add_argument("--output", default=OUTPUT_CSV_DEFAULT, help="Path to write urls.csv")
    parser.add_argument("--allow-duplicates", action="store_true",
                        help="Allow the same URL to appear under multiple locations (disables global de-dup for this run)")
    parser.add_argument("--no-append", dest="append", action="store_false",
                        help="Do not append; overwrite output file")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging")
    parser.add_argument("--throttle", type=float, default=REQUEST_THROTTLE_SECONDS_DEFAULT,
                        help="Seconds to sleep between page requests")
    parser.add_argument("--max-empty-pages", type=int, default=MAX_EMPTY_PAGES_DEFAULT,
                        help="Stop after this many consecutive empty pages for a location")
    parser.add_argument("--max-pages", type=int, default=-1,
                        help="Cap number of pages per location (-1 = unlimited)")
    parser.add_argument("--stop-when-old", type=int, default=STOP_WHEN_OLD_DEFAULT,
                        help="Early-stop a location after N pages in a row with 0 NEW URLs (set 0 to disable)")
    args = parser.parse_args()

    verbose = not args.quiet
    unique_by_url = not args.allow_duplicates
    max_pages = None if args.max_pages is None or args.max_pages < 0 else int(args.max_pages)
    append = args.append

    locations = read_locations_csv(args.input, verbose=verbose)

    # read existing urls.csv so we only emit brand-new URLs when appending
    existing_global = load_existing_urls(args.output, verbose=verbose) if append else set()

    stats: Dict[str, int] = {}
    rows_iter = build_url_rows(
        locations,
        existing_global=existing_global,
        unique_by_url=unique_by_url,
        verbose=verbose,
        max_empty_pages=args.max_empty_pages,
        throttle_s=args.throttle,
        max_pages=max_pages,
        stop_when_old=max(0, args.stop_when_old),
        stats=stats,
    )
    written = write_urls_csv(rows_iter, args.output, append=append, verbose=verbose)

    if verbose:
        if append:
            dprint(True, f"[resume] already in file at start: {stats.get('already_in_file', 0)} URLs")
        if unique_by_url:
            dprint(True, f"[dedupe] skipped {stats.get('skipped_global_duplicates', 0)} duplicate URL rows within this run")
        dprint(True, f"[done] wrote {written} NEW rows → {args.output}")
    else:
        print(f"Wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()