# -*- coding: utf-8 -*-
"""
Fetch raw listing JSON for each URL in urls.csv and save as NDJSON.

Incremental by default:
  • Scans existing listings_raw.jsonl and SKIPS URLs already present.
  • Use --no-resume to ignore the existing file and refetch everything.

Usage:
  python step03_fetch_raw_listings.py
  python step03_fetch_raw_listings.py --no-resume
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import time
from typing import Dict, Iterable, List, Optional, Set

import requests

try:
    from hemnet import get_single_listing as hemnet_get_single_listing
except Exception:
    hemnet_get_single_listing = None

INPUT_URLS_DEFAULT = "urls.csv"
OUTPUT_JSONL_DEFAULT = "listings_raw.jsonl"
THROTTLE_SECONDS_DEFAULT = 0.6
REQUEST_TIMEOUT_DEFAULT = 25

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "sv-SE,sv;q=0.9,en-US;q=0.8,en;q=0.7",
})
NEXT_DATA_RE = re.compile(
    r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>\s*({.*?})\s*</script>',
    re.IGNORECASE | re.DOTALL,
)


def _now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def dprint(verbose: bool, *args, **kwargs) -> None:
    if verbose:
        print(*args, **kwargs)


def read_urls_csv(path: str) -> List[str]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["url"] for row in reader if row.get("url")]


def load_already_fetched(path: str, verbose: bool = False) -> Set[str]:
    seen: Set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    u = obj.get("url")
                    if u:
                        seen.add(u)
                except Exception:
                    continue
        dprint(verbose, f"[resume] loaded {len(seen)} URLs already in {path}")
    except FileNotFoundError:
        pass
    return seen


def _fallback_fetch_apollo_state(url: str, verbose: bool = False) -> Optional[Dict]:
    try:
        resp = SESSION.get(url, timeout=REQUEST_TIMEOUT_DEFAULT)
    except requests.RequestException as e:
        dprint(verbose, f"  [fallback] GET failed: {e}")
        return None
    dprint(verbose, f"  [fallback] GET {resp.url} → {resp.status_code}, {len(resp.text)} bytes")
    if resp.status_code != 200 or not resp.text:
        return None
    m = NEXT_DATA_RE.search(resp.text)
    if not m:
        dprint(verbose, "  [fallback] __NEXT_DATA__ not found")
        return None
    try:
        next_data = json.loads(m.group(1))
    except Exception as e:
        dprint(verbose, f"  [fallback] __NEXT_DATA__ parse failed: {e}")
        return None
    apollo = next_data.get("props", {}).get("pageProps", {}).get("__APOLLO_STATE__")
    return apollo if isinstance(apollo, dict) else None


def fetch_listing_apollo(url: str, verbose: bool = False) -> Optional[Dict]:
    if hemnet_get_single_listing is not None:
        try:
            res = hemnet_get_single_listing(url)
            if isinstance(res, dict):
                if "__APOLLO_STATE__" in res:
                    return res["__APOLLO_STATE__"]
                if "__APOLLO_STATE__" in res.get("pageProps", {}):
                    return res["pageProps"]["__APOLLO_STATE__"]
                if "__APOLLO_STATE__" in res.get("props", {}).get("pageProps", {}):
                    return res["props"]["pageProps"]["__APOLLO_STATE__"]
        except Exception as e:
            dprint(verbose, f"  [hemnet] get_single_listing raised {type(e).__name__}: {e}")
    return _fallback_fetch_apollo_state(url, verbose=verbose)


def write_jsonl(path: str, rows: Iterable[Dict], verbose: bool = False) -> int:
    count = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    dprint(verbose, f"[output] appended {count} rows → {path}")
    return count


def iter_raw_rows(
    urls: List[str],
    already: Set[str],
    throttle_s: float,
    verbose: bool,
    limit: Optional[int],
) -> Iterable[Dict]:
    written = 0
    for i, url in enumerate(urls, start=1):
        if url in already:
            dprint(verbose, f"[skip] already have {url}")
            continue
        dprint(verbose, f"\n[item {i}] fetching {url}")
        apollo = fetch_listing_apollo(url, verbose=verbose)
        if not isinstance(apollo, dict):
            dprint(verbose, "  [warn] no apollo_state extracted; skipping")
        else:
            yield {"url": url, "fetched_at": _now_utc_iso(), "apollo_state": apollo}
            written += 1
        if limit is not None and written >= limit:
            dprint(verbose, f"[stop] reached limit={limit}")
            break
        time.sleep(throttle_s)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch raw Hemnet listing JSON (Apollo state) to NDJSON.")
    p.add_argument("--input", default=INPUT_URLS_DEFAULT, help="Path to urls.csv")
    p.add_argument("--output", default=OUTPUT_JSONL_DEFAULT, help="Path to listings_raw.jsonl")
    p.add_argument("--throttle", type=float, default=THROTTLE_SECONDS_DEFAULT, help="Seconds between requests")
    p.add_argument("--limit", type=int, default=-1, help="Maximum NEW rows to fetch (-1 = unlimited)")
    p.add_argument("--no-resume", action="store_true", help="Ignore existing output; refetch everything")
    p.add_argument("--quiet", action="store_true", help="Reduce logging")
    args = p.parse_args()

    verbose = not args.quiet
    limit = None if args.limit is None or args.limit < 0 else int(args.limit)

    urls = read_urls_csv(args.input)
    already = set() if args.no_resume else load_already_fetched(args.output, verbose=verbose)
    rows_iter = iter_raw_rows(urls, already, throttle_s=args.throttle, verbose=verbose, limit=limit)
    write_jsonl(args.output, rows_iter, verbose=verbose)


if __name__ == "__main__":
    main()