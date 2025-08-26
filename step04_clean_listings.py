# -*- coding: utf-8 -*-
"""
Clean/flatten Hemnet listing JSON (Apollo state) into a tidy table.

Adds floor parsing with Swedish heuristics:
  • "N tr" (trappor upp) → floor_number = N + 1  (e.g., "1 tr" → 2)
  • Ground markers ("BV", "Bottenvåning", "Entréplan", "Markplan") → floor_number = 0
  • Supports "X av Y", "vån/våning X", "plan X", "en trappa upp"

Input: listings_raw.jsonl from step03
Output:
  • listings_clean.csv
  • listings_clean.parquet (if pyarrow is available)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd

# ------------- Defaults -------------
INPUT_JSONL_DEFAULT = "listings_raw.jsonl"
OUTPUT_CSV_DEFAULT = "listings_clean.csv"
OUTPUT_PARQUET_DEFAULT = "listings_clean.parquet"

# ------------- Utils -------------
TRAILING_ID_RE = re.compile(r"(\d+)(?:/)?$")


def dget(obj: Dict, *path, default=None):
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def parse_listing_id_from_url(url: str) -> Optional[str]:
    m = TRAILING_ID_RE.search(url or "")
    return m.group(1) if m else None


def money_amount(val) -> Optional[float]:
    """Normalize Hemnet Money objects (or plain numbers) to float."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        amt = val.get("amount")
        if isinstance(amt, (int, float)):
            return float(amt)
    return None


# ------------- Floor parsing helpers -------------
FLOOR_BOTTLABELS = {"bottenvåning", "bottenplan", "entréplan", "entreplan", "markplan", "bv"}


def parse_floor_from_label(label: str) -> Tuple[Optional[int], Optional[int], Optional[bool]]:
    """
    Parse Swedish 'formattedFloor' strings like:
      - "1 av 3, hiss finns ej"
      - "3 av 5, hiss finns"
      - "vån 2/4"
      - "plan 3"
      - "3 tr"  (→ floor_number = 4)
      - "en trappa upp" (→ floor_number = 2)
      - "BV", "Bottenvåning", "Entréplan", "Markplan" (→ floor_number = 0)
    Returns: (floor_number, floor_total, has_elevator)
    """
    if not label:
        return None, None, None

    s = label.strip()
    low = s.lower()

    # elevator
    has_elevator = None
    if "hiss finns ej" in low:
        has_elevator = False
    elif "hiss finns" in low or ("hiss" in low and "ej" not in low):
        has_elevator = True

    # explicit ground level markers
    if any(tok in low for tok in FLOOR_BOTTLABELS):
        return 0, None, has_elevator

    # "X av Y" or "X/Y"
    m = re.search(r"(\d+)\s*(?:av|/)\s*(\d+)", low)
    if m:
        fn = int(m.group(1))
        ft = int(m.group(2))
        return fn, ft, has_elevator

    # "plan X"
    m = re.search(r"\bplan\s*(\d+)", low)
    if m:
        fn = int(m.group(1))
        return fn, None, has_elevator

    # "vån 3" / "våning 3" / "van 3" (without å)
    m = re.search(r"\bv[åa]n(?:ing)?\s*(\d+)", low)
    if m:
        fn = int(m.group(1))
        return fn, None, has_elevator

    # "3 tr" (3 flights up) → normalize to floor_number = 3 + 1 = 4
    m = re.search(r"\b(\d+)\s*tr\b", low)
    if m:
        fn = int(m.group(1)) + 1
        return fn, None, has_elevator

    # "en trappa upp" / "1 trappa upp" → treat as 1 flight up → floor 2
    if "en trappa upp" in low or re.search(r"\b1\s+trappa\s+upp\b", low):
        return 2, None, has_elevator
    m = re.search(r"\b(\d+)\s+trapp(?:a|or)?\s+upp\b", low)
    if m:
        fn = int(m.group(1)) + 1
        return fn, None, has_elevator

    # Fallback: first number we see (best-effort)
    m = re.search(r"(\d+)", low)
    if m:
        fn = int(m.group(1))
        return fn, None, has_elevator

    return None, None, has_elevator


# ------------- Apollo helpers -------------
def _deref(apollo: Dict, ref: Dict) -> Optional[Dict]:
    """Follow Apollo __ref pointers like {'__ref': 'Location:123'}."""
    if not isinstance(ref, dict):
        return None
    ref_id = ref.get("__ref")
    if ref_id and isinstance(apollo.get(ref_id), dict):
        return apollo[ref_id]
    return None


def pick_best_listing_node(apollo: Dict) -> Optional[Dict]:
    """Choose a listing node (e.g. ActivePropertyListing) with the richest info."""
    best = None
    best_score = -1
    for _, val in apollo.items():
        if not isinstance(val, dict):
            continue
        t = val.get("__typename", "")
        if "PropertyListing" not in t:
            continue
        score = len(val)  # simple heuristic
        if score > best_score:
            best = val
            best_score = score
    return best


def extract_fields(url: str, fetched_at: str, apollo: Dict) -> Dict:
    """Extract stable fields; unpack Money; follow refs; parse floor info."""
    listing = pick_best_listing_node(apollo) or {}

    # Basic IDs / address
    listing_id = listing.get("id") or parse_listing_id_from_url(url)
    address = (
        listing.get("streetAddress")
        or dget(listing, "address", "streetAddress")
        or dget(listing, "displayAddress")
        or listing.get("title")
    )

    # Prices / fees (Money)
    asking_price_sek = money_amount(listing.get("askingPrice")) or money_amount(listing.get("price"))
    fee_sek = money_amount(listing.get("monthlyFee")) or money_amount(dget(listing, "fee"))
    square_meter_price_sek = money_amount(listing.get("squareMeterPrice"))
    running_costs_sek = money_amount(listing.get("runningCosts"))

    # Home characteristics
    rooms = listing.get("numberOfRooms")
    living_area_sqm = listing.get("livingArea") or dget(listing, "livingArea", "value")

    # Year built (consider legacy string)
    year_built = listing.get("constructionYear") or listing.get("yearBuilt") or listing.get("legacyConstructionYear")
    try:
        if isinstance(year_built, str):
            year_built = int(re.sub(r"\D", "", year_built)) if re.search(r"\d", year_built) else None
    except Exception:
        year_built = None

    # Housing form vs tenure
    housing_form_name = dget(listing, "housingForm", "name")
    housing_primary_group = dget(listing, "housingForm", "primaryGroup")
    tenure_name = dget(listing, "tenure", "name")

    # Coordinates
    coords = listing.get("coordinates") or {}
    lat = coords.get("lat") or coords.get("latitude")
    lon = coords.get("long") or coords.get("lng") or coords.get("longitude")

    # Location names (via refs)
    area_name = listing.get("area") or dget(listing, "location", "name")

    municipality_name = None
    muni_ref = listing.get("municipality")
    muni_node = _deref(apollo, muni_ref) if muni_ref else None
    if isinstance(muni_node, dict):
        municipality_name = muni_node.get("fullName") or muni_node.get("name")

    county_name = None
    region_ref = listing.get("region") or listing.get("county")
    region_node = _deref(apollo, region_ref) if region_ref else None
    if isinstance(region_node, dict):
        county_name = region_node.get("fullName") or region_node.get("name")

    districts_names = None
    dist_refs = listing.get("districts") or []
    names: List[str] = []
    for ref in dist_refs:
        node = _deref(apollo, ref)
        if isinstance(node, dict):
            nm = node.get("fullName") or node.get("name")
            if nm:
                names.append(nm)
    if names:
        districts_names = " | ".join(names)

    # Floor info
    floor_label = (
        listing.get("formattedFloor")
        or listing.get("floorLabel")
        or dget(listing, "address", "floorLabel")
    )
    # direct numeric if present
    floor_number = listing.get("floor") or listing.get("floorNumber")
    floor_total = listing.get("floorTotal") or listing.get("floorCount")
    has_elevator = None

    # Parse label as fallback or to fill missing parts (with "tr" → +1 normalization)
    if floor_label and (floor_number is None or floor_total is None or has_elevator is None):
        fn, ft, he = parse_floor_from_label(floor_label)
        if floor_number is None:
            floor_number = fn
        if floor_total is None:
            floor_total = ft
        if has_elevator is None:
            has_elevator = he

    return {
        "id": listing_id,
        "url": url,
        "fetched_at": fetched_at,

        "address": address,
        "area_name": area_name,
        "municipality_name": municipality_name,
        "county_name": county_name,
        "districts_names": districts_names,
        "post_code": listing.get("postCode"),
        "city": dget(listing, "municipality", "name"),

        "housing_form_name": housing_form_name,
        "housing_primary_group": housing_primary_group,
        "tenure_name": tenure_name,

        "rooms": rooms,
        "living_area_sqm": living_area_sqm,

        "asking_price_sek": asking_price_sek,
        "fee_sek": fee_sek,
        "square_meter_price_sek": square_meter_price_sek,
        "running_costs_sek": running_costs_sek,

        "lat": lat,
        "lon": lon,
        "year_built": year_built,

        "floor_label": floor_label,
        "floor_number": floor_number,
        "floor_total": floor_total,
        "has_elevator": has_elevator,
    }


# ------------- IO -------------
def iter_raw_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            yield obj


def build_clean_rows(raw_path: str, verbose: bool = True) -> List[Dict]:
    rows: List[Dict] = []
    n_total = 0
    n_ok = 0
    for obj in iter_raw_jsonl(raw_path):
        n_total += 1
        url = obj.get("url")
        fetched_at = obj.get("fetched_at")
        apollo = obj.get("apollo_state") or {}
        if not isinstance(apollo, dict):
            if verbose:
                print(f"[warn] skipping (no apollo_state): {url}")
            continue
        try:
            row = extract_fields(url=url, fetched_at=fetched_at, apollo=apollo)
            rows.append(row)
            n_ok += 1
        except Exception as e:
            if verbose:
                print(f"[warn] failed to parse listing ({url}): {e}")
            continue

    if verbose:
        print(f"[clean] parsed {n_ok}/{n_total} records")
    return rows


def write_outputs(rows: List[Dict], csv_path: str, parquet_path: str, verbose: bool = True) -> None:
    df = pd.DataFrame(rows)

    # Safe numeric coercions
    num_cols = [
        "asking_price_sek", "fee_sek", "square_meter_price_sek", "running_costs_sek",
        "rooms", "living_area_sqm", "lat", "lon", "year_built",
        "floor_number", "floor_total"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Write CSV
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"[output] wrote CSV → {csv_path} ({len(df)} rows)")

    # Write Parquet if possible
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(parquet_path, index=False)
        if verbose:
            print(f"[output] wrote Parquet → {parquet_path}")
    except Exception as e:
        if verbose:
            print(f"[note] Parquet not written (pyarrow missing or error: {e})")


def main() -> None:
    p = argparse.ArgumentParser(description="Flatten raw Hemnet listings (Apollo state) to CSV/Parquet (with floor info).")
    p.add_argument("--input", default=INPUT_JSONL_DEFAULT, help="Path to listings_raw.jsonl")
    p.add_argument("--csv", default=OUTPUT_CSV_DEFAULT, help="Path to write listings_clean.csv")
    p.add_argument("--parquet", default=OUTPUT_PARQUET_DEFAULT, help="Path to write listings_clean.parquet (if pyarrow available)")
    p.add_argument("--quiet", action="store_true", help="Reduce logging")
    args = p.parse_args()

    verbose = not args.quiet
    rows = build_clean_rows(args.input, verbose=verbose)
    write_outputs(rows, csv_path=args.csv, parquet_path=args.parquet, verbose=verbose)


if __name__ == "__main__":
    main()
