#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step07_score_listings.py

Score apartments into [0, 1] using:
  + living_area_sqm            (positive)
  - asking_price_* / price_*   (negative)
  - fee_*                      (negative)
  - transit_minutes            (negative)
  - tennis_transit_minutes     (negative)

Key features in this version:
- Robust column detection (handles asking_price_sek, etc.)
- Safe parsing of numeric-like strings
- Robust scaling by quantiles (winsorization)
- Weighted average over available dimensions (row-wise normalized)
- Exports all desirability components: d_area, d_price, d_fee, d_commute, d_tennis, and score
- Guardrail: raises a clear error if a chosen dimension has a constant (degenerate) distribution
  so you don't silently get a flat 0.5 for everyone again.

Usage:
  python step07_score_listings.py \
    --input listings_commute.csv \
    --output listings_scored.csv \
    --weights "area=0.30 price=0.30 fee=0.10 commute=0.20 tennis=0.10" \
    --low-q 0.05 --high-q 0.95
"""

from __future__ import annotations
import argparse
import sys
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

VERSION = "step07-2025-08-20a"

# -------------------------------
# Column candidates / detection
# -------------------------------
AREA_CANDIDATES = [
    "living_area_sqm",  # expected
    "boarea", "boarea_kvm", "boyta", "area_sqm", "area"
]
PRICE_CANDIDATES = [
    "asking_price_sek",  # common in your CSV
    "asking_price",
    "price_sek", "price", "starting_price",
    "utgangspris", "utgångspris", "pris",
]
FEE_CANDIDATES = [
    "fee_sek", "fee", "monthly_fee", "avgift", "månadavgift", "manadsavgift", "månadsavgift"
]
COMMUTE_CANDIDATES = [
    "transit_minutes", "commute_minutes", "work_transit_minutes"
]
TENNIS_COMMUTE_CANDIDATES = [
    "tennis_transit_minutes", "tennis_minutes"
]

# -------------------------------
# Helpers
# -------------------------------

def parse_weights(text: Optional[str]) -> Dict[str, float]:
    """Parse weights like "area=0.3 price=0.3 fee=0.1 commute=0.2 tennis=0.1".
    Accepts commas or spaces as separators. Unknown keys are ignored with a warning.
    """
    default = {"area": 0.30, "price": 0.30, "fee": 0.10, "commute": 0.20, "tennis": 0.10}
    if not text:
        return default
    parts = re.split(r"[,\s]+", text.strip())
    out: Dict[str, float] = {}
    for p in parts:
        if not p:
            continue
        m = re.match(r"^(area|price|fee|commute|tennis)=(\d*\.?\d+)$", p)
        if m:
            key, val = m.group(1), float(m.group(2))
            out[key] = val
        else:
            # Ignore unknown, but keep helpful message
            if "=" in p:
                key = p.split("=", 1)[0]
                print(f"[warn] Unknown weight key '{key}' ignored. Allowed: area, price, fee, commute, tennis")
    if not out:
        print("[warn] No valid weights parsed; using defaults.")
        out = default
    # Normalize to sum=1 (and warn if not close)
    s = sum(out.values())
    if s <= 0:
        print("[warn] Sum of weights <= 0; falling back to defaults.")
        out = default
        s = sum(out.values())
    if abs(s - 1.0) > 1e-9:
        out = {k: v / s for k, v in out.items()}
        print(f"[info] Weights renormalized to sum=1: {out}")
    return out


def _to_float_series(s: pd.Series, *, parse_kind: str, index: pd.Index) -> pd.Series:
    """Parse a Series to float with light cleaning.

    parse_kind: 'int' or 'float'
      - 'int': remove all non-digits (and minus); treat thousands separators safely.
      - 'float': allow a single decimal point; remove other junk.
    """
    if not isinstance(s, pd.Series):
        return pd.Series(np.nan, index=index, dtype=float)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    s = s.astype(str)
    s = s.str.replace("\u00a0", " ", regex=False)   # NBSP -> space
    s = s.str.strip()

    if parse_kind == "int":
        # Keep digits and minus only
        s2 = s.str.replace(r"[^\d\-]", "", regex=True)
        return pd.to_numeric(s2, errors="coerce").astype(float)
    else:
        # Float-like: keep digits, one dot for decimals, and minus
        # Remove commas first (thousands in many locales)
        s2 = s.str.replace(",", "", regex=False)
        # Remove everything except digits, dot, minus
        s2 = s2.str.replace(r"[^\d\.\-]", "", regex=True)
        # If multiple dots (e.g. thousands separators), drop all but last
        def _fix_many_dots(x: str) -> str:
            if x.count(".") <= 1:
                return x
            # keep last dot as decimal sep
            parts = x.split(".")
            return "".join(parts[:-1]) + "." + parts[-1]
        s2 = s2.apply(_fix_many_dots)
        return pd.to_numeric(s2, errors="coerce").astype(float)


def pick_numeric_column(
    df: pd.DataFrame,
    candidates: List[str],
    *,
    fuzzy_keywords: Optional[List[str]] = None,
    parse_kind: str = "float",
) -> pd.Series:
    """Return the first matching column parsed as float. If none, try fuzzy keyword match.
    Always returns a Series aligned to df.index (possibly all-NaN).
    """
    # 1) Exact candidate names
    for c in candidates:
        if c in df.columns:
            s = df[c]
            out = _to_float_series(s, parse_kind=parse_kind, index=df.index)
            out.name = c
            return out
    # 2) Fuzzy search
    if fuzzy_keywords:
        low = {c.lower(): c for c in df.columns}
        for name_lc, orig in low.items():
            if any(k in name_lc for k in fuzzy_keywords):
                s = df[orig]
                out = _to_float_series(s, parse_kind=parse_kind, index=df.index)
                out.name = orig
                return out
    # 3) Nothing found
    print(f"[warn] No matching column found among: {candidates} or keywords={fuzzy_keywords}")
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out.name = None
    return out


# -------------------------------
# Scaling to desirability [0, 1]
# -------------------------------

def robust_scaled_desirability(
    s: pd.Series,
    low_q: float,
    high_q: float,
    *,
    positive: bool,
    label: str,
) -> pd.Series:
    """Winsorize by [low_q, high_q] and scale to [0, 1].

    - If the series has *no* valid numbers, returns all-NaN and warns (no raise).
    - If the series has a degenerate distribution (lo==hi), raise ValueError with a helpful message.
    """
    x = pd.to_numeric(s, errors="coerce")
    if not x.notna().any():
        print(f"[warn] '{label}': no numeric values; returning NaN desirability.")
        return pd.Series(np.nan, index=s.index, dtype=float)

    lo = x.quantile(low_q)
    hi = x.quantile(high_q)
    if pd.isna(lo) or pd.isna(hi):
        print(f"[warn] '{label}': quantiles undefined; returning NaN desirability.")
        return pd.Series(np.nan, index=s.index, dtype=float)

    if hi <= lo:
        uniq = pd.unique(x.dropna())
        msg = (
            f"[error] '{label}': degenerate distribution (lo==hi=={lo}). "
            f"Distinct non-NaN values: {len(uniq)}. "
            f"This usually means the column was mis-detected or parsed as a single constant."
        )
        raise ValueError(msg)

    xc = x.clip(lower=lo, upper=hi)
    scaled = (xc - lo) / (hi - lo)
    if not positive:
        scaled = 1.0 - scaled
    # Ensure bounds exactly in [0, 1] for numerical stability
    scaled = scaled.clip(lower=0.0, upper=1.0)

    # Extra guardrail: if, after scaling, all non-NaNs are the same value, raise
    vals = pd.unique(scaled.dropna())
    if len(vals) <= 1:
        raise ValueError(
            f"[error] '{label}': scaled desirability collapsed to a single value ({vals[0] if len(vals)==1 else 'NaN'})."
        )
    return scaled


def weighted_rowwise_mean(desirabilities: Dict[str, pd.Series], weights: Dict[str, float]) -> Tuple[pd.Series, pd.Series]:
    """Compute a weighted mean across desirability series, ignoring NaNs row-wise.

    Returns (score, weight_sum_per_row).
    """
    # Filter to keys that exist in desirabilities and have nonzero weight
    items = [(k, desirabilities[k], weights.get(k, 0.0)) for k in desirabilities.keys() if weights.get(k, 0.0) > 0]
    if not items:
        return pd.Series(np.nan), pd.Series(0.0)

    cols = []
    den_parts = []
    for k, s, w in items:
        cols.append(s * w)
        den_parts.append(s.notna().astype(float) * w)
    num = pd.concat(cols, axis=1).sum(axis=1, skipna=True, min_count=1)
    den = pd.concat(den_parts, axis=1).sum(axis=1)

    score = num / den
    score[den <= 0] = np.nan
    return score, den


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Score listings into [0,1] with robust scaling.")
    ap.add_argument("--input", required=True, help="Input CSV (e.g., listings_commute.csv)")
    ap.add_argument("--output", required=True, help="Output CSV (e.g., listings_scored.csv)")
    ap.add_argument("--weights", default=None, help="Weights like 'area=0.3 price=0.3 fee=0.1 commute=0.2 tennis=0.1' (will be renormalized)")
    ap.add_argument("--low-q", type=float, default=0.05, help="Lower quantile for winsorization (default 0.05)")
    ap.add_argument("--high-q", type=float, default=0.95, help="Upper quantile for winsorization (default 0.95)")
    ap.add_argument("--debug", action="store_true", help="Verbose debug prints")
    args = ap.parse_args()

    low_q = args.low_q
    high_q = args.high_q
    if not (0.0 <= low_q < high_q <= 1.0):
        print(f"[error] Invalid quantile bounds: low_q={low_q}, high_q={high_q}")
        sys.exit(2)

    print(f"[info] Version {VERSION}; quantiles=({low_q}, {high_q})")

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"[error] Could not read input CSV: {e}")
        sys.exit(2)

    n_rows = len(df)
    print(f"[info] Loaded {n_rows} rows from {args.input}")

    weights = parse_weights(args.weights)
    if args.debug:
        print(f"[debug] weights: {weights}")

    # --- Pick columns ---
    area_s  = pick_numeric_column(df, AREA_CANDIDATES,  fuzzy_keywords=["area", "boarea", "kvm", "sqm", "m2", "m²"], parse_kind="float")
    price_s = pick_numeric_column(df, PRICE_CANDIDATES, fuzzy_keywords=["price", "pris"], parse_kind="int")
    fee_s   = pick_numeric_column(df, FEE_CANDIDATES,   fuzzy_keywords=["fee", "avg"],   parse_kind="int")
    comm_s  = pick_numeric_column(df, COMMUTE_CANDIDATES, fuzzy_keywords=["transit", "commute", "work"], parse_kind="float")
    tenn_s  = pick_numeric_column(df, TENNIS_COMMUTE_CANDIDATES, fuzzy_keywords=["tennis"], parse_kind="float")

    print(f"[cols] area={area_s.name} price={price_s.name} fee={fee_s.name} commute={comm_s.name} tennis={tenn_s.name}")

    # --- Scale to desirabilities ---
    try:
        df["d_area"]    = robust_scaled_desirability(area_s,  low_q, high_q, positive=True,  label="area")
        df["d_price"]   = robust_scaled_desirability(price_s, low_q, high_q, positive=False, label="price")
        df["d_fee"]     = robust_scaled_desirability(fee_s,   low_q, high_q, positive=False, label="fee")
        df["d_commute"] = robust_scaled_desirability(comm_s,  low_q, high_q, positive=False, label="commute")
        df["d_tennis"]  = robust_scaled_desirability(tenn_s,  low_q, high_q, positive=False, label="tennis")
    except ValueError as e:
        print(str(e))
        print("[hint] Check that the detected column names are correct and not constant, and that parsing worked.")
        sys.exit(1)

    # --- Compose weighted score ---
    desir = {
        "area":    df["d_area"],
        "price":   df["d_price"],
        "fee":     df["d_fee"],
        "commute": df["d_commute"],
        "tennis":  df["d_tennis"],
    }
    score, wsum = weighted_rowwise_mean(desir, weights)
    df["score"] = score
    df["score_weight_sum"] = wsum

    # Final small sort preview (not persisted unless caller later sorts)
    try:
        top = df.nlargest(5, "score")[['score', 'd_area','d_price','d_fee','d_commute','d_tennis']]
        if args.debug:
            print("[debug] top-5 by score (preview):\n", top)
    except Exception:
        pass

    # --- Save ---
    try:
        df.to_csv(args.output, index=False)
        print(f"[ok] wrote {len(df)} rows -> {args.output}")
    except Exception as e:
        print(f"[error] Could not write output CSV: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
