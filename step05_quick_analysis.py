# -*- coding: utf-8 -*-
"""
Quick exploration of listings_clean.csv:
  • Filters (municipality/area + numeric ranges)
  • Computes price_per_sqm
  • Summary stats to a text file
  • Saves 3 plots: price-vs-area scatter (+ trend), price histogram, area histogram
  • Optional lon/lat scatter (no basemap; just a quick spatial sanity check)

Dependencies:
  - pandas
  - matplotlib

Usage example:
  python step05_quick_analysis.py \
    --input listings_clean.csv \
    --outdir figs \
    --filter-municipality "Stockholms kommun" \
    --filter-area "Årsta" \
    --min-area 15 --max-area 200 \
    --max-ppsqm 200000
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# ---------- Filters ----------
def apply_filters(
    df: pd.DataFrame,
    municipality_substr: Optional[str],
    area_substr: Optional[str],
    min_area: Optional[float],
    max_area: Optional[float],
    min_price: Optional[float],
    max_price: Optional[float],
    max_ppsqm: Optional[float],
) -> pd.DataFrame:
    out = df.copy()

    # numeric coercions
    for col in ["asking_price_sek", "fee_sek", "rooms", "living_area_sqm", "lat", "lon", "year_built"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # compute price per sqm
    out["price_per_sqm"] = out["asking_price_sek"] / out["living_area_sqm"]

    # text filters (case-insensitive 'contains')
    if municipality_substr:
        out = out[out["municipality_name"].fillna("").str.contains(municipality_substr, case=False, na=False)]
    if area_substr:
        # filter on area_name OR address for convenience
        mask_area = out["area_name"].fillna("").str.contains(area_substr, case=False, na=False)
        mask_addr = out["address"].fillna("").str.contains(area_substr, case=False, na=False)
        out = out[mask_area | mask_addr]

    # numeric filters
    if min_area is not None:
        out = out[out["living_area_sqm"] >= min_area]
    if max_area is not None:
        out = out[out["living_area_sqm"] <= max_area]
    if min_price is not None:
        out = out[out["asking_price_sek"] >= min_price]
    if max_price is not None:
        out = out[out["asking_price_sek"] <= max_price]
    if max_ppsqm is not None:
        out = out[out["price_per_sqm"] <= max_ppsqm]

    return out


# ---------- Summaries ----------
def summary_stats(df: pd.DataFrame) -> str:
    def fmt(x: Optional[float]) -> str:
        return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.0f}".replace(",", " ")

    N = len(df)
    price = pd.to_numeric(df["asking_price_sek"], errors="coerce")
    area = pd.to_numeric(df["living_area_sqm"], errors="coerce")
    ppsm = pd.to_numeric(df["price_per_sqm"], errors="coerce")

    def stats(s: pd.Series) -> Tuple[float, float, float, float, float]:
        return (
            float(s.dropna().min()) if s.notna().any() else float("nan"),
            float(s.dropna().quantile(0.25)) if s.notna().any() else float("nan"),
            float(s.dropna().median()) if s.notna().any() else float("nan"),
            float(s.dropna().quantile(0.75)) if s.notna().any() else float("nan"),
            float(s.dropna().max()) if s.notna().any() else float("nan"),
        )

    p_min, p_q1, p_med, p_q3, p_max = stats(price)
    a_min, a_q1, a_med, a_q3, a_max = stats(area)
    s_min, s_q1, s_med, s_q3, s_max = stats(ppsm)

    lines = [
        f"rows: {N}",
        "",
        "asking_price_sek:",
        f"  min / q1 / median / q3 / max: {fmt(p_min)} / {fmt(p_q1)} / {fmt(p_med)} / {fmt(p_q3)} / {fmt(p_max)}",
        "",
        "living_area_sqm:",
        f"  min / q1 / median / q3 / max: {fmt(a_min)} / {fmt(a_q1)} / {fmt(a_med)} / {fmt(a_q3)} / {fmt(a_max)}",
        "",
        "price_per_sqm:",
        f"  min / q1 / median / q3 / max: {fmt(s_min)} / {fmt(s_q1)} / {fmt(s_med)} / {fmt(s_q3)} / {fmt(s_max)}",
    ]
    return "\n".join(lines)


def save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------- Plots ----------
def plot_price_vs_area(df: pd.DataFrame, outpath: str) -> None:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    x = pd.to_numeric(df["living_area_sqm"], errors="coerce")
    y = pd.to_numeric(df["asking_price_sek"], errors="coerce")
    keep = x.notna() & y.notna()
    x = x[keep]
    y = y[keep]

    plt.figure()
    plt.scatter(x, y, s=12, alpha=0.6)
    # simple trend line (degree-1 poly fit)
    try:
        m, b = pd.Series(y).corr(x), None  # this line is placeholder for linter; we compute with polyfit next
        import numpy as np
        coeffs = np.polyfit(x.values, y.values, deg=1)
        xp = np.linspace(float(x.min()), float(x.max()), 100)
        yp = coeffs[0] * xp + coeffs[1]
        plt.plot(xp, yp)  # default color/style
    except Exception:
        pass
    plt.xlabel("Living area (m²)")
    plt.ylabel("Asking price (SEK)")
    plt.title("Price vs Area")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_hist(series: pd.Series, title: str, xlabel: str, outpath: str, bins: int = 60) -> None:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    vals = pd.to_numeric(series, errors="coerce").dropna()
    plt.figure()
    plt.hist(vals, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_lonlat(df: pd.DataFrame, outpath: str) -> None:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    lat = pd.to_numeric(df["lat"], errors="coerce")
    lon = pd.to_numeric(df["lon"], errors="coerce")
    keep = lat.notna() & lon.notna()
    lat = lat[keep]
    lon = lon[keep]

    if len(lat) == 0:
        return  # nothing to plot

    plt.figure()
    plt.scatter(lon, lat, s=6, alpha=0.5)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Lon/Lat scatter (no basemap)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ---------- Orchestration ----------
def main() -> None:
    p = argparse.ArgumentParser(description="Quick analysis of listings_clean.csv")
    p.add_argument("--input", default="listings_clean.csv", help="Path to listings_clean.csv")
    p.add_argument("--outdir", default="figs", help="Directory to write outputs")
    p.add_argument("--filter-municipality", default=None, help="Substring match for municipality_name (case-insensitive)")
    p.add_argument("--filter-area", default=None, help="Substring match for area_name/address (case-insensitive)")
    p.add_argument("--min-area", type=float, default=None)
    p.add_argument("--max-area", type=float, default=None)
    p.add_argument("--min-price", type=float, default=None)
    p.add_argument("--max-price", type=float, default=None)
    p.add_argument("--max-ppsqm", type=float, default=None, help="Filter out listings above this SEK/m²")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)

    df_f = apply_filters(
        df,
        municipality_substr=args.filter_municipality,
        area_substr=args.filter_area,
        min_area=args.min_area,
        max_area=args.max_area,
        min_price=args.min_price,
        max_price=args.max_price,
        max_ppsqm=args.max_ppsqm,
    )

    # Save summary text
    txt = summary_stats(df_f)
    save_text(os.path.join(args.outdir, "summary.txt"), txt)
    print(txt)

    # Plots
    plot_price_vs_area(df_f, os.path.join(args.outdir, "price_vs_area.png"))
    plot_hist(df_f["asking_price_sek"], "Asking price", "SEK", os.path.join(args.outdir, "hist_price.png"))
    plot_hist(df_f["living_area_sqm"], "Living area", "m²", os.path.join(args.outdir, "hist_area.png"))
    plot_lonlat(df_f, os.path.join(args.outdir, "map_scatter_lonlat.png"))

    print(f"\nWrote figures to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
