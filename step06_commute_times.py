# -*- coding: utf-8 -*-
"""
Commute times (Nominatim + ResRobot) + optional tennis-club commute.

Adds, when --tennis is provided:
  • tennis_transit_seconds / tennis_transit_minutes / tennis_transfers / tennis_walk_seconds
  • tennis_commute_provider / tennis_commute_when

Inputs:
  • listings_clean.csv
Outputs:
  • listings_commute.csv
Caches:
  • cache_geocode.csv
  • cache_commute.csv
"""

from __future__ import annotations

import argparse, csv, datetime as dt, math, os, re, time
from typing import Dict, Iterable, List, Optional, Tuple
import requests

# ---------- Config ----------
INPUT_DEFAULT = "listings_clean.csv"
OUTPUT_DEFAULT = "listings_commute.csv"
GEOCODE_CACHE_DEFAULT = "cache_geocode.csv"
COMMUTE_CACHE_DEFAULT = "cache_commute.csv"

MAX_MINUTES_DEFAULT = 180  # 3 hours is beyond any reasonable Stockholm PT trip

# Rate limits
RESROBOT_MIN_INTERVAL = 60.0 / 45.0      # ≤45 req/min
NOMINATIM_MIN_INTERVAL = 1.05            # be nice

STO_TZ = dt.timezone(dt.timedelta(hours=2))  # Stockholm summer time

URL_RE = re.compile(r"^https?://", re.I)
LATLON_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")

SESSION = requests.Session()
SESSION.headers.update({
    "Accept-Language": "sv-SE,sv;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "application/json,text/json;q=0.9,*/*;q=0.8",
    "User-Agent": "hempriser/commute (contact: you@example.com)",
})

def dprint(v: bool, *a, **k):
    if v: print(*a, **k)

def ensure_float(x):
    try: return float(x)
    except: return None

# ---------- Time helpers ----------
def parse_latlon(text: str) -> Optional[Tuple[float, float]]:
    m = LATLON_RE.match(text or "")
    return (float(m.group(1)), float(m.group(2))) if m else None

def next_weekday_time(tag: str) -> dt.datetime:
    if tag == "now":
        return dt.datetime.now(STO_TZ).replace(second=0, microsecond=0)
    m = re.match(r"^weekday_(\d{2})(\d{2})$", tag or "")
    hh, mm = (9, 0) if not m else (int(m.group(1)), int(m.group(2)))
    now = dt.datetime.now(STO_TZ)
    day = now
    while True:
        day += dt.timedelta(days=1)
        if day.weekday() < 5: break
    return day.replace(hour=hh, minute=mm, second=0, microsecond=0)

def iso8601_dur_to_seconds(s: str) -> Optional[int]:
    if not s or not s.startswith("PT"): return None
    secs = 0
    for val, unit in re.findall(r"(\d+)([HMS])", s):
        n = int(val)
        secs += n * (3600 if unit == "H" else 60 if unit == "M" else 1)
    return secs

# ---------- Address sanitation ----------
GENITIVE_FIX = { "Stockholms": "Stockholm", "Göteborgs": "Göteborg", "Malmös": "Malmö" }

def clean_muni(m: str) -> str:
    s = re.sub(r"\s*kommun\s*$", "", (m or ""), flags=re.I).strip()
    s = re.sub(r"\s*tätort\s*$", "", s, flags=re.I).strip()
    return GENITIVE_FIX.get(s, s)

FLOOR_TOKENS = (
    r"vån(?:ing)?", r"\d+\s*tr(?:appa|appor)?", r"\d+\/\d+\s*tr", r"\d+\s*tr",
    r"tr(?:appa|appor)?", r"plan\s*\d+", r"uteplats", r"högst\s*upp!?",
    r"lgh\s*\d+", r"lägenhet\s*\d+"
)
FLOOR_RE = re.compile(r"(?:,\s*)?(?:" + r"|".join(FLOOR_TOKENS) + r")\b", flags=re.IGNORECASE)

def sanitize_street(street: str) -> str:
    s = (street or "").strip()
    if not s: return s
    s = re.sub(r"\s+", " ", s)
    parts = [p.strip() for p in s.split(",")]
    kept = [p for p in parts if not FLOOR_RE.search(p)]
    s = ", ".join(kept) if kept else parts[0]
    s = FLOOR_RE.sub("", s).strip(", ").strip()
    s = re.sub(r"(\d+[A-Za-zÅÄÖåäö]?)\s*-\s*\d+\b", r"\1", s)  # "209-1104" -> "209"
    s = re.sub(r"\b(\d+)\s*([A-Za-zÅÄÖåäö])\b", r"\1\2", s)   # "25 A" -> "25A"
    s = re.sub(r"\s+", " ", s).strip(" ,")
    return s

def normalize_city(city: str) -> str:
    s = (city or "").strip()
    s = re.sub(r"\s*,.*$", "", s)
    s = re.sub(r"\s*tätort\s*$", "", s, flags=re.I).strip()
    return GENITIVE_FIX.get(s, s)

def build_address_variants(row: Dict[str, str]) -> List[str]:
    street = None
    for k in ("address","street_address","streetAddress","title"):
        if row.get(k):
            street = sanitize_street(str(row[k])); break
    if not street: return []
    post = (row.get("post_code") or row.get("postCode") or "").strip()
    city = row.get("city")
    muni = clean_muni(row.get("municipality_name","") or row.get("municipality","") or "")
    county = row.get("county_name") or row.get("county") or ""
    city = normalize_city(city) if city else normalize_city(muni) if muni else ""
    county = re.sub(r"\s*län\s*$", "", county or "", flags=re.I).strip()
    variants = []
    if post and city: variants.append(f"{street}, {post} {city}, Sweden")
    if city: variants.append(f"{street}, {city}, Sweden")
    if muni and muni != city: variants.append(f"{street}, {muni}, Sweden")
    if county: variants.append(f"{street}, {county}, Sweden")
    variants.append(f"{street}, Sweden")
    out, seen = [], set()
    for v in variants:
        if v not in seen: seen.add(v); out.append(v)
    return out

# ---------- Geocoders ----------
def nominatim_geocode(query: str, email: Optional[str], verbose: bool=False) -> Optional[Tuple[float,float]]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query, "format": "jsonv2", "addressdetails": 0,
        "countrycodes": "se", "limit": 1, "email": email or ""
    }
    resp = SESSION.get(url, params=params, timeout=30)
    dprint(verbose, f"[nominatim] {resp.url} -> {resp.status_code}")
    if resp.status_code != 200: return None
    data = resp.json()
    if not isinstance(data, list) or not data: return None
    lat = ensure_float(data[0].get("lat")); lon = ensure_float(data[0].get("lon"))
    if lat is None or lon is None: return None
    return (lat, lon)

def geocode_with_retries(row: Dict[str,str], email: Optional[str], last_ts: float,
                         verbose: bool=False) -> Tuple[Optional[Tuple[float,float]], float]:
    variants = build_address_variants(row)
    for i, q in enumerate(variants, 1):
        dt_wait = NOMINATIM_MIN_INTERVAL - (time.time() - last_ts)
        if dt_wait > 0: time.sleep(dt_wait)
        loc = nominatim_geocode(q, email, verbose=verbose)
        last_ts = time.time()
        if loc:
            if verbose and i > 1: dprint(True, f"[geocode:retry{i}] {q!r} -> {loc}")
            return loc, last_ts
        else:
            dprint(verbose, f"[miss] geocode failed for {q!r}")
    return None, last_ts

# ---------- Transit (ResRobot) ----------
def resrobot_trip(origin: Tuple[float, float], dest: Tuple[float, float],
                  when: dt.datetime, api_key: str, verbose: bool=False,
                  num_alternatives: int = 5) -> Optional[Dict]:
    url = "https://api.resrobot.se/v2.1/trip"
    params = {
        "format": "json",
        "originCoordLat": f"{origin[0]:.6f}", "originCoordLong": f"{origin[1]:.6f}",
        "destCoordLat": f"{dest[0]:.6f}",     "destCoordLong": f"{dest[1]:.6f}",
        "date": when.strftime("%Y-%m-%d"), "time": when.strftime("%H:%M"),
        "lang": "sv", "numF": max(1, int(num_alternatives)),
        "passlist": 0, "accessId": api_key,
    }
    resp = SESSION.get(url, params=params, timeout=40)
    dprint(verbose, f"[resrobot] trip -> {resp.status_code}")
    if resp.status_code != 200: return None
    data = resp.json()
    trips = data.get("Trip") or []
    best = None
    for t in trips:
        secs = iso8601_dur_to_seconds(t.get("duration"))
        if secs is None: continue
        transfers = 0; walk_seconds = 0
        try:
            legs = (t.get("LegList") or {}).get("Leg") or []
            jnys = 0
            for L in legs:
                if L.get("type") == "JNY": jnys += 1
                elif L.get("type") == "WALK":
                    walk_seconds += iso8601_dur_to_seconds(L.get("duration")) or 0
            transfers = max(0, jnys - 1)
        except: pass
        cand = {"seconds": int(secs), "transfers": int(transfers), "walk_seconds": int(walk_seconds)}
        if (best is None) or (cand["seconds"] < best["seconds"]) or \
           (cand["seconds"] == best["seconds"] and cand["transfers"] < best["transfers"]):
            best = cand
    return best

# ---------- Caches ----------
def _norm_addr_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def load_geocode_cache(path: str, verbose: bool=False) -> Dict[str, Tuple[float,float]]:
    cache = {}
    if not os.path.exists(path): return cache
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = row.get("address_key"); lat = ensure_float(row.get("lat")); lon = ensure_float(row.get("lon"))
            if k and lat is not None and lon is not None: cache[k] = (lat, lon)
    dprint(verbose, f"[cache] geocodes: {len(cache)}")
    return cache

def append_geocode_cache(path: str, items: List[Tuple[str,float,float]], verbose: bool=False):
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["address_key","lat","lon"])
        if need_header: w.writeheader()
        for k, lat, lon in items:
            w.writerow({"address_key": k, "lat": f"{lat:.7f}", "lon": f"{lon:.7f}"})
    if items: dprint(verbose, f"[cache] +{len(items)} geocodes")

# replace the old key builder
def commute_cache_key(url: str, provider: str,
                      origin: Tuple[float,float],
                      dest: Tuple[float,float],
                      when_key: str) -> str:
    ol = f"{origin[0]:.5f},{origin[1]:.5f}"
    dl = f"{dest[0]:.5f},{dest[1]:.5f}"
    return f"{provider}|{ol}|{dl}|{when_key}|{url}"

def load_commute_cache(path: str, verbose: bool=False) -> Dict[str, Dict]:
    cache = {}
    if not os.path.exists(path): return cache
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = commute_cache_key(
                    row.get("url",""), row.get("provider",""),
                    (ensure_float(row.get("origin_lat")) or 0.0, ensure_float(row.get("origin_lon")) or 0.0),
                    (ensure_float(row.get("dest_lat")) or 0.0, ensure_float(row.get("dest_lon")) or 0.0),
                    row.get("when_key","")
                )
            except Exception:
                continue
            if row.get("seconds"):
                cache[key] = {
                    "seconds": int(float(row["seconds"])),
                    "transfers": int(row.get("transfers","0") or 0),
                    "walk_seconds": int(row.get("walk_seconds","0") or 0),
                }
    dprint(verbose, f"[cache] commutes: {len(cache)}")
    return cache

def append_commute_cache(path: str, rows: List[Dict], verbose: bool=False):
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "url","provider",
            "origin_lat","origin_lon","dest_lat","dest_lon",
            "when_key","seconds","transfers","walk_seconds","cached_at"
        ])
        if need_header: w.writeheader()
        for r in rows: w.writerow(r)
    if rows: dprint(verbose, f"[cache] +{len(rows)} commutes")

# ---------- IO ----------
def read_listings(path: str, verbose: bool=False) -> List[Dict[str,str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dprint(verbose, f"[input] {len(rows)} rows from {path}")
    return rows

def write_output(path: str, rows: Iterable[Dict[str,str]], fieldnames: List[str], verbose: bool=False) -> int:
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow(r); n += 1
    dprint(verbose, f"[output] wrote {n} rows -> {path}")
    return n

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Annotate listings with commute times (workplace + optional tennis club).")
    ap.add_argument("--input", default=INPUT_DEFAULT)
    ap.add_argument("--output", default=OUTPUT_DEFAULT)

    ap.add_argument("--workplace", required=True, help="Address string or 'lat,lon'")
    ap.add_argument("--when", default="weekday_0900", help="'now' or 'weekday_HHMM'")

    ap.add_argument("--tennis", default=None, help="Tennis club address or 'lat,lon' (optional)")
    ap.add_argument("--tennis-when", default=None, help="'now' or 'weekday_HHMM' (defaults to weekday_0700 if --tennis)")

    ap.add_argument("--provider", choices=["resrobot"], default="resrobot")
    ap.add_argument("--geocode-cache", default=GEOCODE_CACHE_DEFAULT)
    ap.add_argument("--commute-cache", default=COMMUTE_CACHE_DEFAULT)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--contact-email", default=os.getenv("NOMINATIM_EMAIL",""),
                    help="Optional email to include in Nominatim requests (recommended).")
    ap.add_argument("--max-minutes", type=int, default=MAX_MINUTES_DEFAULT,
                help="If a computed trip exceeds this many minutes, retry with more alternatives; if still too large, leave blank and warn.")
    args = ap.parse_args()

    def too_big(secs: Optional[int]) -> bool:
        return secs is not None and secs > args.max_minutes * 60

    verbose = not args.quiet
    resrobot_key = os.getenv("RESROBOT_API_KEY")
    if not resrobot_key:
        raise SystemExit("RESROBOT_API_KEY is required (free key from Trafiklab).")

    # -- Destination 1: Workplace
    dest_work = parse_latlon(args.workplace)
    last_geocode_at = 0.0
    if dest_work is None:
        dest_work, last_geocode_at = geocode_with_retries(
            {"address": args.workplace}, args.contact_email, last_geocode_at, verbose=verbose
        )
        if not dest_work:
            raise SystemExit("Could not geocode workplace. Pass 'lat,lon' or a clearer address.")
    depart_dt_work = next_weekday_time(args.when)
    when_key_work = f"{args.when}|{depart_dt_work.strftime('%Y%m%d%H%M')}"
    dprint(verbose, f"[time] workplace {depart_dt_work.isoformat()} (when_key={when_key_work})")

    # -- Destination 2: Tennis (optional)
    dest_tennis: Optional[Tuple[float,float]] = None
    depart_dt_tennis: Optional[dt.datetime] = None
    when_key_tennis: Optional[str] = None
    if args.tennis:
        dest_tennis = parse_latlon(args.tennis)
        if dest_tennis is None:
            # geocode tennis via same logic
            dest_tennis, last_geocode_at = geocode_with_retries(
                {"address": args.tennis}, args.contact_email, last_geocode_at, verbose=verbose
            )
            if not dest_tennis:
                raise SystemExit("Could not geocode tennis club. Pass 'lat,lon' or a clearer address.")
        tennis_when = args.tennis_when or "weekday_0700"
        depart_dt_tennis = next_weekday_time(tennis_when)
        when_key_tennis = f"{tennis_when}|{depart_dt_tennis.strftime('%Y%m%d%H%M')}"
        dprint(verbose, f"[time] tennis {depart_dt_tennis.isoformat()} (when_key={when_key_tennis})")

    listings = read_listings(args.input, verbose=verbose)
    geoc_cache = load_geocode_cache(args.geocode_cache, verbose=verbose)
    commute_cache = load_commute_cache(args.commute_cache, verbose=verbose)
    new_geoc_rows: List[Tuple[str,float,float]] = []
    new_commute_rows: List[Dict] = []
    out_rows: List[Dict[str,str]] = []

    last_transit_at = 0.0
    processed = 0

    for row in listings:
        if args.limit is not None and args.limit >= 0 and processed >= args.limit:
            break

        url = row.get("url") or ""
        if not URL_RE.search(url):
            dprint(verbose, f"[skip] bad url: {url!r}")
            continue

        # Origin coords: prefer existing lat/lon, else geocode
        lat = ensure_float(row.get("lat") or row.get("latitude"))
        lon = ensure_float(row.get("lon") or row.get("lng") or row.get("longitude"))
        origin = (lat, lon) if lat is not None and lon is not None else None

        if origin is None:
            variants = build_address_variants(row)
            origin_key = _norm_addr_key(variants[0]) if variants else None
            if origin_key and origin_key in geoc_cache:
                origin = geoc_cache[origin_key]
            else:
                origin, last_geocode_at = geocode_with_retries(row, args.contact_email, last_geocode_at, verbose=verbose)
                if origin and origin_key:
                    geoc_cache[origin_key] = origin
                    new_geoc_rows.append((origin_key, origin[0], origin[1]))
                    dprint(verbose, f"[geocode] {variants[0]} -> {origin}")

        out = dict(row)
        if origin and (not row.get("lat") and not row.get("longitude")):
            out["lat"], out["lon"] = f"{origin[0]:.7f}", f"{origin[1]:.7f}"

        # Initialize output fields
        out.update({
            "transit_seconds": "", "transit_minutes": "", "transfers": "", "walk_seconds": "",
            "commute_provider": "resrobot", "commute_when": depart_dt_work.isoformat(),
        })
        if dest_tennis:
            out.update({
                "tennis_transit_seconds": "", "tennis_transit_minutes": "",
                "tennis_transfers": "", "tennis_walk_seconds": "",
                "tennis_commute_provider": "resrobot",
                "tennis_commute_when": depart_dt_tennis.isoformat(),
            })

        if not origin:
            out_rows.append(out); processed += 1; continue

        # --- Workplace commute (cached) ---
        key_w = commute_cache_key(url, "resrobot", origin, dest_work, when_key_work)
        cached_w = commute_cache.get(key_w)
        if cached_w:
            secs = cached_w["seconds"]; mins = int(math.ceil(secs/60.0))
            out.update({
                "transit_seconds": str(secs), "transit_minutes": str(mins),
                "transfers": str(cached_w.get("transfers",0)),
                "walk_seconds": str(cached_w.get("walk_seconds",0)),
            })
            dprint(verbose, f"[hit] workplace cache {url} -> {mins} min")
        else:
            dt_wait = RESROBOT_MIN_INTERVAL - (time.time() - last_transit_at)
            if dt_wait > 0: time.sleep(dt_wait)
            res_w = None
            try:
                res_w = resrobot_trip(origin, dest_work, depart_dt_work, api_key=resrobot_key, verbose=verbose, num_alternatives=1)
            except Exception as e:
                dprint(verbose, f"[warn] resrobot (work) error for {url}: {e}")
            
            # retry with more options if it looks wrong
            if not res_w or too_big(res_w.get("seconds")) or res_w.get("transfers", 0) >= 4:
                dt_wait = RESROBOT_MIN_INTERVAL - (time.time() - last_transit_at)
                if dt_wait > 0: time.sleep(dt_wait)
                res_w = resrobot_trip(origin, dest_work, depart_dt_work, api_key=resrobot_key, verbose=verbose, num_alternatives=6)
                last_transit_at = time.time()

            # if still absurd, leave fields blank and warn (don’t poison the cache)
            if res_w and too_big(res_w.get("seconds")):
                dprint(True, f"[outlier] {url} -> {int(res_w['seconds']/60)} min; skipping & not caching")
                res_w = None  # fall through; fields remain empty
            if res_w:
                secs = int(res_w["seconds"]); mins = int(math.ceil(secs/60.0))
                out.update({
                    "transit_seconds": str(secs), "transit_minutes": str(mins),
                    "transfers": str(res_w.get("transfers",0)),
                    "walk_seconds": str(res_w.get("walk_seconds",0)),
                })
                new_commute_rows.append({
                    "url": url, "provider": "resrobot",
                    "origin_lat": f"{origin[0]:.7f}", "origin_lon": f"{origin[1]:.7f}",
                    "dest_lat": f"{dest_work[0]:.7f}", "dest_lon": f"{dest_work[1]:.7f}",
                    "when_key": when_key_work, "seconds": str(secs),
                    "transfers": str(res_w.get("transfers",0)),
                    "walk_seconds": str(res_w.get("walk_seconds",0)),
                    "cached_at": dt.datetime.utcnow().isoformat() + "Z",
                })
                dprint(verbose, f"[ok] workplace {url} -> {mins} min (transfers={res_w.get('transfers',0)})")
            else:
                dprint(verbose, f"[miss] workplace route {url}")

        # --- Tennis commute (cached) ---
        if dest_tennis and depart_dt_tennis and when_key_tennis:
            key_t = commute_cache_key(url, "resrobot", origin, dest_tennis, when_key_tennis)
            cached_t = commute_cache.get(key_t)
            if cached_t:
                secs = cached_t["seconds"]; mins = int(math.ceil(secs/60.0))
                out.update({
                    "tennis_transit_seconds": str(secs), "tennis_transit_minutes": str(mins),
                    "tennis_transfers": str(cached_t.get("transfers",0)),
                    "tennis_walk_seconds": str(cached_t.get("walk_seconds",0)),
                })
                dprint(verbose, f"[hit] tennis cache {url} -> {mins} min")
            else:
                dt_wait = RESROBOT_MIN_INTERVAL - (time.time() - last_transit_at)
                if dt_wait > 0: time.sleep(dt_wait)
                res_t = None
                try:
                    res_t = resrobot_trip(origin, dest_tennis, depart_dt_tennis, api_key=resrobot_key, verbose=verbose, num_alternatives=1)
                except Exception as e:
                    dprint(verbose, f"[warn] resrobot (tennis) error for {url}: {e}")
                # retry with more options if it looks wrong
                if not res_t or too_big(res_t.get("seconds")) or res_t.get("transfers", 0) >= 4:
                    dt_wait = RESROBOT_MIN_INTERVAL - (time.time() - last_transit_at)
                    if dt_wait > 0: time.sleep(dt_wait)
                    res_t = resrobot_trip(origin, dest_tennis, depart_dt_tennis, api_key=resrobot_key, verbose=verbose, num_alternatives=6)
                    last_transit_at = time.time()

                # if still absurd, leave fields blank and warn (don’t poison the cache)
                if res_t and too_big(res_t.get("seconds")):
                    dprint(True, f"[outlier] {url} -> {int(res_t['seconds']/60)} min; skipping & not caching")
                    res_t = None  # fall through; fields remain empty
                if res_t:
                    secs = int(res_t["seconds"]); mins = int(math.ceil(secs/60.0))
                    out.update({
                        "tennis_transit_seconds": str(secs), "tennis_transit_minutes": str(mins),
                        "tennis_transfers": str(res_t.get("transfers",0)),
                        "tennis_walk_seconds": str(res_t.get("walk_seconds",0)),
                    })
                    new_commute_rows.append({
                        "url": url, "provider": "resrobot",
                        "origin_lat": f"{origin[0]:.7f}", "origin_lon": f"{origin[1]:.7f}",
                        "dest_lat": f"{dest_tennis[0]:.7f}", "dest_lon": f"{dest_tennis[1]:.7f}",
                        "when_key": when_key_tennis, "seconds": str(secs),
                        "transfers": str(res_t.get("transfers",0)),
                        "walk_seconds": str(res_t.get("walk_seconds",0)),
                        "cached_at": dt.datetime.utcnow().isoformat() + "Z",
                    })
                    dprint(verbose, f"[ok] tennis {url} -> {mins} min (transfers={res_t.get('transfers',0)})")
                else:
                    dprint(verbose, f"[miss] tennis route {url}")

        out_rows.append(out)
        processed += 1

    # Persist caches + output
    if new_geoc_rows: append_geocode_cache(GEOCODE_CACHE_DEFAULT, new_geoc_rows, verbose=verbose)
    if new_commute_rows: append_commute_cache(COMMUTE_CACHE_DEFAULT, new_commute_rows, verbose=verbose)

    base_cols = list(listings[0].keys()) if listings else []
    for extra in ("lat","lon"):
        if extra not in base_cols: base_cols.append(extra)

    work_cols = ["transit_seconds","transit_minutes","transfers","walk_seconds","commute_provider","commute_when"]
    for c in work_cols:
        if c not in base_cols: base_cols.append(c)

    if dest_tennis:
        tennis_cols = [
            "tennis_transit_seconds","tennis_transit_minutes","tennis_transfers",
            "tennis_walk_seconds","tennis_commute_provider","tennis_commute_when"
        ]
        for c in tennis_cols:
            if c not in base_cols: base_cols.append(c)

    write_output(args.output, out_rows, base_cols, verbose=verbose)

if __name__ == "__main__":
    main()