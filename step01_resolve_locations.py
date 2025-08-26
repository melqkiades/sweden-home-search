# -*- coding: utf-8 -*-
"""
Resolve Hemnet location IDs for a small set of place names
(e.g., Stockholm, Solna, Årsta) and write them to a CSV.

Note: hemnet.get_location_ids() returns {id, fullName, parentFullName, type}.
We use fullName (not name) to avoid blanks.
"""
from typing import Dict, Iterable, List
import csv
from hemnet import get_location_ids

OUTPUT_CSV = "locations.csv"

# Tell it what you want; tweak prefer_types/must_include to bias the match.
QUERIES: List[Dict] = [
    # {"term": "Stockholm", "prefer_types": {"MUNICIPALITY", "CITY"}, "must_include": ["Stockholm"]},
    {"term": "Stockholm", "prefer_types": {"MUNICIPALITY"}, "must_include": ["Stockholms kommun"]},
    {"term": "Solna",     "prefer_types": {"MUNICIPALITY", "CITY"}, "must_include": ["Solna"]},
    {"term": "Årsta",     "prefer_types": {"DISTRICT", "NEIGHBORHOOD"}, "must_include": ["Årsta", "Stockholm"]},
]

TYPE_ORDER = ["DISTRICT", "NEIGHBORHOOD", "CITY", "MUNICIPALITY", "COUNTY", "REGION", "COUNTRY"]


def type_rank(t: str) -> int:
    return TYPE_ORDER.index(t) if t in TYPE_ORDER else 99


def _disp_name(c: Dict) -> str:
    # Prefer 'fullName' from hemnet.get_location_ids(); fall back to 'name' if present
    return (c.get("fullName") or c.get("name") or "").strip()


def choose_best(candidates: Iterable[Dict], prefer_types: Iterable[str], must_include: Iterable[str]) -> Dict:
    """
    Pick the best candidate:
      1) Filter by must_include substrings (case-insensitive) found in display name.
      2) Prefer desired types.
      3) Break ties by type priority, then shortest display name.
    """
    must_include = [s.lower() for s in must_include]
    prefer_types = set(prefer_types)

    def has_all_needles(c: Dict) -> bool:
        return all(n in _disp_name(c).lower() for n in must_include)

    pool = [c for c in candidates if has_all_needles(c)] or list(candidates)
    preferred = [c for c in pool if c.get("type") in prefer_types] or pool
    preferred.sort(key=lambda c: (type_rank(c.get("type", "")), len(_disp_name(c))))
    return preferred[0]


def resolve_locations(queries: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for q in queries:
        term = q["term"]
        cands = get_location_ids(term)
        chosen = choose_best(cands, q["prefer_types"], q["must_include"])
        rows.append({
            "term": term,
            "id": chosen.get("id"),
            "name": _disp_name(chosen),           # use fullName here
            "type": chosen.get("type"),
        })
    return rows


def write_locations_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["term", "id", "name", "type"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = resolve_locations(QUERIES)
    write_locations_csv(rows, OUTPUT_CSV)
    for r in rows:
        print(f"[picked] {r['term']:<10} → {r['name']} [{r['type']}] id={r['id']}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()