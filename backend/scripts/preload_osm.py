"""Prefetch a Geofabrik OSM extract so road density never needs Overpass.

    uv run scripts/preload_osm.py mumbai            # a preset
    uv run scripts/preload_osm.py 72.70,18.85,73.30,19.35
    uv run scripts/preload_osm.py --list            # candidate extracts + bboxes

Downloads land in backend/cache/pbf/ (git-ignored) and are reused by every
subsequent Fetch.  Any bbox no extract covers falls back to Overpass.
"""

from __future__ import annotations

import sys

from backend.config import PRESETS
from backend.osm_local import REGIONS, pick_region, preload


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    if "--list" in flags or not args:
        from backend.osm_local import region_index

        idx = region_index()
        for key, url in REGIONS.items():
            b = (idx.get(url) or {}).get("bbox")
            print(f"{key:24s} {b}  {url.rsplit('/', 1)[-1]}")
        if not args:
            print("\nusage: preload_osm.py <preset|lon0,lat0,lon1,lat1> [--list]")
        return 0

    arg = args[0]
    if arg in PRESETS:
        bbox = PRESETS[arg]["bbox"]
    else:
        bbox = [float(x) for x in arg.split(",")]
    print(f"bbox = {bbox}")
    print(f"region = {pick_region(bbox)}")
    return 0 if preload(bbox) else 1


if __name__ == "__main__":
    raise SystemExit(main())
