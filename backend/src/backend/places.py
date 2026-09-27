"""Hyperlocal place naming for the point inspector — in **every** city.

The inspector used to answer "where am I?" from a Mumbai-only table, so a
click in Delhi / Nagpur / London reported a Mumbai landmark thousands of
kilometres away.  :func:`resolve_place` answers with the first source that
works:

1. **Curated POIs** — the team-hub landmarks and CPCB stations — whenever one
   is within :data:`CURATED_RADIUS_KM` of the click.  Exact, offline, and it
   keeps the TSEC / Khar / Santacruz demo spots front and centre.
2. **OpenStreetMap reverse geocoding** through the keyless Nominatim public
   API (~0.7 KiB and ~1 s per answer at ``zoom=16``, which resolves suburb /
   neighbourhood level).  This is what makes the inspector work for all 129
   registry cities.  Answers are cached per 0.01° cell on disk
   (``cache/geocode/``) and in memory, and calls are throttled to Nominatim's
   published 1 request/second usage policy.
3. **Nearest registered city centre** — fully offline fallback so the panel
   still shows something sensible when the network is down.

Failures never raise: the endpoint degrades to the next source.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from .config import CACHE_DIR, USER_AGENT

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
GEOCODE_DIR: Path = CACHE_DIR / "geocode"
GEOCODE_DIR.mkdir(parents=True, exist_ok=True)

MIN_INTERVAL_S = 1.05  # Nominatim public usage policy: max 1 request/second
REQUEST_TIMEOUT_S = 6.0
COOLDOWN_S = 60.0  # stay quiet for a minute after a failed call
CURATED_RADIUS_KM = 2.0  # a curated POI this close wins over the geocoder
CURATED_FALLBACK_KM = 30.0  # offline-fallback ceiling for curated POIs
# Presets that are hyperlocal zoom levels rather than cities — naming one as
# the "nearest city" would read "Near Mumbai Suburbs (Bandra / Khar / BKC)".
NON_CITY_SLUGS = {"mumbai_suburbs"}

# Address keys in priority order — "Khar" (suburb) reads better than an admin
# ward, and cities without a suburb simply fall through to the next key.
LOCALITY_KEYS: tuple[tuple[str, str], ...] = (
    ("suburb", "Suburb"),
    ("neighbourhood", "Neighbourhood"),
    ("quarter", "Quarter"),
    ("borough", "Borough"),
    ("city_district", "City district"),
    ("town", "Town"),
    ("village", "Village"),
    ("city", "City"),
    ("county", "County"),
    ("state", "State"),
)
CITY_KEYS = ("city", "town", "village", "municipality", "county", "state")

_lock = threading.Lock()
_last_call = 0.0
_fail_until = 0.0
_mem: dict[str, dict] = {}


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _nearest_curated(lat: float, lon: float) -> tuple[dict | None, float]:
    """Closest built-in POI (Mumbai landmarks + CPCB stations), if any."""
    from .stations import MUMBAI_LANDMARKS, MUMBAI_STATIONS

    best: dict | None = None
    best_km = float("inf")
    for p in (*MUMBAI_LANDMARKS, *MUMBAI_STATIONS):
        d = _km(lat, lon, float(p["lat"]), float(p["lon"]))
        if d < best_km:
            best_km = d
            best = p
    if best is None:
        return None, float("inf")
    return {
        "name": best["name"],
        "type": str(best.get("category") or best.get("type") or "Landmark"),
        "source": "Curated POI",
    }, best_km


def _cache_path(lat: float, lon: float) -> Path:
    # One file per 0.01° cell — the same resolution as the output grid.
    return GEOCODE_DIR / f"{lat:.2f}_{lon:.2f}.json"


def _build_place(data: dict, lat: float, lon: float) -> dict | None:
    """Turn a Nominatim response into the inspector's place record."""
    addr = data.get("address") or {}
    key = next((k for k, _ in LOCALITY_KEYS if addr.get(k)), None)
    locality = str(addr[key]) if key else ""
    city = next((str(addr[k]) for k in CITY_KEYS if addr.get(k)), "")

    if locality:
        name = f"{locality}, {city}" if city and city.lower() != locality.lower() else locality
        kind = dict(LOCALITY_KEYS)[key]
    else:
        name = str(data.get("display_name") or "").split(",")[0].strip()
        kind = str(data.get("type") or "Place").replace("_", " ").title()
    if not name:
        return None

    # 0 km when the click is inside the matched area's bbox (the bbox is the
    # area's own extent, so "inside" really means "you are in this locality").
    dist = 0.0
    bb = data.get("boundingbox")
    try:
        if bb and len(bb) == 4:
            s, n, w, e = (float(v) for v in bb)
            if not (s <= lat <= n and w <= lon <= e):
                dist = _km(lat, lon, float(data["lat"]), float(data["lon"]))
        else:
            dist = _km(lat, lon, float(data["lat"]), float(data["lon"]))
    except (KeyError, TypeError, ValueError):
        dist = 0.0

    return {
        "name": name,
        "type": kind,
        "distance_km": round(dist, 2),
        "source": "OpenStreetMap (Nominatim)",
    }


def _nominatim(lat: float, lon: float) -> dict | None:
    """Reverse-geocode one point (cached, throttled). Returns None on failure."""
    global _last_call, _fail_until

    key = f"{lat:.2f}_{lon:.2f}"
    if key in _mem:
        return _mem[key]
    path = _cache_path(lat, lon)
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            _mem[key] = cached
            return cached
        except (OSError, ValueError):
            pass

    if time.monotonic() < _fail_until:  # recent failure — skip the slow timeout
        return None

    with _lock:
        wait = _last_call + MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            with httpx.Client(
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT_S,
                follow_redirects=True,
            ) as client:
                resp = client.get(
                    NOMINATIM_URL,
                    params={
                        "lat": lat,
                        "lon": lon,
                        "format": "jsonv2",
                        "zoom": 16,
                        "addressdetails": 1,
                        "accept-language": "en",
                    },
                )
            if resp.status_code != 200:
                # 404 = "unable to geocode" (sea, empty desert) is a normal
                # answer, not an outage — only 429/5xx arm the cooldown.
                if resp.status_code == 429 or resp.status_code >= 500:
                    _fail_until = time.monotonic() + COOLDOWN_S
                return None
            place = _build_place(resp.json(), lat, lon)
        except Exception:  # noqa: BLE001 - offline must never break the inspector
            _fail_until = time.monotonic() + COOLDOWN_S
            return None

    if place is None:
        return None
    _mem[key] = place
    try:
        path.write_text(json.dumps(place, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # cache is best-effort
    return place


def _nearest_city_centre(lat: float, lon: float) -> dict | None:
    """Offline fallback: nearest registered city centre (129-city registry)."""
    from .cities import CITY_BBOXES
    from .config import PRESETS

    best_name, best_km = "", float("inf")
    seen: set[str] = set()
    for src in (PRESETS, CITY_BBOXES):
        for slug, cfg in src.items():
            if slug in seen or slug in NON_CITY_SLUGS:
                continue
            seen.add(slug)
            centre = cfg.get("center")
            if not centre and all(k in cfg for k in ("min_lat", "min_lon", "max_lat", "max_lon")):
                centre = [
                    (float(cfg["min_lat"]) + float(cfg["max_lat"])) / 2,
                    (float(cfg["min_lon"]) + float(cfg["max_lon"])) / 2,
                ]
            if not centre:
                continue
            d = _km(lat, lon, float(centre[0]), float(centre[1]))
            if d < best_km:
                best_km = d
                best_name = str(cfg.get("label") or slug)
    if not best_name:
        return None
    return {
        "name": f"Near {best_name}",
        "type": "City centre",
        "distance_km": round(best_km, 1),
        "source": "Offline city registry",
    }


def resolve_place(lat: float, lon: float) -> dict[str, Any]:
    """Best-effort hyperlocal place record for a map click (never raises)."""
    curated, curated_km = _nearest_curated(lat, lon)
    if curated is not None and curated_km <= CURATED_RADIUS_KM:
        return {**curated, "distance_km": round(curated_km, 2)}

    geocoded = _nominatim(lat, lon)
    if geocoded:
        return geocoded

    if curated is not None and curated_km <= CURATED_FALLBACK_KM:
        return {**curated, "distance_km": round(curated_km, 2)}

    return _nearest_city_centre(lat, lon) or {
        "name": "Local Cell",
        "type": "Urban",
        "distance_km": 0.0,
        "source": "Grid cell",
    }
