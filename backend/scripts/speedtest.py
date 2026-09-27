"""Rough latency/size probe for every upstream the pipeline talks to.

Usage:  uv run scripts/_speedtest.py [mumbai|london]
Deliberately small: 1-2 calls per endpoint, Overpass mirrors in configured order.
"""

from __future__ import annotations

import sys
import time

import httpx

from backend.config import OVERPASS_URLS, USER_AGENT, WX_VARIABLES

HDRS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

BBOX = {
    # lon_min, lat_min, lon_max, lat_max
    "mumbai": [72.70, 18.85, 73.30, 19.35],
    "london": [-0.51, 51.28, 0.33, 51.76],
}


def timed(label: str, fn) -> None:
    t0 = time.perf_counter()
    try:
        info = fn()
    except Exception as exc:  # noqa: BLE001
        print(f"{label:46s} FAIL  {time.perf_counter() - t0:6.2f}s  {exc}")
        return
    dt = time.perf_counter() - t0
    print(f"{label:46s} {dt:6.2f}s  {info}")


def probe_points() -> tuple[list[float], list[float]]:
    """40 points on a 0.05 deg lattice (one AQ/WX chunk)."""
    lon0, lat0, lon1, lat1 = BBOX[ARGS]
    lats, lons = [], []
    for j in range(5):
        for i in range(8):
            lats.append(lat0 + j * 0.05)
            lons.append(lon0 + i * 0.05)
    return lats, lons


ARGS = sys.argv[1] if len(sys.argv) > 1 else "mumbai"
LATS, LONS = probe_points()
DAYS = 7  # 168 hourly steps, i.e. the full default date window
START, END = "2026-09-20", "2026-09-26"


def _hourly(payload) -> dict:
    # One location -> dict; many locations -> list of dicts.
    if isinstance(payload, list):
        payload = payload[0]
    return payload.get("hourly", {}) or {}


def aq(domain: str) -> str:
    r = httpx.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": ",".join(f"{x:.5f}" for x in LATS),
            "longitude": ",".join(f"{x:.5f}" for x in LONS),
            "hourly": "nitrogen_dioxide",
            "start_date": START,
            "end_date": END,
            "domains": domain,
        },
        headers=HDRS,
        timeout=60,
    )
    r.raise_for_status()
    return f"HTTP {r.status_code}  {len(r.content) / 1024:7.1f} KiB  {len(_hourly(r.json()).get('time', []))} steps"


def wx() -> str:
    r = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": ",".join(f"{x:.5f}" for x in LATS),
            "longitude": ",".join(f"{x:.5f}" for x in LONS),
            "hourly": ",".join(WX_VARIABLES),
            "start_date": START,
            "end_date": END,
        },
        headers=HDRS,
        timeout=60,
    )
    r.raise_for_status()
    return f"HTTP {r.status_code}  {len(r.content) / 1024:7.1f} KiB  {len(_hourly(r.json()).get('time', []))} steps"


def elev() -> str:
    n = 100
    r = httpx.get(
        "https://api.open-meteo.com/v1/elevation",
        params={
            "latitude": ",".join(f"{LATS[i % len(LATS)]:.5f}" for i in range(n)),
            "longitude": ",".join(f"{LONS[i % len(LONS)]:.5f}" for i in range(n)),
        },
        headers=HDRS,
        timeout=60,
    )
    r.raise_for_status()
    return f"HTTP {r.status_code}  {len(r.content) / 1024:7.1f} KiB  {len(r.json().get('elevation', []))} pts"


def overpass(url: str, quadrant: tuple[float, float, float, float]) -> str:
    query = (
        "[out:json][timeout:40];"
        'way["highway"~"^(motorway|trunk|primary|secondary)$"]'
        f"({quadrant[0]},{quadrant[1]},{quadrant[2]},{quadrant[3]});"
        "out center;"
    )
    r = httpx.post(url, data={"data": query}, headers=HDRS, timeout=45)
    r.raise_for_status()
    n = len(r.json().get("elements", []))
    return f"HTTP {r.status_code}  {len(r.content) / 1024:7.1f} KiB  {n} ways"


def main() -> None:
    lon0, lat0, lon1, lat1 = BBOX[ARGS]
    q = ((lat0 + lat1) / 2, lon0, lat1, (lon0 + lon1) / 2)  # one real quadrant
    print(f"== {ARGS} bbox={BBOX[ARGS]}  window={START}..{END} ({DAYS} days, 40 pts) ==")
    timed("open-meteo air-quality cams_global (NO2)", lambda: aq("cams_global"))
    timed("open-meteo air-quality cams_europe (NO2)", lambda: aq("cams_europe"))
    timed("open-meteo weather (7 hourly vars)", wx)
    timed("open-meteo elevation (100 pts)", elev)
    if "--no-overpass" in sys.argv:
        return
    for url in OVERPASS_URLS:
        host = url.split("/")[2]
        timed(f"overpass {host:28s} quadrant", lambda u=url: overpass(u, q))


if __name__ == "__main__":
    main()
