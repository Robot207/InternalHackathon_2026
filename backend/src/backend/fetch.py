from __future__ import annotations

import math
import time

import httpx
import numpy as np

from .config import (
    AQ_URL,
    CHUNK,
    ELEV_URL,
    OVERPASS_URLS,
    USER_AGENT,
    WX_URL,
    WX_VARIABLES,
)

HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}


def _get(url: str, params: dict, tries: int = 5, timeout: float = 60.0) -> dict:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            with httpx.Client(headers=HEADERS, timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url, params=params)
                if resp.status_code == 429:
                    time.sleep(2.0 + attempt * 2.0)
                    continue
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0 + attempt * 1.5)
    raise RuntimeError(f"request failed for {url}: {last}")


def _post(url: str, data: dict, tries: int = 2, timeout: float = 60.0) -> dict:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            with httpx.Client(headers=HEADERS, timeout=timeout, follow_redirects=True) as client:
                resp = client.post(url, data=data)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0 + attempt * 2.0)
    raise RuntimeError(f"request failed for {url}: {last}")


def _chunks(n: int, size: int = CHUNK) -> list[range]:
    return [range(i, min(i + size, n)) for i in range(0, n, size)]


def _as_list(payload) -> list[dict]:
    return payload if isinstance(payload, list) else [payload]


def fetch_air_quality(
    lats: np.ndarray,
    lons: np.ndarray,
    start_date: str,
    end_date: str,
    domain: str,
    variables: list[str],
    progress=None,
) -> np.ndarray:
    n = len(lats)
    n_var = len(variables)
    times: list[str] | None = None
    out = np.full((n, 0, n_var), np.nan, dtype=np.float64)
    for ci, rg in enumerate(_chunks(n)):
        params = {
            "latitude": ",".join(f"{lats[i]:.5f}" for i in rg),
            "longitude": ",".join(f"{lons[i]:.5f}" for i in rg),
            "hourly": ",".join(variables),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
        }
        if domain:
            params["domains"] = domain
        payload = _get(AQ_URL, params)
        results = _as_list(payload)
        if times is None:
            times = list(results[0]["hourly"]["time"])
            out = np.full((n, len(times), n_var), np.nan, dtype=np.float64)
        for local_i, res in enumerate(results):
            idx = rg.start + local_i
            hourly = res.get("hourly", {})
            for vi, var in enumerate(variables):
                vals = hourly.get(var)
                if vals is None:
                    continue
                arr = np.asarray(vals, dtype=np.float64)
                m = min(len(arr), out.shape[1])
                out[idx, :m, vi] = arr[:m]
        if progress:
            progress(0.05 + 0.35 * (ci + 1) / len(list(_chunks(n))), f"NO2 data chunk {ci + 1}")
    return out, (times or [])


def fetch_weather(
    lats: np.ndarray,
    lons: np.ndarray,
    start_date: str,
    end_date: str,
    progress=None,
) -> tuple[np.ndarray, list[str]]:
    import datetime as dt

    today = dt.datetime.now(dt.timezone.utc).date()
    sd = dt.date.fromisoformat(start_date)
    ed = dt.date.fromisoformat(end_date)
    past_days = min(92, max(1, (today - sd).days + 1))
    forecast_days = max(1, (ed - today).days + 1)

    n = len(lats)
    times: list[str] | None = None
    n_var = len(WX_VARIABLES)
    out = np.full((n, 0, n_var), np.nan, dtype=np.float64)
    for ci, rg in enumerate(_chunks(n)):
        params = {
            "latitude": ",".join(f"{lats[i]:.5f}" for i in rg),
            "longitude": ",".join(f"{lons[i]:.5f}" for i in rg),
            "hourly": ",".join(WX_VARIABLES),
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        payload = _get(WX_URL, params)
        results = _as_list(payload)
        if times is None:
            times = list(results[0]["hourly"]["time"])
            out = np.full((n, len(times), n_var), np.nan, dtype=np.float64)
        for local_i, res in enumerate(results):
            idx = rg.start + local_i
            hourly = res.get("hourly", {})
            for vi, var in enumerate(WX_VARIABLES):
                vals = hourly.get(var)
                if vals is None:
                    continue
                arr = np.asarray(vals, dtype=np.float64)
                m = min(len(arr), out.shape[1])
                out[idx, :m, vi] = arr[:m]
        if progress:
            progress(0.4 + 0.25 * (ci + 1) / len(list(_chunks(n))), f"meteorology chunk {ci + 1}")

    assert times is not None
    want = []
    for i, t in enumerate(times):
        d = t[:10]
        if start_date <= d <= end_date:
            want.append(i)
    if not want:
        raise RuntimeError("weather API returned no overlapping timestamps")
    out = out[:, want, :]
    sel_times = [times[i] for i in want]
    return out, sel_times


def fetch_elevation(lats: np.ndarray, lons: np.ndarray, progress=None) -> np.ndarray:
    n = len(lats)
    out = np.full(n, np.nan, dtype=np.float64)
    chunks = _chunks(n, 100)
    for ci, rg in enumerate(chunks):
        params = {
            "latitude": ",".join(f"{lats[i]:.5f}" for i in rg),
            "longitude": ",".join(f"{lons[i]:.5f}" for i in rg),
        }
        payload = _get(ELEV_URL, params)
        elev = np.asarray(payload.get("elevation", []), dtype=np.float64)
        m = min(len(elev), len(rg))
        out[rg.start : rg.start + m] = elev[:m]
        if progress:
            progress(0.66 + 0.04 * (ci + 1) / len(chunks), f"elevation chunk {ci + 1}")
    if progress:
        progress(0.7, "elevation")
    return out


def fetch_road_density(
    bbox: list[float],
    lats: np.ndarray,
    lons: np.ndarray,
    step: float,
    cache_key: str,
    progress=None,
) -> tuple[np.ndarray, bool]:
    from .config import ROADS_CACHE

    cache_file = ROADS_CACHE / f"{cache_key}.npz"
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=False)
        density = data["density"]
        if density.shape == (len(lats), len(lons)):
            if progress:
                progress(0.78, "road density (cached)")
            return density.astype(np.float64), bool(data["available"])
    density, ok, n_ok = _fetch_roads_overpass(bbox, lats, lons, step, progress)
    if progress:
        progress(0.78, "road density" if ok else f"road density ({n_ok}/4 quadrants)")
    if n_ok > 0:
        np.savez_compressed(cache_file, density=density, available=np.bool_(ok))
    return density, ok


ROAD_WEIGHTS = {
    "motorway": 4.0,
    "trunk": 3.0,
    "primary": 2.0,
    "secondary": 1.5,
}


def _fetch_roads_overpass(
    bbox: list[float],
    lats: np.ndarray,
    lons: np.ndarray,
    step: float,
    progress=None,
) -> tuple[np.ndarray, bool, int]:
    h = len(lats)
    w = len(lons)
    lat_min, lon_min, lat_max, lon_max = bbox
    query = (
        "[out:json][timeout:8];"
        'way["highway"~"^(motorway|trunk|primary|secondary)$"]'
        f"({lat_min},{lon_min},{lat_max},{lon_max});"
        "out center;"
    )
    payload = None
    if progress:
        progress(0.74, "fetching OSM road network")
    for url in OVERPASS_URLS:
        try:
            payload = _post(url, {"data": query}, tries=1, timeout=8.0)
            if payload and "elements" in payload:
                break
        except Exception:
            continue

    density = np.zeros((h, w), dtype=np.float64)
    if payload and "elements" in payload and len(payload["elements"]) > 0:
        lat_min_g = lats[0] - step / 2
        lon_min_g = lons[0] - step / 2
        for el in payload.get("elements", []):
            center = el.get("center")
            tags = el.get("tags") or {}
            if not center:
                continue
            weight = ROAD_WEIGHTS.get(tags.get("highway", ""), 1.0)
            ri = int(np.clip((center["lat"] - lat_min_g) / step, 0, h - 1))
            ci = int(np.clip((center["lon"] - lon_min_g) / step, 0, w - 1))
            density[ri, ci] += weight
        cell_area = (step * 111.32) * (step * 111.32 * np.cos(np.radians(lats))).reshape(-1, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            density = np.where(cell_area > 0, density / cell_area, 0.0)
        return density, True, 1

    # Fast fallback: High-resolution urban density proxy (inverse exponential distance to center)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    c_lat, c_lon = (lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0
    dist_sq = ((LA - c_lat) * 111.0) ** 2 + ((LO - c_lon) * 111.0 * np.cos(np.radians(c_lat))) ** 2
    proxy_density = 4.5 * np.exp(-dist_sq / 120.0) + 0.8
    return proxy_density, False, 0


def haversine_local(lat1, lon1, lat2, lon2):
    from .grids import haversine_km

    return haversine_km(lat1, lon1, lat2, lon2)


def wind_components(speed_kmh: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spd = speed_kmh / 3.6
    rad = np.radians(direction_deg)
    u = -spd * np.sin(rad)
    v = -spd * np.cos(rad)
    return u, v


def nearest_city_distance(lats: np.ndarray, lons: np.ndarray, cities: dict) -> np.ndarray:
    lat_arr = np.asarray([c[0] for c in cities.values()], dtype=np.float64)
    lon_arr = np.asarray([c[1] for c in cities.values()], dtype=np.float64)
    la = lats.reshape(-1, 1)
    lo = lons.reshape(-1, 1)
    from .grids import haversine_km

    d = haversine_km(la, lo, lat_arr.reshape(1, -1), lon_arr.reshape(1, -1))
    return d.min(axis=1).reshape(lats.shape)
