from __future__ import annotations

import math
import random
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


def _backoff(attempt: int, cap: float = 25.0) -> float:
    """Exponential backoff with jitter. 1s, 2s, 4s, 8s, 16s, 25s..."""
    base = min(cap, 1.0 * (2**attempt))
    return base + random.uniform(0.0, 0.4 * base)


def _retry_after(resp: "httpx.Response", attempt: int) -> float:
    """Honour an explicit Retry-After header when the server sends one."""
    raw = resp.headers.get("Retry-After")
    if raw:
        try:
            return min(120.0, max(1.0, float(raw)))
        except ValueError:
            pass
    return _backoff(attempt)


def _limit_reason(resp: "httpx.Response") -> str:
    """Extract the provider's own explanation from a 429/5xx body."""
    try:
        body = resp.json()
        if isinstance(body, dict):
            reason = body.get("reason") or body.get("error")
            if isinstance(reason, str) and reason:
                return reason
    except Exception:  # noqa: BLE001
        pass
    return resp.reason_phrase or "rate limited"


def _get(url: str, params: dict, tries: int = 7, timeout: float = 60.0) -> dict:
    """GET with exponential backoff.

    At 0.01 deg the fine grid is split into many chunks, so Open-Meteo/Overpass
    will rate limit (HTTP 429) part-way through a fetch. Back off and retry
    instead of failing the whole dataset build. When the provider states an
    hourly quota, retrying inside the same hour cannot help, so fail fast with
    that reason instead of burning the remaining tries.
    """
    last: Exception | None = None
    for attempt in range(tries):
        try:
            with httpx.Client(headers=HEADERS, timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url, params=params)
                if resp.status_code == 429 or resp.status_code >= 500:
                    detail = _limit_reason(resp)
                    last = RuntimeError(f"HTTP {resp.status_code} for {url}: {detail}")
                    if resp.status_code == 429 and "next hour" in detail:
                        raise last  # quota window exceeded; retrying is pointless
                    time.sleep(_retry_after(resp, attempt))
                    continue
                resp.raise_for_status()
                return resp.json()
        except RuntimeError as exc:
            if exc is last and "next hour" in str(exc):
                raise
            last = exc
            if attempt < tries - 1:
                time.sleep(_backoff(attempt))
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < tries - 1:
                time.sleep(_backoff(attempt))
    raise RuntimeError(f"request failed for {url}: {last}")


def _post(url: str, data: dict, tries: int = 5, timeout: float = 60.0) -> dict:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            with httpx.Client(headers=HEADERS, timeout=timeout, follow_redirects=True) as client:
                resp = client.post(url, data=data)
                if resp.status_code == 429 or resp.status_code >= 500:
                    last = RuntimeError(f"HTTP {resp.status_code} for {url}")
                    time.sleep(_retry_after(resp, attempt))
                    continue
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < tries - 1:
                time.sleep(_backoff(attempt))
    raise RuntimeError(f"request failed for {url}: {last}")


def _chunks(n: int, size: int = CHUNK) -> list[range]:
    return [range(i, min(i + size, n)) for i in range(0, n, size)]


# Small pause between chunks: at 0.01 deg a fetch is dozens of sequential
# requests, and the upstream API rate limits bursts.
CHUNK_PAUSE = 0.2


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
        if ci + 1 < len(list(_chunks(n))):
            time.sleep(CHUNK_PAUSE)
    # `out` is filled as (point, time, var); every consumer reads time-major
    # (T, h, w), so transpose here rather than at each call site.
    return np.ascontiguousarray(out.transpose(1, 0, 2)), (times or [])


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
        if ci + 1 < len(list(_chunks(n))):
            time.sleep(CHUNK_PAUSE)

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
    # Same contract as fetch_air_quality: fill (point, time, var), return time-major.
    return np.ascontiguousarray(out.transpose(1, 0, 2)), sel_times


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
        if ci + 1 < len(chunks):
            time.sleep(CHUNK_PAUSE)
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
) -> tuple[np.ndarray, bool, str]:
    """road_density grid → (density, available, source).

    Preference order: a good cached grid, then a *local* Geofabrik extract read
    with DuckDB (osm_local), then whatever a previous run cached, then Overpass
    itself.  The mirrors time out (45-50 s / HTTP 504) and issue no API key, so
    the local extract is what keeps this feature alive.
    """
    from .config import ROADS_CACHE
    from . import osm_local

    cache_file = ROADS_CACHE / f"{cache_key}.npz"
    cached: np.ndarray | None = None
    cached_ok = False
    if cache_file.exists():
        # Close before np.savez_compressed below: an open NpzFile keeps the zip
        # locked and the overwrite fails outright on Windows.
        with np.load(cache_file, allow_pickle=True) as data:
            if data["density"].shape == (len(lats), len(lons)):
                cached = data["density"]
                cached_ok = bool(data["available"])
                if cached_ok:
                    if progress:
                        progress(0.78, "road density (cached)")
                    return cached.astype(np.float64), True, "cached"

    local = osm_local.road_density(bbox, lats, lons, step, progress)
    if local is not None:
        density, ok = local
        np.savez_compressed(cache_file, density=density, available=np.bool_(ok))
        if progress:
            progress(0.78, "road density (local OSM extract)" if ok else "road density (local extract empty)")
        return density, ok, "OpenStreetMap Geofabrik extract (ODbL, local)"

    if cached is not None:
        if progress:
            progress(0.78, "road density (previous run, degraded)")
        return cached.astype(np.float64), cached_ok, "OpenStreetMap Overpass (ODbL, cached)"

    density, ok, n_ok = _fetch_roads_overpass(bbox, lats, lons, step, progress)
    if progress:
        progress(0.78, "road density" if ok else f"road density ({n_ok}/4 quadrants)")
    if n_ok > 0:
        np.savez_compressed(cache_file, density=density, available=np.bool_(ok))
    return density, ok, "OpenStreetMap Overpass (ODbL)"


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
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_mid = (lat_min + lat_max) / 2
    lon_mid = (lon_min + lon_max) / 2
    quadrants = [
        (lat_min, lon_min, lat_mid, lon_mid),
        (lat_min, lon_mid, lat_mid, lon_max),
        (lat_mid, lon_min, lat_max, lon_mid),
        (lat_mid, lon_mid, lat_max, lon_max),
    ]
    h = len(lats)
    w = len(lons)
    density = np.zeros((h, w), dtype=np.float64)
    lat_min_g, lat_max_g = lats[0] - step / 2, lats[-1] + step / 2
    lon_min_g, lon_max_g = lons[0] - step / 2, lons[-1] + step / 2
    ok_quadrants = 0
    for qi, q in enumerate(quadrants):
        if progress:
            progress(0.71 + 0.06 * qi / len(quadrants), f"road density (OSM {qi + 1}/4)")
        query = (
            "[out:json][timeout:40];"
            'way["highway"~"^(motorway|trunk|primary|secondary)$"]'
            f"({q[0]},{q[1]},{q[2]},{q[3]});"
            "out center;"
        )
        payload = None
        for url in OVERPASS_URLS:
            try:
                payload = _post(url, {"data": query}, tries=1, timeout=25.0)
                break
            except Exception:  # noqa: BLE001
                continue
        if payload is None:
            continue
        ok_quadrants += 1
        for el in payload.get("elements", []):
            center = el.get("center")
            tags = el.get("tags") or {}
            if not center:
                continue
            weight = ROAD_WEIGHTS.get(tags.get("highway", ""), 1.0)
            ri = int(np.clip((center["lat"] - lat_min_g) / step, 0, h - 1))
            ci = int(np.clip((center["lon"] - lon_min_g) / step, 0, w - 1))
            density[ri, ci] += weight
    if ok_quadrants == 0:
        return np.zeros((h, w), dtype=np.float64), False, 0
    if ok_quadrants < len(quadrants):
        return density, False, ok_quadrants
    cell_area = (step * 111.32) * (step * 111.32 * np.cos(np.radians(lats))).reshape(-1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(cell_area > 0, density / cell_area, 0.0)
    return density, True, ok_quadrants


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
