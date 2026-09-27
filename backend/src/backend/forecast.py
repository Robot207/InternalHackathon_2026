"""72-hour NO2 projection driven by live, keyless Open-Meteo forecasts.

Page 2 of the UI used to carry the last analysis forward with hand-written
growth numbers (1.07 at +12 h ... 1.46 at +72 h), so the driver panel, the
stagnation index and the GRAP alert were fixed regardless of the weather.
This module replaces them with the actual forecast for the active domain:

* meteorology — a 3x3 sample over the bbox from the Open-Meteo Forecast API
  (hourly, keyless, non-commercial friendly: 10k requests/day) giving
  temperature, relative humidity, wind speed/direction, cloud, precipitation,
  boundary-layer height and surface pressure;
* an emission x dispersion model, deliberately transparent:

      C(t) = C0 * (E(t)/E0) * (D0/Dt)

  E  emission index  = 0.45 background + 0.55 traffic, where traffic follows a
     weekday/weekend diurnal profile (morning and evening rush peaks);
  D  dispersion      = 0.55*clip(wind/6 m/s) + 0.45*clip(BLH/1200 m)
                       + a rain washout term (precipitation scavenges NO2);
* a stagnation index in [0,1] and a CPCB-referenced alert rule per hour.

Only "now .. +72 h" is returned (hourly), so the five slider stops map onto
real timestamps in the city's own timezone.
"""

from __future__ import annotations

import bisect
import math
import time
from datetime import datetime, timedelta, timezone as dt_tz

import numpy as np

from .artifacts import latest_dir
from .config import PRESETS, WX_URL
from .fetch import _as_list, _get

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "precipitation",
    "boundary_layer_height",
    "surface_pressure",
]
N_HOURS = 73  # now .. +72 h inclusive
SAMPLES = 3  # 3x3 sample over the bbox (one request)

# --- model constants (documented, not fitted) -------------------------------
TRAFFIC_SHARE = 0.55  # share of NO2 emissions attributable to road traffic
WIND_FULL = 6.0  # m/s at which ventilation is considered complete
BLH_FULL = 1200.0  # m mixing height at which dilution is considered complete
RAIN_FULL = 1.0  # mm/h that washes NO2 out of the air
VENT_WEIGHT = 0.55
MIX_WEIGHT = 0.45
WASH_WEIGHT = 0.35

ALERT_STAGNATION = 0.70  # poor dispersion + humidity -> alert
ALERT_LEVEL = 80.0  # ug/m3, start of the CPCB "Moderate" band for NO2
# No lead-time clause: an earlier draft also armed the widget from T+48 h at
# stagnation 0.50, which made every horizon past +48 h read red on its own.
DEFAULT_BASELINE = 40.0  # ug/m3 fallback when no analysis exists

CACHE_TTL = 1800.0  # seconds; forecasts change slowly
_CACHE: dict[str, tuple[float, dict]] = {}


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _traffic_index(hour: int, dow: int) -> float:
    """Relative traffic emissions for a local hour (0-23) and weekday (0=Mon)."""
    am = math.exp(-0.5 * ((hour - 8.5) / 1.4) ** 2)  # morning rush
    pm = math.exp(-0.5 * ((hour - 19.0) / 1.6) ** 2)  # evening rush
    midday = 0.30 * math.exp(-0.5 * ((hour - 13.0) / 3.0) ** 2)
    idx = 0.35 + 1.00 * am + 1.15 * pm + midday
    if dow >= 5:  # Saturday / Sunday — no commuter peaks
        idx = 0.45 + 0.55 * idx
    return idx


def _sample_points(bbox: list[float]) -> tuple[list[float], list[float]]:
    lon0, lat0, lon1, lat1 = bbox
    lons = np.linspace(lon0, lon1, SAMPLES)
    lats = np.linspace(lat0, lat1, SAMPLES)
    grid = [(float(la), float(lo)) for la in lats for lo in lons]
    return [g[0] for g in grid], [g[1] for g in grid]


def _met(bbox: list[float]) -> dict:
    """Fetch `N_HOURS` of forecast met for a 3x3 sample and average it."""
    lats, lons = _sample_points(bbox)
    payload = _get(
        WX_URL,
        {
            "latitude": ",".join(f"{v:.4f}" for v in lats),
            "longitude": ",".join(f"{v:.4f}" for v in lons),
            "hourly": ",".join(HOURLY_VARS),
            "forecast_days": 5,  # 120 h of margin around now..+72 h
            "timezone": "auto",  # local timestamps + utc offset
        },
        tries=3,
        timeout=30.0,
    )
    rows = _as_list(payload)
    if not rows or not isinstance(rows[0].get("hourly"), dict):
        raise RuntimeError("Open-Meteo forecast returned no hourly data")

    hours: list[str] = list(rows[0]["hourly"]["time"])
    offset = int(rows[0].get("utc_offset_seconds") or 0)
    tzname = str(rows[0].get("timezone") or "UTC")

    def cols(name: str) -> list[np.ndarray]:
        out = []
        for r in rows:
            vals = r.get("hourly", {}).get(name)
            if vals:
                out.append(np.asarray(vals, dtype=float))
        if not out:
            raise RuntimeError(f"Open-Meteo forecast missing '{name}'")
        n = min(min(len(c) for c in out), len(hours))
        return [c[:n] for c in out]

    def series(name: str) -> np.ndarray:
        return np.nanmean(np.vstack(cols(name)), axis=0)

    # Wind is averaged as a vector across the sample (averaging degrees of
    # direction is meaningless — N and S would average to E) — and the mean is
    # taken per hour, not over the whole window, or the forecast would read as
    # one constant breeze for three days.
    ws = np.vstack(cols("wind_speed_10m"))
    wd = np.vstack(cols("wind_direction_10m"))
    rad = np.radians(wd)
    u = np.nanmean(-ws * np.sin(rad), axis=0)
    v = np.nanmean(-ws * np.cos(rad), axis=0)
    wind = np.hypot(u, v)
    wdir = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0

    n = min(len(hours), len(wind))
    hours = hours[:n]
    return {
        "hours": hours,
        "timezone": tzname,
        "utc_offset_seconds": offset,
        "temperature_c": series("temperature_2m")[:n],
        "humidity_pct": series("relative_humidity_2m")[:n],
        "wind_ms": wind[:n],
        "wind_dir_deg": wdir[:n],
        "cloud_pct": series("cloud_cover")[:n],
        "precip_mm": series("precipitation")[:n],
        "blh_m": np.nan_to_num(series("boundary_layer_height")[:n], nan=800.0),
        "pressure_hpa": series("surface_pressure")[:n],
    }


def _now_index(hours: list[str], offset: int) -> int:
    """Index of the current local hour inside the returned hourly series."""
    local = datetime.now(dt_tz.utc) + timedelta(seconds=offset)
    key = local.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M")
    idx = bisect.bisect_left(hours, key)
    if idx < len(hours) and hours[idx] == key:
        return idx
    return min(max(idx, 0), len(hours) - 1)


def _baseline(summary: dict | None, preset: str) -> tuple[float, str]:
    """Mean NO2 of the latest analysis's last frame, else a labelled default.

    The baseline has to be the mean of the very frame the UI draws at "Now",
    otherwise the projected numbers and the map would disagree — so it is read
    from the active analysis whatever city it was built for.
    """
    npz = latest_dir() / "predictions.npz"
    if summary and npz.exists():
        try:
            data = np.load(npz, allow_pickle=False)
            last = np.asarray(data["pred"], dtype=float)[-1]
            finite = last[np.isfinite(last)]
            if finite.size:
                src = f"mean of the latest analysis frame ({summary.get('preset')})"
                if summary.get("preset") != preset:
                    src += f" — mapped onto {preset}"
                return round(float(finite.mean()), 2), src
        except Exception:  # noqa: BLE001 - fall through to the default
            pass
    return DEFAULT_BASELINE, f"default for {preset} (no analysis for this city yet)"


def _project(met: dict, start: int) -> dict:
    """Run the emission x dispersion model over `hours[start:start+N_HOURS]`."""
    hours = met["hours"][start : start + N_HOURS]
    n = len(hours)
    if n == 0:
        raise RuntimeError("forecast window is empty")

    traffic = np.array(
        [
            _traffic_index(
                int(h[11:13]),
                datetime.strptime(h[:10], "%Y-%m-%d").weekday(),
            )
            for h in hours
        ]
    )
    tbar = float(np.mean(traffic)) or 1.0
    traffic_rel = traffic / tbar

    wind = np.asarray(met["wind_ms"][start : start + n], dtype=float)
    blh = np.asarray(met["blh_m"][start : start + n], dtype=float)
    rh = np.asarray(met["humidity_pct"][start : start + n], dtype=float)
    rain = np.asarray(met["precip_mm"][start : start + n], dtype=float)

    vent = np.clip(wind / WIND_FULL, 0.0, 1.0)
    mix = np.clip(blh / BLH_FULL, 0.0, 1.0)
    wash = np.clip(rain / RAIN_FULL, 0.0, 1.0)
    disp = np.minimum(1.0, VENT_WEIGHT * vent + MIX_WEIGHT * mix + WASH_WEIGHT * wash)

    emit = (1.0 - TRAFFIC_SHARE) + TRAFFIC_SHARE * traffic_rel
    emit_mit = (1.0 - TRAFFIC_SHARE) + 0.6 * TRAFFIC_SHARE * traffic_rel

    d0 = float(disp[0]) or 1e-3
    e0 = float(emit[0]) or 1.0
    factor = np.clip((emit / e0) * (d0 / np.maximum(disp, 1e-3)), 0.35, 3.0)
    factor_mit = np.clip((emit_mit / e0) * (d0 / np.maximum(disp, 1e-3)), 0.35, 3.0)

    # Stagnation: what is left after dispersion, sharpened by humidity/haze.
    stag = (1.0 - disp) * (0.85 + 0.15 * np.clip(rh, 0.0, 100.0) / 100.0)
    hazy = (rh >= 92.0) & (wind < 3.0)
    stag = np.clip(np.where(hazy, stag + 0.08, stag), 0.0, 1.0)

    return {
        "hours": hours,
        "factor": [round(float(x), 3) for x in factor],
        "factor_mitigated": [round(float(x), 3) for x in factor_mit],
        "stagnation": [round(float(x), 3) for x in stag],
        "emission_index": [round(float(x), 3) for x in emit],
        "dispersion_index": [round(float(x), 3) for x in disp],
        "traffic_index": [round(float(x), 3) for x in traffic_rel],
        "_factor": factor,
        "_stag": stag,
        "_wind": wind,
        "_blh": blh,
        "_rh": rh,
        "_rain": rain,
    }


def _alert(model: dict, projected: np.ndarray) -> tuple[list[bool], list[str]]:
    armed: list[bool] = []
    reasons: list[str] = []
    for i in range(len(projected)):
        stag = float(model["_stag"][i])
        wind = float(model["_wind"][i])
        blh = float(model["_blh"][i])
        level = float(projected[i])
        why: str | None = None
        if stag >= ALERT_STAGNATION:
            why = (
                f"stagnation {stag:.2f} — wind {wind:.1f} m/s, "
                f"mixing height {round(blh)} m"
            )
        elif level >= ALERT_LEVEL:
            why = f"projected {level:.0f} µg/m³ ≥ {ALERT_LEVEL:.0f} (CPCB Moderate band)"
        armed.append(why is not None)
        reasons.append(why or "")
    return armed, reasons


def _build(preset: str, bbox: list[float], summary: dict | None) -> dict:
    met = _met(bbox)
    start = _now_index(met["hours"], int(met["utc_offset_seconds"]))
    model = _project(met, start)
    n = len(model["hours"])

    baseline, baseline_source = _baseline(summary, preset)
    factor = np.asarray(model.pop("_factor"))
    stag = np.asarray(model.pop("_stag"))
    # private numpy scratch — never serialise it
    for k in [k for k in model if k.startswith("_")]:
        model.pop(k)
    projected = baseline * factor
    wind = _met_wind(met, start, n)
    blh = _met_blh(met, start, n)
    armed, reasons = _alert({"_stag": stag, "_wind": wind, "_blh": blh}, projected)

    sl = slice(start, start + n)
    return {
        "preset": preset,
        "bbox": bbox,
        "generated_at": datetime.now(dt_tz.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "timezone": met["timezone"],
        "utc_offset_seconds": met["utc_offset_seconds"],
        "source": "Open-Meteo Forecast (keyless, hourly)",
        "baseline_ugm3": baseline,
        "baseline_source": baseline_source,
        "horizons": [0, 12, 24, 48, 72],
        "hours": model["hours"],
        "met": {
            "temperature_c": _round(met["temperature_c"][sl]),
            "humidity_pct": _round(met["humidity_pct"][sl]),
            "wind_ms": _round(wind),
            "wind_dir_deg": _round(_met_wind_dir(met, start, n)),
            "blh_m": _round(blh),
            "cloud_pct": _round(met["cloud_pct"][sl]),
            "precip_mm": _round(met["precip_mm"][sl], 2),
            "pressure_hpa": _round(met["pressure_hpa"][sl]),
        },
        "model": {
            **model,
            "alert": armed,
            "alert_reason": reasons,
            "projected_ugm3": [round(float(x), 1) for x in projected],
            "baseline_ugm3": baseline,
        },
        "stale": False,
    }


def _met_wind(met: dict, start: int, n: int) -> np.ndarray:
    return np.asarray(met["wind_ms"][start : start + n], dtype=float)


def _met_wind_dir(met: dict, start: int, n: int) -> np.ndarray:
    return np.asarray(met["wind_dir_deg"][start : start + n], dtype=float)


def _met_blh(met: dict, start: int, n: int) -> np.ndarray:
    return np.asarray(met["blh_m"][start : start + n], dtype=float)


def _round(arr, digits: int = 1) -> list[float]:
    return [round(float(x), digits) if np.isfinite(x) else None for x in np.asarray(arr, dtype=float)]


def build(preset: str, summary: dict | None = None) -> dict:
    """Forecast payload for `preset`, cached for 30 minutes."""
    if preset not in PRESETS:
        raise KeyError(f"unknown preset '{preset}'")
    key = f"{preset}|{(summary or {}).get('key', '')}"
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < CACHE_TTL:
        return hit[1]
    bbox = list(PRESETS[preset]["bbox"])
    try:
        payload = _build(preset, bbox, summary)
    except Exception:
        if hit:  # network hiccup — serve the last good run rather than an error
            stale = dict(hit[1])
            stale["stale"] = True
            return stale
        raise
    _CACHE[key] = (now, payload)
    return payload
