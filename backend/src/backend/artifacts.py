from __future__ import annotations

import json

import numpy as np

from .config import ARTIFACT_DIR
from .grids import block_upsample, make_grid, upsample

LAYER_NAMES = ["prediction", "reference", "coarse", "coarse_bilinear", "residual", "cloud_gap"]
STATIC_NAMES = ["elevation", "road_density"]


def latest_dir() -> "object":
    return ARTIFACT_DIR / "latest"


def has_predictions() -> bool:
    return (latest_dir() / "predictions.npz").exists()


def has_model() -> bool:
    return (latest_dir() / "model.joblib").exists()


def read_meta() -> dict | None:
    path = latest_dir() / "meta.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def write_meta(meta: dict) -> None:
    latest_dir().mkdir(parents=True, exist_ok=True)
    (latest_dir() / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_layers(summary: dict) -> dict:
    data = np.load(latest_dir() / "predictions.npz", allow_pickle=False)
    pred = data["pred"].astype(np.float64)
    ref = data["ref"].astype(np.float64)
    c_up = data["c_up"].astype(np.float64)
    gap = data["gap_up"].astype(np.float64)
    if ref.shape != pred.shape:
        ref = np.full_like(pred, np.nan)
    residual = pred - ref

    grid = make_grid(summary["bbox"], float(summary["fine_step"]))
    c_blocks = block_upsample(_centers(c_up, grid), grid)
    gap_blocks = block_upsample(_centers(gap, grid), grid)

    raw = {
        "prediction": pred,
        "reference": ref,
        "coarse": c_blocks,
        "coarse_bilinear": c_up,
        "residual": residual,
        "cloud_gap": gap_blocks,
    }
    ranges = {}
    for name, arr in raw.items():
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            ranges[name] = None
            continue
        ranges[name] = [
            round(float(np.percentile(finite, 2)), 3),
            round(float(np.percentile(finite, 98)), 3),
        ]
    layers = {name: _round(arr) for name, arr in raw.items()}
    lats = np.asarray(summary["lats"], dtype=float)
    lons = np.asarray(summary["lons"], dtype=float)
    return {
        "times": summary["times"],
        "lats": lats.tolist(),
        "lons": lons.tolist(),
        "t_len": pred.shape[0],
        "layers": layers,
        "ranges": ranges,
        "static": {
            "elevation": _round(_static_elevation(summary)),
            "road_density": _round(_static_roads(summary), 3),
        },
    }


def _centers(field: np.ndarray, grid: dict) -> np.ndarray:
    """Sample a fine-grid (bilinear) field at coarse-cell centers (exact block values)."""
    iy = np.abs(grid["lats"][:, None] - grid["clats"][None, :]).argmin(axis=0)
    ix = np.abs(grid["lons"][:, None] - grid["clons"][None, :]).argmin(axis=0)
    return field[:, iy][:, :, ix]


def _static_elevation(summary: dict) -> np.ndarray:
    data = np.load(latest_dir() / "predictions.npz", allow_pickle=False)
    if "elev" in data.files:
        return data["elev"].astype(np.float64)
    return np.zeros((len(summary["lats"]), len(summary["lons"])), dtype=np.float64)


def _static_roads(summary: dict) -> np.ndarray:
    data = np.load(latest_dir() / "predictions.npz", allow_pickle=False)
    if "roads" in data.files:
        return data["roads"].astype(np.float64)
    return np.zeros((len(summary["lats"]), len(summary["lons"])), dtype=np.float64)


def _round(arr: np.ndarray, digits: int = 2) -> list:
    out = np.round(arr, digits)
    out = np.where(np.isfinite(out), out, None)
    return out.tolist()


def write_layers(summary: dict) -> dict:
    layers = build_layers(summary)
    latest_dir().mkdir(parents=True, exist_ok=True)
    (latest_dir() / "layers.json").write_text(json.dumps(layers), encoding="utf-8")
    return layers


def export_netcdf(summary: dict) -> "object":
    import xarray as xr

    data = np.load(latest_dir() / "predictions.npz", allow_pickle=False)
    pred = data["pred"].astype(np.float32)
    ref = data["ref"].astype(np.float32)
    c_up = data["c_up"].astype(np.float32)
    gap = data["gap_up"].astype(np.float32)
    if ref.shape != pred.shape:
        ref = np.full_like(pred, np.nan)
    lats = np.asarray(summary["lats"], dtype=np.float64)
    lons = np.asarray(summary["lons"], dtype=np.float64)
    times = [t.replace(":00Z", ":00:00") for t in summary["times"]]
    elev = _static_elevation(summary).astype(np.float32)
    roads = _static_roads(summary).astype(np.float32)
    meta = read_meta() or {}
    ds = xr.Dataset(
        data_vars={
            "no2_predicted": (("time", "lat", "lon"), pred, {"units": "ug m-3", "long_name": "ML downscaled NO2"}),
            "no2_reference": (("time", "lat", "lon"), ref, {"units": "ug m-3", "long_name": "independent fine reference"}),
            "no2_coarse_bilinear": (("time", "lat", "lon"), c_up, {"units": "ug m-3", "long_name": "gap-filled coarse field, bilinear"}),
            "cloud_gap_fraction": (("time", "lat", "lon"), gap, {"units": "1", "long_name": "coarse cloud gap mask"}),
            "elevation": (("lat", "lon"), elev, {"units": "m"}),
            "road_density": (("lat", "lon"), roads, {"units": "km km-2"}),
        },
        coords={
            "time": np.asarray(times, dtype="datetime64[ns]"),
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "title": "NO2 satellite downscaling demonstration",
            "model": str(meta.get("model_name", "")),
            "split": str(meta.get("split", "")),
            "metrics": json.dumps(meta.get("metrics", {})),
            "sources": json.dumps(summary.get("sources", {})),
            "license": "data: Copernicus/Open-Meteo/OSM open licences",
        },
    )
    path = latest_dir() / "export.nc"
    ds.to_netcdf(path)
    return path


def layers_file() -> "object":
    return latest_dir() / "layers.json"


def get_aqi_category(val: float) -> dict[str, str]:
    from .config import AQI_NO2_BREAKPOINTS

    if not np.isfinite(val) or val < 0:
        return {"category": "Unknown", "color": "#888888", "description": "No data"}
    for low, high, label, color, desc in AQI_NO2_BREAKPOINTS:
        if low <= val <= high:
            return {"category": label, "color": color, "description": desc}
    return {
        "category": "Severe",
        "color": "#7e0023",
        "description": "Critical atmospheric pollution level",
    }


def export_geojson(summary: dict, time_idx: int = -1) -> "object":
    data = np.load(latest_dir() / "predictions.npz", allow_pickle=False)
    pred = data["pred"].astype(np.float64)
    c_up = data["c_up"].astype(np.float64)
    lats = np.asarray(summary["lats"], dtype=np.float64)
    lons = np.asarray(summary["lons"], dtype=np.float64)
    times = summary.get("times", [])

    t = (time_idx % pred.shape[0]) if pred.shape[0] > 0 else 0
    t_str = times[t] if t < len(times) else "latest"

    dlat = (lats[1] - lats[0]) / 2.0 if len(lats) > 1 else 0.025
    dlon = (lons[1] - lons[0]) / 2.0 if len(lons) > 1 else 0.025

    features = []
    for r, lat in enumerate(lats):
        for c, lon in enumerate(lons):
            p_val = pred[t, r, c]
            b_val = c_up[t, r, c]
            if not np.isfinite(p_val):
                continue
            aqi = get_aqi_category(float(p_val))
            # Polygon box for the grid cell
            poly = [
                [
                    [round(lon - dlon, 5), round(lat - dlat, 5)],
                    [round(lon + dlon, 5), round(lat - dlat, 5)],
                    [round(lon + dlon, 5), round(lat + dlat, 5)],
                    [round(lon - dlon, 5), round(lat + dlat, 5)],
                    [round(lon - dlon, 5), round(lat - dlat, 5)],
                ]
            ]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": poly},
                    "properties": {
                        "lat": round(lat, 4),
                        "lon": round(lon, 4),
                        "time": t_str,
                        "downscaled_no2": round(float(p_val), 2),
                        "baseline_no2": round(float(b_val), 2),
                        "aqi_category": aqi["category"],
                        "aqi_color": aqi["color"],
                    },
                }
            )

    fc = {
        "type": "FeatureCollection",
        "name": f"NO2_Downscaled_{summary.get('preset', 'export')}",
        "metadata": {
            "time": t_str,
            "region": summary.get("preset_label", ""),
            "units": "µg/m³",
        },
        "features": features,
    }
    path = latest_dir() / "export.geojson"
    path.write_text(json.dumps(fc, indent=2), encoding="utf-8")
    return path


def export_csv(summary: dict) -> "object":
    import pandas as pd

    data = np.load(latest_dir() / "predictions.npz", allow_pickle=False)
    pred = data["pred"].astype(np.float64)
    c_up = data["c_up"].astype(np.float64)
    lats = np.asarray(summary["lats"], dtype=np.float64)
    lons = np.asarray(summary["lons"], dtype=np.float64)
    times = summary.get("times", [])

    rows = []
    # If time series is long, export the latest 24 hours to keep file responsive
    t_start = max(0, pred.shape[0] - 24)
    for t in range(t_start, pred.shape[0]):
        t_str = times[t] if t < len(times) else f"t_{t}"
        for r, lat in enumerate(lats):
            for c, lon in enumerate(lons):
                p_val = pred[t, r, c]
                b_val = c_up[t, r, c]
                if not np.isfinite(p_val):
                    continue
                aqi = get_aqi_category(float(p_val))
                rows.append(
                    {
                        "time": t_str,
                        "lat": round(lat, 5),
                        "lon": round(lon, 5),
                        "downscaled_no2_ugm3": round(float(p_val), 2),
                        "baseline_no2_ugm3": round(float(b_val), 2),
                        "aqi_category": aqi["category"],
                    }
                )

    df = pd.DataFrame(rows)
    path = latest_dir() / "export.csv"
    df.to_csv(path, index=False)
    return path


def inspect_point(lat: float, lon: float, summary: dict) -> dict:
    from .stations import MUMBAI_LANDMARKS, MUMBAI_STATIONS

    data = np.load(latest_dir() / "predictions.npz", allow_pickle=False)
    pred = data["pred"].astype(np.float64)
    c_up = data["c_up"].astype(np.float64)
    gap = data["gap_up"].astype(np.float64)
    elev = _static_elevation(summary)
    roads = _static_roads(summary)

    lats = np.asarray(summary["lats"], dtype=float)
    lons = np.asarray(summary["lons"], dtype=float)
    times = summary.get("times", [])

    ri = int(np.argmin(np.abs(lats - lat)))
    ci = int(np.argmin(np.abs(lons - lon)))

    matched_lat = float(lats[ri])
    matched_lon = float(lons[ci])

    # Find nearest landmark or station
    all_places = MUMBAI_LANDMARKS + MUMBAI_STATIONS
    nearest_place = None
    min_dist_km = 9999.0
    for p in all_places:
        d_km = 111.0 * np.sqrt((lat - p["lat"]) ** 2 + ((lon - p["lon"]) * np.cos(np.radians(lat))) ** 2)
        if d_km < min_dist_km:
            min_dist_km = d_km
            nearest_place = p

    # Current time (latest frame)
    t_curr = pred.shape[0] - 1
    p_curr = float(pred[t_curr, ri, ci]) if np.isfinite(pred[t_curr, ri, ci]) else None
    b_curr = float(c_up[t_curr, ri, ci]) if np.isfinite(c_up[t_curr, ri, ci]) else None
    g_curr = bool(gap[t_curr, ri, ci] > 0.5)

    aqi_info = get_aqi_category(p_curr) if p_curr is not None else get_aqi_category(-1)

    # 24-hour diurnal profile
    t_window = min(24, pred.shape[0])
    series_times = times[-t_window:] if times else []
    series_downscaled = [
        round(float(v), 2) if np.isfinite(v) else None for v in pred[-t_window:, ri, ci]
    ]
    series_baseline = [
        round(float(v), 2) if np.isfinite(v) else None for v in c_up[-t_window:, ri, ci]
    ]

    elev_val = float(elev[ri, ci]) if np.isfinite(elev[ri, ci]) else 0.0
    road_val = float(roads[ri, ci]) if np.isfinite(roads[ri, ci]) else 0.0

    return {
        "query": {"lat": round(lat, 5), "lon": round(lon, 5)},
        "cell": {"lat": round(matched_lat, 5), "lon": round(matched_lon, 5), "r": ri, "c": ci},
        "nearest_landmark": {
            "name": nearest_place["name"] if nearest_place else "Local Cell",
            "distance_km": round(min_dist_km, 2),
            "type": nearest_place.get("type") or nearest_place.get("category") or "Urban",
        },
        "current": {
            "downscaled_no2": round(p_curr, 2) if p_curr is not None else None,
            "baseline_no2": round(b_curr, 2) if b_curr is not None else None,
            "cloud_gap_repaired": g_curr,
            "aqi": aqi_info,
        },
        "static_features": {
            "elevation_m": round(elev_val, 1),
            "road_density_km_km2": round(road_val, 3),
        },
        "diurnal_24h": {
            "times": series_times,
            "downscaled": series_downscaled,
            "baseline": series_baseline,
        },
    }

