from __future__ import annotations

import datetime as dt

import numpy as np

from .config import CITIES
from .fetch import nearest_city_distance
from .grids import neighbor_stats, upsample
from .inversion import vcd_to_surface_concentration

STATIC_FEATURES = ["elevation", "road_density", "lon_norm", "lat_norm", "dist_city_km"]
TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
MET_FEATURES = ["temp_c", "wind_u", "wind_v", "wind_speed", "humidity", "blh_m", "precip_mm", "cloud_cover"]
COARSE_FEATURES = [
    "coarse_no2",
    "coarse_nb_mean",
    "coarse_nb_std",
    "coarse_nb_min",
    "coarse_nb_max",
    "cloud_gap_frac",
]
PHYSICS_FEATURES = ["vcd_surface_inv", "ventilation_coeff"]
FEATURE_NAMES = COARSE_FEATURES + MET_FEATURES + STATIC_FEATURES + TIME_FEATURES + PHYSICS_FEATURES


def build_features(data: dict) -> dict:
    coarse_filled = np.asarray(data["coarse_filled"], dtype=np.float64)
    gap = np.asarray(data["cloud_gap"], dtype=np.float64)
    elev = np.asarray(data["elev"], dtype=np.float64)
    roads = np.asarray(data["roads"], dtype=np.float64)
    lats = np.asarray(data["lats"], dtype=np.float64)
    lons = np.asarray(data["lons"], dtype=np.float64)
    times = [str(t) for t in np.asarray(data["times"]).tolist()]
    grid = _grid_from(data)

    c_up = upsample(coarse_filled, grid)
    nb = neighbor_stats(coarse_filled)
    nb_up = {k: upsample(v, grid) for k, v in nb.items()}
    gap_up = upsample(gap, grid)

    wx = {k: np.asarray(data[k], dtype=np.float64) for k in ("temp", "u", "v", "spd", "rh", "cloud", "precip", "blh")}

    t_len = c_up.shape[0]
    h, w = elev.shape
    lat_n = ((lats - lats.min()) / max(lats.max() - lats.min(), 1e-9)).reshape(1, h, 1)
    lon_n = ((lons - lons.min()) / max(lons.max() - lons.min(), 1e-9)).reshape(1, 1, w)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    dist = nearest_city_distance(LA.reshape(-1), LO.reshape(-1), CITIES).reshape(h, w)
    elev_b = np.broadcast_to(elev.reshape(1, h, w), (t_len, h, w))
    roads_b = np.broadcast_to(roads.reshape(1, h, w), (t_len, h, w))
    dist_b = np.broadcast_to(dist.reshape(1, h, w), (t_len, h, w))
    lat_b = np.broadcast_to(lat_n, (t_len, h, w))
    lon_b = np.broadcast_to(lon_n, (t_len, h, w))

    hours, dows = _time_parts(times)
    hour_sin = np.broadcast_to(np.sin(2 * np.pi * hours / 24.0).reshape(t_len, 1, 1), (t_len, h, w))
    hour_cos = np.broadcast_to(np.cos(2 * np.pi * hours / 24.0).reshape(t_len, 1, 1), (t_len, h, w))
    dow_sin = np.broadcast_to(np.sin(2 * np.pi * dows / 7.0).reshape(t_len, 1, 1), (t_len, h, w))
    dow_cos = np.broadcast_to(np.cos(2 * np.pi * dows / 7.0).reshape(t_len, 1, 1), (t_len, h, w))

    # Atmospheric physics features
    vcd_proxy = (c_up / 46.0055) * (np.maximum(wx["blh"], 80.0) / 1000.0)
    vcd_surface = vcd_to_surface_concentration(vcd_proxy, wx["blh"], wx["temp"], elev_b)
    ventilation = (wx["spd"] * np.maximum(wx["blh"], 80.0)) / 1000.0

    stack = [
        c_up,
        nb_up["mean"],
        nb_up["std"],
        nb_up["min"],
        nb_up["max"],
        gap_up,
        wx["temp"],
        wx["u"],
        wx["v"],
        wx["spd"],
        wx["rh"],
        wx["blh"],
        wx["precip"],
        wx["cloud"],
        elev_b,
        roads_b,
        lon_b,
        lat_b,
        dist_b,
        hour_sin,
        hour_cos,
        dow_sin,
        dow_cos,
        vcd_surface,
        ventilation,
    ]
    X = np.stack(stack, axis=-1).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = 0.0

    ref = data.get("ref")
    y = np.full((t_len, h, w), np.nan, dtype=np.float64)
    if ref is not None:
        ref_arr = np.asarray(ref, dtype=np.float64)
        if ref_arr.shape == (t_len, h, w):
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.log(np.where(ref_arr > 0.5, ref_arr, np.nan)) - np.log(
                    np.where(c_up > 0.5, c_up, np.nan)
                )
    y_target_valid = np.isfinite(y)
    pred_valid = np.isfinite(c_up) & (c_up > 0.5)

    return {
        "X": X,
        "names": list(FEATURE_NAMES),
        "y": y.astype(np.float32),
        "y_valid": y_target_valid,
        "pred_valid": pred_valid,
        "c_up": c_up.astype(np.float32),
        "gap_up": gap_up.astype(np.float32),
        "times": times,
        "lats": lats,
        "lons": lons,
        "grid": grid,
        "t_len": t_len,
        "H": h,
        "W": w,
    }


def _grid_from(data: dict) -> dict:
    lats = np.asarray(data["lats"], dtype=np.float64)
    lons = np.asarray(data["lons"], dtype=np.float64)
    return {
        "lats": lats,
        "lons": lons,
        "clats": np.asarray(data["clats"], dtype=np.float64),
        "clons": np.asarray(data["clons"], dtype=np.float64),
        "Hc": len(data["clats"]),
        "Wc": len(data["clons"]),
        "block_idx": np.asarray(data["block_idx"]),
        "n_blocks": len(data["clats"]) * len(data["clons"]),
    }


def _time_parts(times: list[str]) -> tuple[np.ndarray, np.ndarray]:
    hours = np.zeros(len(times), dtype=np.float64)
    dows = np.zeros(len(times), dtype=np.float64)
    for i, t in enumerate(times):
        stamp = dt.datetime.fromisoformat(t.replace("Z", "+00:00"))
        hours[i] = stamp.hour
        dows[i] = stamp.weekday()
    return hours, dows
