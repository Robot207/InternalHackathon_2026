from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .artifacts import latest_dir, read_meta
from .config import DATASET_DIR
from .training import metrics


def _pick(cols: dict[str, str], aliases: list[str]) -> str | None:
    for a in aliases:
        if a in cols:
            return cols[a]
    return None


def validate_station_csv(content: bytes) -> dict:
    df = pd.read_csv(io.BytesIO(content))
    if df.empty:
        raise ValueError("CSV contains no rows")
    cols = {str(c).lower().strip(): c for c in df.columns}
    lon_c = _pick(cols, ["lon", "longitude", "long", "x"])
    lat_c = _pick(cols, ["lat", "latitude", "y"])
    val_c = _pick(cols, ["no2", "no2_ugm3", "no2_ug_m3", "value", "obs", "observed", "concentration"])
    time_c = _pick(cols, ["time", "datetime", "date", "timestamp"])
    missing = [n for n, c in (("lon", lon_c), ("lat", lat_c), ("no2", val_c)) if c is None]
    if missing:
        raise ValueError(
            f"missing required columns: {', '.join(missing)} (accepted: lon/lat/no2, optional time)"
        )

    meta = read_meta()
    if not meta or not meta.get("summary_key"):
        raise ValueError("no trained/predicted result available yet")
    summary_path = DATASET_DIR / f"{meta['summary_key']}.summary.json"
    if not summary_path.exists():
        raise ValueError("dataset summary for current result not found")
    import json

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    npz = np.load(latest_dir() / "predictions.npz", allow_pickle=False)
    pred = npz["pred"].astype(np.float64)
    base = npz["c_up"].astype(np.float64)

    lats = np.asarray(summary["lats"], dtype=float)
    lons = np.asarray(summary["lons"], dtype=float)
    times = pd.to_datetime(np.asarray(summary["times"], dtype=str).str.replace("Z", "", regex=False))

    obs_lat = pd.to_numeric(df[lat_c], errors="coerce").to_numpy(dtype=float)
    obs_lon = pd.to_numeric(df[lon_c], errors="coerce").to_numpy(dtype=float)
    obs_val = pd.to_numeric(df[val_c], errors="coerce").to_numpy(dtype=float)

    use_time = time_c is not None
    obs_time = None
    if use_time:
        obs_time = pd.to_datetime(df[time_c], errors="coerce", utc=True).dt.tz_localize(None)
        if obs_time.isna().all():
            use_time = False

    pred_vals, base_vals, used = [], [], []
    t_mean = np.nanmean(pred, axis=0)
    b_mean = np.nanmean(base, axis=0)
    for i in range(len(df)):
        la, lo, ov = obs_lat[i], obs_lon[i], obs_val[i]
        if not np.isfinite(la) or not np.isfinite(lo) or not np.isfinite(ov):
            continue
        ri = int(np.argmin(np.abs(lats - la)))
        ci = int(np.argmin(np.abs(lons - lo)))
        if use_time and obs_time is not None and pd.notna(obs_time.iloc[i]):
            ti = int(np.argmin(np.abs(times - obs_time.iloc[i])))
            p, b = pred[ti, ri, ci], base[ti, ri, ci]
        else:
            p, b = t_mean[ri, ci], b_mean[ri, ci]
        if not np.isfinite(p):
            continue
        pred_vals.append(float(p))
        base_vals.append(float(b) if np.isfinite(b) else float(p))
        used.append(ov)

    if len(used) < 3:
        raise ValueError("fewer than 3 usable observations matched to the result grid")
    pred_arr = np.asarray(pred_vals)
    base_arr = np.asarray(base_vals)
    obs_arr = np.asarray(used)
    result = metrics(pred_arr, obs_arr, base_arr)
    return {
        "n_matched": len(used),
        "temporal_match": use_time,
        "metrics": result,
        "note": "independent station observations matched to nearest grid cell"
        + ("" if use_time else " (period mean; no usable time column)"),
    }
