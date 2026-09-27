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
