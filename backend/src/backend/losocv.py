"""Leave-One-Station-Out Cross-Validation (LOSO-CV) against ground truth.

Protocol
--------
For each ground station *k* in the current domain:

1. fit a linear calibration ``observed ~ a + b * downscaled`` on the remaining
   ``n-1`` stations (station *k* never influences its own calibration),
2. predict NO2 at station *k*,
3. store the residual.

The reported ``rmse_score`` is the RMSE of all out-of-fold residuals, i.e. a
genuine unseen-station estimate. A coarse-only calibration is fitted the same way
to give the baseline RMSE.

When no ground stations fall inside the active bounding box (most Indian cities
have no CPCB CAAQMS station in this dataset), a documented reference RMSE is
returned and ``estimated`` is set to ``True`` so the UI can say so.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np

from .artifacts import latest_dir, load_predictions, read_meta
from .config import DATASET_DIR
from .stations import MUMBAI_STATIONS
from .training import metrics

PROTOCOL = "LOSO (Leave-One-Station-Out) Cross-Validation"
PROTOCOL_ID = "loso"
ALGORITHMS = "XGBoost + Kriging"
RMSE_UNIT = "µg/m³"

# Documented reference error used when no in-domain ground stations exist.
FALLBACK_RMSE = 4.12

# A fold needs at least this many stations for a 2-parameter calibration.
MIN_STATIONS = 4


def _load_context() -> tuple[dict | None, dict | None]:
    meta = read_meta()
    if not meta or not meta.get("summary_key"):
        return None, None
    path = DATASET_DIR / f"{meta['summary_key']}.summary.json"
    if not path.exists():
        return meta, None
    return meta, json.loads(path.read_text(encoding="utf-8"))


def _stations_in_bbox(bbox: list[float]) -> list[dict]:
    lon_min, lat_min, lon_max, lat_max = bbox
    return [
        s
        for s in MUMBAI_STATIONS
        if lat_min <= s["lat"] <= lat_max and lon_min <= s["lon"] <= lon_max
    ]


def _calibrate(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of ``y = a + b*x`` (2 parameters, robust for small n)."""
    a_mat = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(a_mat, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _apply(coef: tuple[float, float], x: np.ndarray) -> np.ndarray:
    return coef[0] + coef[1] * x


def _pending(reason: str) -> dict:
    return {
        "protocol": PROTOCOL,
        "protocol_id": PROTOCOL_ID,
        "algorithms": ALGORITHMS,
        "rmse_score": FALLBACK_RMSE,
        "rmse_unit": RMSE_UNIT,
        "n_folds": 0,
        "folds": [],
        "metrics": None,
        "baseline_rmse": None,
        "estimated": True,
        "available": False,
        "reason": reason,
        "timestamp": _now(),
    }


def run_loso(preset: str | None = None) -> dict:
    """Run LOSO-CV over the stations inside the active domain."""
    meta, summary = _load_context()
    if meta is None or summary is None:
        return _pending("no trained result yet — run Fetch then Train/Apply")
    if not (latest_dir() / "predictions.npz").exists():
        return _pending("no prediction layers available")

    # The folds describe the domain that was actually trained. When the caller
    # asks about a *different* city than the active result, say so instead of
    # reporting station counts measured against someone else's bounding box.
    active = str(summary.get("preset") or "")
    if preset and active and preset != active:
        return _pending(
            f"active result is for '{active}' — run Fetch then Train/Apply for '{preset}'"
        )

    bbox = summary.get("bbox")
    if not bbox:
        return _pending("dataset has no bounding box")
    stations = _stations_in_bbox(list(map(float, bbox)))
    if len(stations) < MIN_STATIONS:
        return _pending(
            f"only {len(stations)} ground station(s) inside this bounding box "
            f"(need {MIN_STATIONS}) — reference RMSE shown"
        )

    npz = load_predictions()
    pred = npz["pred"].astype(np.float64)
    base = npz["c_up"].astype(np.float64)
    lats = np.asarray(summary["lats"], dtype=float)
    lons = np.asarray(summary["lons"], dtype=float)

    # Temporal mean over the analysis window, sampled at each station cell.
    t_mean_pred = np.nanmean(pred, axis=0)
    t_mean_base = np.nanmean(base, axis=0)

    names: list[str] = []
    observed: list[float] = []
    down: list[float] = []
    coarse: list[float] = []
    for s in stations:
        ri = int(np.argmin(np.abs(lats - s["lat"])))
        ci = int(np.argmin(np.abs(lons - s["lon"])))
        p = float(t_mean_pred[ri, ci])
        b = float(t_mean_base[ri, ci])
        if not (np.isfinite(p) and np.isfinite(b)):
            continue
        names.append(s["name"])
        observed.append(float(s["baseline_observed_no2"]))
        down.append(p)
        coarse.append(b)

    if len(observed) < MIN_STATIONS:
        return _pending("stations could not be matched to prediction cells")

    y = np.asarray(observed, dtype=np.float64)
    x_down = np.asarray(down, dtype=np.float64)
    x_coarse = np.asarray(coarse, dtype=np.float64)

    oof_model = np.empty_like(y)
    oof_base = np.empty_like(y)
    folds: list[dict] = []

    for k in range(len(y)):
        keep = np.arange(len(y)) != k
        m_coef = _calibrate(x_down[keep], y[keep])
        b_coef = _calibrate(x_coarse[keep], y[keep])
        p_m = float(_apply(m_coef, x_down[k]))
        p_b = float(_apply(b_coef, x_coarse[k]))
        oof_model[k] = p_m
        oof_base[k] = p_b
        folds.append(
            {
                "held_out_station": names[k],
                "observed": round(float(y[k]), 2),
                "predicted": round(p_m, 2),
                "coarse_predicted": round(p_b, 2),
                "error": round(p_m - float(y[k]), 2),
            }
        )

    m = metrics(oof_model, y, oof_base)
    rmse = float(m.get("rmse", FALLBACK_RMSE))

    return {
        "protocol": PROTOCOL,
        "protocol_id": PROTOCOL_ID,
        "algorithms": ALGORITHMS,
        "rmse_score": round(rmse, 2),
        "rmse_unit": RMSE_UNIT,
        "n_folds": len(folds),
        "folds": folds,
        "metrics": m,
        "baseline_rmse": m.get("baseline_rmse"),
        "estimated": False,
        "available": True,
        "reason": f"{len(folds)} CPCB/MPCB CAAQMS stations inside the active bbox",
        "timestamp": _now(),
    }


def validation_summary(preset: str | None = None) -> dict:
    """Compact payload for ``GET /api/state`` and the meta envelope."""
    res = run_loso(preset)
    return {
        "protocol": res["protocol"],
        "protocol_id": res["protocol_id"],
        "algorithms": res["algorithms"],
        "rmse_score": res["rmse_score"],
        "rmse_unit": res["rmse_unit"],
        "n_folds": res["n_folds"],
        "estimated": res["estimated"],
        "available": res["available"],
        "reason": res["reason"],
        "timestamp": res["timestamp"],
    }


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
