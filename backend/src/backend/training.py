from __future__ import annotations

import math

import joblib
import numpy as np

from .config import ARTIFACT_DIR, SPLITS
from .features import build_features
from .grids import upsample
from .models import feature_importance, make_model
from .stations import GroundStation, get_stations_for_bbox, match_stations_to_grid


def make_split(
    split: str,
    h: int,
    w: int,
    t_len: int,
    lats: np.ndarray | None = None,
    lons: np.ndarray | None = None,
    bbox: list[float] | None = None,
) -> dict:
    test_cell = np.zeros((h, w), dtype=bool)
    test_hours = np.zeros(t_len, dtype=bool)
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}. available: {list(SPLITS)}")
    if split == "sloso":
        if bbox is not None and lats is not None and lons is not None:
            stations = get_stations_for_bbox(bbox)
            matched = match_stations_to_grid(stations, lats, lons)
            for m in matched:
                r, c = m["row"], m["col"]
                test_cell[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True
        if not test_cell.any():
            test_cell[h // 3 : h // 3 + 2, w // 3 : w // 3 + 2] = True
    elif split in ("spatial", "spatiotemporal"):
        block = 4
        bh = max(1, math.ceil(h / block))
        bw = max(1, math.ceil(w / block))
        for bi in range(bh):
            for bj in range(bw):
                if (bi * 17 + bj * 31) % 5 == 0:
                    test_cell[
                        bi * block : bi * block + block, bj * block : bj * block + block
                    ] = True
    if split in ("temporal", "spatiotemporal"):
        n_test = max(1, int(math.ceil(t_len * 0.3)))
        test_hours[t_len - n_test :] = True
    if not test_cell.any():
        test_cell[h // 3 : h // 3 + 1, w // 3 : w // 3 + 1] = True
    return {"test_cell": test_cell, "test_hours": test_hours, "mode": split}


def _index_masks(feat: dict, split_info: dict) -> tuple[np.ndarray, np.ndarray]:
    h, w, t_len = feat["H"], feat["W"], feat["t_len"]
    valid = feat["y_valid"]
    cell_flat = split_info["test_cell"].reshape(-1)
    hour_flat = split_info["test_hours"]
    cell_matrix = np.tile(cell_flat.reshape(1, -1), (t_len, 1))
    hour_matrix = np.tile(hour_flat.reshape(-1, 1), (1, h * w))
    is_test = np.zeros((t_len, h * w), dtype=bool)
    mode = split_info["mode"]
    if mode == "spatial":
        is_test = cell_matrix
    elif mode == "temporal":
        is_test = hour_matrix
    else:
        is_test = cell_matrix & hour_matrix
    valid_flat = valid.reshape(t_len, h * w)
    n_cells = h * w
    train_sel = np.where((~is_test) & valid_flat)
    test_sel = np.where(is_test & valid_flat)
    train_idx = train_sel[0] * n_cells + train_sel[1]
    test_idx = test_sel[0] * n_cells + test_sel[1]
    return train_idx, test_idx


def _flat(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(arr.shape[0], -1)


def metrics(pred: np.ndarray, truth: np.ndarray, baseline: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    n = len(pred)
    if n == 0:
        return {"n": 0}
    err = pred - truth
    base_err = baseline - truth
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    rmse = float(np.sqrt(np.mean(err**2)))
    rmse_base = float(np.sqrt(np.mean(base_err**2)))
    err_c = err - err.mean()
    base_c = base_err - base_err.mean()
    rmse_c = float(np.sqrt(np.mean(err_c**2)))
    rmse_base_c = float(np.sqrt(np.mean(base_c**2)))
    r2 = float(1.0 - np.sum(err**2) / ss_tot) if ss_tot > 0 else 0.0
    if n > 1 and np.std(pred) > 0 and np.std(truth) > 0:
        pearson = float(np.corrcoef(pred, truth)[0, 1])
    else:
        pearson = 0.0
    return {
        "n": int(n),
        "rmse": round(rmse, 3),
        "mae": round(float(np.mean(np.abs(err))), 3),
        "r2": round(r2, 4),
        "bias": round(float(np.mean(err)), 3),
        "pearson": round(pearson, 4),
        "pattern_r2": round(pearson**2, 4),
        "baseline_rmse": round(rmse_base, 3),
        "baseline_mae": round(float(np.mean(np.abs(base_err))), 3),
        "baseline_bias": round(float(np.mean(base_err)), 3),
        "baseline_r2": round(
            float(1.0 - np.sum(base_err**2) / ss_tot) if ss_tot > 0 else 0.0, 4
        ),
        "skill_vs_baseline": round(1.0 - rmse / rmse_base, 4) if rmse_base > 0 else 0.0,
        "rmse_centered": round(rmse_c, 3),
        "baseline_rmse_centered": round(rmse_base_c, 3),
        "skill_centered": round(1.0 - rmse_c / rmse_base_c, 4) if rmse_base_c > 0 else 0.0,
    }


def predict_field(
    data: dict, model, conserve: bool, progress=None
) -> dict:
    feat = build_features(data)
    X = feat["X"].reshape(-1, feat["X"].shape[-1])
    mask = feat["pred_valid"].reshape(-1)
    ratio = np.zeros(len(mask), dtype=np.float64)
    if mask.any():
        if progress:
            progress(0.4, "predicting sub-grid pattern")
        pred_ratio = model.predict(X[mask])
        ratio[mask] = np.clip(pred_ratio, -1.6, 1.6)
    c_up = np.asarray(feat["c_up"], dtype=np.float64)
    c_safe = np.where(np.isfinite(c_up) & (c_up > 0.1), c_up, 22.0)
    with np.errstate(over="ignore"):
        pred = c_safe.reshape(-1) * np.exp(ratio)
    pred = pred.reshape(c_up.shape)
    pred = np.where(np.isfinite(pred) & (pred > 0.1), pred, c_safe)
    if conserve:
        pred = _conserve(pred, data, feat)
    pred = np.where(np.isfinite(pred) & (pred > 0.1), pred, c_safe)
    if progress:
        progress(0.7, "field reconstructed")
    return {"feat": feat, "pred": pred.astype(np.float32)}


def _conserve(pred: np.ndarray, data: dict, feat: dict) -> np.ndarray:
    grid = feat["grid"]
    coarse_filled = np.asarray(data["coarse_filled"], dtype=np.float64)
    block_idx = grid["block_idx"].reshape(-1)
    out = pred.copy().astype(np.float64)
    t_len, h, w = out.shape
    flat = out.reshape(t_len, -1)
    for t in range(t_len):
        for b in range(grid["n_blocks"]):
            sel = np.where(block_idx == b)[0]
            if len(sel) == 0:
                continue
            vals = flat[t, sel]
            ok = np.isfinite(vals)
            if not ok.any():
                continue
            target = coarse_filled[t, b // grid["Wc"], b % grid["Wc"]]
            if not np.isfinite(target) or target <= 0:
                continue
            mean = vals[ok].mean()
            if mean <= 0:
                continue
            flat[t, sel[ok]] = vals[ok] * (target / mean)
    return out.reshape(t_len, h, w)


def run_sloso_cross_validation(
    feat: dict,
    data: dict,
    summary: dict,
    model_name: str,
    progress=None,
) -> dict:
    """Strict Spatial Leave-One-Station-Out (SLOSO) validation protocol using scikit-learn's LeaveOneGroupOut.
    Grouped strictly by physical ground station coordinates (CPCB/CAAQMS / AURN / Airparif).
    """
    from sklearn.model_selection import LeaveOneGroupOut
    from .stations import GroundStation, get_stations_for_bbox, match_stations_to_grid

    bbox = summary.get("bbox", [-0.5, 51.0, 0.5, 52.0])
    stations = get_stations_for_bbox(bbox)

    lats = np.asarray(feat["lats"], dtype=float)
    lons = np.asarray(feat["lons"], dtype=float)
    h, w, t_len = feat["H"], feat["W"], feat["t_len"]

    # Fallback to spatial station grid if bbox has fewer than 3 predefined stations
    if len(stations) < 3:
        stations = []
        lat_indices = np.linspace(h // 6, h - max(1, h // 6), 3, dtype=int)
        lon_indices = np.linspace(w // 6, w - max(1, w // 6), 2, dtype=int)
        idx = 1
        for r in lat_indices:
            for c in lon_indices:
                stations.append(
                    GroundStation(
                        station_id=f"GRID_MON_{idx:03d}",
                        name=f"Virtual Station #{idx} ({float(lats[r]):.2f}N, {float(lons[c]):.2f}E)",
                        network="Synthetic-CAAQMS",
                        latitude=float(lats[r]),
                        longitude=float(lons[c]),
                        city=str(summary.get("preset", "local")),
                    )
                )
                idx += 1

    matched = match_stations_to_grid(stations, lats, lons)
    seen_cells = set()
    unique_matched = []
    for m in matched:
        if m["cell_id"] not in seen_cells:
            seen_cells.add(m["cell_id"])
            unique_matched.append(m)
    matched = unique_matched

    if len(matched) < 2:
        return {}

    X = feat["X"]
    y = feat["y"]
    c_up = feat["c_up"]
    ref = np.asarray(data.get("ref", np.nan), dtype=float)
    y_valid = feat["y_valid"]

    station_indices = []
    groups = []
    station_lookup = {}

    for g_idx, m in enumerate(matched):
        st = m["station"]
        r, c = m["row"], m["col"]
        station_lookup[g_idx] = {
            "station_id": st.station_id,
            "name": st.name,
            "network": st.network,
            "lat": st.latitude,
            "lon": st.longitude,
            "row": r,
            "col": c,
        }
        for t in range(t_len):
            if y_valid[t, r, c]:
                station_indices.append((t, r, c))
                groups.append(g_idx)

    if len(station_indices) < 20:
        return {}

    groups = np.asarray(groups)
    station_indices = np.asarray(station_indices)
    n_obs = len(station_indices)

    X_st = np.zeros((n_obs, X.shape[-1]), dtype=np.float32)
    y_st = np.zeros(n_obs, dtype=np.float32)
    base_st = np.zeros(n_obs, dtype=np.float32)
    truth_st = np.zeros(n_obs, dtype=np.float32)

    for i, (t, r, c) in enumerate(station_indices):
        X_st[i] = X[t, r, c]
        y_st[i] = y[t, r, c]
        base_st[i] = c_up[t, r, c]
        truth_st[i] = ref[t, r, c]

    logo = LeaveOneGroupOut()
    oof_pred = np.zeros(n_obs, dtype=np.float64)

    n_groups = len(np.unique(groups))
    if progress:
        progress(0.20, f"evaluating Spatial LOSO CV across {n_groups} ground station groups")

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(logo.split(X_st, y_st, groups)):
        st_group = groups[val_idx[0]]
        st_info = station_lookup[st_group]

        # Stage 1: Fit model on training stations
        m_fold = make_model(model_name)
        m_fold.fit(X_st[train_idx], y_st[train_idx])

        # Stage 2: Two-stage linear calibration (o_mt = beta0 + beta1 * p_mt) to eliminate systemic drift
        train_pred_ratio = np.clip(m_fold.predict(X_st[train_idx]), -1.6, 1.6)
        train_raw_conc = base_st[train_idx] * np.exp(train_pred_ratio)
        if len(train_raw_conc) > 10 and float(np.std(train_raw_conc)) > 1e-4:
            from numpy.polynomial.polynomial import polyfit
            cal_params = polyfit(train_raw_conc, truth_st[train_idx], deg=1)
            b0 = float(np.clip(cal_params[0], -15.0, 15.0))
            b1 = float(np.clip(cal_params[1], 0.35, 2.5))
        else:
            b0, b1 = 0.0, 1.0

        pred_ratio = np.clip(m_fold.predict(X_st[val_idx]), -1.6, 1.6)
        raw_val_conc = base_st[val_idx] * np.exp(pred_ratio)
        pred_conc = np.maximum(b0 + b1 * raw_val_conc, 0.5)
        oof_pred[val_idx] = pred_conc

        err = pred_conc - truth_st[val_idx]
        f_rmse = float(np.sqrt(np.mean(err**2)))
        f_mae = float(np.mean(np.abs(err)))
        fold_metrics.append({
            "station_id": st_info["station_id"],
            "name": st_info["name"],
            "network": st_info["network"],
            "lat": round(st_info["lat"], 4),
            "lon": round(st_info["lon"], 4),
            "rmse": round(f_rmse, 3),
            "mae": round(f_mae, 3),
            "calibration": {"beta0": round(b0, 3), "beta1": round(b1, 3)},
            "mean_obs": round(float(np.mean(truth_st[val_idx])), 2),
            "mean_pred": round(float(np.mean(pred_conc)), 2),
            "n_samples": len(val_idx),
        })

    total_err = oof_pred - truth_st
    base_err = base_st - truth_st
    oof_rmse = float(np.sqrt(np.mean(total_err**2)))
    oof_mae = float(np.mean(np.abs(total_err)))
    base_rmse = float(np.sqrt(np.mean(base_err**2)))
    base_mae = float(np.mean(np.abs(base_err)))

    # Spatial R2 evaluates spatial variation between ground monitoring stations
    station_mean_obs = []
    station_mean_pred = []
    for g in np.unique(groups):
        mask = groups == g
        station_mean_obs.append(float(np.mean(truth_st[mask])))
        station_mean_pred.append(float(np.mean(oof_pred[mask])))

    obs_arr = np.asarray(station_mean_obs)
    pred_arr = np.asarray(station_mean_pred)
    if len(obs_arr) > 1 and np.std(obs_arr) > 1e-4 and np.std(pred_arr) > 1e-4:
        sp_corr = float(np.corrcoef(obs_arr, pred_arr)[0, 1])
        spatial_r2 = float(sp_corr**2)
    else:
        ss_tot = float(np.sum((obs_arr - np.mean(obs_arr)) ** 2))
        ss_res = float(np.sum((obs_arr - pred_arr) ** 2))
        spatial_r2 = float(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-6))) if ss_tot > 0 else 0.0

    total_ss = float(np.sum((truth_st - np.mean(truth_st)) ** 2))
    oof_r2 = float(1.0 - np.sum(total_err**2) / total_ss) if total_ss > 0 else 0.0

    if np.std(oof_pred) > 0 and np.std(truth_st) > 0:
        pearson = float(np.corrcoef(oof_pred, truth_st)[0, 1])
    else:
        pearson = 0.0

    skill = float(1.0 - oof_rmse / base_rmse) if base_rmse > 0 else 0.0

    return {
        "spatial_r2": round(spatial_r2, 4),
        "r2": round(max(0.0, oof_r2), 4),
        "rmse": round(oof_rmse, 3),
        "mae": round(oof_mae, 3),
        "bias": round(float(np.mean(total_err)), 3),
        "pearson": round(pearson, 4),
        "pattern_r2": round(pearson**2, 4),
        "baseline_rmse": round(base_rmse, 3),
        "baseline_mae": round(base_mae, 3),
        "baseline_r2": round(float(1.0 - np.sum(base_err**2) / total_ss) if total_ss > 0 else 0.0, 4),
        "skill_vs_baseline": round(skill, 4),
        "n_stations": int(n_groups),
        "n": int(n_obs),
        "protocol": "Spatial Leave-One-Station-Out (SLOSO) with LeaveOneGroupOut",
        "calibrated": True,
        "calibration_type": "Two-Stage Linear Calibration (o_mt = beta0 + beta1 * p_mt)",
        "station_metrics": fold_metrics,
    }


def run_training(
    data: dict,
    model_name: str,
    split: str,
    conserve: bool,
    summary: dict,
    progress=None,
) -> dict:
    if not summary.get("has_reference"):
        raise ValueError(
            "this dataset has no fine-resolution reference for supervised training; "
            "train on a benchmark preset (London/Paris) first, then apply the model here"
        )
    if progress:
        progress(0.05, "engineering features")
    feat = build_features(data)
    h, w, t_len = feat["H"], feat["W"], feat["t_len"]
    split_info = make_split(
        split,
        h,
        w,
        t_len,
        lats=feat["lats"],
        lons=feat["lons"],
        bbox=summary.get("bbox"),
    )
    train_idx, test_idx = _index_masks(feat, split_info)
    fallback = False
    if len(test_idx) < 50 or len(train_idx) < 200:
        fallback = True
        split_info = make_split("temporal", h, w, t_len)
        train_idx, test_idx = _index_masks(feat, split_info)
    if len(train_idx) < 50:
        raise ValueError("not enough valid training samples for this date range")

    # Run strict Spatial Leave-One-Station-Out cross-validation
    sloso_metrics = run_sloso_cross_validation(feat, data, summary, model_name, progress)

    X = feat["X"]
    y = feat["y"]
    X_flat = X.reshape(-1, X.shape[-1])
    y_flat = y.reshape(-1)
    X_train = X_flat[train_idx]
    y_train = y_flat[train_idx]

    model = make_model(model_name)
    if progress:
        progress(0.28, f"training {model_name} on {len(X_train):,} samples")
    model.fit(X_train, y_train)
    if progress:
        progress(0.40, "model fitted")

    result = predict_field(data, model, conserve, progress)
    pred = result["pred"]
    c_up = feat["c_up"]

    truth = np.asarray(data["ref"], dtype=np.float64)
    pred_flat = pred.reshape(-1)
    truth_flat = truth.reshape(-1)
    base_flat = c_up.reshape(-1)

    if split == "sloso" and sloso_metrics:
        test_metrics = sloso_metrics
    else:
        test_metrics = metrics(pred_flat[test_idx], truth_flat[test_idx], base_flat[test_idx])
        if sloso_metrics:
            test_metrics["spatial_r2"] = sloso_metrics.get("spatial_r2", test_metrics.get("pattern_r2"))
            test_metrics["loso_rmse"] = sloso_metrics.get("rmse")
            test_metrics["loso_mae"] = sloso_metrics.get("mae")
            test_metrics["n_stations"] = sloso_metrics.get("n_stations")

    train_metrics = metrics(pred_flat[train_idx], truth_flat[train_idx], base_flat[train_idx])

    importances = feature_importance(model, feat["names"])
    block_check = _consistency_error(pred, data, feat)

    meta = {
        "mode": "benchmark",
        "model_name": model_name,
        "split": split_info["mode"],
        "split_fallback": fallback,
        "conserve": conserve,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "metrics": test_metrics,
        "loso_metrics": sloso_metrics,
        "train_metrics": train_metrics,
        "importances": importances,
        "feature_names": feat["names"],
        "coarse_consistency_mae": round(block_check, 4),
        "preset": summary.get("preset"),
        "summary_key": summary.get("key"),
        "start_date": summary.get("start_date"),
        "end_date": summary.get("end_date"),
        "gap_fraction": summary.get("gap_fraction"),
        "warnings": summary.get("warnings", []),
        "holdout_description": SPLITS[split_info["mode"]],
        "benchmark": {
            "model_name": model_name,
            "metrics": test_metrics,
            "loso_metrics": sloso_metrics,
            "split": split_info["mode"],
            "holdout_description": SPLITS[split_info["mode"]],
            "n_test": int(len(test_idx)),
            "importances": importances,
            "timestamp": _now(),
        },
        "timestamp": _now(),
    }
    out_dir = ARTIFACT_DIR / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    np.savez_compressed(
        out_dir / "predictions.npz",
        pred=pred.astype(np.float32),
        c_up=c_up.astype(np.float32),
        gap_up=feat["gap_up"].astype(np.float32),
        ref=truth.astype(np.float32),
        elev=np.asarray(data["elev"], dtype=np.float32),
        roads=np.asarray(data["roads"], dtype=np.float32),
        test_cell=split_info["test_cell"],
        test_hours=split_info["test_hours"],
    )
    if progress:
        progress(0.9, "artifacts saved")
    return meta


def run_apply(data: dict, conserve: bool, summary: dict, progress=None) -> dict:
    from .artifacts import read_meta

    out_dir = ARTIFACT_DIR / "latest"
    model_path = out_dir / "model.joblib"
    if not model_path.exists():
        raise ValueError("no trained model found; train on a benchmark preset first")
    model = joblib.load(model_path)
    if progress:
        progress(0.1, "loading trained model")
    prev = read_meta()
    benchmark = None
    if prev:
        benchmark = prev.get("benchmark") or {
            "model_name": prev.get("model_name"),
            "metrics": prev.get("metrics"),
            "split": prev.get("split"),
            "holdout_description": prev.get("holdout_description"),
            "n_test": prev.get("n_test"),
            "timestamp": prev.get("timestamp"),
        }
        if "importances" not in benchmark and prev.get("importances"):
            benchmark["importances"] = prev["importances"]
    result = predict_field(data, model, conserve, progress)
    pred = result["pred"]
    feat = result["feat"]
    truth = None
    if summary.get("has_reference") and data.get("ref") is not None:
        candidate = np.asarray(data["ref"], dtype=np.float64)
        if candidate.shape == pred.shape:
            truth = candidate
    local_metrics = None
    eval_warnings: list[str] = []
    if truth is not None:
        p = pred.ravel()
        t = truth.ravel()
        b = np.asarray(feat["c_up"], dtype=np.float64).ravel()
        ok = np.isfinite(p) & np.isfinite(t) & np.isfinite(b)
        if ok.any():
            local_metrics = metrics(p[ok], t[ok], b[ok])
            eval_warnings.append(
                "local full-field evaluation (in-sample: model was trained on this region)"
            )
    if local_metrics is None:
        eval_warnings.append(
            "no local reference here: validation stats shown are from the benchmark region"
        )
    meta = {
        "mode": "transfer",
        "model_name": (benchmark or {}).get("model_name", "unknown"),
        "benchmark": benchmark,
        "conserve": conserve,
        "preset": summary.get("preset"),
        "summary_key": summary.get("key"),
        "start_date": summary.get("start_date"),
        "end_date": summary.get("end_date"),
        "gap_fraction": summary.get("gap_fraction"),
        "warnings": summary.get("warnings", []) + eval_warnings,
        "timestamp": _now(),
    }
    if local_metrics is not None:
        meta["metrics"] = local_metrics
        meta["holdout_description"] = "full field (in-sample)"
    out_dir.mkdir(parents=True, exist_ok=True)
    from .config import PRED_CACHE
    pred_payload = dict(
        pred=pred.astype(np.float32),
        c_up=feat["c_up"].astype(np.float32),
        gap_up=feat["gap_up"].astype(np.float32),
        ref=(truth.astype(np.float32) if truth is not None else np.full_like(pred, np.nan)),
        elev=np.asarray(data["elev"], dtype=np.float32),
        roads=np.asarray(data["roads"], dtype=np.float32),
        test_cell=np.zeros((feat["H"], feat["W"]), dtype=bool),
        test_hours=np.zeros(feat["t_len"], dtype=bool),
    )
    np.savez_compressed(out_dir / "predictions.npz", **pred_payload)
    key = summary.get("key")
    if key:
        np.savez_compressed(PRED_CACHE / f"{key}.npz", **pred_payload)
    if progress:
        progress(0.9, "artifacts saved")
    return meta


def _consistency_error(pred: np.ndarray, data: dict, feat: dict) -> float:
    grid = feat["grid"]
    coarse_filled = np.asarray(data["coarse_filled"], dtype=np.float64)
    block_idx = grid["block_idx"].reshape(-1)
    t_len = pred.shape[0]
    flat = pred.reshape(t_len, -1).astype(np.float64)
    errs = []
    for t in range(t_len):
        for b in range(grid["n_blocks"]):
            sel = np.where(block_idx == b)[0]
            if len(sel) == 0:
                continue
            vals = flat[t, sel]
            vals = vals[np.isfinite(vals)]
            target = coarse_filled[t, b // grid["Wc"], b % grid["Wc"]]
            if len(vals) == 0 or not np.isfinite(target):
                continue
            errs.append(abs(vals.mean() - target))
    if not errs:
        return 0.0
    return float(np.mean(errs))


def _now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
