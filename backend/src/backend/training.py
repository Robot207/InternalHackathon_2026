from __future__ import annotations

import math

import joblib
import numpy as np

from .config import ARTIFACT_DIR, SPLITS
from .features import build_features
from .grids import upsample
from .models import feature_importance, make_model


def make_split(split: str, h: int, w: int, t_len: int) -> dict:
    test_cell = np.zeros((h, w), dtype=bool)
    test_hours = np.zeros(t_len, dtype=bool)
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}. available: {list(SPLITS)}")
    if split in ("spatial", "spatiotemporal"):
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
    ratio = np.full(len(mask), np.nan, dtype=np.float64)
    if mask.any():
        if progress:
            progress(0.4, "predicting sub-grid pattern")
        pred_ratio = model.predict(X[mask])
        ratio[mask] = np.clip(pred_ratio, -1.6, 1.6)
    c_up = feat["c_up"]
    with np.errstate(over="ignore"):
        pred = c_up.reshape(-1) * np.exp(ratio)
    pred = pred.reshape(c_up.shape)
    pred = np.where(np.isfinite(pred) & np.isfinite(c_up), pred, np.nan)
    if conserve:
        pred = _conserve(pred, data, feat)
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
    split_info = make_split(split, h, w, t_len)
    train_idx, test_idx = _index_masks(feat, split_info)
    fallback = False
    if len(test_idx) < 100 or len(train_idx) < 500:
        fallback = True
        split_info = make_split("temporal", h, w, t_len)
        train_idx, test_idx = _index_masks(feat, split_info)
    if len(train_idx) < 100:
        raise ValueError("not enough valid training samples for this date range")

    X = feat["X"]
    y = feat["y"]
    X_flat = X.reshape(-1, X.shape[-1])
    y_flat = y.reshape(-1)
    X_train = X_flat[train_idx]
    y_train = y_flat[train_idx]

    model = make_model(model_name)
    if progress:
        progress(0.15, f"training {model_name} on {len(X_train):,} samples")
    model.fit(X_train, y_train)
    if progress:
        progress(0.35, "model fitted")

    result = predict_field(data, model, conserve, progress)
    pred = result["pred"]
    c_up = feat["c_up"]

    truth = np.asarray(data["ref"], dtype=np.float64)
    pred_flat = pred.reshape(-1)
    truth_flat = truth.reshape(-1)
    base_flat = c_up.reshape(-1)
    test_metrics = metrics(pred_flat[test_idx], truth_flat[test_idx], base_flat[test_idx])
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
    np.savez_compressed(
        out_dir / "predictions.npz",
        pred=pred.astype(np.float32),
        c_up=feat["c_up"].astype(np.float32),
        gap_up=feat["gap_up"].astype(np.float32),
        ref=(truth.astype(np.float32) if truth is not None else np.full_like(pred, np.nan)),
        elev=np.asarray(data["elev"], dtype=np.float32),
        roads=np.asarray(data["roads"], dtype=np.float32),
        test_cell=np.zeros((feat["H"], feat["W"]), dtype=bool),
        test_hours=np.zeros(feat["t_len"], dtype=bool),
    )
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


def run_benchmark_arena(
    data: dict,
    split: str,
    conserve: bool,
    summary: dict,
    progress=None,
) -> dict:
    """Benchmark all available ML models on the identical holdout split.

    Ranks models by RMSE, MAE, R², Pearson correlation, and skill vs coarse baseline.
    """
    import time
    from .config import MODELS

    if not summary.get("has_reference"):
        raise ValueError(
            "Multi-model arena requires a reference dataset (e.g. London or Paris) "
            "to evaluate accuracy on independent unseen validation data."
        )

    if progress:
        progress(0.05, "engineering features for arena")
    feat = build_features(data)
    h, w, t_len = feat["H"], feat["W"], feat["t_len"]
    split_info = make_split(split, h, w, t_len)
    train_idx, test_idx = _index_masks(feat, split_info)
    if len(test_idx) < 100 or len(train_idx) < 500:
        split_info = make_split("temporal", h, w, t_len)
        train_idx, test_idx = _index_masks(feat, split_info)

    X = feat["X"]
    y = feat["y"]
    X_flat = X.reshape(-1, X.shape[-1])
    y_flat = y.reshape(-1)
    X_train = X_flat[train_idx]
    y_train = y_flat[train_idx]
    c_up = feat["c_up"]
    truth = np.asarray(data["ref"], dtype=np.float64)

    truth_test = truth.reshape(-1)[test_idx]
    base_test = c_up.reshape(-1)[test_idx]

    candidate_models = list(MODELS.keys())
    leaderboard = []

    total_models = len(candidate_models)
    for idx, m_id in enumerate(candidate_models):
        if progress:
            progress(
                0.1 + 0.8 * (idx / total_models),
                f"evaluating {MODELS[m_id]} ({idx+1}/{total_models})",
            )
        t0 = time.time()
        try:
            model = make_model(m_id)
            model.fit(X_train, y_train)
            fit_time = round(time.time() - t0, 2)

            pred_res = predict_field(data, model, conserve, None)
            pred = pred_res["pred"]
            pred_test = pred.reshape(-1)[test_idx]

            m_metrics = metrics(pred_test, truth_test, base_test)
            block_err = _consistency_error(pred, data, feat)

            leaderboard.append(
                {
                    "model_id": m_id,
                    "model_name": MODELS[m_id],
                    "metrics": m_metrics,
                    "fit_time_sec": fit_time,
                    "coarse_consistency_mae": round(block_err, 4),
                }
            )
        except Exception as exc:  # noqa: BLE001
            leaderboard.append(
                {
                    "model_id": m_id,
                    "model_name": MODELS[m_id],
                    "error": str(exc),
                    "fit_time_sec": round(time.time() - t0, 2),
                }
            )

    valid_entries = [e for e in leaderboard if "metrics" in e]
    valid_entries.sort(
        key=lambda x: (
            -x["metrics"].get("skill_vs_baseline", -999),
            x["metrics"].get("rmse", 999),
        )
    )

    winner = valid_entries[0]["model_id"] if valid_entries else "none"

    if progress:
        progress(1.0, f"arena complete: optimal model is {MODELS.get(winner, winner)}")

    return {
        "split": split_info["mode"],
        "holdout_description": SPLITS[split_info["mode"]],
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "winner_id": winner,
        "winner_name": MODELS.get(winner, winner),
        "leaderboard": valid_entries,
        "timestamp": _now(),
    }

