from __future__ import annotations

import datetime as dt
import io
import json

import numpy as np
import pandas as pd

from .artifacts import _mean_frame, has_predictions, load_predictions, read_meta
from .config import DATASET_DIR, SPLITS
from .losocv import _stations_in_bbox
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
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    npz = load_predictions()
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


# ---------------------------------------------------------------------------
# Per-frame "validation on unseen data" for the timeline the map is painting
# ---------------------------------------------------------------------------
#
# The panel used to quote ``meta.metrics``: one number computed at train time
# over the *whole* holdout, so scrubbing the timeline changed nothing — and on
# a transfer city those numbers came from the benchmark region (London) while
# being read as if they validated the city on screen.  ``frame_validation``
# re-scores the unseen samples (or the ground stations) of the frame the UI is
# actually showing, and says plainly when the active region has no independent
# truth at all.

STATION_FRAME_CAVEAT = (
    "Station readings are published period means, so this compares the frame's "
    "field against them rather than a same-hour reading."
)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _answer(**kw) -> dict:
    out: dict = {
        "available": False,
        "source": "none",
        "source_label": None,
        "reason": None,
        "note": None,
        "frame": None,
        "split": None,
        "holdout_description": None,
        "n": 0,
        "metrics": None,
        "timestamp": _now(),
    }
    out.update(kw)
    return out


def _frame_parts(times: list, t_len: int, time_idx: int | None) -> tuple[bool, int, dict]:
    """Resolve the timeline index exactly like ``inspect_point`` does.

    ``0..n-1`` is that hour, ``n`` (slider end stop), a negative index or an
    omitted index is the "period mean" frame the map parks on at start-up.
    """
    if time_idx is not None and 0 <= int(time_idx) < t_len:
        t = int(time_idx)
        label = times[t].replace("T", " ").replace(":00Z", "Z") if times else f"hour {t}"
        stamp = str(times[t]) if times else None
        return False, t, {"index": t, "is_mean": False, "label": label, "time": stamp}
    if times:
        label = f"period mean · {times[0][:10]} → {times[-1][:10]}"
    else:
        label = "period mean"
    return True, 0, {"index": None, "is_mean": True, "label": label, "time": None}


def _rows_mean(arr: np.ndarray, rows: np.ndarray | None) -> np.ndarray:
    """Temporal mean of ``arr`` over ``rows`` (all hours when ``rows`` is None)."""
    return _mean_frame(arr if rows is None else arr[rows])


def frame_validation(time_idx: int | None = None) -> dict:
    """Score the *unseen* data for timeline frame ``time_idx``.

    Returns ``available: true`` with frame-level ``metrics`` when the active
    region has independent truth (the 0.1° fine reference, or ground stations
    inside the bbox); otherwise ``available: false`` with a ``reason`` so the
    UI can say so instead of passing another region's numbers off as local
    validation.  Any unexpected error degrades to the same ``available:
    false`` shape rather than failing the endpoint.
    """
    try:
        return _frame_validation(time_idx)
    except Exception as exc:  # noqa: BLE001 — a broken result must not blank the panel
        return _answer(reason=f"frame validation error: {exc}")


def _frame_validation(time_idx: int | None = None) -> dict:
    meta = read_meta()
    if not meta or not meta.get("summary_key"):
        return _answer(reason="no trained/predicted result yet — run Fetch → Train first")
    summary_path = DATASET_DIR / f"{meta['summary_key']}.summary.json"
    if not summary_path.exists():
        return _answer(reason="dataset summary for the current result not found")
    if not has_predictions():
        return _answer(reason="no prediction layers yet — run Train or Apply first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    data = load_predictions()
    pred = np.asarray(data["pred"], dtype=np.float64)
    base = np.asarray(data["c_up"], dtype=np.float64)
    ref_raw = data.get("ref")
    ref = np.asarray(ref_raw, dtype=np.float64) if ref_raw is not None else None
    if ref is None or ref.shape != pred.shape:
        # Transfer datasets carry an empty reference cube (has_reference=False).
        ref = np.full(pred.shape, np.nan)

    times = [str(t) for t in (summary.get("times") or [])]
    t_len = pred.shape[0]
    is_mean, t, frame = _frame_parts(times, t_len, time_idx)

    # Which samples were withheld from training? Transfer runs ship all-False
    # masks (nothing was withheld here), so treat them as "no local holdout".
    test_cell = data.get("test_cell")
    test_hours = data.get("test_hours")
    split = meta.get("split") or (meta.get("benchmark") or {}).get("split")
    has_holdout = (
        test_cell is not None
        and test_hours is not None
        and bool(np.asarray(test_cell).any() or np.asarray(test_hours).any())
    )
    if not has_holdout:
        split = None  # nothing was withheld for this result
    holdout_description = (
        meta.get("holdout_description")
        or (SPLITS.get(split) if split else "full field (in-sample)")
        or "full field (in-sample)"
    )

    common = {"frame": frame, "split": split, "holdout_description": holdout_description}

    # --- 1. independent fine reference (London / Paris, any has_reference city)
    if bool(np.isfinite(ref).any()):
        cell_mask = None
        rows = None  # None = every hour counts towards the mean frame
        if split:
            if split in ("spatial", "spatiotemporal"):
                cell_mask = np.asarray(test_cell, dtype=bool)
            if split in ("temporal", "spatiotemporal"):
                hours = np.asarray(test_hours, dtype=bool)
                if is_mean:
                    rows = np.where(hours)[0] if hours.any() else None
                elif not hours[t]:
                    return _answer(
                        **common,
                        source="reference",
                        source_label="independent fine reference",
                        reason=(
                            f"hour {t} sits in the training window of the "
                            f"{holdout_description.lower()} holdout — no unseen samples at this frame"
                        ),
                        note="Cards below show the whole-period holdout instead.",
                    )

        if is_mean:
            p_f, b_f, r_f = _rows_mean(pred, rows), _rows_mean(base, rows), _rows_mean(ref, rows)
        else:
            p_f, b_f, r_f = pred[t], base[t], ref[t]

        sel = np.isfinite(p_f) & np.isfinite(b_f) & np.isfinite(r_f)
        if cell_mask is not None:
            sel &= cell_mask
        n = int(sel.sum())
        if n == 0:
            return _answer(
                **common,
                source="reference",
                source_label="independent fine reference",
                reason="no unseen reference samples on this frame",
                note="Cards below show the whole-period holdout instead.",
            )
        m = metrics(p_f[sel], r_f[sel], b_f[sel])
        return _answer(
            **common,
            available=True,
            source="reference",
            source_label=f"independent fine reference · {n} unseen cells",
            n=int(m["n"]),
            metrics=m,
            reason=None,
        )

    # --- 2. ground stations inside the active bbox (Mumbai CPCB / MPCB)
    stations = _stations_in_bbox(list(map(float, summary["bbox"])))
    if stations:
        p_f = _rows_mean(pred, None) if is_mean else pred[t]
        b_f = _rows_mean(base, None) if is_mean else base[t]
        lats = np.asarray(summary["lats"], dtype=float)
        lons = np.asarray(summary["lons"], dtype=float)
        pred_vals, base_vals, obs_vals = [], [], []
        for s in stations:
            if not (lats.min() <= s["lat"] <= lats.max() and lons.min() <= s["lon"] <= lons.max()):
                continue
            ri = int(np.argmin(np.abs(lats - s["lat"])))
            ci = int(np.argmin(np.abs(lons - s["lon"])))
            p, b = p_f[ri, ci], b_f[ri, ci]
            if not (np.isfinite(p) and np.isfinite(b)):
                continue
            pred_vals.append(float(p))
            base_vals.append(float(b))
            obs_vals.append(float(s["baseline_observed_no2"]))
        if len(obs_vals) >= 3:
            m = metrics(np.asarray(pred_vals), np.asarray(obs_vals), np.asarray(base_vals))
            return _answer(
                frame=frame,
                split=split,
                holdout_description="ground-station check · period-mean readings",
                available=True,
                source="stations",
                source_label=f"{m['n']} ground stations",
                n=int(m["n"]),
                metrics=m,
                note=STATION_FRAME_CAVEAT,
                reason=None,
            )

    # --- 3. nothing independent to score this region against
    return _answer(
        **common,
        source="none",
        reason=(
            "no independent local reference (fine satellite product or ground "
            "station) for this region"
        ),
        note="Any cards shown are the benchmark-region holdout, not this city.",
    )
