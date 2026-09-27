from __future__ import annotations

import datetime as dt
import json

import numpy as np

from . import fetch as fch
from .config import CITIES, DATASET_DIR, DEFAULT_CLOUD_THRESHOLD, MAX_DAYS, PRESETS
from .grids import block_mean, make_grid, upsample


def dataset_key(preset: str, start: str, end: str, step: float, cloud_threshold: float) -> str:
    return f"{preset}_{start}_{end}_s{step:.3f}_c{cloud_threshold:.0f}"


def dataset_paths(key: str) -> tuple:
    return DATASET_DIR / f"{key}.npz", DATASET_DIR / f"{key}.summary.json"


def validate_dates(start_date: str, end_date: str) -> None:
    sd = dt.date.fromisoformat(start_date)
    ed = dt.date.fromisoformat(end_date)
    if ed < sd:
        raise ValueError("end_date must be on or after start_date")
    if (ed - sd).days + 1 > MAX_DAYS:
        raise ValueError(f"date range limited to {MAX_DAYS} days")
    age = (dt.datetime.now(dt.timezone.utc).date() - sd).days
    if age > 92:
        raise ValueError("start_date is older than 92 days (meteorology API limit)")


def build_dataset(
    preset: str,
    start_date: str,
    end_date: str,
    fine_step: float,
    cloud_threshold: float = DEFAULT_CLOUD_THRESHOLD,
    force: bool = False,
    progress=None,
    city: str | None = None,
) -> dict:
    from .indian_cities import INDIAN_CITIES, get_city_bbox

    city_target = city or preset
    if city is not None or preset not in PRESETS:
        city_info = get_city_bbox(city_target)
        preset = city_info["id"]
        cfg = {
            "label": city_info["name"],
            "bbox": city_info["bbox"],
            "center": city_info["center"],
            "zoom": city_info["zoom"],
            "fine_reference": False,
            "notes": f"Indian City domain ({city_info['name']}): 0.01° (~1km) hyper-local inference",
        }
    else:
        cfg = PRESETS[preset]

    validate_dates(start_date, end_date)
    key = dataset_key(preset, start_date, end_date, fine_step, cloud_threshold)
    npz_path, sum_path = dataset_paths(key)
    if npz_path.exists() and sum_path.exists() and not force:
        summary = json.loads(sum_path.read_text(encoding="utf-8"))
        summary["cached"] = True
        return summary

    def report(p: float, stage: str):
        if progress:
            progress(p, stage)

    grid = make_grid(cfg["bbox"], fine_step)
    h, w = grid["H"], grid["W"]
    clats = grid["clats"]
    clons = grid["clons"]
    Hc, Wc = len(clats), len(clons)
    clat_mesh, clon_mesh = np.meshgrid(clats, clons, indexing="ij")
    clats_flat = clat_mesh.ravel()
    clons_flat = clon_mesh.ravel()
    warnings: list[str] = []

    report(0.08, "fetching coarse satellite NO2 (0.25 deg CAMS)")
    coarse_flat, times = fch.fetch_air_quality(
        clats_flat, clons_flat, start_date, end_date, "cams_global", ["nitrogen_dioxide"], report
    )
    n_t = coarse_flat.shape[1]
    coarse_blocks = coarse_flat[:, :, 0].T.reshape(n_t, Hc, Wc)
    coarse_fine = upsample(coarse_blocks, grid)

    ref = None

    report(0.28, "fetching meteorology (ECMWF blend)")
    wx_flat, wx_times = fch.fetch_weather(clats_flat, clons_flat, start_date, end_date, report)
    if wx_times != times:
        if not wx_times:
            raise RuntimeError("meteorology returned no data")
        common = [t for t in times if t in set(wx_times)]
        if not common:
            raise RuntimeError("NO2 and meteorology timestamps do not overlap")
        warnings.append("trimmed timestamps to meteorology overlap")
        pos = {t: i for i, t in enumerate(times)}
        keep = [pos[t] for t in common]
        times = common
        coarse_fine = coarse_fine[keep]
        coarse_blocks = coarse_blocks[keep]
        wx_keep = [wx_times.index(t) for t in common]
        wx_flat = wx_flat[wx_keep]

    n_t = len(times)
    wx_coarse = wx_flat.transpose(1, 0, 2).reshape(n_t, Hc, Wc, -1)
    wx = np.stack([upsample(wx_coarse[..., vi], grid) for vi in range(wx_coarse.shape[-1])], axis=-1)

    elev_raw = fch.fetch_elevation(clats_flat, clons_flat, report)
    elev_coarse = elev_raw.reshape(1, Hc, Wc)
    elev = upsample(elev_coarse, grid)[0]

    roads, roads_ok = fch.fetch_road_density(
        cfg["bbox"], grid["lats"], grid["lons"], fine_step, preset, report
    )
    if not roads_ok:
        warnings.append("OSM road density unavailable (Overpass outage); feature set degraded")

    temp = wx[..., 0]
    spd = wx[..., 1]
    direction = wx[..., 2]
    u, v = fch.wind_components(spd, direction)
    rh = wx[..., 3]
    cloud = wx[..., 4]
    precip = wx[..., 5]
    blh = wx[..., 6]

    report(0.8, "building coarse grid")
    cloud_blocks = wx_coarse[..., 4]
    gap = np.where(
        np.isfinite(cloud_blocks), cloud_blocks > cloud_threshold, False
    ).astype(bool)
    observed = np.where(gap, np.nan, coarse_blocks)
    filled = gapfill(observed, coarse_blocks)

    if ref is not None:
        bad = int(np.sum(~np.isfinite(ref)))
        if bad == ref.size:
            warnings.append("fine reference contains no valid values")
            ref = None

    report(0.9, "saving dataset")
    np.savez_compressed(
        npz_path,
        times=np.asarray(times),
        lats=grid["lats"],
        lons=grid["lons"],
        clats=grid["clats"],
        clons=grid["clons"],
        block_idx=grid["block_idx"],
        coarse=coarse_blocks.astype(np.float32),
        coarse_filled=filled.astype(np.float32),
        cloud_gap=gap,
        cloud_blocks=cloud_blocks.astype(np.float32),
        ref=(ref.astype(np.float32) if ref is not None else np.full((1, 1, 1), np.nan, np.float32)),
        temp=temp.astype(np.float32),
        u=u.astype(np.float32),
        v=v.astype(np.float32),
        spd=spd.astype(np.float32),
        rh=rh.astype(np.float32),
        cloud=cloud.astype(np.float32),
        precip=precip.astype(np.float32),
        blh=blh.astype(np.float32),
        elev=elev.astype(np.float32),
        roads=roads.astype(np.float32),
        bbox=np.asarray(cfg["bbox"], dtype=np.float64),
        step=np.asarray([fine_step]),
        has_ref=np.asarray([ref is not None]),
    )

    summary = {
        "key": key,
        "preset": preset,
        "preset_label": cfg["label"],
        "bbox": cfg["bbox"],
        "start_date": start_date,
        "end_date": end_date,
        "fine_step": fine_step,
        "coarse_step": 0.25,
        "cloud_threshold": cloud_threshold,
        "n_times": n_t,
        "n_lat": h,
        "n_lon": w,
        "n_coarse": grid["n_blocks"],
        "times": [t + ":00Z" for t in times],
        "lats": [round(float(x), 5) for x in grid["lats"]],
        "lons": [round(float(x), 5) for x in grid["lons"]],
        "clats": [round(float(x), 5) for x in grid["clats"]],
        "clons": [round(float(x), 5) for x in grid["clons"]],
        "has_reference": bool(ref is not None),
        "gap_fraction": round(float(np.mean(gap)), 4),
        "road_density_available": roads_ok,
        "sources": {
            "coarse_no2": "Open-Meteo CAMS global (~0.1 deg native, aggregated to 0.25 deg)",
            "fine_reference": "Open-Meteo CAMS European (0.1 deg)" if ref is not None else None,
            "meteorology": "Open-Meteo (ECMWF/KNMI model blend)",
            "elevation": "Open-Meteo DEM",
            "roads": "OpenStreetMap Overpass (ODbL)" if roads_ok else None,
        },
        "warnings": warnings,
        "cached": False,
    }
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report(1.0, "dataset ready")
    return summary


def load_dataset(key: str) -> dict:
    npz_path, _ = dataset_paths(key)
    if not npz_path.exists():
        raise FileNotFoundError(f"dataset not found: {key}")
    data = np.load(npz_path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    summary_path = npz_path.with_suffix(".summary.json")
    out["summary"] = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return out


def gapfill(observed: np.ndarray, coarse_raw: np.ndarray | None = None) -> np.ndarray:
    t_len, hc, wc = observed.shape
    arr = observed.astype(np.float64).copy()
    for i in range(hc):
        for j in range(wc):
            col = arr[:, i, j]
            ok = np.isfinite(col)
            if ok.all() or not ok.any():
                continue
            idx = np.arange(t_len)
            col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
            arr[:, i, j] = col
    still = ~np.isfinite(arr)
    if still.any():
        filled = arr.copy()
        for _ in range(hc + wc):
            if not still.any():
                break
            base_acc = np.where(np.isfinite(filled), filled, 0.0)
            base_cnt = np.where(np.isfinite(filled), 1.0, 0.0)
            acc = np.zeros_like(base_acc)
            cnt = np.zeros_like(base_cnt)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a_s, c_s = _shift(base_acc, base_cnt, di, dj)
                acc += a_s
                cnt += c_s
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(cnt > 0, acc / np.maximum(cnt, 1e-9), np.nan)
            new_vals = np.where(still & np.isfinite(mean), mean, np.nan)
            filled = np.where(np.isfinite(new_vals), new_vals, filled)
            still = ~np.isfinite(filled)
        
        fallback_val = 22.0
        if coarse_raw is not None and np.isfinite(coarse_raw).any():
            fallback_val = float(np.nanmean(coarse_raw))
        elif np.isfinite(arr).any():
            fallback_val = float(np.nanmean(arr))
        if not np.isfinite(fallback_val) or fallback_val <= 1.0:
            fallback_val = 22.0
        filled = np.where(np.isfinite(filled), filled, fallback_val)
        arr = filled
    arr = np.where(np.isfinite(arr) & (arr > 0.5), arr, 20.0)
    return arr


def _shift(acc: np.ndarray, cnt: np.ndarray, di: int, dj: int):
    out_a = np.zeros_like(acc)
    out_c = np.zeros_like(cnt)
    src_r = slice(max(0, -di), acc.shape[0] - max(0, di))
    dst_r = slice(max(0, di), acc.shape[0] - max(0, -di))
    src_c = slice(max(0, -dj), acc.shape[1] - max(0, dj))
    dst_c = slice(max(0, dj), acc.shape[1] - max(0, -dj))
    out_a[dst_r, dst_c] = acc[src_r, src_c]
    out_c[dst_r, dst_c] = cnt[src_r, src_c]
    return out_a, out_c
