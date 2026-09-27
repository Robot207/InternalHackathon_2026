from __future__ import annotations

import datetime as dt
import json

import numpy as np

from . import fetch as fch
from .config import (
    CITIES,
    DATASET_DIR,
    DEFAULT_CLOUD_THRESHOLD,
    FETCH_STEP,
    MAX_DAYS,
    PRESETS,
)
from .grids import block_mean, make_grid, upsample


def _resample(arr: np.ndarray, fgrid: dict, grid: dict) -> np.ndarray:
    """Bilinearly interpolate fetch-grid samples (axis 1/2) onto the fine grid."""
    if fgrid is grid:
        return arr
    return upsample(
        arr,
        {
            "clats": fgrid["lats"],
            "clons": fgrid["lons"],
            "lats": grid["lats"],
            "lons": grid["lons"],
        },
    )


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
) -> dict:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset: {preset}")
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

    cfg = PRESETS[preset]
    grid = make_grid(cfg["bbox"], fine_step)
    h, w = grid["H"], grid["W"]
    # Data are fetched on the (much smaller) 0.05 deg fetch grid and bilinearly
    # resampled onto the 0.01 deg analysis grid. At 0.01 deg a city is thousands
    # of points, which is dozens of chunked upstream requests per pass and trips
    # the provider's HTTP 429 limiter; the fetch grid keeps each pass at a handful
    # of requests while the output resolution stays at ~1 km.
    fetch_step = max(fine_step, FETCH_STEP)
    if fetch_step == fine_step:
        fgrid = grid
    else:
        # Grow the fetch bbox by one full fetch cell so the fetch cell centres
        # always bracket every fine cell centre. Without a margin the fine
        # grid's outer ring lies outside the fetch grid and bilinear
        # interpolation clamps it to the nearest edge value (a constant border).
        pad = fetch_step
        lo0, la0, lo1, la1 = cfg["bbox"]
        fgrid = make_grid([lo0 - pad, la0 - pad, lo1 + pad, la1 + pad], fetch_step)
    if fgrid["H"] < 2 or fgrid["W"] < 2:
        fgrid = grid  # bbox smaller than one fetch cell: fetch at full resolution
    fh, fw = fgrid["H"], fgrid["W"]
    lats = np.repeat(fgrid["lats"], fw)
    lons = np.tile(fgrid["lons"], fh)
    warnings: list[str] = []

    report(0.02, "starting fetch")
    ref = np.full((0, h, w), np.nan)
    if cfg["fine_reference"]:
        try:
            ref_flat, times = fch.fetch_air_quality(
                lats, lons, start_date, end_date, "cams_europe", ["nitrogen_dioxide"], report
            )
            ref = ref_flat[:, :, 0].reshape(-1, fh, fw)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"fine reference fetch failed: {exc}")
    if ref.shape[0] == 0:
        ref = None
        if cfg["fine_reference"]:
            warnings.append("fine reference unavailable, running in coarse-only mode")

    coarse_flat, times = fch.fetch_air_quality(
        lats, lons, start_date, end_date, "cams_global", ["nitrogen_dioxide"], report
    )
    coarse_fetch = coarse_flat[:, :, 0].reshape(-1, fh, fw)

    wx_flat, wx_times = fch.fetch_weather(lats, lons, start_date, end_date, report)
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
        coarse_fetch = coarse_fetch[keep]
        if ref is not None:
            ref = ref[keep]
        wx_keep = [wx_times.index(t) for t in common]
        wx_flat = wx_flat[wx_keep]
    n_t = len(times)

    # Everything above was fetched on the coarse fetch grid; resample it onto
    # the 0.01 deg analysis grid that the model actually trains and renders on.
    coarse_fine = _resample(coarse_fetch, fgrid, grid)
    if ref is not None:
        ref = _resample(ref, fgrid, grid)
    wx = _resample(wx_flat.reshape(n_t, fh, fw, -1), fgrid, grid)

    elev = _resample(fch.fetch_elevation(lats, lons, report).reshape(1, fh, fw), fgrid, grid)[0]
    roads, roads_ok, roads_source = fch.fetch_road_density(
        cfg["bbox"], grid["lats"], grid["lons"], fine_step, preset, report
    )
    if not roads_ok:
        warnings.append(
            "OSM road density unavailable (no local extract and Overpass outage); feature set degraded"
        )

    temp = wx[..., 0]
    spd = wx[..., 1]
    direction = wx[..., 2]
    u, v = fch.wind_components(spd, direction)
    rh = wx[..., 3]
    cloud = wx[..., 4]
    precip = wx[..., 5]
    blh = wx[..., 6]

    report(0.8, "building coarse grid & physics-informed gap repair")
    coarse_blocks = block_mean(coarse_fine, grid)
    cloud_blocks = block_mean(cloud, grid)
    blh_blocks = block_mean(blh, grid)
    spd_blocks = block_mean(spd, grid)
    gap = np.where(
        np.isfinite(cloud_blocks), cloud_blocks > cloud_threshold, False
    ).astype(bool)
    observed = np.where(gap, np.nan, coarse_blocks)
    filled, gap_confidence = gapfill_meteo(observed, blh_blocks, spd_blocks, times)

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
        gap_confidence=gap_confidence.astype(np.float32),
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
        "gap_recovery_confidence": round(float(np.mean(gap_confidence)), 3),
        "gapfill_method": "physics-informed meteorological + spatiotemporal",
        "road_density_available": roads_ok,
        "sources": {
            "coarse_no2": "Open-Meteo CAMS global (~0.1 deg native, aggregated to 0.25 deg)",
            "fine_reference": "Open-Meteo CAMS European (0.1 deg)" if ref is not None else None,
            "meteorology": "Open-Meteo (ECMWF/KNMI model blend)",
            "elevation": "Open-Meteo DEM",
            "roads": roads_source if roads_ok else None,
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


def gapfill_meteo(
    observed: np.ndarray,
    blh_blocks: np.ndarray | None = None,
    spd_blocks: np.ndarray | None = None,
    times: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Physics-informed meteorological gap filling for satellite NO2 observations.

    Handles gaps under cloudy conditions using temporal interpolation modulated by
    boundary-layer height (trapping), wind speed (dispersion), and diurnal emission
    patterns, followed by spatial multi-directional neighbor diffusion.
    """
    t_len, hc, wc = observed.shape
    arr = observed.astype(np.float64).copy()
    confidence = np.ones((t_len, hc, wc), dtype=np.float32)

    hours = np.zeros(t_len, dtype=float)
    if times and len(times) == t_len:
        for ti, t_str in enumerate(times):
            try:
                part = t_str.split("T")[-1].split(":")[0]
                hours[ti] = float(part)
            except Exception:
                hours[ti] = float(ti % 24)
    else:
        hours = np.arange(t_len) % 24

    # Diurnal vehicular & industrial emission curves (morning rush 8-10, evening 18-21)
    diurnal_profile = (
        1.0
        + 0.35 * np.exp(-0.5 * ((hours - 9.0) / 2.0) ** 2)
        + 0.45 * np.exp(-0.5 * ((hours - 19.0) / 2.5) ** 2)
    )

    for i in range(hc):
        for j in range(wc):
            col = arr[:, i, j]
            ok = np.isfinite(col)
            if ok.all():
                continue
            if not ok.any():
                confidence[:, i, j] = 0.40
                continue

            idx = np.arange(t_len)
            interpolated = np.interp(idx, idx[ok], col[ok])

            # For cloud gaps, modulate by atmospheric ventilation index V = BLH * WindSpeed
            if blh_blocks is not None and spd_blocks is not None:
                blh_col = np.maximum(blh_blocks[:, i, j], 50.0)
                spd_col = np.maximum(spd_blocks[:, i, j], 0.5)
                dispersion_idx = 1.0 / np.sqrt(blh_col * spd_col)
                norm_disp = dispersion_idx / max(float(np.mean(dispersion_idx)), 1e-6)
                gap_mask = ~ok
                mod_factor = 0.5 * norm_disp[gap_mask] + 0.5 * (
                    diurnal_profile[gap_mask] / max(float(np.mean(diurnal_profile)), 1e-6)
                )
                mod_factor = np.clip(mod_factor, 0.55, 1.80)
                interpolated[gap_mask] = interpolated[gap_mask] * mod_factor

            for ti in range(t_len):
                if not ok[ti]:
                    min_dist = int(np.min(np.abs(idx[ok] - ti)))
                    confidence[ti, i, j] = float(max(0.45, 0.95 - 0.04 * min_dist))

            arr[:, i, j] = interpolated

    # Spatial neighbor diffusion for any remaining voids
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
            confidence[still & np.isfinite(new_vals)] = 0.55
            still = ~np.isfinite(filled)
        global_mean = np.nanmean(arr) if np.isfinite(arr).any() else 0.0
        filled = np.where(
            np.isfinite(filled), filled, global_mean if np.isfinite(global_mean) else 0.0
        )
        confidence[~np.isfinite(arr)] = 0.30
        arr = filled

    return arr, confidence


def gapfill(observed: np.ndarray) -> np.ndarray:
    return gapfill_meteo(observed)[0]


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

