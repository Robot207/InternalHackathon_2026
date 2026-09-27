from __future__ import annotations

import numpy as np


def make_grid(bbox: list[float], step: float) -> dict:
    lon_min, lat_min, lon_max, lat_max = bbox
    lats = np.arange(lat_min + step / 2.0, lat_max, step, dtype=np.float64)
    lons = np.arange(lon_min + step / 2.0, lon_max, step, dtype=np.float64)
    coarse_edges_lat = _edges(lat_min, lat_max, 0.25)
    coarse_edges_lon = _edges(lon_min, lon_max, 0.25)
    clats = (coarse_edges_lat[:-1] + coarse_edges_lat[1:]) / 2.0
    clons = (coarse_edges_lon[:-1] + coarse_edges_lon[1:]) / 2.0
    lat_idx = np.clip(
        np.searchsorted(coarse_edges_lat, lats, side="right") - 1, 0, len(clats) - 1
    )
    lon_idx = np.clip(
        np.searchsorted(coarse_edges_lon, lons, side="right") - 1, 0, len(clons) - 1
    )
    block_idx = np.add.outer(lat_idx * len(clons), lon_idx).astype(np.int32)
    return {
        "bbox": list(bbox),
        "step": step,
        "lats": lats,
        "lons": lons,
        "H": len(lats),
        "W": len(lons),
        "clats": clats,
        "clons": clons,
        "Hc": len(clats),
        "Wc": len(clons),
        "edges_lat": coarse_edges_lat,
        "edges_lon": coarse_edges_lon,
        "block_idx": block_idx,
        "n_blocks": len(clats) * len(clons),
    }


def _edges(lo: float, hi: float, step: float) -> np.ndarray:
    edges = [lo]
    x = lo
    while x + step < hi - 1e-9:
        x += step
        edges.append(x)
    edges.append(hi)
    return np.asarray(edges, dtype=np.float64)


def block_mean(arr: np.ndarray, grid: dict) -> np.ndarray:
    t_len = arr.shape[0]
    flat_idx = grid["block_idx"].ravel()
    n = grid["n_blocks"]
    out = np.full((t_len, n), np.nan, dtype=np.float64)
    vals = arr.reshape(t_len, -1)
    for t in range(t_len):
        v = vals[t]
        ok = np.isfinite(v)
        if not ok.any():
            continue
        num = np.bincount(flat_idx, weights=np.where(ok, v, 0.0), minlength=n)
        den = np.bincount(flat_idx, weights=ok.astype(np.float64), minlength=n)
        good = den > 0
        out[t, good] = num[good] / den[good]
    return out.reshape(t_len, grid["Hc"], grid["Wc"])


def _interp_axis(values: np.ndarray, src: np.ndarray, dst: np.ndarray, axis: int) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    order = np.argsort(src)
    src_s = src[order]
    idx = np.clip(np.searchsorted(src_s, dst), 1, len(src_s) - 1)
    lo = idx - 1
    hi = idx
    x0, x1 = src_s[lo], src_s[hi]
    span = np.where(x1 > x0, x1 - x0, 1.0)
    w_hi = np.clip((dst - x0) / span, 0.0, 1.0)
    w_lo = 1.0 - w_hi
    shape = (len(dst),) + (1,) * (values.ndim - 1)
    w_lo = w_lo.reshape(shape)
    w_hi = w_hi.reshape(shape)
    moved = np.moveaxis(values, axis, 0)
    res = moved[order][lo] * w_lo + moved[order][hi] * w_hi
    return np.moveaxis(res, 0, axis)


def upsample(coarse: np.ndarray, grid: dict) -> np.ndarray:
    step1 = _interp_axis(coarse, grid["clats"], grid["lats"], axis=1)
    return _interp_axis(step1, grid["clons"], grid["lons"], axis=2)


def block_upsample(coarse: np.ndarray, grid: dict) -> np.ndarray:
    """Nearest/block replication: every coarse cell becomes a crisp square."""
    t = coarse.shape[0]
    flat = np.asarray(coarse, dtype=np.float64).reshape(t, -1)
    idx = grid["block_idx"].ravel()
    return flat[:, idx].reshape(t, grid["H"], grid["W"])


def neighbor_stats(coarse: np.ndarray) -> dict:
    t_len, hc, wc = coarse.shape
    stack = []
    for dlat in (-1, 0, 1):
        for dlon in (-1, 0, 1):
            shifted = np.full_like(coarse, np.nan)
            rows_src = slice(max(0, -dlat), hc - max(0, dlat))
            rows_dst = slice(max(0, dlat), hc - max(0, -dlat))
            cols_src = slice(max(0, -dlon), wc - max(0, dlon))
            cols_dst = slice(max(0, dlon), wc - max(0, -dlon))
            shifted[:, rows_dst, cols_dst] = coarse[:, rows_src, cols_src]
            stack.append(shifted)
    arr = np.stack(stack, axis=0)
    with np.errstate(all="ignore"):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        mn = np.nanmin(arr, axis=0)
        mx = np.nanmax(arr, axis=0)
    return {
        "mean": np.where(np.isfinite(mean), mean, np.nan),
        "std": np.where(np.isfinite(std), std, np.nan),
        "min": np.where(np.isfinite(mn), mn, np.nan),
        "max": np.where(np.isfinite(mx), mx, np.nan),
    }


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
