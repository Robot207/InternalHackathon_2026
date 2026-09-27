"""Ordinary Kriging upsampling: coarse 0.25 deg satellite pixels -> fine 0.01 deg grid.

This replaces the plain bilinear ``grids.upsample`` used to build the base field
``c_up`` so the pipeline is genuinely "XGBoost + Kriging":

* **Kriging** carries the coarse field onto the 1 km grid (geostatistical,
  variogram-weighted, exact at coarse cell centres).
* **XGBoost** then predicts the sub-grid log-residual ratio on top of it.

The variogram geometry is static for a given grid, so the kriging weight matrix
is solved ONCE per grid and reused for every timestep (see ``kriging_weights``).
The linear system is shared by all targets, so a single ``np.linalg.solve`` of a
``(n_coarse+1, n_coarse+1)`` system yields every fine cell's weights at once.

Pure numpy — no extra dependency (``pykrige``/``rasterio`` are not installed).
"""

from __future__ import annotations

import numpy as np

from .grids import haversine_km, upsample

# Standardised variogram: sill = 1, small nugget so the system stays well posed.
SILL = 1.0
NUGGET = 0.05

_WEIGHT_CACHE: dict[tuple, np.ndarray] = {}


def _hw(grid: dict) -> tuple[int, int]:
    """Fine grid shape, tolerating callers that omit explicit H/W (features._grid_from)."""
    if "H" in grid and "W" in grid:
        return int(grid["H"]), int(grid["W"])
    return len(grid["lats"]), len(grid["lons"])


def _grid_key(grid: dict) -> tuple:
    h, w = _hw(grid)
    return (
        h,
        w,
        int(grid["Hc"]),
        int(grid["Wc"]),
        float(np.min(grid["lats"])),
        float(np.max(grid["lats"])),
        float(np.min(grid["lons"])),
        float(np.max(grid["lons"])),
    )


def _spherical_cov(h_km: np.ndarray, rng_km: float) -> np.ndarray:
    """Spherical covariance C(h), with C(0) = sill."""
    if rng_km <= 0:
        return np.where(h_km <= 1e-9, SILL, 0.0)
    x = np.clip(h_km / rng_km, 0.0, 1.0)
    rho = 1.0 - 1.5 * x + 0.5 * x**3
    cov = (SILL - NUGGET) * rho
    cov = np.where(h_km <= 1e-9, SILL, cov)
    return np.where(x >= 1.0, 0.0, cov)


def kriging_weights(grid: dict) -> np.ndarray:
    """Return the ``(H*W, Hc*Wc)`` ordinary-kriging weight matrix for this grid.

    Rows sum to 1 (the unbiasedness constraint). Cached because the geometry
    never changes for a given bbox/resolution.
    """
    key = _grid_key(grid)
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    # Fine and coarse cell CENTRES (row-major: index = i * W + j, matching the
    # (H,W) / (Hc,Wc) array layout used everywhere else in the pipeline).
    f_lat, f_lon = np.meshgrid(
        np.asarray(grid["lats"], dtype=np.float64),
        np.asarray(grid["lons"], dtype=np.float64),
        indexing="ij",
    )
    c_lat, c_lon = np.meshgrid(
        np.asarray(grid["clats"], dtype=np.float64),
        np.asarray(grid["clons"], dtype=np.float64),
        indexing="ij",
    )
    fine_lat = f_lat.reshape(-1)
    fine_lon = f_lon.reshape(-1)
    coarse_lat = c_lat.reshape(-1)
    coarse_lon = c_lon.reshape(-1)

    n_f = fine_lat.size
    n_c = coarse_lat.size

    # Practical range = half the bbox diagonal, in km.
    span_lat = float(np.max(coarse_lat) - np.min(coarse_lat))
    span_lon = float(np.max(coarse_lon) - np.min(coarse_lon))
    diag_km = float(
        np.hypot(span_lat * 111.19, span_lon * 111.19 * np.cos(np.radians(float(np.mean(coarse_lat)))))
    )
    rng = max(diag_km * 0.5, 1.0)

    # Coarse <-> coarse covariance block (identical for every fine target).
    h_cc = haversine_km(
        coarse_lat[:, None], coarse_lon[:, None], coarse_lat[None, :], coarse_lon[None, :]
    )
    a = np.empty((n_c + 1, n_c + 1), dtype=np.float64)
    a[:n_c, :n_c] = _spherical_cov(h_cc, rng)
    a[np.arange(n_c), np.arange(n_c)] = SILL  # C(0) = sill (nugget discontinuity)
    a[:n_c, n_c] = 1.0
    a[n_c, :n_c] = 1.0
    a[n_c, n_c] = 0.0
    a[np.arange(n_c + 1), np.arange(n_c + 1)] += 1e-9  # ridge: guard singularity

    # RHS: covariance of each fine target with every coarse point, plus the
    # unbiasedness constraint. One shared solve covers all fine cells.
    h_fc = haversine_km(
        fine_lat[:, None], fine_lon[:, None], coarse_lat[None, :], coarse_lon[None, :]
    )
    rhs = np.empty((n_c + 1, n_f), dtype=np.float64)
    rhs[:n_c, :] = _spherical_cov(h_fc, rng).T
    rhs[n_c, :] = 1.0

    try:
        sol = np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError:  # pragma: no cover - numerical fallback
        sol = np.linalg.lstsq(a, rhs, rcond=None)[0]

    w = sol[:n_c, :].T  # (n_fine, n_coarse)

    # Renormalise against round-off so the unbiasedness constraint holds exactly.
    row = w.sum(axis=1, keepdims=True)
    w = w / np.where(np.abs(row) < 1e-12, 1.0, row)

    out = np.ascontiguousarray(w, dtype=np.float64)
    _WEIGHT_CACHE[key] = out
    return out


def kriging_upsample(coarse: np.ndarray, grid: dict, weights: np.ndarray | None = None) -> np.ndarray:
    """Ordinary-Kriging a ``(T, Hc, Wc)`` coarse stack onto the fine ``(T, H, W)`` grid.

    Cells whose kriging estimate is non-finite or non-positive fall back to the
    bilinear result so the downstream ``c_up > 0.5`` validity test never loses
    valid pixels.
    """
    coarse = np.asarray(coarse, dtype=np.float64)
    if coarse.ndim != 3:
        raise ValueError(f"expected (T,Hc,Wc), got shape {coarse.shape}")
    t_len = coarse.shape[0]
    w = kriging_weights(grid) if weights is None else weights
    n_f = w.shape[0]
    m = w.shape[1]

    flat = coarse.reshape(t_len, m)
    out = np.empty((t_len, n_f), dtype=np.float64)

    if np.isfinite(flat).all():
        # Fast path: no gaps -> a single matmul per timestep.
        out = flat @ w.T
    else:
        finite = np.isfinite(flat)
        for t in range(t_len):
            mask = finite[t].astype(np.float64)
            wt = w * mask[None, :]
            den = wt.sum(axis=1, keepdims=True)
            wt = wt / np.where(np.abs(den) < 1e-12, 1.0, den)
            out[t] = wt @ np.where(finite[t], flat[t], 0.0)

    out = out.reshape(t_len, *_hw(grid))

    bad = ~np.isfinite(out) | (out <= 0.0)
    if bad.any():
        with np.errstate(invalid="ignore"):
            bil = upsample(coarse, grid)
        out = np.where(bad, bil, out)
        out = np.where(np.isfinite(out), out, np.nan)

    return out
