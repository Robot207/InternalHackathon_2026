"""Point inspector vs painted map: cross-city consistency check.

The UI paints ``layers.json`` frames on the canvas and quotes the same cell in
the HYPERLOCAL POINT INSPECTOR.  For every cached city this script rebuilds a
prediction cube (synthetic but deterministic, so no model run is needed), builds
the layers exactly like ``/api/result/layers`` does, and asserts that
``inspect_point(..., time_idx=t)`` returns *the value of frame t* — including
the special "period mean" frame the slider parks on at start-up.

It also reports what the pre-fix code would have shown (always the last hour),
which is the ~30 µg/m³ disagreement visible in the app.

Usage:
    uv run scripts/check_inspect.py            # every cached city
    uv run scripts/check_inspect.py delhi mumbai
"""

from __future__ import annotations

import gc
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from backend import artifacts
from backend.config import DATASET_DIR
from backend.dataset import load_dataset
from backend.grids import make_grid, upsample

# Both sides round to 2 dp, so 0.011 covers the worst-case rounding pair.
TOL = 0.011
N_RANDOM_POINTS = 10


def value_at(cube: list, t: int, t_len: int, r: int, c: int) -> float | None:
    """Value of the frame the map draws at (r, c); None where there is no data."""
    if 0 <= t < t_len:
        v = cube[t][r][c]
        return float(v) if isinstance(v, (int, float)) else None
    vals = [fr[r][c] for fr in cube if isinstance(fr[r][c], (int, float))]
    return float(np.mean(vals)) if vals else None


def synthetic_cube(summary: dict, key: str, out_dir: Path) -> dict:
    """Write a predictions.npz for ``key`` without running the model.

    The field only has to be deterministic and spatially structured: the check
    compares two readers of the *same* array, so the numbers themselves are
    irrelevant (a flat field would hide index bugs).
    """
    data = load_dataset(key)
    grid = make_grid(summary["bbox"], float(summary.get("fine_step") or 0.01))
    coarse = np.asarray(data["coarse_filled"], dtype=np.float64)
    gap = np.asarray(data["cloud_gap"], dtype=np.float64)
    c_up = upsample(coarse, grid)
    gap_up = upsample(gap, grid)

    lats, lons = grid["lats"], grid["lons"]
    LA, LO = np.meshgrid(lats - lats.min(), lons - lons.min(), indexing="ij")
    pattern = 1.0 + 0.35 * np.sin(LA * 60.0) * np.cos(LO * 45.0)
    phase = np.linspace(0.0, 2.0 * np.pi, c_up.shape[0]).reshape(-1, 1, 1)
    pred = np.where(np.isfinite(c_up), c_up * pattern * (1.0 + 0.2 * np.sin(phase)), np.nan)

    ref = data.get("ref")
    ref_arr = np.asarray(ref, dtype=np.float32) if ref is not None else None
    if ref_arr is None or ref_arr.shape != pred.shape:
        ref_arr = np.full(pred.shape, np.nan, dtype=np.float32)

    elev = np.asarray(data["elev"], dtype=np.float32)
    roads = np.asarray(data["roads"], dtype=np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions.npz",
        pred=pred.astype(np.float32),
        c_up=c_up.astype(np.float32),
        gap_up=gap_up.astype(np.float32),
        ref=ref_arr,
        elev=elev,
        roads=roads,
        test_cell=np.zeros(elev.shape, dtype=bool),
        test_hours=np.zeros(c_up.shape[0], dtype=bool),
    )
    return {"summary": summary, "grid": grid, "pred": pred}


def check_city(summary: dict, key: str, scratch: Path) -> tuple[bool, str, dict]:
    synthetic_cube(summary, key, scratch)
    layers = artifacts.build_layers(summary)
    # keep only the layers this check reads — the others cost ~100 MB each
    for name in list(layers["layers"]):
        if name not in ("prediction", "coarse", "coarse_bilinear"):
            del layers["layers"][name]
    pred_cube = layers["layers"]["prediction"]
    base_cube = layers["layers"]["coarse_bilinear"]
    block_cube = layers["layers"]["coarse"]
    t_len = len(pred_cube)
    lats = np.asarray(summary["lats"], dtype=float)
    lons = np.asarray(summary["lons"], dtype=float)

    rng = np.random.default_rng(7)
    coords: list[tuple[float, float]] = []
    for _ in range(N_RANDOM_POINTS):
        r = int(rng.integers(0, len(lats)))
        c = int(rng.integers(0, len(lons)))
        coords.append((float(lats[r]), float(lons[c])))
    coords.append((float(lats[len(lats) // 2]), float(lons[len(lons) // 2])))

    times = summary.get("times") or []
    probes = [0, 1, t_len // 2, t_len - 1, t_len, t_len + 3]
    probes += [int(rng.integers(0, t_len)) for _ in range(3)]

    worst = 0.0
    worst_where = ""
    checks = 0
    legacy_spread: list[float] = []
    legacy_note = ""

    for lat, lon in coords:
        ri = int(np.argmin(np.abs(lats - lat)))
        ci = int(np.argmin(np.abs(lons - lon)))
        for t in probes:
            res = artifacts.inspect_point(lat, lon, summary, time_idx=t)
            if not res["in_domain"]:
                return False, f"point ({lat}, {lon}) reported outside its own grid", {}
            got = res["current"]["downscaled_no2"]
            exp = value_at(pred_cube, t, t_len, ri, ci)
            if (got is None) != (exp is None):
                return False, f"t={t} cell=({ri},{ci}): got {got}, map paints {exp}", {}
            if got is not None and exp is not None:
                d = abs(got - exp)
                checks += 1
                if d > worst:
                    worst, worst_where = d, f"t={t} cell=({ri},{ci}) {got} vs {exp}"
                if d > TOL:
                    return False, f"ML value off by {d} at t={t} cell=({ri},{ci}): {got} vs {exp}", {}
                b = res["current"]["baseline_no2"]
                eb = value_at(base_cube, t, t_len, ri, ci)
                if b is not None and eb is not None and abs(b - eb) > TOL:
                    return False, f"baseline off by {abs(b - eb)} at t={t} cell=({ri},{ci})", {}
                blk = res["current"]["baseline_block_no2"]
                ebk = value_at(block_cube, t, t_len, ri, ci)
                if blk is not None and ebk is not None and abs(blk - ebk) > TOL:
                    return False, f"block baseline off by {abs(blk - ebk)} at t={t} cell=({ri},{ci})", {}
                if 0 <= t < t_len and t < len(times):
                    want = times[t].replace("T", " ").replace(":00Z", "Z")
                    if res["frame"]["label"] != want:
                        return False, f"frame label {res['frame']['label']} != {want}", {}
            # what the old (pre-fix) inspector always answered: the last hour
            if got is not None and 0 <= t < t_len and t != t_len - 1:
                old = pred_cube[t_len - 1][ri][ci]
                if isinstance(old, (int, float)):
                    legacy_spread.append(abs(float(old) - float(got)))

        # the period-mean frame must be labelled as such
        mean_res = artifacts.inspect_point(lat, lon, summary, time_idx=t_len)
        if not mean_res["frame"]["is_mean"] or mean_res["frame"]["index"] is not None:
            return False, "period-mean frame not flagged as mean", {}

    # an out-of-domain click must not answer with a neighbouring cell
    far = artifacts.inspect_point(float(lats[0]) + 5.0, float(lons[0]) + 5.0, summary, time_idx=0)
    if far["in_domain"] or far["current"]["downscaled_no2"] is not None:
        return False, "out-of-domain click returned a value", {}

    if legacy_spread:
        legacy_note = (
            f"pre-fix drift (last hour vs shown frame): mean {np.mean(legacy_spread):.1f}, "
            f"max {np.max(legacy_spread):.1f} ug/m3"
        )
    info = {"checks": checks, "worst": worst, "where": worst_where, "legacy": legacy_note}
    return True, "", info


def main() -> int:
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    keys = sorted(p.name[: -len(".summary.json")] for p in DATASET_DIR.glob("*.summary.json"))
    if only:
        keys = [k for k in keys if k.split("_")[0] in only]
    if not keys:
        print("no cached datasets found")
        return 2

    failures = 0
    with tempfile.TemporaryDirectory(prefix="inspect_check_") as tmp:
        scratch = Path(tmp)
        real_artifact_dir = artifacts.ARTIFACT_DIR
        try:
            # redirect writes so the running app's artifacts/latest is untouched
            artifacts.ARTIFACT_DIR = scratch
            for key in keys:
                summary = json.loads(
                    (DATASET_DIR / f"{key}.summary.json").read_text(encoding="utf-8")
                )
                preset = summary.get("preset") or key.split("_")[0]
                ok, why, info = check_city(summary, key, scratch / "latest")
                mark = "PASS" if ok else "FAIL"
                if ok:
                    print(
                        f"[{mark}] {preset:<16} {info['checks']:>4} value checks - "
                        f"max|d|={info['worst']:.3f} ug/m3 - {info['legacy']}"
                    )
                else:
                    failures += 1
                    print(f"[{mark}] {preset:<16} {why}")
                gc.collect()
        finally:
            artifacts.ARTIFACT_DIR = real_artifact_dir

    total = len(keys)
    print(f"\n{total - failures}/{total} cities agree with the painted map")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
