"""Panel "4 · Validation on unseen data" — per-frame, per-city check.

The panel used to re-render ``meta.metrics``: one number computed **once** at
train time, so scrubbing the timeline changed nothing, and on a transfer city
those numbers came from the benchmark region (London) while being read as if
they validated the city on screen.

``/api/validation/frame?t=`` re-scores the *unseen* samples of the frame the
map is painting.  This script rebuilds a deterministic result per cached city
(synthetic but structured, so no model run is needed), writes the meta.json a
trained result would have, and asserts for **every** cached city:

* the payload always answers — right shape, right frame label, never an error;
* where independent truth exists (fine reference, or ground stations inside
  the bbox) the metrics are finite **and actually differ between frames**;
* where none exists the answer is an honest ``available: false`` with a
  reason, instead of another region's numbers being passed off as local;
* a frame outside a temporal holdout says so rather than scoring seen data.

Usage:
    uv run scripts/check_frame_validation.py            # every cached city
    uv run scripts/check_frame_validation.py london mumbai
"""

from __future__ import annotations

import gc
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from backend import artifacts, validation
from backend.config import DATASET_DIR, SPLITS
from backend.dataset import load_dataset
from backend.grids import make_grid, upsample
from backend.losocv import _stations_in_bbox

# The default split withholds the tail of the timeline.
SPLIT = "spatiotemporal"
HOLDOUT_FRAC = 0.3

REQUIRED_KEYS = {
    "available",
    "source",
    "reason",
    "note",
    "frame",
    "split",
    "holdout_description",
    "n",
    "metrics",
}
METRIC_KEYS = {"n", "rmse", "mae", "skill_vs_baseline", "pattern_r2", "baseline_rmse"}


def build_result(summary: dict, key: str, out_dir: Path) -> dict:
    """Write ``predictions.npz`` + ``meta.json`` for ``key`` without a model run.

    The field only has to be deterministic and spatially structured: the check
    compares *scoring of frames against each other*, so a flat field would hide
    the exact bug being tested (identical metrics at every frame).
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

    ref_raw = data.get("ref")
    ref = np.asarray(ref_raw, dtype=np.float32) if ref_raw is not None else None
    if ref is None or ref.shape != pred.shape:
        ref = np.full(pred.shape, np.nan, dtype=np.float32)

    elev = np.asarray(data["elev"], dtype=np.float32)
    roads = np.asarray(data["roads"], dtype=np.float32)

    # Holdout: every 3rd cell is an unseen block, the tail of the timeline is
    # unseen in time — the same shape a trained benchmark result ships with.
    t_len = pred.shape[0]
    n_hold_hours = max(1, int(round(t_len * HOLDOUT_FRAC)))
    test_hours = np.zeros(t_len, dtype=bool)
    test_hours[t_len - n_hold_hours :] = True
    test_cell = np.zeros(elev.shape, dtype=bool)
    test_cell[::3] = True
    n_test = int((test_cell[None, :, :] & test_hours[:, None, None]).sum())

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions.npz",
        pred=pred.astype(np.float32),
        c_up=c_up.astype(np.float32),
        gap_up=gap_up.astype(np.float32),
        ref=ref,
        elev=elev,
        roads=roads,
        test_cell=test_cell,
        test_hours=test_hours,
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "mode": "benchmark",
                "model_name": "random_forest",
                "split": SPLIT,
                "holdout_description": SPLITS[SPLIT],
                "conserve": True,
                "n_test": n_test,
                "summary_key": key,
                "preset": summary.get("preset"),
                "gap_fraction": summary.get("gap_fraction"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "t_len": t_len,
        "hold_hours": n_hold_hours,
        "has_ref": bool(np.isfinite(ref).any()),
    }


def _ok_metrics(res: dict, where: str) -> str | None:
    m = res.get("metrics")
    if not isinstance(m, dict):
        return f"{where}: available but metrics missing"
    missing = METRIC_KEYS - set(m)
    if missing:
        return f"{where}: metrics missing {sorted(missing)}"
    if int(m["n"]) <= 0:
        return f"{where}: n={m['n']} cannot validate anything"
    for k in METRIC_KEYS:
        v = m[k]
        if k != "n" and not np.isfinite(float(v)):
            return f"{where}: {k}={v} is not finite"
    return None


def check_city(summary: dict, key: str, scratch: Path) -> tuple[bool, str, dict]:
    info = build_result(summary, key, scratch / "latest")
    t_len = info["t_len"]

    has_ref = info["has_ref"]
    stations = _stations_in_bbox([float(v) for v in summary["bbox"]])
    truth = "reference" if has_ref else ("stations" if len(stations) >= 3 else "none")

    probes = [0, t_len // 4, t_len // 2, t_len - 1, t_len, t_len + 3, -1]
    rmses: list[float] = []
    scored = 0
    unscored_reason = ""

    for t in probes:
        res = validation.frame_validation(t)  # must never raise
        missing = REQUIRED_KEYS - set(res)
        if missing:
            return False, f"t={t}: payload missing {sorted(missing)}", info
        frame = res.get("frame")
        if not isinstance(frame, dict) or not frame.get("label"):
            return False, f"t={t}: no frame label — the panel cannot say what it scored", info
        is_mean = t >= t_len or t < 0
        if bool(frame["is_mean"]) != is_mean:
            return False, f"t={t}: frame flagged is_mean={frame['is_mean']}", info
        if not res.get("holdout_description"):
            return False, f"t={t}: holdout_description empty", info

        if truth == "none":
            if res["available"] or res["source"] != "none":
                return False, f"t={t}: claimed validation without any local truth", info
            if not res.get("reason"):
                return False, f"t={t}: available=false but no reason to show the user", info
            continue

        if res["available"]:
            bad = _ok_metrics(res, f"t={t}")
            if bad:
                return False, bad, info
            if res["source"] != truth:
                return False, f"t={t}: source={res['source']}, expected {truth}", info
            scored += 1
            rmses.append(float(res["metrics"]["rmse"]))
        else:
            # Legal only for a frame outside the temporal holdout window.
            unscored_reason = res.get("reason") or ""
            if "holdout" not in unscored_reason and "unseen" not in unscored_reason:
                return False, f"t={t}: unexplained refusal: {unscored_reason}", info

    if truth == "none":
        return True, "no local truth — honest unavailable state at every frame", info
    if scored < 2:
        return False, f"only {scored} frames could be scored ({unscored_reason})", info
    spread = max(rmses) - min(rmses)
    if spread <= 0:
        return False, f"metrics identical at every frame (spread={spread:.3f}) — the bug is back", info
    return (
        True,
        f"{scored}/{len(probes)} frames scored, RMSE spread {spread:.3f} ug/m3 across the timeline",
        info,
    )


def main() -> int:
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    keys = sorted(p.name[: -len(".summary.json")] for p in DATASET_DIR.glob("*.summary.json"))
    if only:
        keys = [k for k in keys if k.split("_")[0] in only]
    if not keys:
        print("no cached datasets found")
        return 2

    failures = 0
    with tempfile.TemporaryDirectory(prefix="frameval_check_") as tmp:
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
                try:
                    ok, why, _info = check_city(summary, key, scratch)
                except Exception as exc:  # noqa: BLE001 — one city must not hide the rest
                    ok, why = False, f"raised {type(exc).__name__}: {exc}"
                if ok:
                    print(f"[PASS] {preset:<16} {why}")
                else:
                    failures += 1
                    print(f"[FAIL] {preset:<16} {why}")
                gc.collect()
        finally:
            artifacts.ARTIFACT_DIR = real_artifact_dir

    total = len(keys)
    print(f"\n{total - failures}/{total} cities score each timeline frame honestly")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
