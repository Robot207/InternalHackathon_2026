from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import artifacts, forecast, jobs
from .cities import DEFAULT_CITY, city_list, register_cities, resolve_city
from .config import ARTIFACT_DIR, BASE_DIR, DEFAULT_FINE_STEP, MODELS, PRESETS, SPLITS
from .dataset import build_dataset, dataset_paths, load_dataset
from .losocv import ALGORITHMS, run_loso, validation_summary
from .schemas import ApplyRequest, FetchRequest, TrainRequest
from .stations import evaluate_built_in_stations, get_stations_and_landmarks
from .training import run_apply, run_benchmark_arena, run_training
from .validation import validate_station_csv

# Fold the Indian-city bounding boxes into PRESETS before any request is served.
register_cities()

app = FastAPI(title="NO2 Satellite Downscaling API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: dict = {"summary": None, "meta": None}


def _restore_state() -> None:
    meta = artifacts.read_meta()
    if meta and meta.get("summary_key"):
        _, summary_path = dataset_paths(meta["summary_key"])
        if summary_path.exists():
            STATE["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    STATE["meta"] = meta


_restore_state()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "no2-downscale", "version": "0.1.0"}


@app.get("/api/config")
def get_config() -> dict:
    return {
        "models": [{"id": k, "label": v} for k, v in MODELS.items()],
        "splits": [{"id": k, "label": v} for k, v in SPLITS.items()],
        "presets": [
            {
                "id": k,
                "label": v["label"],
                "bbox": v["bbox"],
                "center": v["center"],
                "zoom": v["zoom"],
                "fine_reference": v["fine_reference"],
                "notes": v["notes"],
            }
            for k, v in PRESETS.items()
        ],
        "defaults": {
            "model": "xgboost",
            "split": "spatiotemporal",
            "preset": "mumbai",
            "city": DEFAULT_CITY,
        },
        "target_resolution": {
            "fine_step": DEFAULT_FINE_STEP,
            "label": f"{DEFAULT_FINE_STEP}° / ~1 km hyper-local",
            "coarse_step": 0.25,
        },
        "algorithms": ALGORITHMS,
        "n_cities": len(PRESETS),
    }


@app.get("/api/cities")
def get_cities() -> dict:
    """Bounding boxes (min_lat/max_lat/min_lon/max_lon) for every indexed city."""
    return {
        "default": DEFAULT_CITY,
        "count": len(city_list()),
        "cities": city_list(),
    }


@app.get("/api/validation/loso")
def get_loso(preset: str | None = None) -> dict:
    """Leave-One-Station-Out cross-validation metrics for the active domain."""
    return run_loso(preset)


@app.get("/api/forecast")
def get_forecast(preset: str = "mumbai", city: str | None = None) -> dict:
    """72-hour NO2 projection driven by the live Open-Meteo forecast.

    `city` wins over `preset`, exactly like /api/fetch, so the horizon is
    always computed for the bbox the UI is actually showing.
    """
    target = preset
    if city:
        resolved = resolve_city(city)
        if not resolved:
            raise HTTPException(400, f"unknown city '{city}'")
        target = resolved[0]
    if target not in PRESETS:
        raise HTTPException(400, f"unknown preset '{target}'")
    try:
        return forecast.build(target, STATE.get("summary"))
    except Exception as exc:  # noqa: BLE001 - provider outage must not 500 the UI
        raise HTTPException(502, f"forecast unavailable: {exc}") from exc


@app.get("/api/state")
def get_state() -> dict:
    summary = STATE.get("summary")
    return {
        "summary": summary,
        "meta": STATE.get("meta"),
        "has_model": artifacts.has_model(),
        "has_predictions": artifacts.has_predictions(),
        "has_layers": artifacts.layers_file().exists(),
        "validation": validation_summary(summary.get("preset") if summary else None),
    }


@app.post("/api/fetch")
def fetch(req: FetchRequest) -> dict:
    # `city` wins over `preset` so the UI can crop to an arbitrary Indian bbox.
    target = req.preset
    if req.city:
        resolved = resolve_city(req.city)
        if not resolved:
            raise HTTPException(400, f"unknown city '{req.city}'")
        target = resolved[0]
    if target not in PRESETS:
        raise HTTPException(400, f"unknown preset '{target}'")
    start, end = req.resolve_dates()

    def job(progress) -> dict:
        summary = build_dataset(
            target,
            start,
            end,
            req.fine_step,
            cloud_threshold=req.cloud_threshold,
            force=req.force,
            progress=progress,
        )
        STATE["summary"] = summary
        return summary

    job_id = jobs.submit("fetch", job)
    return {"job_id": job_id}


@app.post("/api/train")
def train(req: TrainRequest) -> dict:
    summary = STATE.get("summary")
    if not summary:
        raise HTTPException(400, "fetch a dataset before training")
    if req.model not in MODELS:
        raise HTTPException(400, f"unknown model '{req.model}'")
    if req.split not in SPLITS:
        raise HTTPException(400, f"unknown split '{req.split}'")
    key = summary["key"]

    def job(progress) -> dict:
        data = load_dataset(key)
        meta = run_training(data, req.model, req.split, req.conserve, summary, progress)
        artifacts.write_meta(meta)
        artifacts.write_layers(summary)
        STATE["meta"] = meta
        progress(1.0, "done")
        return meta

    job_id = jobs.submit("train", job)
    return {"job_id": job_id}


@app.post("/api/apply")
def apply_model(req: ApplyRequest) -> dict:
    summary = STATE.get("summary")
    if not summary:
        raise HTTPException(400, "fetch a dataset before running inference")
    if not artifacts.has_model():
        raise HTTPException(400, "no trained model available; train on a benchmark preset first")
    key = summary["key"]

    def job(progress) -> dict:
        data = load_dataset(key)
        meta = run_apply(data, req.conserve, summary, progress)
        artifacts.write_meta(meta)
        artifacts.write_layers(summary)
        STATE["meta"] = meta
        progress(1.0, "done")
        return meta

    job_id = jobs.submit("apply", job)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/result/meta")
def result_meta() -> dict:
    meta = artifacts.read_meta()
    if not meta:
        raise HTTPException(404, "no result yet")
    # Every result carries its validation envelope so the UI never has to guess.
    meta = dict(meta)
    meta["validation"] = validation_summary(meta.get("preset"))
    return meta


@app.get("/api/result/layers")
def result_layers() -> FileResponse:
    summary = STATE.get("summary")
    if not artifacts.has_predictions() or summary is None:
        raise HTTPException(404, "no prediction layers yet")
    # The UI frames the map on whatever grid comes back from here. Handing out
    # the previous run's file while another city is active teleports the view
    # back to that city, so only serve layers produced for the active dataset.
    meta = artifacts.read_meta()
    if not meta or meta.get("summary_key") != summary.get("key"):
        raise HTTPException(
            404,
            "prediction layers belong to a previous dataset — run Apply for the active city first",
        )
    path = artifacts.layers_file()
    pred_path = artifacts.latest_dir() / "predictions.npz"
    if not path.exists() or path.stat().st_mtime < pred_path.stat().st_mtime:
        artifacts.write_layers(summary)
    return FileResponse(path, media_type="application/json")


@app.get("/api/export/netcdf")
def export_netcdf() -> FileResponse:
    if not artifacts.has_predictions() or STATE.get("summary") is None:
        raise HTTPException(404, "no prediction layers yet")
    path = artifacts.latest_dir() / "export.nc"
    pred_path = artifacts.latest_dir() / "predictions.npz"
    try:
        if not path.exists() or path.stat().st_mtime < pred_path.stat().st_mtime:
            artifacts.export_netcdf(STATE["summary"])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"NetCDF export failed: {exc}") from exc
    return FileResponse(path, media_type="application/x-netcdf", filename="no2_downscaled.nc")


@app.post("/api/validate/stations")
async def validate_stations(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        result = validate_station_csv(content)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return result


@app.get("/api/stations/benchmark")
def get_benchmark_stations(preset: str = "mumbai") -> dict:
    return get_stations_and_landmarks(preset)


@app.post("/api/stations/benchmark/validate")
def validate_benchmark_stations(preset: str = "mumbai") -> dict:
    try:
        return evaluate_built_in_stations(preset)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/benchmark/models")
def benchmark_models(split: str = "spatiotemporal", conserve: bool = True) -> dict:
    summary = STATE.get("summary")
    if not summary:
        raise HTTPException(400, "fetch a dataset before running model benchmark")
    if not summary.get("has_reference"):
        raise HTTPException(
            400,
            "Multi-model arena requires a reference preset (London/Paris) to evaluate unseen validation accuracy.",
        )
    key = summary["key"]

    def job(progress) -> dict:
        data = load_dataset(key)
        arena_res = run_benchmark_arena(data, split, conserve, summary, progress)
        return arena_res

    job_id = jobs.submit("arena", job)
    return {"job_id": job_id}


@app.get("/api/point/inspect")
def inspect_point_endpoint(lat: float, lon: float) -> dict:
    summary = STATE.get("summary")
    if not summary or not artifacts.has_predictions():
        raise HTTPException(404, "No downscaling prediction available to inspect.")
    try:
        return artifacts.inspect_point(lat, lon, summary)
    except Exception as exc:
        raise HTTPException(400, f"Point inspection error: {exc}") from exc


@app.get("/api/export/geojson")
def export_geojson_endpoint() -> FileResponse:
    summary = STATE.get("summary")
    if not artifacts.has_predictions() or summary is None:
        raise HTTPException(404, "no prediction layers yet")
    path = artifacts.export_geojson(summary)
    return FileResponse(
        path,
        media_type="application/geo+json",
        filename=f"no2_downscaled_{summary.get('preset', 'mumbai')}.geojson",
    )


@app.get("/api/export/csv")
def export_csv_endpoint() -> FileResponse:
    summary = STATE.get("summary")
    if not artifacts.has_predictions() or summary is None:
        raise HTTPException(404, "no prediction layers yet")
    path = artifacts.export_csv(summary)
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"no2_downscaled_{summary.get('preset', 'mumbai')}.csv",
    )


@app.get("/api/export/geotiff")
def export_geotiff_endpoint() -> FileResponse:
    """GeoTIFF export.

    Requires ``rasterio`` (``uv add rasterio``). Until it is present the endpoint
    answers 501 with a clear message so the UI can fall back to NetCDF rather
    than silently serving a mislabelled file.
    """
    summary = STATE.get("summary")
    if not artifacts.has_predictions() or summary is None:
        raise HTTPException(404, "no prediction layers yet")
    try:
        path = artifacts.export_geotiff(summary)
    except ImportError as exc:
        raise HTTPException(501, f"GeoTIFF unavailable: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"GeoTIFF export failed: {exc}") from exc
    return FileResponse(
        path,
        media_type="image/tiff",
        filename=f"no2_downscaled_{summary.get('preset', 'city')}.tif",
    )



@app.exception_handler(Exception)
async def unhandled(request, exc):  # noqa: ANN001
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
