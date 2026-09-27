# NO₂ Downscale · PSAI02

**AI/ML downscaling of satellite-based NO₂ air-quality maps.**
Satellites like Sentinel‑5P/TROPOMI measure nitrogen dioxide as *big* pixels
(~0.25°, ~25 km) and clouds wipe out large parts of every overpass. This tool
takes that coarse, patchy record and produces a finer (0.05°, ~5 km) hourly
NO₂ field — repairing cloud gaps first, then sharpening the field with a
learned ML model — shown side-by-side against the raw satellite pixel so the
result is inspectable, not just plausible.

![pipeline](docs/pipeline_stages.png)

## What the app does

| Left map — *raw satellite pixel* | Right map — *our model* |
| --- | --- |
| Coarse 0.25° pixels (crisp block view), cloud-gap mask, optional bilinear baseline, independent 0.1° reference | ML-downscaled 0.05° prediction, error vs reference, per-pixel comparison |

Both maps are **synced** (pan/zoom one, the other follows), carry pixel-edge
gridlines, and share a time slider + play button over the fetched window.
The sidebar runs the whole workflow: fetch → train → validate → export.

![block view](docs/block_check.png)

## Pipeline

1. **Fetch** (keyless open data)
   - Coarse NO₂: Open‑Meteo Air Quality (CAMS global, ~0.1° native) aggregated to 0.25° satellite-pixel blocks.
   - Fine reference (Europe only): CAMS European analysis at 0.1° — an *independent* product to validate against.
   - Meteorology: temperature, wind, humidity, cloud cover, precipitation, boundary-layer height (Open‑Meteo).
   - Static: elevation (Open‑Meteo DEM), OSM road density (Overpass, optional), nearest-city distance.
2. **Cloud-gap repair** — pixels whose cloud cover exceeds the cutoff (default 60%) are masked, then gap-filled by temporal interpolation within each block plus spatial neighbour blending. The mask itself is exposed in the UI as a layer (the UI reports the repaired fraction, e.g. 41% for London, 66% for monsoon Mumbai).
3. **Feature engineering** — coarse NO₂ + neighbour statistics, gap fraction, met features, cyclic hour/weekday, elevation, road density, coordinates, city distance (~23 features).
4. **Model** — scikit-learn/XGBoost regressors (`random_forest`, `extra_trees`, `hist_gradient_boosting`, `xgboost`, `mlp`) trained on the **log-ratio** between the fine truth and the bilinear coarse baseline, so the model learns sub-pixel *structure* rather than the smooth background.
5. **Conservation** — predictions are re-pinned so every 0.25° block keeps its original satellite total; the model may only redistribute mass inside the pixel.
6. **Validation** — held-out spatial blocks, a held-out time window, or both; scored against the bilinear baseline (RMSE/MAE/pattern r²/skill). Station CSV upload gives a fully independent check.
7. **Serve** — FastAPI job API + Vite/Leaflet dual-map frontend; results exportable as NetCDF.

## Available solutions (prior art) — and the gap

Research for the statement's "Available Solutions" section found only *partial*
components, nothing that combines gap repair + sharpening + honest validation
in one runnable tool:

| Project | Covers | Gap |
| --- | --- | --- |
| `ibm-granite/granite-geospatial-wxc-downscaling` (HF, TerraTorch) | General climate-variable downscaling | Not NO₂/air-quality; heavy pretrained setup |
| `Tasfiya025/ClimateSimulation_Downscaling_GAN` | GAN downscaling of climate simulations | Synthetic data, no satellite/cloud repair |
| `bstdenis/unet-rdps-to-hrdps-downscaling` | U-Net precipitation downscaling | Precipitation-specific |
| `minsughim/downNO2_XGB` (Kim et al., 2021 RSE) | XGBoost NO₂ downscaling | Ground-station supervision; no cloud repair |
| `HSG-AIML/Global-NO2-Estimation` | ML global NO₂ estimation | Estimation, not gap repair + sharpening |
| `masawdah/air_quality` (AI4EO/ESA) | EO air-quality enhancement | Narrow scope |
| `MariaDaneseISPC/rome-s5p-high-resolution-pollutants` | Precomputed Rome S5P products | Static outputs, no pipeline |
| `rbngz/IMP-2023` | Interpolation methods | Generic, not NO₂-aware |

## Quick start

Prerequisites: [`uv`](https://docs.astral.sh/uv/) (Python ≥ 3.13) and Node ≥ 20.

```bash
# 1. API + ML backend  (terminal A)
cd backend
uv sync
uv run serve            # http://127.0.0.1:8000

# 2. Web UI  (terminal B)
cd frontend
npm install
npm run dev             # http://localhost:5173  (proxies /api → :8000)
```

Open http://localhost:5173, then:

1. **Fetch coarse data** — the region defaults to **Mumbai** (our local test bed); London/Paris are the validation benchmarks, Delhi is transfer-only.
2. **Train & downscale** — pick a model + holdout split; watch progress live.
3. Explore layers, scrub the timeline, read the metrics cards, and **Download NetCDF**.

> **Fresh checkout note:** Mumbai has no independent local reference, so its *Train*
> button is disabled on a fresh clone. Fetch **London** (7 days) and *Train* once
> to create the model + benchmark metrics — then Apply works on Mumbai/Delhi.

Headless end-to-end check (no UI):

```bash
cd backend
uv run scripts/smoke.py london 7          # preset, days, [model]
```

`cd frontend && npm run build` type-checks and builds the UI for production.

## Results (reference run)

London, 7 days (2026‑09‑19 → 09‑25), Random Forest, holdout = unseen blocks + time,
scoring on 2,040 held-out samples:

| Metric | Model | Bilinear baseline |
| --- | --- | --- |
| RMSE (µg/m³) | **11.9** | 13.1 |
| MAE (µg/m³) | **9.4** | 11.3 |
| Skill vs baseline | **+9.1 %** | — |
| Pattern r² / Pearson r | **0.72 / 0.85** | — |

Notes: absolute values carry a systematic offset (~+9 µg/m³) because the coarse
input (CAMS *global*) and the reference (CAMS *European*) are different product
generations — the UI's residual panel makes this visible rather than hiding it.
Outside Europe there is no independent 0.1° reference, so those regions run in
**transfer mode**: the London-trained model is applied as-is (metrics cards
then show the benchmark-region stats, and a local full-field evaluation runs
whenever a reference exists).

## API

| Method & path | Purpose |
| --- | --- |
| `GET /api/health`, `GET /api/config` | Liveness; models, splits, presets |
| `GET /api/state` | Current dataset summary + result meta |
| `POST /api/fetch` | Download + gap-fill a region/date window (job) |
| `POST /api/train` | Train model, predict field, validate (job) |
| `POST /api/apply` | Apply trained model to the loaded dataset (job) |
| `POST /api/benchmark/models` | Multi-model arena leaderboard across all algorithms (job) |
| `GET /api/jobs/{id}` | Job progress/stage polling |
| `GET /api/result/meta`, `GET /api/result/layers` | Metrics/importances; grid layers for the map |
| `GET /api/point/inspect?lat=&lon=` | Hyperlocal point inspector (AQI, diurnal curve, landmark) |
| `GET /api/stations/benchmark` | Built-in CPCB CAAQMS stations & landmark pins (Mumbai) |
| `POST /api/stations/benchmark/validate` | 1-Click evaluate model against Mumbai CPCB ground sensors |
| `POST /api/validate/stations` | Custom CSV (`lon,lat,no2[,time]`) independent validation |
| `GET /api/export/netcdf` | NetCDF export of the current result |
| `GET /api/export/geojson` | GeoJSON polygon feature collection export |
| `GET /api/export/csv` | Tabular CSV export of downscaled predictions |

All long operations are jobs: they return `{job_id}` immediately and report
`progress` + human-readable `stage` (e.g. `evaluating Random Forest (1/6)`).


## Data sources

Everything is **API-key-free**: Open‑Meteo Air Quality & Forecast (CAMS,
ECMWF), Open‑Meteo DEM, OpenStreetMap Overpass (road density) and OSM raster
tiles. Data © Copernicus/Open‑Meteo open services, © OpenStreetMap contributors
(ODbL).

## Repository layout

```
backend/
  src/backend/         FastAPI app + pipeline modules
    config.py          presets, models, splits, API endpoints
    fetch.py           keyless downloads (NO₂/met/DEM/OSM)
    dataset.py         orchestration, cloud mask, gap-fill, caching
    features.py        23-feature matrix builder
    training.py        splits, fit, predict, conservation, metrics
    artifacts.py       meta/layers/NetCDF export
    validation.py      station CSV validation
  scripts/smoke.py     headless end-to-end test
  cache/, artifacts/   generated (git-ignored)
frontend/
  src/main.ts          workflow glue, job polling, map rendering
  src/map.ts           synced dual Leaflet maps + gridline pane
  src/gridImage.ts     grid arrays → PNG overlays
docs/                  pipeline & result screenshots
```

## Project conventions

- **Backend**: Python ≥ 3.13 managed by **uv only** (`uv sync`, `uv run …`) — never `pip install`. FastAPI app under `backend/src/backend/` with one module per pipeline stage (`src/` layout).
- **Frontend**: **vanilla TypeScript + Vite** (no React or other framework), **Leaflet** for the synced dual maps, **OpenStreetMap** raster tiles as the keyless basemap (dark styling via CSS filter). The dev server proxies `/api` → `127.0.0.1:8000`.
- **Checks**: `cd frontend && npm run build` is the quality gate (strict `tsc` + Vite bundle; no linter is configured). `uv run scripts/smoke.py <preset> <days>` verifies the ML pipeline end-to-end headlessly.
- **Generated data** — `backend/cache/` and `backend/artifacts/` are git-ignored; always regenerate via Fetch/Train, never commit them.
- **Regions** — the default preset is **Mumbai** (transfer mode: no local reference). London/Paris are the supervised-training benchmarks; train on one of those before Apply works elsewhere.

## Limitations

- Independent fine reference exists **only over the CAMS European domain**;
  elsewhere validation is transfer-mode or station-CSV based.
- Heavily overcast windows (e.g. Mumbai monsoon) end up mostly gap-filled —
  the UI always shows the repaired fraction so this can't hide.
- OSM road density is best-effort (Overpass is flaky; failures are cached and
  reported as a degraded-feature warning).
- Research prototype — not a health advisory.
