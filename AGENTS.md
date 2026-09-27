This is a project for the hackathon problem statement
Title: Downscaling of Satellite based air quality map using 

Description: Develop an AI/ML (Artificial Intelligence/Machine Learning) model to generate fine spatial resolution air quality map from coarse resolution satellite data. It should utilize existing python-based ML libraries. Developed model need to be validated with unseen independent data. Challenge: To utilize large satellite data having gaps under cloudy conditions To select suitable ML algorithm and ensure optimal fitting of ML model for desired accuracy To validate model output with unseen independent data Usage: To enhance air quality knowledge, Sharpen focus at local level Users: Researchers and government bodies monitoring/working on air quality assessment Available Solutions (if Yes, reasons for not using them): Individual components are available, comprehensive and proven solution does not exist. Desired Outcome: Fine resolution air quality map of NO2. 

Expected Solution: The proposed solution is an AI/ML-based air quality mapping system that generates high spatial resolution NO₂ concentration maps from coarse-resolution satellite observations. The system will preprocess and integrate satellite data with relevant environmental and meteorological parameters, while handling missing or incomplete observations caused by cloud cover through suitable data-processing and gap-filling techniques. Multiple Python-based machine learning algorithms can be evaluated and optimized to identify the model that provides the most accurate spatial enhancement of NO₂ concentrations. The trained model will then generate fineresolution NO₂ maps at the local level, providing a clearer representation of air quality variations within smaller geographic areas. To ensure reliability, the model outputs will be validated using unseen independent datasets and appropriate accuracy metrics. The resulting interactive maps and downloadable datasets can support researchers and government agencies in understanding local air pollution patterns, monitoring NO₂ levels, and making informed decisions for air quality assessment and management. 

Use uv
The college is TSEC, Bandra West and this hackathon team is in Khar West(which is very close by) so we are testing in Mumbai mostly

For a layman's analogy we have taken the following example
Setting up individual sensors everywhere to measure NO2 is expensive
One satellite can measure all this however it's so high up a area like Bandra and Santacruz appear the same
So we use a ml model to zoom in and estimate how does NO2 differ between these regions
A satellite also cannot measure during cloud cover so the cloud gaps have to be filled in 

The world bank has a nice relevant but very long [paper](https://documents1.worldbank.org/curated/en/099756509142664774/pdf/IDU-e9220273-ec3a-4910-baa6-197f26f8511a.pdf) if needed or asked fetch it albeit its long and in pdf format

## Project conventions

- Use **uv** for the backend: `cd backend && uv run serve` (FastAPI on :8000), `uv run scripts/smoke.py <preset> <days>` for headless pipeline checks. Never `pip install`.
- Frontend is **vanilla TypeScript + Vite** (no React), **Leaflet** dual synced maps, **OpenStreetMap** tiles as the keyless basemap: `cd frontend && npm run dev` (:5173, proxies `/api` to :8000). `npm run build` is the strict-tsc typecheck gate (no linter configured).
- Default preset is **Mumbai** (transfer mode — no local reference, so Train is disabled there). **London/Paris** are the supervised benchmarks: train there once, then Apply to Mumbai/Delhi. This matches "we are testing in Mumbai mostly" *and* the statement's independent-validation requirement.
- `backend/cache/` and `backend/artifacts/` are generated and git-ignored — regenerate, never commit. Do not commit or push anything unless explicitly asked.
- Full docs, API table, prior-art table and results live in `README.md`.