from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "cache"
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATASET_DIR = CACHE_DIR / "datasets"
ROADS_CACHE = CACHE_DIR / "roads"
for _d in (CACHE_DIR, ARTIFACT_DIR, DATASET_DIR, ROADS_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://api.open-meteo.com/v1/forecast"
ELEV_URL = "https://api.open-meteo.com/v1/elevation"
OVERPASS_URLS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

USER_AGENT = "no2-downscale-hackathon/0.1 (research prototype)"
CHUNK = 40
COARSE_STEP = 0.25
DEFAULT_FINE_STEP = 0.01  # Strictly 0.01 deg (~1km hyper-local resolution)
DEFAULT_CLOUD_THRESHOLD = 60.0
MAX_DAYS = 30
MAX_HISTORY_DAYS = 92

WX_VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
    "cloud_cover",
    "precipitation",
    "boundary_layer_height",
]

CITIES = {
    "delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "kolkata": (22.5726, 88.3639),
    "chennai": (13.0827, 80.2707),
    "bengaluru": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867),
    "karachi": (24.8607, 67.0011),
    "lahore": (31.5204, 74.3587),
    "dhaka": (23.8103, 90.4125),
    "london": (51.5074, -0.1278),
    "paris": (48.8566, 2.3522),
    "berlin": (52.5200, 13.4050),
    "madrid": (40.4168, -3.7038),
    "rome": (41.9028, 12.4964),
    "amsterdam": (52.3676, 4.9041),
    "brussels": (50.8503, 4.3517),
    "zurich": (47.3769, 8.5417),
    "new_york": (40.7128, -74.0060),
    "los_angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "mexico_city": (19.4326, -99.1332),
    "sao_paulo": (-23.5505, -46.6333),
    "buenos_aires": (-34.6037, -58.3816),
    "cairo": (30.0444, 31.2357),
    "lagos": (6.5244, 3.3792),
    "johannesburg": (-26.2041, 28.0473),
    "nairobi": (-1.2921, 36.8219),
    "moscow": (55.7558, 37.6173),
    "istanbul": (41.0082, 28.9784),
    "dubai": (25.2048, 55.2708),
    "riyadh": (24.7136, 46.6753),
    "tehran": (35.6892, 51.3890),
    "beijing": (39.9042, 116.4074),
    "shanghai": (31.2304, 121.4737),
    "guangzhou": (23.1291, 113.2644),
    "tokyo": (35.6762, 139.6503),
    "seoul": (37.5665, 126.9780),
    "bangkok": (13.7563, 100.5018),
    "jakarta": (-6.2088, 106.8456),
    "singapore": (1.3521, 103.8198),
    "manila": (14.5995, 120.9842),
    "sydney": (-33.8688, 151.2093),
    "melbourne": (-37.8136, 144.9631),
    "toronto": (43.6532, -79.3832),
}

PRESETS: dict[str, dict] = {
    "nagpur": {
        "label": "Nagpur, India (Default)",
        "bbox": [78.98, 21.04, 79.20, 21.26],
        "center": [21.1458, 79.0882],
        "zoom": 11,
        "fine_reference": False,
        "notes": "Default Indian city: 0.01° (~1km) hyper-local downscaled grid",
    },
    "london": {
        "label": "London, UK",
        "bbox": [-0.51, 51.28, 0.33, 51.76],
        "center": [51.52, -0.09],
        "zoom": 9,
        "fine_reference": True,
        "notes": "CAMS European (0.1 deg) fine reference available",
    },
    "paris": {
        "label": "Paris, France",
        "bbox": [1.95, 48.65, 2.65, 49.05],
        "center": [48.86, 2.35],
        "zoom": 9,
        "fine_reference": True,
        "notes": "CAMS European (0.1 deg) fine reference available",
    },
    "delhi": {
        "label": "Delhi NCR, India",
        "bbox": [76.95, 28.30, 77.85, 28.95],
        "center": [28.61, 77.21],
        "zoom": 10,
        "fine_reference": False,
        "notes": "Outside CAMS European domain: transfer inference only",
    },
    "mumbai": {
        "label": "Mumbai, India",
        "bbox": [72.70, 18.85, 73.30, 19.35],
        "center": [19.10, 73.00],
        "zoom": 10,
        "fine_reference": False,
        "notes": "Outside CAMS European domain: transfer inference only",
    },
}

MODELS = {
    "lightgbm": "Spatial LightGBM (Gradient Boosting)",
    "xgboost": "XGBoost",
    "hist_gradient_boosting": "HistGradientBoosting",
    "extra_trees": "Extra Trees",
    "mlp": "MLP Neural Net",
    "random_forest": "Random Forest (Deprecated)",
}

SPLITS = {
    "sloso": "Spatial Leave-One-Station-Out (SLOSO)",
    "spatial": "Unseen spatial blocks",
    "temporal": "Unseen time window",
    "spatiotemporal": "Unseen blocks + time",
}
