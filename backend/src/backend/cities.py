"""Bounding boxes for Indian cities.

Injected into ``config.PRESETS`` by :func:`register_cities` so every downstream
consumer (dataset build, artifacts, stations, ``GET /api/config``) keeps working
without modification.

Each city is stored as ``(label, lat, lon, radius_deg)`` and expanded into an
explicit bounding box with the keys requested by the problem statement::

    {"min_lat": ..., "max_lat": ..., "min_lon": ..., "max_lon": ...}

The canonical ``bbox`` list written into PRESETS keeps the existing wire format
``[lon_min, lat_min, lon_max, lat_max]`` used by ``config.PRESETS``.
"""

from __future__ import annotations

from .config import PRESETS

# Fallback used when a city lookup misses or the request omits both city/preset.
DEFAULT_CITY = "nagpur"

# slug: (label, latitude, longitude, half-extent in degrees)
# Radius 0.25 deg (~28 km box) by default; larger metros get a wider box.
_INDIAN_CITIES: dict[str, tuple[str, float, float, float]] = {
    "nagpur": ("Nagpur, Maharashtra", 21.1458, 79.0882, 0.28),
    "mumbai": ("Mumbai, Maharashtra", 19.0760, 72.8777, 0.30),
    "delhi": ("Delhi NCR, India", 28.6139, 77.2090, 0.35),
    "bengaluru": ("Bengaluru, Karnataka", 12.9716, 77.5946, 0.30),
    "hyderabad": ("Hyderabad, Telangana", 17.3850, 78.4867, 0.30),
    "chennai": ("Chennai, Tamil Nadu", 13.0827, 80.2707, 0.30),
    "kolkata": ("Kolkata, West Bengal", 22.5726, 88.3639, 0.30),
    "pune": ("Pune, Maharashtra", 18.5204, 73.8567, 0.28),
    "ahmedabad": ("Ahmedabad, Gujarat", 23.0225, 72.5714, 0.28),
    "jaipur": ("Jaipur, Rajasthan", 26.9124, 75.7873, 0.28),
    "surat": ("Surat, Gujarat", 21.1702, 72.8311, 0.26),
    "lucknow": ("Lucknow, Uttar Pradesh", 26.8467, 80.9462, 0.28),
    "bhopal": ("Bhopal, Madhya Pradesh", 23.2599, 77.4126, 0.28),
    "patna": ("Patna, Bihar", 25.5941, 85.1376, 0.28),
    "visakhapatnam": ("Visakhapatnam, Andhra Pradesh", 17.6868, 83.2185, 0.28),
    "vadodara": ("Vadodara, Gujarat", 22.3072, 73.1812, 0.25),
    "kochi": ("Kochi, Kerala", 9.9312, 76.2673, 0.25),
    "coimbatore": ("Coimbatore, Tamil Nadu", 11.0168, 76.9558, 0.26),
    "chandigarh": ("Chandigarh, India", 30.7333, 76.7794, 0.25),
    "guwahati": ("Guwahati, Assam", 26.1445, 91.7362, 0.26),
    "bhubaneswar": ("Bhubaneswar, Odisha", 20.2961, 85.8245, 0.26),
    "mysuru": ("Mysuru, Karnataka", 12.2958, 76.6394, 0.25),
    "tiruchirappalli": ("Tiruchirappalli, Tamil Nadu", 10.7905, 78.7047, 0.25),
    "madurai": ("Madurai, Tamil Nadu", 9.9252, 78.1198, 0.25),
    "vijayawada": ("Vijayawada, Andhra Pradesh", 16.5062, 80.6480, 0.25),
    "agra": ("Agra, Uttar Pradesh", 27.1767, 78.0081, 0.25),
    "kanpur": ("Kanpur, Uttar Pradesh", 26.4499, 80.3319, 0.28),
    "varanasi": ("Varanasi, Uttar Pradesh", 25.3176, 82.9739, 0.25),
    "meerut": ("Meerut, Uttar Pradesh", 28.9845, 77.7064, 0.25),
    "rajkot": ("Rajkot, Gujarat", 22.3039, 70.8022, 0.26),
    "jodhpur": ("Jodhpur, Rajasthan", 26.2389, 73.0243, 0.26),
    "ranchi": ("Ranchi, Jharkhand", 23.3441, 85.3096, 0.26),
    "raipur": ("Raipur, Chhattisgarh", 21.2514, 81.6296, 0.26),
    "dehradun": ("Dehradun, Uttarakhand", 30.3165, 78.0322, 0.25),
    "noida": ("Noida, Uttar Pradesh", 28.5355, 77.3910, 0.25),
    "gurugram": ("Gurugram, Haryana", 28.4595, 77.0266, 0.25),
    "faridabad": ("Faridabad, Haryana", 28.4089, 77.3178, 0.25),
    "ghaziabad": ("Ghaziabad, Uttar Pradesh", 28.6692, 77.4538, 0.25),
    "thiruvananthapuram": ("Thiruvananthapuram, Kerala", 8.5241, 76.9366, 0.25),
    "kozhikode": ("Kozhikode, Kerala", 11.2588, 75.7804, 0.25),
    "thrissur": ("Thrissur, Kerala", 10.5276, 76.2144, 0.24),
    "mangaluru": ("Mangaluru, Karnataka", 12.9141, 74.8560, 0.25),
    "hubballi": ("Hubballi, Karnataka", 15.3647, 75.1240, 0.25),
    "belagavi": ("Belagavi, Karnataka", 15.8497, 74.4977, 0.25),
    "nashik": ("Nashik, Maharashtra", 19.9975, 73.7898, 0.26),
    "aurangabad": ("Chhatrapati Sambhajinagar, Maharashtra", 19.8762, 75.3433, 0.26),
    "solapur": ("Solapur, Maharashtra", 17.6599, 75.9064, 0.25),
    "amravati": ("Amravati, Maharashtra", 20.9374, 77.7796, 0.25),
    "kolhapur": ("Kolhapur, Maharashtra", 16.7050, 74.2433, 0.25),
    "sangli": ("Sangli, Maharashtra", 16.8524, 74.5815, 0.24),
    "rajahmundry": ("Rajahmundry, Andhra Pradesh", 17.0005, 81.8040, 0.25),
    "guntur": ("Guntur, Andhra Pradesh", 16.3067, 80.4365, 0.25),
    "warangal": ("Warangal, Telangana", 17.9689, 79.5941, 0.25),
    "tirupati": ("Tirupati, Andhra Pradesh", 13.6288, 79.4192, 0.25),
    "salem": ("Salem, Tamil Nadu", 11.6643, 78.1460, 0.25),
    "erode": ("Erode, Tamil Nadu", 11.3410, 77.7172, 0.24),
    "thoothukudi": ("Thoothukudi, Tamil Nadu", 8.7642, 78.1348, 0.24),
    "tirunelveli": ("Tirunelveli, Tamil Nadu", 8.7139, 77.7567, 0.24),
    "jamshedpur": ("Jamshedpur, Jharkhand", 22.8046, 86.2029, 0.25),
    "dhanbad": ("Dhanbad, Jharkhand", 23.7957, 86.4304, 0.25),
    "gaya": ("Gaya, Bihar", 24.7955, 85.0002, 0.24),
    "bhagalpur": ("Bhagalpur, Bihar", 25.2425, 86.9740, 0.24),
    "muzaffarpur": ("Muzaffarpur, Bihar", 26.1209, 85.3647, 0.24),
    "purnia": ("Purnia, Bihar", 25.7770, 87.4750, 0.24),
    "siliguri": ("Siliguri, West Bengal", 26.7271, 88.3953, 0.25),
    "darjeeling": ("Darjeeling, West Bengal", 27.0360, 88.2627, 0.24),
    "agartala": ("Agartala, Tripura", 23.8315, 91.2868, 0.25),
    "imphal": ("Imphal, Manipur", 24.8170, 93.9368, 0.25),
    "shillong": ("Shillong, Meghalaya", 25.5788, 91.8933, 0.25),
    "aizawl": ("Aizawl, Mizoram", 23.7271, 92.7176, 0.25),
    "kohima": ("Kohima, Nagaland", 25.6751, 94.1086, 0.24),
    "itanagar": ("Itanagar, Arunachal Pradesh", 27.0844, 93.6053, 0.25),
    "gangtok": ("Gangtok, Sikkim", 27.3314, 88.6138, 0.24),
    "panaji": ("Panaji, Goa", 15.4909, 73.8278, 0.24),
    "margao": ("Margao, Goa", 15.2830, 73.9862, 0.24),
    "bhuj": ("Bhuj, Gujarat", 23.2420, 69.6669, 0.25),
    "jamnagar": ("Jamnagar, Gujarat", 22.4707, 70.0577, 0.25),
    "gandhinagar": ("Gandhinagar, Gujarat", 23.2156, 72.6369, 0.25),
    "anand": ("Anand, Gujarat", 22.5645, 72.9289, 0.24),
    "bhavnagar": ("Bhavnagar, Gujarat", 21.7645, 72.1519, 0.25),
    "udaipur": ("Udaipur, Rajasthan", 24.5854, 73.7125, 0.25),
    "ajmer": ("Ajmer, Rajasthan", 26.4499, 74.6399, 0.25),
    "bikaner": ("Bikaner, Rajasthan", 28.0229, 73.3119, 0.25),
    "alwar": ("Alwar, Rajasthan", 27.5530, 76.6346, 0.24),
    "kota": ("Kota, Rajasthan", 25.2138, 75.8648, 0.25),
    "jammu": ("Jammu, J&K", 32.7266, 74.8570, 0.25),
    "srinagar": ("Srinagar, J&K", 34.0837, 74.7973, 0.25),
    "leh": ("Leh, Ladakh", 34.1526, 77.5771, 0.25),
    "shimla": ("Shimla, Himachal Pradesh", 31.1048, 77.1734, 0.24),
    "haridwar": ("Haridwar, Uttarakhand", 29.9457, 78.1642, 0.24),
    "roorkee": ("Roorkee, Uttarakhand", 29.8543, 77.8880, 0.24),
    "aligarh": ("Aligarh, Uttar Pradesh", 27.8974, 78.0880, 0.24),
    "moradabad": ("Moradabad, Uttar Pradesh", 28.8386, 78.7733, 0.24),
    "bareilly": ("Bareilly, Uttar Pradesh", 28.3670, 79.4304, 0.24),
    "gorakhpur": ("Gorakhpur, Uttar Pradesh", 26.7606, 83.3732, 0.25),
    "prayagraj": ("Prayagraj, Uttar Pradesh", 25.4358, 81.8463, 0.25),
    "jhansi": ("Jhansi, Uttar Pradesh", 25.4484, 78.5685, 0.24),
    "gwalior": ("Gwalior, Madhya Pradesh", 26.2183, 78.1828, 0.25),
    "jabalpur": ("Jabalpur, Madhya Pradesh", 23.1815, 79.9864, 0.25),
    "ujjain": ("Ujjain, Madhya Pradesh", 23.1793, 75.7849, 0.24),
    "indore": ("Indore, Madhya Pradesh", 22.7196, 75.8577, 0.28),
    "sagar": ("Sagar, Madhya Pradesh", 23.8388, 78.7378, 0.24),
    "rewa": ("Rewa, Madhya Pradesh", 24.5362, 81.3038, 0.24),
    "bilaspur": ("Bilaspur, Chhattisgarh", 22.0797, 82.1409, 0.25),
    "cuttack": ("Cuttack, Odisha", 20.4625, 85.8828, 0.25),
    "rourkela": ("Rourkela, Odisha", 22.2604, 84.8536, 0.25),
    "puri": ("Puri, Odisha", 19.8135, 85.8312, 0.24),
    "kharagpur": ("Kharagpur, West Bengal", 22.3460, 87.2320, 0.24),
    "asansol": ("Asansol, West Bengal", 23.6739, 86.9524, 0.24),
    "durgapur": ("Durgapur, West Bengal", 23.5204, 87.3119, 0.24),
    "haldia": ("Haldia, West Bengal", 22.0600, 88.1100, 0.24),
    "baharampur": ("Baharampur, West Bengal", 24.0988, 88.2679, 0.24),
    "pondicherry": ("Puducherry, India", 11.9416, 79.8083, 0.24),
    "tanjore": ("Thanjavur, Tamil Nadu", 10.7905, 79.1390, 0.24),
    "karur": ("Karur, Tamil Nadu", 10.9600, 78.0700, 0.24),
    "cuddalore": ("Cuddalore, Tamil Nadu", 11.7480, 79.7560, 0.24),
    "nellore": ("Nellore, Andhra Pradesh", 14.4426, 79.9865, 0.25),
    "kurnool": ("Kurnool, Andhra Pradesh", 15.8281, 78.0373, 0.25),
    "anantapur": ("Anantapur, Andhra Pradesh", 14.6819, 77.6006, 0.25),
    "kadapa": ("Kadapa, Andhra Pradesh", 14.4674, 78.8241, 0.25),
    "karimnagar": ("Karimnagar, Telangana", 18.4386, 79.1288, 0.25),
    "nizamabad": ("Nizamabad, Telangana", 18.6725, 78.0941, 0.25),
    "khammam": ("Khammam, Telangana", 17.2473, 80.1514, 0.25),
    "vizianagaram": ("Vizianagaram, Andhra Pradesh", 18.1067, 83.3956, 0.24),
    "eluru": ("Eluru, Andhra Pradesh", 16.7107, 81.0952, 0.24),
    "machilipatnam": ("Machilipatnam, Andhra Pradesh", 16.1900, 81.1400, 0.24),
    "kanchipuram": ("Kanchipuram, Tamil Nadu", 12.8342, 79.7036, 0.24),
    "hazari": ("Hazaribagh, Jharkhand", 23.9960, 85.3690, 0.24),
    "kumbakonam": ("Kumbakonam, Tamil Nadu", 10.9600, 79.3800, 0.24),
}

CITY_BBOXES: dict[str, dict] = {}


def _bbox(lat: float, lon: float, r: float) -> dict[str, float]:
    return {
        "min_lat": round(lat - r, 4),
        "max_lat": round(lat + r, 4),
        "min_lon": round(lon - r, 4),
        "max_lon": round(lon + r, 4),
    }


def city_list() -> list[dict]:
    """Cities in the shape ``GET /api/cities`` and the UI combobox consume."""
    out = []
    for slug, (label, lat, lon, r) in sorted(_INDIAN_CITIES.items()):
        bb = _bbox(lat, lon, r)
        out.append(
            {
                "id": slug,
                "label": label,
                "center": [round(lat, 4), round(lon, 4)],
                "zoom": 11,
                **bb,
            }
        )
    return out


def resolve_city(name: str | None) -> tuple[str, dict] | None:
    """Return ``(preset_id, preset_cfg)`` for a city name, or ``None`` if unknown.

    Matching is case/separator insensitive so "Nagpur", "nagpur" and
    "Nagpur, Maharashtra" all resolve.
    """
    if not name:
        return None
    key = name.strip().lower()
    if key in CITY_BBOXES:
        return key, CITY_BBOXES[key]
    if key in PRESETS:
        return key, PRESETS[key]
    for slug, (label, lat, lon, r) in _INDIAN_CITIES.items():
        short = label.lower().split(",")[0].strip()
        if key in (slug, label.lower(), short):
            cfg = CITY_BBOXES.get(slug) or PRESETS.get(slug)
            if cfg:
                return slug, cfg
    return None


def register_cities() -> int:
    """Merge city bounding boxes into ``config.PRESETS``.

    Existing entries (mumbai / delhi / london / paris / mumbai_suburbs) keep their
    curated bbox, ``fine_reference`` flag and notes — they are never overwritten.
    Safe to call more than once.
    """
    added = 0
    for slug, (label, lat, lon, r) in _INDIAN_CITIES.items():
        bb = _bbox(lat, lon, r)
        CITY_BBOXES[slug] = {
            "label": label,
            "bbox": [bb["min_lon"], bb["min_lat"], bb["max_lon"], bb["max_lat"]],
            "center": [round(lat, 4), round(lon, 4)],
            "zoom": 11,
            "fine_reference": False,
            "notes": "Indian city: outside CAMS European domain, transfer inference only",
            **bb,
        }
        if slug in PRESETS:
            continue
        PRESETS[slug] = {
            "label": label,
            "bbox": CITY_BBOXES[slug]["bbox"],
            "center": CITY_BBOXES[slug]["center"],
            "zoom": 11,
            "fine_reference": False,
            "notes": CITY_BBOXES[slug]["notes"],
        }
        added += 1
    return added


def city_bbox(name: str | None) -> list[float] | None:
    """Bounding box ``[lon_min, lat_min, lon_max, lat_max]`` for a city, if known.

    The curated PRESETS bbox wins over the derived one when both exist.
    """
    resolved = resolve_city(name)
    if not resolved:
        return None
    slug, cfg = resolved
    if slug in PRESETS:
        return list(PRESETS[slug]["bbox"])
    return list(cfg["bbox"])
