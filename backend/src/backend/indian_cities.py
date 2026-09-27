"""Indian Cities Bounding Box (BBox) Database.

Provides bounding boxes (min_lat, max_lat, min_lon, max_lon) and Leaflet/CAMS
format [min_lon, min_lat, max_lon, max_lat] for 110+ top Indian cities.
Default city: Nagpur (Maharashtra, India).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class CityInfo(TypedDict):
    id: str
    name: str
    state: str
    center: list[float]  # [lat, lon]
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    bbox: list[float]  # [min_lon, min_lat, max_lon, max_lat]
    zoom: int


# Base list of 110+ top Indian cities with precise coordinates
_RAW_CITIES: list[tuple[str, str, str, float, float, float, float, int]] = [
    # id, name, state, lat, lon, lat_span, lon_span, zoom
    ("nagpur", "Nagpur", "Maharashtra", 21.1458, 79.0882, 0.22, 0.22, 11),
    ("mumbai", "Mumbai", "Maharashtra", 19.0760, 72.8777, 0.40, 0.35, 10),
    ("delhi", "Delhi NCR", "Delhi", 28.6139, 77.2090, 0.45, 0.50, 10),
    ("bengaluru", "Bengaluru", "Karnataka", 12.9716, 77.5946, 0.35, 0.35, 11),
    ("hyderabad", "Hyderabad", "Telangana", 17.3850, 78.4867, 0.35, 0.35, 11),
    ("ahmedabad", "Ahmedabad", "Gujarat", 23.0225, 72.5714, 0.30, 0.30, 11),
    ("chennai", "Chennai", "Tamil Nadu", 13.0827, 80.2707, 0.35, 0.30, 11),
    ("kolkata", "Kolkata", "West Bengal", 22.5726, 88.3639, 0.35, 0.30, 11),
    ("surat", "Surat", "Gujarat", 21.1702, 72.8311, 0.25, 0.25, 11),
    ("pune", "Pune", "Maharashtra", 18.5204, 73.8567, 0.30, 0.30, 11),
    ("jaipur", "Jaipur", "Rajasthan", 26.9124, 75.7873, 0.25, 0.25, 11),
    ("lucknow", "Lucknow", "Uttar Pradesh", 26.8467, 80.9462, 0.25, 0.25, 11),
    ("kanpur", "Kanpur", "Uttar Pradesh", 26.4499, 80.3319, 0.25, 0.25, 11),
    ("indore", "Indore", "Madhya Pradesh", 22.7196, 75.8577, 0.25, 0.25, 11),
    ("thane", "Thane", "Maharashtra", 19.2183, 72.9781, 0.20, 0.20, 11),
    ("bhopal", "Bhopal", "Madhya Pradesh", 23.2599, 77.4126, 0.25, 0.25, 11),
    ("visakhapatnam", "Visakhapatnam", "Andhra Pradesh", 17.6868, 83.2185, 0.25, 0.25, 11),
    ("pimpri_chinchwad", "Pimpri-Chinchwad", "Maharashtra", 18.6298, 73.7997, 0.20, 0.20, 11),
    ("patna", "Patna", "Bihar", 25.5941, 85.1376, 0.22, 0.25, 11),
    ("vadodara", "Vadodara", "Gujarat", 22.3072, 73.1812, 0.22, 0.22, 11),
    ("ghaziabad", "Ghaziabad", "Uttar Pradesh", 28.6692, 77.4538, 0.20, 0.20, 11),
    ("ludhiana", "Ludhiana", "Punjab", 30.9010, 75.8573, 0.22, 0.22, 11),
    ("agra", "Agra", "Uttar Pradesh", 27.1767, 78.0081, 0.22, 0.22, 11),
    ("nashik", "Nashik", "Maharashtra", 19.9975, 73.7898, 0.22, 0.22, 11),
    ("faridabad", "Faridabad", "Haryana", 28.4089, 77.3178, 0.20, 0.20, 11),
    ("meerut", "Meerut", "Uttar Pradesh", 28.9845, 77.7064, 0.20, 0.20, 11),
    ("rajkot", "Rajkot", "Gujarat", 22.3039, 70.8022, 0.20, 0.20, 11),
    ("varanasi", "Varanasi", "Uttar Pradesh", 25.3176, 82.9739, 0.20, 0.20, 11),
    ("srinagar", "Srinagar", "Jammu and Kashmir", 34.0837, 74.7973, 0.22, 0.22, 11),
    ("aurangabad", "Chhatrapati Sambhajinagar", "Maharashtra", 19.8762, 75.3433, 0.22, 0.22, 11),
    ("dhanbad", "Dhanbad", "Jharkhand", 23.7957, 86.4304, 0.20, 0.20, 11),
    ("amritsar", "Amritsar", "Punjab", 31.6340, 74.8723, 0.20, 0.20, 11),
    ("navi_mumbai", "Navi Mumbai", "Maharashtra", 19.0330, 73.0297, 0.22, 0.20, 11),
    ("prayagraj", "Prayagraj", "Uttar Pradesh", 25.4358, 81.8463, 0.22, 0.22, 11),
    ("ranchi", "Ranchi", "Jharkhand", 23.3441, 85.3096, 0.22, 0.22, 11),
    ("howrah", "Howrah", "West Bengal", 22.5958, 88.2636, 0.20, 0.20, 11),
    ("coimbatore", "Coimbatore", "Tamil Nadu", 11.0168, 76.9558, 0.25, 0.25, 11),
    ("jabalpur", "Jabalpur", "Madhya Pradesh", 23.1815, 79.9864, 0.22, 0.22, 11),
    ("gwalior", "Gwalior", "Madhya Pradesh", 26.2183, 78.1828, 0.22, 0.22, 11),
    ("vijayawada", "Vijayawada", "Andhra Pradesh", 16.5062, 80.6480, 0.22, 0.22, 11),
    ("jodhpur", "Jodhpur", "Rajasthan", 26.2389, 73.0243, 0.22, 0.22, 11),
    ("madurai", "Madurai", "Tamil Nadu", 9.9252, 78.1198, 0.22, 0.22, 11),
    ("raipur", "Raipur", "Chhattisgarh", 21.2514, 81.6296, 0.22, 0.22, 11),
    ("kota", "Kota", "Rajasthan", 25.2138, 75.8648, 0.20, 0.20, 11),
    ("guwahati", "Guwahati", "Assam", 26.1445, 91.7362, 0.22, 0.25, 11),
    ("chandigarh", "Chandigarh", "Chandigarh", 30.7333, 76.7794, 0.20, 0.20, 11),
    ("solapur", "Solapur", "Maharashtra", 17.6599, 75.9064, 0.20, 0.20, 11),
    ("hubballi", "Hubballi-Dharwad", "Karnataka", 15.3647, 75.1240, 0.22, 0.22, 11),
    ("bareilly", "Bareilly", "Uttar Pradesh", 28.3670, 79.4304, 0.20, 0.20, 11),
    ("moradabad", "Moradabad", "Uttar Pradesh", 28.8386, 78.7733, 0.20, 0.20, 11),
    ("mysuru", "Mysuru", "Karnataka", 12.2958, 76.6394, 0.20, 0.20, 11),
    ("gurugram", "Gurugram", "Haryana", 28.4595, 77.0266, 0.20, 0.20, 11),
    ("aligarh", "Aligarh", "Uttar Pradesh", 27.8974, 78.0880, 0.20, 0.20, 11),
    ("jalandhar", "Jalandhar", "Punjab", 31.3260, 75.5762, 0.20, 0.20, 11),
    ("tiruchirappalli", "Tiruchirappalli", "Tamil Nadu", 10.7905, 78.7047, 0.20, 0.20, 11),
    ("bhubaneswar", "Bhubaneswar", "Odisha", 20.2961, 85.8245, 0.22, 0.22, 11),
    ("salem", "Salem", "Tamil Nadu", 11.6643, 78.1460, 0.20, 0.20, 11),
    ("warangal", "Warangal", "Telangana", 17.9689, 79.5941, 0.20, 0.20, 11),
    ("thiruvananthapuram", "Thiruvananthapuram", "Kerala", 8.5241, 76.9366, 0.22, 0.20, 11),
    ("bhiwandi", "Bhiwandi", "Maharashtra", 19.2967, 73.0631, 0.18, 0.18, 12),
    ("saharanpur", "Saharanpur", "Uttar Pradesh", 29.9671, 77.5452, 0.20, 0.20, 11),
    ("guntur", "Guntur", "Andhra Pradesh", 16.3067, 80.4365, 0.20, 0.20, 11),
    ("amravati", "Amravati", "Maharashtra", 20.9320, 77.7523, 0.20, 0.20, 11),
    ("bikaner", "Bikaner", "Rajasthan", 28.0229, 73.3119, 0.20, 0.20, 11),
    ("noida", "Noida", "Uttar Pradesh", 28.5355, 77.3910, 0.20, 0.20, 11),
    ("jamshedpur", "Jamshedpur", "Jharkhand", 22.8046, 86.2029, 0.20, 0.20, 11),
    ("bhilai", "Bhilai", "Chhattisgarh", 21.1938, 81.3509, 0.20, 0.20, 11),
    ("cuttack", "Cuttack", "Odisha", 20.4625, 85.8828, 0.20, 0.20, 11),
    ("firozabad", "Firozabad", "Uttar Pradesh", 27.1593, 78.3957, 0.18, 0.18, 12),
    ("kochi", "Kochi", "Kerala", 9.9312, 76.2673, 0.22, 0.20, 11),
    ("nellore", "Nellore", "Andhra Pradesh", 14.4426, 79.9865, 0.20, 0.20, 11),
    ("bhavnagar", "Bhavnagar", "Gujarat", 21.7645, 72.1519, 0.20, 0.20, 11),
    ("dehradun", "Dehradun", "Uttarakhand", 30.3165, 78.0322, 0.22, 0.22, 11),
    ("durgapur", "Durgapur", "West Bengal", 23.5204, 87.3119, 0.20, 0.20, 11),
    ("asansol", "Asansol", "West Bengal", 23.6739, 86.9524, 0.20, 0.20, 11),
    ("rourkela", "Rourkela", "Odisha", 22.2604, 84.8536, 0.20, 0.20, 11),
    ("nanded", "Nanded", "Maharashtra", 19.1383, 77.3210, 0.20, 0.20, 11),
    ("kolhapur", "Kolhapur", "Maharashtra", 16.7050, 74.2433, 0.20, 0.20, 11),
    ("ajmer", "Ajmer", "Rajasthan", 26.4499, 74.6399, 0.20, 0.20, 11),
    ("akola", "Akola", "Maharashtra", 20.7002, 77.0082, 0.20, 0.20, 11),
    ("kalaburagi", "Kalaburagi (Gulbarga)", "Karnataka", 17.3297, 76.8343, 0.20, 0.20, 11),
    ("jamnagar", "Jamnagar", "Gujarat", 22.4707, 70.0577, 0.20, 0.20, 11),
    ("ujjain", "Ujjain", "Madhya Pradesh", 23.1765, 75.7885, 0.20, 0.20, 11),
    ("siliguri", "Siliguri", "West Bengal", 26.7271, 88.3953, 0.20, 0.20, 11),
    ("jhansi", "Jhansi", "Uttar Pradesh", 25.4484, 78.5685, 0.20, 0.20, 11),
    ("ulhasnagar", "Ulhasnagar", "Maharashtra", 19.2215, 73.1645, 0.18, 0.18, 12),
    ("jammu", "Jammu", "Jammu and Kashmir", 32.7266, 74.8570, 0.22, 0.22, 11),
    ("sangli", "Sangli-Miraj", "Maharashtra", 16.8524, 74.5815, 0.20, 0.20, 11),
    ("mangaluru", "Mangaluru", "Karnataka", 12.9141, 74.8560, 0.20, 0.20, 11),
    ("erode", "Erode", "Tamil Nadu", 11.3410, 77.7172, 0.20, 0.20, 11),
    ("belagavi", "Belagavi", "Karnataka", 15.8497, 74.4977, 0.20, 0.20, 11),
    ("tirunelveli", "Tirunelveli", "Tamil Nadu", 8.7139, 77.7567, 0.20, 0.20, 11),
    ("malegaon", "Malegaon", "Maharashtra", 20.5579, 74.5287, 0.18, 0.18, 12),
    ("gaya", "Gaya", "Bihar", 24.7914, 85.0002, 0.20, 0.20, 11),
    ("jalgaon", "Jalgaon", "Maharashtra", 21.0077, 75.5626, 0.20, 0.20, 11),
    ("udaipur", "Udaipur", "Rajasthan", 24.5854, 73.7125, 0.20, 0.20, 11),
    ("davanagere", "Davanagere", "Karnataka", 14.4644, 75.9218, 0.20, 0.20, 11),
    ("kozhikode", "Kozhikode", "Kerala", 11.2588, 75.7804, 0.20, 0.20, 11),
    ("kurnool", "Kurnool", "Andhra Pradesh", 15.8281, 78.0373, 0.20, 0.20, 11),
    ("rajahmundry", "Rajahmundry", "Andhra Pradesh", 17.0005, 81.8040, 0.20, 0.20, 11),
    ("bokaro", "Bokaro Steel City", "Jharkhand", 23.6693, 86.1511, 0.20, 0.20, 11),
    ("ballari", "Ballari", "Karnataka", 15.1394, 76.9214, 0.20, 0.20, 11),
    ("patiala", "Patiala", "Punjab", 30.3398, 76.3869, 0.20, 0.20, 11),
    ("agartala", "Agartala", "Tripura", 23.8315, 91.2868, 0.20, 0.20, 11),
    ("bhagalpur", "Bhagalpur", "Bihar", 25.2425, 86.9842, 0.20, 0.20, 11),
    ("muzaffarpur", "Muzaffarpur", "Bihar", 26.1209, 85.3647, 0.20, 0.20, 11),
    ("latur", "Latur", "Maharashtra", 18.4088, 76.5604, 0.20, 0.20, 11),
    ("dhule", "Dhule", "Maharashtra", 20.9042, 74.7749, 0.20, 0.20, 11),
    ("rohtak", "Rohtak", "Haryana", 28.8955, 76.6066, 0.20, 0.20, 11),
    ("korba", "Korba", "Chhattisgarh", 22.3595, 82.7501, 0.20, 0.20, 11),
    ("bhilwara", "Bhilwara", "Rajasthan", 25.3407, 74.6313, 0.20, 0.20, 11),
    ("berhampur", "Berhampur", "Odisha", 19.3150, 84.7941, 0.20, 0.20, 11),
    ("muzaffarnagar", "Muzaffarnagar", "Uttar Pradesh", 29.4727, 77.7085, 0.20, 0.20, 11),
    ("ahmednagar", "Ahmednagar", "Maharashtra", 19.0948, 74.7480, 0.20, 0.20, 11),
    ("mathura", "Mathura", "Uttar Pradesh", 27.4924, 77.6737, 0.20, 0.20, 11),
    ("kollam", "Kollam", "Kerala", 8.8932, 76.6141, 0.20, 0.20, 11),
    ("kadapa", "Kadapa", "Andhra Pradesh", 14.4673, 78.8242, 0.20, 0.20, 11),
    ("sambalpur", "Sambalpur", "Odisha", 21.4669, 83.9812, 0.20, 0.20, 11),
    ("bilaspur", "Bilaspur", "Chhattisgarh", 22.0797, 82.1409, 0.20, 0.20, 11),
    ("shahjahanpur", "Shahjahanpur", "Uttar Pradesh", 27.8804, 79.9077, 0.20, 0.20, 11),
    ("satara", "Satara", "Maharashtra", 17.6805, 73.9930, 0.18, 0.18, 12),
    ("vijayapura", "Vijayapura (Bijapur)", "Karnataka", 16.8302, 75.7100, 0.20, 0.20, 11),
    ("rampur", "Rampur", "Uttar Pradesh", 28.8154, 79.0257, 0.18, 0.18, 12),
    ("shivamogga", "Shivamogga", "Karnataka", 13.9299, 75.5681, 0.20, 0.20, 11),
    ("chandrapur", "Chandrapur", "Maharashtra", 19.9615, 79.2961, 0.20, 0.20, 11),
    ("junagadh", "Junagadh", "Gujarat", 21.5222, 70.4579, 0.20, 0.20, 11),
    ("thrissur", "Thrissur", "Kerala", 10.5276, 76.2144, 0.20, 0.20, 11),
    ("alwar", "Alwar", "Rajasthan", 27.5530, 76.6346, 0.20, 0.20, 11),
    ("bardhaman", "Bardhaman", "West Bengal", 23.2324, 87.8615, 0.20, 0.20, 11),
    ("kakinada", "Kakinada", "Andhra Pradesh", 16.9891, 82.2475, 0.20, 0.20, 11),
    ("nizamabad", "Nizamabad", "Telangana", 18.6725, 78.0941, 0.20, 0.20, 11),
    ("panipat", "Panipat", "Haryana", 29.3909, 76.9635, 0.20, 0.20, 11),
    ("darbhanga", "Darbhanga", "Bihar", 26.1542, 85.8918, 0.20, 0.20, 11),
    ("kharagpur", "Kharagpur", "West Bengal", 22.3460, 87.2320, 0.20, 0.20, 11),
    ("aizawl", "Aizawl", "Mizoram", 23.7271, 92.7176, 0.20, 0.20, 11),
    ("imphal", "Imphal", "Manipur", 24.8170, 93.9368, 0.20, 0.20, 11),
    ("shillong", "Shillong", "Meghalaya", 25.5788, 91.8933, 0.20, 0.20, 11),
    ("gangtok", "Gangtok", "Sikkim", 27.3389, 88.6065, 0.18, 0.18, 12),
    ("shimla", "Shimla", "Himachal Pradesh", 31.1048, 77.1734, 0.18, 0.18, 12),
    ("panaji", "Panaji", "Goa", 15.4909, 73.8278, 0.18, 0.18, 12),
    ("port_blair", "Port Blair", "Andaman and Nicobar", 11.6234, 92.7265, 0.20, 0.20, 11),
    ("puducherry", "Puducherry", "Puducherry", 11.9416, 79.8083, 0.18, 0.18, 12),
    ("khar_bandra", "Khar / Bandra (Mumbai)", "Maharashtra", 19.0600, 72.8350, 0.12, 0.12, 13),
]


def _build_database() -> dict[str, CityInfo]:
    db: dict[str, CityInfo] = {}
    for cid, name, state, lat, lon, lat_span, lon_span, zoom in _RAW_CITIES:
        min_lat = round(lat - lat_span / 2.0, 4)
        max_lat = round(lat + lat_span / 2.0, 4)
        min_lon = round(lon - lon_span / 2.0, 4)
        max_lon = round(lon + lon_span / 2.0, 4)
        db[cid] = {
            "id": cid,
            "name": f"{name}, {state}",
            "state": state,
            "center": [lat, lon],
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
            "bbox": [min_lon, min_lat, max_lon, max_lat],
            "zoom": zoom,
        }
    return db


INDIAN_CITIES: dict[str, CityInfo] = _build_database()
DEFAULT_CITY = "nagpur"


def get_city_bbox(city_name_or_id: str) -> CityInfo:
    """Retrieve city bounding box and metadata, checking INDIAN_CITIES, PRESETS, and custom coords."""
    key = city_name_or_id.lower().strip().replace(" ", "_").replace("-", "_")
    if key in INDIAN_CITIES:
        return INDIAN_CITIES[key]
    from .config import PRESETS
    if key in PRESETS:
        p = PRESETS[key]
        return {
            "id": key,
            "name": p["label"],
            "state": "International" if key in ["london", "paris"] else "India",
            "center": p["center"],
            "min_lat": p["bbox"][1],
            "max_lat": p["bbox"][3],
            "min_lon": p["bbox"][0],
            "max_lon": p["bbox"][2],
            "bbox": p["bbox"],
            "zoom": p["zoom"],
        }
    # Check partial match on name in INDIAN_CITIES
    for cid, info in INDIAN_CITIES.items():
        if key in cid or key in info["name"].lower():
            return info
    # Check partial match on PRESETS
    for pid, p in PRESETS.items():
        if key in pid or key in p["label"].lower():
            return {
                "id": pid,
                "name": p["label"],
                "state": "International" if pid in ["london", "paris"] else "India",
                "center": p["center"],
                "min_lat": p["bbox"][1],
                "max_lat": p["bbox"][3],
                "min_lon": p["bbox"][0],
                "max_lon": p["bbox"][2],
                "bbox": p["bbox"],
                "zoom": p["zoom"],
            }
    # Check custom coordinate pattern like "custom_19.07_72.87" or "coord_19.07_72.87"
    if key.startswith("custom_") or key.startswith("coord_"):
        parts = key.split("_")
        if len(parts) >= 3:
            try:
                lat = float(parts[1])
                lon = float(parts[2])
                d = 0.20
                return {
                    "id": key,
                    "name": f"Regional Site ({lat:.2f}°N, {lon:.2f}°E)",
                    "state": "Custom Region",
                    "center": [lat, lon],
                    "min_lat": round(lat - d, 4),
                    "max_lat": round(lat + d, 4),
                    "min_lon": round(lon - d, 4),
                    "max_lon": round(lon + d, 4),
                    "bbox": [round(lon - d, 4), round(lat - d, 4), round(lon + d, 4), round(lat + d, 4)],
                    "zoom": 11,
                }
            except Exception:
                pass
    return INDIAN_CITIES[DEFAULT_CITY]


def export_cities_json(target_path: Path | None = None) -> Path:
    """Export the 100+ Indian cities dictionary as JSON for frontend/API consumption."""
    p = target_path or (Path(__file__).resolve().parent / "indian_cities.json")
    data = {
        "default": DEFAULT_CITY,
        "count": len(INDIAN_CITIES),
        "cities": [
            {
                "value": info["id"],
                "label": info["name"],
                "state": info["state"],
                "center": info["center"],
                "min_lat": info["min_lat"],
                "max_lat": info["max_lat"],
                "min_lon": info["min_lon"],
                "max_lon": info["max_lon"],
                "bbox": info["bbox"],
                "zoom": info["zoom"],
            }
            for info in INDIAN_CITIES.values()
        ],
    }
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


# Auto-generate JSON config on import
JSON_PATH = export_cities_json()
