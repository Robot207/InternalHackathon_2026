"""Physical Ground Station Registry and Grouping for Spatial LOSO Validation.

Contains verified coordinates of physical monitoring stations (CPCB/CAAQMS
in India, AURN in London, Airparif in Paris) used to enforce strict
Spatial Leave-One-Station-Out (SLOSO) cross-validation via LeaveOneGroupOut.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GroundStation:
    station_id: str
    name: str
    network: str
    latitude: float
    longitude: float
    city: str


# Physical ground station network coordinates
GROUND_STATIONS: list[GroundStation] = [
    # Mumbai CAAQMS / CPCB Network (Maharashtra Pollution Control Board & SAFAR)
    GroundStation("IN_MUM_001", "Bandra West (TSEC/Linking Rd)", "CPCB/MPCB", 19.0596, 72.8295, "mumbai"),
    GroundStation("IN_MUM_002", "Khar West / Santacruz (S.V. Road)", "CPCB/MPCB", 19.0800, 72.8360, "mumbai"),
    GroundStation("IN_MUM_003", "Bandra Kurla Complex (BKC)", "CPCB/MPCB", 19.0664, 72.8687, "mumbai"),
    GroundStation("IN_MUM_004", "Kurla West", "CPCB/MPCB", 19.0657, 72.8794, "mumbai"),
    GroundStation("IN_MUM_005", "Worli Seaface", "CPCB/MPCB", 19.0178, 72.8170, "mumbai"),
    GroundStation("IN_MUM_006", "Colaba (Navy Nagar)", "CPCB/MPCB", 18.9067, 72.8147, "mumbai"),
    GroundStation("IN_MUM_007", "Sion Circle", "CPCB/MPCB", 19.0434, 72.8634, "mumbai"),
    GroundStation("IN_MUM_008", "Andheri West", "CPCB/MPCB", 19.1197, 72.8464, "mumbai"),
    GroundStation("IN_MUM_009", "Borivali East (Sanjay Gandhi NP)", "CPCB/MPCB", 19.2291, 72.8575, "mumbai"),
    GroundStation("IN_MUM_010", "Powai (IIT Bombay)", "CPCB/MPCB", 19.1232, 72.9094, "mumbai"),
    GroundStation("IN_MUM_011", "Chembur Mahul", "CPCB/MPCB", 19.0522, 72.8995, "mumbai"),
    GroundStation("IN_MUM_012", "Mulund West", "CPCB/MPCB", 19.1726, 72.9565, "mumbai"),
    GroundStation("IN_MUM_013", "Navi Mumbai (Vashi Sector 17)", "CPCB/MPCB", 19.0771, 72.9986, "mumbai"),
    GroundStation("IN_MUM_014", "Thane Naupada", "CPCB/MPCB", 19.1860, 72.9757, "mumbai"),

    # Delhi NCR CAAQMS Network
    GroundStation("IN_DEL_001", "Anand Vihar CAAQMS", "CPCB/DPCC", 28.6476, 77.3158, "delhi"),
    GroundStation("IN_DEL_002", "R K Puram CAAQMS", "CPCB/DPCC", 28.5632, 77.1869, "delhi"),
    GroundStation("IN_DEL_003", "Punjabi Bagh", "CPCB/DPCC", 28.6740, 77.1310, "delhi"),
    GroundStation("IN_DEL_004", "Mandir Marg", "CPCB/DPCC", 28.6364, 77.2010, "delhi"),
    GroundStation("IN_DEL_005", "IGI Airport T3", "CPCB/DPCC", 28.5627, 77.1180, "delhi"),
    GroundStation("IN_DEL_006", "ITO Crossroad", "CPCB/DPCC", 28.6286, 77.2410, "delhi"),
    GroundStation("IN_DEL_007", "Okhla Phase 2", "CPCB/DPCC", 28.5308, 77.2713, "delhi"),
    GroundStation("IN_DEL_008", "Bawana Industrial Area", "CPCB/DPCC", 28.7762, 77.0511, "delhi"),

    # London AURN (Automatic Urban and Rural Network)
    GroundStation("UK_LON_001", "London Marylebone Road", "AURN", 51.5225, -0.1546, "london"),
    GroundStation("UK_LON_002", "London Bloomsbury", "AURN", 51.5223, -0.1259, "london"),
    GroundStation("UK_LON_003", "London Westminster", "AURN", 51.4947, -0.1319, "london"),
    GroundStation("UK_LON_004", "London N. Kensington", "AURN", 51.5211, -0.2135, "london"),
    GroundStation("UK_LON_005", "London Bexley", "AURN", 51.4660, 0.1486, "london"),
    GroundStation("UK_LON_006", "London Eltham", "AURN", 51.4526, 0.0708, "london"),
    GroundStation("UK_LON_007", "London Haringey Priory Park", "AURN", 51.5993, -0.1062, "london"),
    GroundStation("UK_LON_008", "London Hillingdon", "AURN", 51.4963, -0.4608, "london"),
    GroundStation("UK_LON_009", "London Southwark", "AURN", 51.4906, -0.0988, "london"),
    GroundStation("UK_LON_010", "London Tower Hamlets", "AURN", 51.5170, -0.0430, "london"),

    # Paris Airparif Network
    GroundStation("FR_PAR_001", "Paris 1er Les Halles", "Airparif", 48.8614, 2.3470, "paris"),
    GroundStation("FR_PAR_002", "Paris 7eme Eiffel", "Airparif", 48.8584, 2.2945, "paris"),
    GroundStation("FR_PAR_003", "Paris 12eme Bercy", "Airparif", 48.8398, 2.3831, "paris"),
    GroundStation("FR_PAR_004", "Paris 18eme Blvd Peripherique", "Airparif", 48.8988, 2.3488, "paris"),
    GroundStation("FR_PAR_005", "Bobigny Centre", "Airparif", 48.9086, 2.4397, "paris"),
    GroundStation("FR_PAR_006", "Vitry-sur-Seine", "Airparif", 48.7874, 2.3927, "paris"),
    GroundStation("FR_PAR_007", "Neuilly-sur-Seine", "Airparif", 48.8848, 2.2687, "paris"),
    GroundStation("FR_PAR_008", "Gennevilliers", "Airparif", 48.9298, 2.2938, "paris"),
]


def get_stations_for_bbox(bbox: list[float] | tuple[float, ...]) -> list[GroundStation]:
    """Retrieve all physical ground stations strictly within the [min_lon, min_lat, max_lon, max_lat] bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    # Add a small buffer of 0.05 degrees to catch border stations
    buf = 0.05
    return [
        s for s in GROUND_STATIONS
        if (min_lat - buf <= s.latitude <= max_lat + buf)
        and (min_lon - buf <= s.longitude <= max_lon + buf)
    ]


def find_nearest_station(lat: float, lon: float, max_dist_km: float = 60.0) -> tuple[GroundStation | None, float]:
    """Find the closest physical ground station to any coordinate (lat, lon).

    Returns (GroundStation, distance_in_km).
    """
    if not GROUND_STATIONS:
        return None, float("inf")

    # Haversine approximate distance in km
    lat1, lon1 = np.radians(lat), np.radians(lon)
    best_st = None
    best_dist = float("inf")

    for st in GROUND_STATIONS:
        lat2, lon2 = np.radians(st.latitude), np.radians(st.longitude)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
        dist_km = 6371.0 * c
        if dist_km < best_dist:
            best_dist = dist_km
            best_st = st

    if best_dist <= max_dist_km:
        return best_st, round(best_dist, 2)
    return best_st, round(best_dist, 2)


def match_stations_to_grid(
    stations: list[GroundStation],
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
) -> list[dict]:
    """Map each physical ground station to its closest grid pixel coordinates (row, col)."""
    matched = []
    h = len(grid_lats)
    w = len(grid_lons)

    for st in stations:
        row = int(np.argmin(np.abs(grid_lats - st.latitude)))
        col = int(np.argmin(np.abs(grid_lons - st.longitude)))
        cell_id = row * w + col
        dist_lat_deg = abs(grid_lats[row] - st.latitude)
        dist_lon_deg = abs(grid_lons[col] - st.longitude)
        dist_km = float(np.sqrt((dist_lat_deg * 111.0) ** 2 + (dist_lon_deg * 111.0 * np.cos(np.radians(st.latitude))) ** 2))
        matched.append({
            "station": st,
            "row": row,
            "col": col,
            "cell_id": cell_id,
            "dist_km": round(dist_km, 2),
            "lat": float(grid_lats[row]),
            "lon": float(grid_lons[col]),
        })
    return matched
