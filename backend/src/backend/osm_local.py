"""Road density from a *local* Geofabrik OSM extract instead of the Overpass API.

Why: the four Overpass mirrors we query time out (45-50 s) or answer HTTP 504,
so one of the model's five static features (`road_density`) silently degrades to
zeros.  Overpass issues no API key at all, so there is no way to buy our way out
of the rate limit — but Geofabrik publishes plain HTTPS extracts (no key, no
rate limit) which DuckDB's spatial extension reads directly:

    https://download.geofabrik.de/asia/india/western-zone-latest.osm.pbf   (210 MB)
    https://download.geofabrik.de/europe/united-kingdom/england/greater-london-latest.osm.pbf  (123 MB)

The flow is:

1. `_remote_bbox()` reads just the first few KB of each candidate file (the PBF
   header stores the extract's own bounding box) and caches it in
   `cache/pbf/regions.json` — so region selection costs ~8 KB per region and
   never a download.
2. `pick_region()` keeps the smallest extract whose bbox fully covers the
   requested one, downloads it once to `cache/pbf/`, and reuses it forever.
3. `road_density()` runs one DuckDB query that joins the four road classes'
   node references onto their coordinates and bins the way centres onto the
   sampling grid — the same numbers the Overpass path produces.

Any problem (no covering extract, download failure, DuckDB/spatial unavailable)
returns ``None`` and the caller falls back to the original Overpass path.
"""

from __future__ import annotations

import json
import struct
import time
import zlib
from pathlib import Path

import httpx
import numpy as np

from .config import CACHE_DIR, USER_AGENT

# Same weighting as fetch.ROAD_WEIGHTS / the Overpass query, so the feature
# keeps its scale whichever source produced it.
ROAD_WEIGHTS = {"motorway": 4.0, "trunk": 3.0, "primary": 2.0, "secondary": 1.5}
ROAD_CLASSES = tuple(ROAD_WEIGHTS)

PBF_DIR = CACHE_DIR / "pbf"
PBF_TMP = PBF_DIR / "duckdb_tmp"
REGIONS_FILE = PBF_DIR / "regions.json"
# Geographic edits are rare compared with traffic: a monthly refresh is plenty.
MAX_AGE_DAYS = 30
# Refuse a partial extract before it starts to matter at the map edges.
MIN_COVERAGE = 0.5
# Downloading inside a fetch job is bounded; Overpass itself is slower, but a
# demo should not stall for ten minutes on a bad connection.
DOWNLOAD_TIMEOUT = 300.0
DOWNLOAD_LIMIT_MB = 600

REGIONS: dict[str, str] = {
    # India is published as six "zones" (state groupings), not as states.
    "india-western-zone": "https://download.geofabrik.de/asia/india/western-zone-latest.osm.pbf",
    "india-northern-zone": "https://download.geofabrik.de/asia/india/northern-zone-latest.osm.pbf",
    "india-central-zone": "https://download.geofabrik.de/asia/india/central-zone-latest.osm.pbf",
    "india-eastern-zone": "https://download.geofabrik.de/asia/india/eastern-zone-latest.osm.pbf",
    "india-southern-zone": "https://download.geofabrik.de/asia/india/southern-zone-latest.osm.pbf",
    "india-north-eastern-zone": "https://download.geofabrik.de/asia/india/north-eastern-zone-latest.osm.pbf",
    "greater-london": (
        "https://download.geofabrik.de/europe/united-kingdom/england/"
        "greater-london-latest.osm.pbf"
    ),
    "ile-de-france": "https://download.geofabrik.de/europe/france/ile-de-france-latest.osm.pbf",
}


# --------------------------------------------------------------------------- #
# PBF header (bounding box) parsing — first bytes only, no download
# --------------------------------------------------------------------------- #
def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = val = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


def _fields(buf: bytes):
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _varint(buf, i)
        elif wire == 2:
            size, i = _varint(buf, i)
            val = buf[i : i + size]
            i += size
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            val = buf[i : i + size]
            i += size
        else:
            return  # reserved wire type: nothing useful follows
        yield field, wire, val


def _bbox_from_head(head: bytes) -> dict | None:
    """Parse [4-byte len][BlobHeader][Blob] → HeaderBlock.bbox, or None."""
    if len(head) < 6:
        return None
    try:
        (head_len,) = struct.unpack(">I", head[:4])
        datasize = next(v for f, w, v in _fields(head[4 : 4 + head_len]) if f == 3)
        blob = head[4 + head_len : 4 + head_len + datasize]
        raw = None
        for f, _w, v in _fields(blob):
            if f == 1:
                raw = v
            elif f == 3:
                raw = zlib.decompress(v)
            elif f == 4:
                raw = zlib.decompress(v)  # brotli is unsupported; skip instead
        if raw is None:
            return None
        for f, _w, v in _fields(raw):
            if f != 1:
                continue
            names = {1: "left", 2: "right", 3: "top", 4: "bottom"}
            out: dict[str, float] = {}
            for bf, _bw, bv in _fields(v):
                if bf in names:
                    out[names[bf]] = ((bv >> 1) ^ -(bv & 1)) / 1e9  # sint64, 1e-9 deg
            if {"left", "right", "top", "bottom"} <= out.keys():
                return out
    except Exception:  # noqa: BLE001 — a malformed header just means "unknown"
        return None
    return None


def remote_bbox(url: str, timeout: float = 20.0) -> dict | None:
    """Bounding box of a remote extract, read from its first few KB."""
    try:
        with httpx.stream(
            "GET", url, timeout=timeout, follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as r:
            r.raise_for_status()
            head = b""
            for chunk in r.iter_bytes(4096):
                head += chunk
                if len(head) >= 16384:
                    break
    except Exception:  # noqa: BLE001
        return None
    return _bbox_from_head(head)


def _load_index() -> dict:
    try:
        return json.loads(REGIONS_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_index(idx: dict) -> None:
    PBF_DIR.mkdir(parents=True, exist_ok=True)
    REGIONS_FILE.write_text(json.dumps(idx, indent=2), encoding="utf-8")


def _report(progress, frac: float, stage: str) -> None:
    """The pipeline's hook signature is ``progress(fraction, stage)`` (as in fetch).

    Calling it with a single string raises TypeError inside the job, which the
    broad ``except`` below would happily turn into "no local source" and drop
    the whole feature back onto the stalled Overpass path.
    """
    if progress is not None:
        progress(frac, stage)


def region_index(timeout: float = 20.0, progress=None) -> dict:
    """url → bbox for every candidate region (cached; ~8 KB per uncached one)."""
    idx = _load_index()
    now = time.time()
    for key, url in REGIONS.items():
        ent = idx.get(url)
        if ent and now - ent.get("checked", 0) < MAX_AGE_DAYS * 86400:
            continue
        if progress:
            _report(progress, 0.70, f"probing OSM extract: {key}")
        bbox = remote_bbox(url, timeout)
        if bbox is None:
            # keep a stale entry rather than losing it to a transient error
            if ent is None:
                continue
        else:
            ent = {"bbox": bbox, "checked": now}
            idx[url] = ent
            _save_index(idx)
    return idx


def pick_region(bbox: list[float], progress=None) -> tuple[str, str, dict] | None:
    """Best cached extract for ``bbox`` = (lon_min, lat_min, lon_max, lat_max).

    Ranked by share of the requested bbox the extract covers (the boundary of a
    zone rarely coincides with a preset bbox — Greater London stops 8 km short
    of the preset's northern edge), ties broken by the smaller extract.
    """
    lon0, lat0, lon1, lat1 = bbox
    area = max((lon1 - lon0) * (lat1 - lat0), 1e-12)
    idx = region_index(progress=progress)
    best: tuple[float, float, str, str, dict] | None = None
    for key, url in REGIONS.items():
        b = (idx.get(url) or {}).get("bbox")
        if not b:
            continue
        dx = min(lon1, b["right"]) - max(lon0, b["left"])
        dy = min(lat1, b["top"]) - max(lat0, b["bottom"])
        # both factors must be positive on their own: a doubly-negative product
        # looks like a huge overlap and would pick an unrelated continent.
        if dx <= 0 or dy <= 0:
            continue
        score = dx * dy / area
        if score < MIN_COVERAGE:
            continue
        own = (b["right"] - b["left"]) * (b["top"] - b["bottom"])
        cand = (score, -own, key, url, b)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        return None
    return best[2], best[3], best[4]


def ensure_extract(url: str, progress=None) -> Path | None:
    """Download an extract once into ``cache/pbf/`` (returns None on failure)."""
    PBF_DIR.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    path = PBF_DIR / name
    if path.exists() and path.stat().st_size > 1_000_000:
        return path  # street networks change slowly; refresh by deleting the file
    tmp = path.with_suffix(path.suffix + ".part")
    done = 0
    t0 = time.perf_counter()
    try:
        with httpx.stream(
            "GET", url, timeout=60.0, follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length") or 0)
            if total and total > DOWNLOAD_LIMIT_MB * 1e6:
                return None
            last = t0
            with tmp.open("wb") as fh:
                for chunk in r.iter_bytes(1 << 16):
                    fh.write(chunk)
                    done += len(chunk)
                    if time.perf_counter() - t0 > DOWNLOAD_TIMEOUT:
                        raise TimeoutError("extract download exceeded budget")
                    if progress is not None and time.perf_counter() - last > 4.0:
                        last = time.perf_counter()
                        mb = done / 1e6
                        eta = (total - done) / (done / max(time.perf_counter() - t0, 0.1))
                        _report(
                            progress,
                            0.72,
                            f"OSM extract {mb:.0f}/{total / 1e6:.0f} MB (~{eta:.0f}s left)",
                        )
        if done < 1_000_000:
            tmp.unlink(missing_ok=True)
            return None
        tmp.replace(path)
    except Exception:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return None
    _report(progress, 0.77, f"OSM extract ready ({done / 1e6:.0f} MB)")
    return path


# --------------------------------------------------------------------------- #
# Density from a local extract
# --------------------------------------------------------------------------- #
def _connect():
    import duckdb

    con = duckdb.connect()
    # An accidental nested-loop join happily fills the drive (DuckDB defaults its
    # temp directory to the volume root: "103.8 GiB/103.8 GiB used").
    con.execute("SET memory_limit='3GB'")
    con.execute(f"SET temp_directory='{str(PBF_TMP).replace(chr(92), '/')}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    return con


def _density_from_pbf(
    path: Path, bbox: list[float], lats: np.ndarray, lons: np.ndarray, step: float
) -> np.ndarray:
    """Bin weighted way centres onto the sampling grid (identical to Overpass)."""
    lon0, lat0, lon1, lat1 = bbox
    cls_list = ",".join(f"'{c}'" for c in ROAD_CLASSES)
    sql = f"""
    WITH ways AS (
      SELECT id, tags['highway'] AS highway, refs
      FROM st_readosm(?)
      WHERE kind = 'way'
        AND tags['highway'] IN ({cls_list})
        AND len(refs) >= 2
    ),
    expanded AS (
      SELECT w.id, w.highway, u.nid FROM ways w, UNNEST(w.refs) AS u(nid)
    ),
    nodes AS (
      SELECT id, lat, lon FROM st_readosm(?)
      WHERE kind = 'node'
        AND lat BETWEEN {lat0} AND {lat1}
        AND lon BETWEEN {lon0} AND {lon1}
    )
    SELECT e.highway, avg(n.lat) AS clat, avg(n.lon) AS clon
    FROM expanded e JOIN nodes n ON n.id = e.nid
    GROUP BY e.id, e.highway
    """
    con = _connect()
    try:
        rows = con.execute(sql, [str(path), str(path)]).fetchall()
    finally:
        con.close()

    h, w = len(lats), len(lons)
    density = np.zeros((h, w), dtype=np.float64)
    lat_min_g = lats[0] - step / 2
    lon_min_g = lons[0] - step / 2
    for highway, clat, clon in rows:
        weight = ROAD_WEIGHTS.get(highway, 1.0)
        ri = int(np.clip((clat - lat_min_g) / step, 0, h - 1))
        ci = int(np.clip((clon - lon_min_g) / step, 0, w - 1))
        density[ri, ci] += weight
    cell_area = (step * 111.32) * (step * 111.32 * np.cos(np.radians(lats))).reshape(-1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(cell_area > 0, density / cell_area, 0.0)
    return density


def road_density(
    bbox: list[float],
    lats: np.ndarray,
    lons: np.ndarray,
    step: float,
    progress=None,
) -> tuple[np.ndarray, bool] | None:
    """Local OSM extract → (density, available). ``None`` ⇒ use Overpass."""
    try:
        picked = pick_region(bbox, progress)
        if picked is None:
            _report(progress, 0.70, "no local OSM extract for this bbox (Overpass fallback)")
            return None
        key, url, rbbox = picked
        dx = min(bbox[2], rbbox["right"]) - max(bbox[0], rbbox["left"])
        dy = min(bbox[3], rbbox["top"]) - max(bbox[1], rbbox["bottom"])
        coverage = max(dx, 0.0) * max(dy, 0.0)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area <= 0 or coverage / area < MIN_COVERAGE:
            return None
        _report(progress, 0.71, f"road density (local OSM extract: {key})")
        path = ensure_extract(url, progress)
        if path is None:
            return None
        density = _density_from_pbf(path, bbox, lats, lons, step)
    except Exception as exc:  # noqa: BLE001 — never fail a fetch job over roads
        # keep the reason visible: a silent return here looks like "Overpass is
        # still the problem" when it is really a local extract/query failure.
        _report(progress, 0.78, f"local OSM extract failed ({type(exc).__name__}); using Overpass")
        return None
    _report(progress, 0.78, f"road density ({key}, {coverage / area:.0%} bbox coverage)")
    return density, True


def preload(bbox: list[float], progress=print) -> str | None:
    """Prefetch the extract for a bbox (``scripts/preload_osm.py`` entry point)."""

    def note(frac: float, stage: str) -> None:  # adapt the (fraction, stage) hook
        progress(f"[{frac:.2f}] {stage}")

    picked = pick_region(bbox, note)
    if picked is None:
        progress(f"no Geofabrik extract covers {bbox}; Overpass will be used")
        return None
    key, url, rbbox = picked
    progress(f"{key}: bbox={rbbox}")
    progress(url)
    path = ensure_extract(url, note)
    if path is None:
        progress("download failed")
        return None
    t0 = time.time()
    probe = road_density(bbox, np.array([bbox[1]]), np.array([bbox[0]]), 1.0, note)
    progress(
        f"extract {path.name} = {path.stat().st_size / 1e6:.0f} MB, "
        f"query test {time.time() - t0:.1f}s -> {'ok' if probe else 'no coverage'}"
    )
    return str(path) if probe else None
