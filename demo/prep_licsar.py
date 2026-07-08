#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prep_licsar_mintpy_loopqc.py

Convert a single-frame LiCSAR GeoTIFF directory into MintPy-ready HDF5:
  - inputs/ifgramStack.h5
  - inputs/geometryGeo.h5
  - config/mintpy_licsar.cfg

Expected directory layout:
<frame_dir>/
├── interferograms/
│   ├── YYYYMMDD_YYYYMMDD/
│   │   ├── YYYYMMDD_YYYYMMDD.geo.unw.tif
│   │   ├── YYYYMMDD_YYYYMMDD.geo.cc.tif
│   │   └── YYYYMMDD_YYYYMMDD.geo.diff_pha.tif   [optional]
├── metadata/
│   ├── <frame>.geo.E.tif
│   ├── <frame>.geo.N.tif
│   ├── <frame>.geo.U.tif
│   ├── <frame>.geo.hgt.tif
│   ├── <frame>.geo.landmask.tif                 [optional]
│   ├── baselines
│   └── metadata.txt
└── epochs/                                      [optional for later use]

Main behavior
-------------
1) Writes MintPy-style HDF5 via mintpy.utils.writefile, not direct h5py calls.
2) Scales LiCSAR coherence based on dtype / metadata instead of blindly using 255.
3) Performs systematic interferogram pre-screening using:
     - valid unwrapped-pixel percentage within the trusted geometry mask;
     - average coherence within valid unwrapped pixels.
4) Performs conservative loop-closure screening on initially retained IFGs and
   drops IFGs that repeatedly participate in bad closure loops.
5) Never writes fake / dummy connectComponent. MintPy unwrap-error correction is
   kept OFF in the generated config.
6) Uses ALOOKS/RLOOKS from metadata.txt if available; otherwise uses LiCSAR defaults ALOOKS=4, RLOOKS=20.
7) hgt/E/N/U are strictly grid-checked; landmask/unw/cc are automatically
   resampled to the hgt reference grid whenever their pixel centers are not identical.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mintpy.utils import readfile, writefile


# ============================================================
# Constants
# ============================================================

DEFAULT_WAVELENGTH = 0.05546576   # Sentinel-1 wavelength [m]

# LiCSAR Sentinel-1 default multilooking factors used when metadata.txt
# does not explicitly provide ALOOKS/RLOOKS.
DEFAULT_ALOOKS = 4
DEFAULT_RLOOKS = 20

LZF = "lzf"
TWO_PI = float(2.0 * np.pi)


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class GridSpec:
    width: int
    length: int
    x_first: float
    y_first: float
    x_step: float
    y_step: float
    x_unit: str
    y_unit: str
    crs_key: str
    nodata: Optional[float]
    dtype: str


@dataclass
class PairRecord:
    date1: str
    date2: str
    pair_dir: Path
    unw_file: Path
    cc_file: Path
    diff_file: Optional[Path] = None


class GridCompatibilityError(ValueError):
    """Raised when a LiCSAR raster is not compatible with the reference grid."""


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare LiCSAR single-frame data for MintPy with IFG QA and loop closure screening."
    )
    p.add_argument("frame_dir", type=Path, help="LiCSAR frame root directory")
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory, default: <frame_dir>/mintpy",
    )
    p.add_argument(
        "--wavelength",
        type=float,
        default=DEFAULT_WAVELENGTH,
        help="Radar wavelength in meters (default: Sentinel-1)",
    )
    p.add_argument(
        "--cc-max",
        type=float,
        default=255.0,
        help="Fallback coherence divisor only when dtype/metadata cannot determine scaling (default: 255).",
    )
    p.add_argument(
        "--min-unw-valid-ratio",
        "--min-valid-ratio",
        dest="min_unw_valid_ratio",
        type=float,
        default=0.05,
        help="Minimum valid unwrapped-pixel percentage inside the trusted geometry mask to initially keep an IFG.",
    )
    p.add_argument(
        "--min-mean-coherence",
        "--min-mean-coh",
        dest="min_mean_coh",
        type=float,
        default=0.05,
        help="Minimum average coherence inside valid unwrapped pixels to initially keep an IFG.",
    )
    p.add_argument(
        "--unw-zero-eps",
        type=float,
        default=1e-6,
        help="Absolute unwrapPhase <= this value is treated as invalid when --keep-zero-unw-valid is not set.",
    )
    p.add_argument(
        "--keep-zero-unw-valid",
        action="store_true",
        help="Treat finite zero-valued unwrapPhase pixels as valid. Default treats LiCSAR unw=0 background as invalid.",
    )
    p.add_argument(
        "--disable-loop-closure",
        action="store_true",
        help="Disable loop-closure screening and only use initial coverage/coherence QA.",
    )
    p.add_argument(
        "--loop-min-valid-ratio",
        type=float,
        default=0.02,
        help="Minimum common valid-pixel ratio inside the trusted mask for a closure loop to be scored.",
    )
    p.add_argument(
        "--loop-phase-threshold",
        type=float,
        default=float(np.pi),
        help="Pixel is considered loop-bad if abs(closure phase) exceeds this threshold in radians.",
    )
    p.add_argument(
        "--loop-bad-pixel-ratio",
        type=float,
        default=0.15,
        help="Closure loop is bad if its bad-pixel ratio exceeds this value.",
    )
    p.add_argument(
        "--loop-rms-threshold",
        type=float,
        default=3.0,
        help="Closure loop is bad if its direct closure RMS exceeds this value in radians.",
    )
    p.add_argument(
        "--loop-min-tested-loops",
        type=int,
        default=2,
        help="Minimum number of scored loops required before an IFG can be rejected by loop closure.",
    )
    p.add_argument(
        "--loop-min-bad-loops",
        type=int,
        default=2,
        help="Minimum number of bad loops required before an IFG can be rejected by loop closure.",
    )
    p.add_argument(
        "--loop-bad-loop-fraction",
        type=float,
        default=0.5,
        help="Minimum fraction of scored loops that must be bad before an IFG is rejected by loop closure.",
    )
    p.add_argument(
        "--max-loop-count",
        type=int,
        default=0,
        help="Maximum number of closure loops to score; 0 means no explicit cap.",
    )
    p.add_argument(
        "--force-avg-incidence",
        action="store_true",
        help="Ignore U-derived incidence angle and fill a constant avg_incidence_angle from metadata.txt",
    )
    p.add_argument(
        "--orbit-direction",
        choices=["ASCENDING", "DESCENDING", "UNKNOWN"],
        default="UNKNOWN",
        help="Optional manual orbit direction metadata",
    )
    # Backward compatibility with old command lines. It is deliberately ignored.
    p.add_argument("--write-dummy-conncomp", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return p


# ============================================================
# Logging
# ============================================================

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


# ============================================================
# Basic parsing
# ============================================================

def parse_value(v: str):
    v = v.strip()
    if re.fullmatch(r"[+-]?\d+", v):
        return int(v)
    try:
        return float(v)
    except ValueError:
        return v


def parse_metadata_txt(path: Path) -> Dict[str, object]:
    meta = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            meta[k.strip()] = parse_value(v)
    return meta


def hms_to_seconds(hms: str) -> float:
    parts = hms.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid center_time: {hms}")
    hh = int(parts[0])
    mm = int(parts[1])
    ss = float(parts[2])
    return hh * 3600 + mm * 60 + ss


def parse_baselines(path: Path) -> Dict[str, object]:
    """
    Expected common LiCSAR baseline format:
        master_date acquisition_date bperp_to_master_m temporal_baseline_days

    Example:
        20190504 20250613 75 2232
    """
    master_dates = set()
    bperp_by_date: Dict[str, float] = {}
    tbase_by_date: Dict[str, float] = {}

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                logging.warning("Skip malformed baselines line %d: %s", i, line)
                continue

            master_date, acq_date, bperp, tbase = parts[:4]
            master_dates.add(master_date)
            bperp_by_date[acq_date] = float(bperp)
            tbase_by_date[acq_date] = float(tbase)

    if not bperp_by_date:
        raise ValueError("No valid lines parsed from baselines file.")

    if len(master_dates) != 1:
        logging.warning("Multiple master dates found in baselines: %s", master_dates)

    master_date = sorted(master_dates)[0]
    return {
        "master_date": master_date,
        "bperp_by_date": bperp_by_date,
        "tbase_by_date": tbase_by_date,
    }


def _lookup_meta_key(meta_txt: Dict[str, object], keys: List[str]) -> Optional[object]:
    """Case-insensitive lookup in metadata.txt parsed dictionary."""
    lower_map = {str(k).lower(): k for k in meta_txt.keys()}
    for key in keys:
        real_key = lower_map.get(key.lower())
        if real_key is not None:
            return meta_txt[real_key]
    return None


def _lookup_meta_float(meta_txt: Dict[str, object], keys: List[str]) -> Optional[float]:
    val = _lookup_meta_key(meta_txt, keys)
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _lookup_meta_int(meta_txt: Dict[str, object], keys: List[str]) -> Optional[int]:
    val = _lookup_meta_key(meta_txt, keys)
    if val is None:
        return None
    try:
        return int(round(float(val)))
    except Exception:
        return None


def get_looks_from_metadata(meta_txt: Dict[str, object]) -> Tuple[int, int, Dict[str, object]]:
    """
    Read ALOOKS/RLOOKS from metadata.txt, with LiCSAR Sentinel-1 defaults.

    Priority:
      1) If metadata.txt explicitly provides ALOOKS/RLOOKS or common aliases,
         use those values.
      2) If missing, use the LiCSAR default multilooking factors:
             ALOOKS = 4
             RLOOKS = 20

    The fallback is deliberately recorded in looks_info so downstream users know
    whether the values came from metadata.txt or from LiCSAR defaults.
    """
    alooks = _lookup_meta_int(
        meta_txt,
        [
            "ALOOKS", "azimuth_looks", "azimuthLooks", "azimuth_multilook",
            "azimuth_multilooks", "nlooks_azimuth", "nlook_azimuth", "alks",
        ],
    )
    rlooks = _lookup_meta_int(
        meta_txt,
        [
            "RLOOKS", "range_looks", "rangeLooks", "range_multilook",
            "range_multilooks", "nlooks_range", "nlook_range", "rlks",
        ],
    )

    alooks_source = "metadata.txt"
    rlooks_source = "metadata.txt"

    if alooks is None:
        alooks = DEFAULT_ALOOKS
        alooks_source = "LiCSAR_default"
        logging.warning(
            "ALOOKS was not found in metadata.txt; use LiCSAR default ALOOKS=%d.",
            alooks,
        )

    if rlooks is None:
        rlooks = DEFAULT_RLOOKS
        rlooks_source = "LiCSAR_default"
        logging.warning(
            "RLOOKS was not found in metadata.txt; use LiCSAR default RLOOKS=%d.",
            rlooks,
        )

    info = {
        "ALOOKS": int(alooks),
        "RLOOKS": int(rlooks),
        "ALOOKS_source": alooks_source,
        "RLOOKS_source": rlooks_source,
        "default_ALOOKS": int(DEFAULT_ALOOKS),
        "default_RLOOKS": int(DEFAULT_RLOOKS),
    }
    return int(alooks), int(rlooks), info


# ============================================================
# MintPy-based raster I/O
# ============================================================

def _get_meta_float(meta: Dict[str, str], keys: List[str], default: Optional[float] = None) -> Optional[float]:
    for k in keys:
        if k in meta and meta[k] not in [None, "", "None", "nan", "NaN"]:
            try:
                return float(meta[k])
            except Exception:
                continue
    return default


def _get_meta_str(meta: Dict[str, str], keys: List[str], default: str = "") -> str:
    for k in keys:
        if k in meta and meta[k] is not None:
            return str(meta[k])
    return default


def _clean_crs_part(value: object) -> str:
    """Normalize blank-like CRS metadata values to an empty string."""
    text = str(value or "").strip()
    if text.lower() in {"none", "nan", "null", "n/a", "na", "unknown"}:
        return ""
    return text


def _normalize_epsg_code(epsg: str) -> str:
    """Normalize common EPSG string forms to a bare numeric code."""
    epsg = _clean_crs_part(epsg)
    if epsg.upper().startswith("EPSG:"):
        epsg = epsg.split(":", 1)[1].strip()
    return epsg


def _is_degree_unit(unit: str) -> bool:
    """Return True for common degree-unit strings from GDAL/MintPy metadata."""
    return _clean_crs_part(unit).lower() in {"degree", "degrees", "deg"}


def _is_geographic_degree_metadata(
    epsg: str,
    utm_zone: str,
    projection: str,
    geocoor_ref: str,
    x_unit: str,
    y_unit: str,
) -> bool:
    """
    Return True for LiCSAR-style geographic lon/lat rasters.

    Why this exists:
        Some LiCSAR/GAMMA GeoTIFFs contain nested WKT authority identifiers such
        as ID["EPSG",9122] under ANGLEUNIT["degree", ...] or ID["EPSG",9001]
        under LENGTHUNIT["metre", ...]. MintPy/GDAL's simplified metadata may
        expose one of these nested IDs as meta["EPSG"], even though it is not the
        CRS EPSG code. For a non-projected degree grid, these nested unit/axis IDs
        must not affect CRS compatibility checks.

    This function is intentionally conservative for this LiCSAR converter:
        - no UTM zone / projection tag;
        - both axes are angular degrees, or the WKT/reference text clearly says
          WGS 84 / GEOGCRS / geographic.
    """
    if utm_zone or projection:
        return False

    ref_txt = str(geocoor_ref or "").lower()
    units_are_degrees = _is_degree_unit(x_unit) and _is_degree_unit(y_unit)
    wkt_looks_geographic = any(token in ref_txt for token in ["geogcrs", "geogcs", "wgs 84", "wgs_1984"])

    # EPSG:9122 is the angular unit degree; EPSG:4326 is the WGS84 geographic CRS.
    # For LiCSAR .geo files, both cases represent the same lon/lat degree grid
    # as long as there is no projection/UTM tag and the raster transform agrees.
    return units_are_degrees or wkt_looks_geographic or epsg in {"4326", "9122"}


def _geographic_degree_crs_key(x_unit: str, y_unit: str) -> str:
    """
    CRS key for unprojected LiCSAR geographic-degree rasters.

    This intentionally does not preserve the EPSG field from MintPy metadata.
    A nested WKT ID["EPSG",9122] is a unit authority code for degree, not a CRS;
    preserving it would incorrectly make the same WGS84 grid look incompatible
    with rasters whose metadata report EPSG:4326 or no CRS EPSG at all.
    """
    xu = "degrees" if _is_degree_unit(x_unit) or not _clean_crs_part(x_unit) else _clean_crs_part(x_unit)
    yu = "degrees" if _is_degree_unit(y_unit) or not _clean_crs_part(y_unit) else _clean_crs_part(y_unit)
    return "|".join(["GEOG_DEGREES", "", "", "", xu, yu])


def _build_crs_key(meta: Dict[str, str]) -> str:
    """
    Build a lightweight CRS signature from MintPy/GDAL metadata.

    Important LiCSAR/GAMMA GeoTIFF caveat:
        WKT may contain nested authority identifiers such as ID["EPSG",9122]
        under ANGLEUNIT["degree", ...]. These identify a unit/axis component,
        not the raster CRS. For LiCSAR .geo rasters in geographic degrees, this
        function ignores such nested IDs and returns a canonical geographic-degree
        key so that coherence, unwrapped phase, height, and landmask rasters are
        compared by their actual grid geometry rather than by incidental WKT IDs.
    """
    epsg = _normalize_epsg_code(_get_meta_str(meta, ["EPSG"], ""))
    utm_zone = _clean_crs_part(_get_meta_str(meta, ["UTM_ZONE"], ""))
    projection = _clean_crs_part(_get_meta_str(meta, ["PROJECTION"], ""))
    geocoor_ref = _clean_crs_part(_get_meta_str(meta, ["GEOCOOR_REF"], ""))
    x_unit = _clean_crs_part(_get_meta_str(meta, ["X_UNIT"], "degrees"))
    y_unit = _clean_crs_part(_get_meta_str(meta, ["Y_UNIT"], "degrees"))

    if _is_geographic_degree_metadata(epsg, utm_zone, projection, geocoor_ref, x_unit, y_unit):
        return _geographic_degree_crs_key(x_unit, y_unit)

    parts = [epsg, utm_zone, projection, geocoor_ref, x_unit, y_unit]
    return "|".join(parts)


def _canonicalize_crs_key(key: str) -> str:
    """
    Canonicalize lightweight CRS keys before comparing rasters.

    This is a second-line safeguard for already-created GridSpec objects or old
    logs/metadata strings. It maps keys such as:
        4326||||degrees|degrees
        9122||||degrees|degrees
        ||||degrees|degrees
    to the same internal LiCSAR geographic-degree CRS key, preventing nested WKT
    unit IDs from being treated as a true CRS mismatch.
    """
    parts = str(key or "").split("|")
    while len(parts) < 6:
        parts.append("")
    epsg = _normalize_epsg_code(parts[0])
    utm_zone = _clean_crs_part(parts[1])
    projection = _clean_crs_part(parts[2])
    geocoor_ref = _clean_crs_part(parts[3])
    x_unit = _clean_crs_part(parts[4])
    y_unit = _clean_crs_part(parts[5])

    if _is_geographic_degree_metadata(epsg, utm_zone, projection, geocoor_ref, x_unit, y_unit):
        return _geographic_degree_crs_key(x_unit, y_unit)

    return "|".join([epsg, utm_zone, projection, geocoor_ref, x_unit, y_unit])


def read_tif_with_metadata(path: Path) -> Tuple[np.ndarray, GridSpec, Dict[str, str]]:
    """Read GeoTIFF via MintPy's readfile.read() and return array, grid spec, and metadata."""
    arr, meta = readfile.read(str(path), print_msg=False)

    width = int(meta["WIDTH"])
    length = int(meta["LENGTH"])
    x_first = float(meta["X_FIRST"])
    y_first = float(meta["Y_FIRST"])
    x_step = float(meta["X_STEP"])
    y_step = float(meta["Y_STEP"])

    x_unit = _get_meta_str(meta, ["X_UNIT"], "degrees")
    y_unit = _get_meta_str(meta, ["Y_UNIT"], "degrees")
    nodata = _get_meta_float(meta, ["NO_DATA_VALUE", "_FillValue", "missing_value"], None)
    dtype = str(arr.dtype)

    spec = GridSpec(
        width=width,
        length=length,
        x_first=x_first,
        y_first=y_first,
        x_step=x_step,
        y_step=y_step,
        x_unit=x_unit,
        y_unit=y_unit,
        crs_key=_build_crs_key(meta),
        nodata=nodata,
        dtype=dtype,
    )
    return arr, spec, meta


def read_tif(path: Path) -> Tuple[np.ndarray, GridSpec]:
    """Read GeoTIFF via MintPy's readfile.read()."""
    arr, spec, _ = read_tif_with_metadata(path)
    return arr, spec


def assert_same_grid(ref: GridSpec, cur: GridSpec, path: Path) -> None:
    if ref.width != cur.width or ref.length != cur.length:
        raise ValueError(f"Grid size mismatch in {path}")

    ref_crs = _canonicalize_crs_key(ref.crs_key)
    cur_crs = _canonicalize_crs_key(cur.crs_key)
    if ref_crs != cur_crs:
        if ref_crs or cur_crs:
            raise ValueError(f"CRS mismatch in {path}: ref CRS={ref.crs_key} canonical={ref_crs}, cur CRS={cur.crs_key} canonical={cur_crs}")

    vals_ref = np.array([ref.x_first, ref.y_first, ref.x_step, ref.y_step], dtype=np.float64)
    vals_cur = np.array([cur.x_first, cur.y_first, cur.x_step, cur.y_step], dtype=np.float64)
    if not np.allclose(vals_ref, vals_cur, atol=1e-12):
        raise ValueError(f"Geo grid mismatch in {path}")


def _same_crs(ref: GridSpec, cur: GridSpec) -> bool:
    """Return True when CRS signatures are compatible for grid-to-grid resampling."""
    ref_key = _canonicalize_crs_key(ref.crs_key)
    cur_key = _canonicalize_crs_key(cur.crs_key)
    if ref_key or cur_key:
        return ref_key == cur_key
    return _clean_crs_part(ref.x_unit) == _clean_crs_part(cur.x_unit) and _clean_crs_part(ref.y_unit) == _clean_crs_part(cur.y_unit)


def _pixel_center_origin(spec: GridSpec) -> np.ndarray:
    """Return the map coordinate of pixel (0, 0) center using the script convention."""
    return np.array(
        [spec.x_first + 0.5 * spec.x_step, spec.y_first + 0.5 * spec.y_step],
        dtype=np.float64,
    )


def grids_have_identical_pixel_centers(
    ref: GridSpec,
    cur: GridSpec,
    step_tol: float = 1e-8,
    center_tol_pixel: float = 0.005,
) -> bool:
    """
    Return True only when two rasters represent the same pixel-center grid.

    This deliberately does NOT accept a half-pixel center/corner offset. If the
    pixel centers do not match, the caller should resample the source raster to
    the reference grid instead of using the source array directly.
    """
    if ref.width != cur.width or ref.length != cur.length:
        return False
    if not _same_crs(ref, cur):
        return False

    ref_step = np.array([ref.x_step, ref.y_step], dtype=np.float64)
    cur_step = np.array([cur.x_step, cur.y_step], dtype=np.float64)
    if not np.allclose(ref_step, cur_step, atol=step_tol, rtol=0.0):
        return False

    pix = max(abs(ref.x_step), abs(ref.y_step), abs(cur.x_step), abs(cur.y_step))
    center_tol = pix * center_tol_pixel
    return np.allclose(_pixel_center_origin(ref), _pixel_center_origin(cur), atol=center_tol, rtol=0.0)


def assert_compatible_grid(
    ref: GridSpec,
    cur: GridSpec,
    path: Path,
    layer_name: str = "raster",
    step_tol: float = 1e-8,
    center_tol_pixel: float = 0.005,
) -> None:
    """
    Strictly check that a raster already lies on the reference pixel-center grid.

    Unlike the previous implementation, this function does not accept a
    half-pixel center/corner metadata offset and never permits a shifted array to
    be used directly. Use align_array_to_reference_grid() for layers that may
    need resampling to the hgt reference grid.
    """
    if ref.width != cur.width or ref.length != cur.length:
        raise GridCompatibilityError(
            f"{layer_name} grid size mismatch in {path}: "
            f"ref size=({ref.length}, {ref.width}), "
            f"cur size=({cur.length}, {cur.width})"
        )

    if not _same_crs(ref, cur):
        raise GridCompatibilityError(
            f"{layer_name} CRS mismatch in {path}: "
            f"ref CRS={ref.crs_key} canonical={_canonicalize_crs_key(ref.crs_key)}, "
            f"cur CRS={cur.crs_key} canonical={_canonicalize_crs_key(cur.crs_key)}"
        )

    ref_step = np.array([ref.x_step, ref.y_step], dtype=np.float64)
    cur_step = np.array([cur.x_step, cur.y_step], dtype=np.float64)
    if not np.allclose(ref_step, cur_step, atol=step_tol, rtol=0.0):
        raise GridCompatibilityError(
            f"{layer_name} pixel size mismatch in {path}: "
            f"ref step={ref_step}, cur step={cur_step}"
        )

    pix = max(abs(ref.x_step), abs(ref.y_step), abs(cur.x_step), abs(cur.y_step))
    center_tol = pix * center_tol_pixel
    ref_center0 = _pixel_center_origin(ref)
    cur_center0 = _pixel_center_origin(cur)
    if not np.allclose(ref_center0, cur_center0, atol=center_tol, rtol=0.0):
        raise GridCompatibilityError(
            f"{layer_name} pixel-center grid mismatch in {path}: "
            f"ref center0={ref_center0}, cur center0={cur_center0}, "
            f"ref step={ref_step}, cur step={cur_step}, "
            f"center_tol={center_tol}"
        )


def _reference_pixel_centers(ref: GridSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Return 1-D x/y coordinate vectors for reference pixel centers."""
    cols = np.arange(ref.width, dtype=np.float64)
    rows = np.arange(ref.length, dtype=np.float64)
    x = ref.x_first + (cols + 0.5) * ref.x_step
    y = ref.y_first + (rows + 0.5) * ref.y_step
    return x, y


def _source_fractional_indices(src: GridSpec, ref: GridSpec) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map reference pixel centers to floating source column/row indices.

    The mapping assumes both rasters are in the same CRS. It supports different
    origins, dimensions, and pixel spacing. CRS transformation is intentionally
    not attempted here; a CRS mismatch is treated as an error.
    """
    x_ref, y_ref = _reference_pixel_centers(ref)
    src_x0, src_y0 = _pixel_center_origin(src)
    src_cols = (x_ref - src_x0) / src.x_step
    src_rows = (y_ref - src_y0) / src.y_step
    return src_cols, src_rows


def _resample_nearest(
    arr: np.ndarray,
    src: GridSpec,
    ref: GridSpec,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Nearest-neighbor resampling to the reference grid."""
    src_cols_f, src_rows_f = _source_fractional_indices(src, ref)
    src_cols = np.rint(src_cols_f).astype(np.int64)
    src_rows = np.rint(src_rows_f).astype(np.int64)

    out = np.full((ref.length, ref.width), fill_value, dtype=np.float32)
    col_ok = (src_cols >= 0) & (src_cols < src.width)
    row_ok = (src_rows >= 0) & (src_rows < src.length)
    if not np.any(col_ok) or not np.any(row_ok):
        return out

    rr = np.where(row_ok)[0]
    cc = np.where(col_ok)[0]
    out[np.ix_(rr, cc)] = arr[np.ix_(src_rows[rr], src_cols[cc])].astype(np.float32)
    return out


def _resample_bilinear_nan(
    arr: np.ndarray,
    src: GridSpec,
    ref: GridSpec,
    nodata: Optional[float] = None,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    NaN-aware bilinear resampling to the reference grid.

    Invalid / nodata source samples do not contribute to the weighted average.
    Pixels with no finite contributing samples are filled with fill_value.
    """
    src_cols_f, src_rows_f = _source_fractional_indices(src, ref)

    c0 = np.floor(src_cols_f).astype(np.int64)
    r0 = np.floor(src_rows_f).astype(np.int64)
    c1 = c0 + 1
    r1 = r0 + 1

    wx = (src_cols_f - c0).astype(np.float64)
    wy = (src_rows_f - r0).astype(np.float64)

    col_ok = (c0 >= 0) & (c1 < src.width)
    row_ok = (r0 >= 0) & (r1 < src.length)

    out = np.full((ref.length, ref.width), fill_value, dtype=np.float32)
    rr = np.where(row_ok)[0]
    cc = np.where(col_ok)[0]
    if rr.size == 0 or cc.size == 0:
        return out

    a = arr.astype(np.float32, copy=False)
    if nodata is not None and np.isfinite(nodata):
        a = a.copy()
        a[a == nodata] = np.nan

    rr0 = r0[rr]
    rr1 = r1[rr]
    cc0 = c0[cc]
    cc1 = c1[cc]

    v00 = a[np.ix_(rr0, cc0)]
    v01 = a[np.ix_(rr0, cc1)]
    v10 = a[np.ix_(rr1, cc0)]
    v11 = a[np.ix_(rr1, cc1)]

    wx2 = wx[cc][np.newaxis, :]
    wy2 = wy[rr][:, np.newaxis]
    w00 = (1.0 - wx2) * (1.0 - wy2)
    w01 = wx2 * (1.0 - wy2)
    w10 = (1.0 - wx2) * wy2
    w11 = wx2 * wy2

    values = [v00, v01, v10, v11]
    weights = [w00, w01, w10, w11]
    numerator = np.zeros((rr.size, cc.size), dtype=np.float64)
    denominator = np.zeros((rr.size, cc.size), dtype=np.float64)

    for val, weight in zip(values, weights):
        finite = np.isfinite(val)
        numerator += np.where(finite, val.astype(np.float64) * weight, 0.0)
        denominator += np.where(finite, weight, 0.0)

    block = np.full((rr.size, cc.size), fill_value, dtype=np.float32)
    good = denominator > 0.0
    block[good] = (numerator[good] / denominator[good]).astype(np.float32)
    out[np.ix_(rr, cc)] = block
    return out


def align_array_to_reference_grid(
    arr: np.ndarray,
    src: GridSpec,
    ref: GridSpec,
    path: Path,
    layer_name: str = "raster",
    resampling: str = "bilinear",
    nodata: Optional[float] = None,
    fill_value: float = np.nan,
    step_tol: float = 1e-8,
    center_tol_pixel: float = 0.005,
) -> Tuple[np.ndarray, bool, Dict[str, object]]:
    """
    Return arr on the hgt reference grid.

    If pixel centers already match, arr is returned unchanged. If dimensions,
    origin, pixel spacing, or pixel centers differ, the array is resampled to the
    reference grid. CRS mismatch is not silently handled and raises an error.
    """
    if not _same_crs(ref, src):
        raise GridCompatibilityError(
            f"{layer_name} CRS mismatch in {path}: "
            f"ref CRS={ref.crs_key} canonical={_canonicalize_crs_key(ref.crs_key)}, "
            f"cur CRS={src.crs_key} canonical={_canonicalize_crs_key(src.crs_key)}"
        )

    same_centers = grids_have_identical_pixel_centers(
        ref, src, step_tol=step_tol, center_tol_pixel=center_tol_pixel
    )
    if same_centers:
        return arr.astype(np.float32, copy=False), False, {
            "resampled": False,
            "resampling_method": "none",
            "source_shape": [int(src.length), int(src.width)],
            "target_shape": [int(ref.length), int(ref.width)],
            "source_center0": _pixel_center_origin(src).tolist(),
            "target_center0": _pixel_center_origin(ref).tolist(),
            "source_step": [float(src.x_step), float(src.y_step)],
            "target_step": [float(ref.x_step), float(ref.y_step)],
        }

    if resampling not in {"nearest", "bilinear"}:
        raise ValueError(f"Unsupported resampling method for {layer_name}: {resampling}")

    logging.warning(
        "%s grid pixel centers differ from hgt reference grid; resample to hgt grid: %s "
        "method=%s, source_shape=(%d, %d), target_shape=(%d, %d), "
        "source_center0=%s, target_center0=%s, source_step=(%.12g, %.12g), target_step=(%.12g, %.12g)",
        layer_name,
        path,
        resampling,
        src.length, src.width, ref.length, ref.width,
        _pixel_center_origin(src), _pixel_center_origin(ref),
        src.x_step, src.y_step, ref.x_step, ref.y_step,
    )

    if resampling == "nearest":
        out = _resample_nearest(arr, src, ref, fill_value=fill_value)
    else:
        out = _resample_bilinear_nan(arr, src, ref, nodata=nodata, fill_value=fill_value)

    return out.astype(np.float32), True, {
        "resampled": True,
        "resampling_method": resampling,
        "source_shape": [int(src.length), int(src.width)],
        "target_shape": [int(ref.length), int(ref.width)],
        "source_center0": _pixel_center_origin(src).tolist(),
        "target_center0": _pixel_center_origin(ref).tolist(),
        "source_step": [float(src.x_step), float(src.y_step)],
        "target_step": [float(ref.x_step), float(ref.y_step)],
    }


def build_lat_lon(spec: GridSpec) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build 2D latitude/longitude arrays from MintPy/GDAL geo metadata.

    Pixel-center convention:
        x = X_FIRST + (col + 0.5) * X_STEP
        y = Y_FIRST + (row + 0.5) * Y_STEP
    """
    cols = np.arange(spec.width, dtype=np.float64)
    rows = np.arange(spec.length, dtype=np.float64)

    x = spec.x_first + (cols + 0.5) * spec.x_step
    y = spec.y_first + (rows + 0.5) * spec.y_step

    lon = np.tile(x[np.newaxis, :], (spec.length, 1)).astype(np.float32)
    lat = np.tile(y[:, np.newaxis], (1, spec.width)).astype(np.float32)
    return lat, lon


# ============================================================
# Directory scanning
# ============================================================

PAIR_RE = re.compile(r"^(\d{8})_(\d{8})$")


def find_single_file(folder: Path, pattern: str) -> Optional[Path]:
    matches = sorted(folder.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {pattern} in {folder}: {matches}")
    return matches[0]


def scan_interferograms(frame_dir: Path) -> List[PairRecord]:
    ifg_root = frame_dir / "interferograms"
    if not ifg_root.exists():
        raise FileNotFoundError(f"Missing directory: {ifg_root}")

    pairs: List[PairRecord] = []
    for d in sorted(ifg_root.iterdir()):
        if not d.is_dir():
            continue
        m = PAIR_RE.match(d.name)
        if not m:
            logging.warning("Skip non-pair directory: %s", d.name)
            continue

        date1, date2 = m.group(1), m.group(2)
        if date1 >= date2:
            raise ValueError(f"Invalid pair ordering in {d.name}; expected date1 < date2")

        unw = find_single_file(d, "*.geo.unw.tif")
        cc = find_single_file(d, "*.geo.cc.tif")
        diff = find_single_file(d, "*.geo.diff_pha.tif")

        if unw is None or cc is None:
            logging.warning("Skip pair %s because unw/cc file missing", d.name)
            continue

        pairs.append(PairRecord(date1, date2, d, unw, cc, diff))

    if not pairs:
        raise RuntimeError("No valid interferogram pairs found.")
    return pairs


def scan_metadata_files(frame_dir: Path) -> Dict[str, Path]:
    meta_dir = frame_dir / "metadata"
    if not meta_dir.exists():
        raise FileNotFoundError(f"Missing directory: {meta_dir}")

    files = {
        "metadata_txt": meta_dir / "metadata.txt",
        "baselines": meta_dir / "baselines",
        "E": find_single_file(meta_dir, "*.geo.E.tif"),
        "N": find_single_file(meta_dir, "*.geo.N.tif"),
        "U": find_single_file(meta_dir, "*.geo.U.tif"),
        "hgt": find_single_file(meta_dir, "*.geo.hgt.tif"),
        "landmask": find_single_file(meta_dir, "*.geo.landmask.tif"),
        "azirg_csv": find_single_file(meta_dir, "*.azirg.csv"),
    }

    if not files["metadata_txt"].exists():
        raise FileNotFoundError(files["metadata_txt"])
    if not files["baselines"].exists():
        raise FileNotFoundError(files["baselines"])
    if files["E"] is None or files["N"] is None or files["U"] is None or files["hgt"] is None:
        raise FileNotFoundError("Missing one or more geometry files among E/N/U/hgt.")
    return files


# ============================================================
# Geometry utilities
# ============================================================

def clean_invalid(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    arr = arr.astype(np.float32)
    bad = ~np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        bad |= (arr == nodata)
    out = arr.astype(np.float32).copy()
    out[bad] = np.nan
    return out


def normalize_enu(
    e: np.ndarray,
    n: np.ndarray,
    u: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize ENU LOS vectors only on valid pixels.

    Returns
    -------
    e2, n2, u2 : normalized components, NaN outside valid pixels
    norm       : original norm
    valid      : boolean mask of valid vector pixels
    """
    norm = np.sqrt(e**2 + n**2 + u**2).astype(np.float32)
    valid = np.isfinite(norm) & (norm > 1e-6)

    e2 = np.full_like(e, np.nan, dtype=np.float32)
    n2 = np.full_like(n, np.nan, dtype=np.float32)
    u2 = np.full_like(u, np.nan, dtype=np.float32)

    e2[valid] = e[valid] / norm[valid]
    n2[valid] = n[valid] / norm[valid]
    u2[valid] = u[valid] / norm[valid]
    return e2, n2, u2, norm, valid


def incidence_from_u(
    u: np.ndarray,
    valid: np.ndarray,
    avg_incidence_angle: Optional[float] = None,
) -> np.ndarray:
    """
    MintPy incidenceAngle = angle from local vertical at target, in degree.

    Use abs(u) to reduce sign-convention ambiguity in LiCSAR LOS vectors.
    """
    inc = np.full_like(u, np.nan, dtype=np.float32)
    uu = np.clip(np.abs(u[valid]), 0.0, 1.0)
    inc[valid] = np.degrees(np.arccos(uu)).astype(np.float32)

    if np.any(valid):
        med = float(np.nanmedian(inc[valid]))
        if avg_incidence_angle is not None and abs(med - avg_incidence_angle) > 15.0:
            logging.warning(
                "Derived incidence median=%.2f differs strongly from metadata avg_incidence_angle=%.2f",
                med,
                avg_incidence_angle,
            )
    return inc


def azimuth_from_en(e: np.ndarray, n: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    MintPy azimuthAngle: angle measured from North, anti-clockwise positive.
    Formula: az = atan2(-E, N)
    """
    az = np.full_like(e, np.nan, dtype=np.float32)
    az[valid] = np.degrees(np.arctan2(-e[valid], n[valid])).astype(np.float32)
    return az


def infer_landmask_to_watermask(landmask: np.ndarray) -> np.ndarray:
    """
    MintPy /waterMask convention: non-zero value for pixels on land.
    LiCSAR file is named landmask, so likely mapping is land != 0 -> True.
    """
    finite = np.isfinite(landmask)
    unique_vals = np.unique(landmask[finite]) if np.any(finite) else np.array([])
    logging.info("landmask unique values (sample): %s", unique_vals[:20])
    return finite & (landmask != 0)


def fill_nan_inside_mask(
    arr: np.ndarray,
    mask: np.ndarray,
    fallback_value: Optional[float] = None,
) -> np.ndarray:
    """Fill NaN only inside a trusted geometry mask and keep outside-mask pixels as NaN."""
    out = arr.astype(np.float32).copy()
    mask = mask.astype(bool)

    inside_finite = mask & np.isfinite(out)
    if np.any(inside_finite):
        fill_value = float(np.nanmedian(out[inside_finite]))
    elif fallback_value is not None:
        fill_value = float(fallback_value)
    else:
        fill_value = 0.0

    repair = mask & ~np.isfinite(out)
    out[repair] = fill_value
    out[~mask] = np.nan
    return out.astype(np.float32)


def apply_mask_keep_nan(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep values inside mask and set outside-mask pixels to NaN."""
    out = arr.astype(np.float32).copy()
    out[~mask.astype(bool)] = np.nan
    return out.astype(np.float32)


def build_geometry(
    frame_dir: Path,
    meta_files: Dict[str, Path],
    meta_txt: Dict[str, object],
    force_avg_incidence: bool = False,
) -> Tuple[Dict[str, np.ndarray], GridSpec]:
    """
    Build MintPy geometry layers with a unified valid geometry mask.

    valid_geom = valid_enu_vector & valid_height & land_mask_if_available
    """
    hgt_raw, hgt_spec = read_tif(meta_files["hgt"])
    e_raw, e_spec = read_tif(meta_files["E"])
    n_raw, n_spec = read_tif(meta_files["N"])
    u_raw, u_spec = read_tif(meta_files["U"])

    assert_same_grid(hgt_spec, e_spec, meta_files["E"])
    assert_same_grid(hgt_spec, n_spec, meta_files["N"])
    assert_same_grid(hgt_spec, u_spec, meta_files["U"])

    hgt = clean_invalid(hgt_raw, hgt_spec.nodata)
    e = clean_invalid(e_raw, e_spec.nodata)
    n = clean_invalid(n_raw, n_spec.nodata)
    u = clean_invalid(u_raw, u_spec.nodata)

    e2, n2, u2, norm, valid_vec = normalize_enu(e, n, u)

    if np.any(valid_vec):
        logging.info(
            "ENU norm statistics on valid ENU pixels: min=%.4f, median=%.4f, max=%.4f",
            float(np.nanmin(norm[valid_vec])),
            float(np.nanmedian(norm[valid_vec])),
            float(np.nanmax(norm[valid_vec])),
        )
        logging.info(
            "Valid ENU pixel ratio: %.3f",
            float(np.count_nonzero(valid_vec)) / float(valid_vec.size),
        )
    else:
        raise RuntimeError("No valid ENU pixels found in E/N/U geometry files.")

    valid_height = np.isfinite(hgt)

    if meta_files["landmask"] is not None:
        landmask_raw, lm_spec = read_tif(meta_files["landmask"])
        landmask_aligned, landmask_resampled, _landmask_grid_info = align_array_to_reference_grid(
            arr=landmask_raw,
            src=lm_spec,
            ref=hgt_spec,
            path=meta_files["landmask"],
            layer_name="landmask",
            resampling="nearest",
            nodata=lm_spec.nodata,
            fill_value=np.nan,
        )
        if landmask_resampled:
            logging.info("landmask was resampled to the hgt reference grid before building waterMask.")
        landmask = clean_invalid(landmask_aligned, lm_spec.nodata)
        land_mask = infer_landmask_to_watermask(landmask)
        logging.info(
            "Landmask valid-land pixel ratio: %.3f",
            float(np.count_nonzero(land_mask)) / float(land_mask.size),
        )
    else:
        land_mask = np.ones((hgt_spec.length, hgt_spec.width), dtype=bool)
        logging.warning(
            "No landmask found. Unified valid geometry mask will use only valid ENU and valid height pixels."
        )

    valid_geom = valid_vec & valid_height & land_mask
    valid_geom_ratio = float(np.count_nonzero(valid_geom)) / float(valid_geom.size)
    logging.info("Unified valid geometry pixel ratio: %.3f", valid_geom_ratio)

    if not np.any(valid_geom):
        raise RuntimeError(
            "Unified valid geometry mask is empty. Check E/N/U, height, and landmask definitions."
        )

    avg_inc = float(meta_txt["avg_incidence_angle"]) if "avg_incidence_angle" in meta_txt else None
    if avg_inc is None:
        avg_inc = 39.0
        logging.warning(
            "avg_incidence_angle missing in metadata.txt; use fallback %.2f degree for internal fill only.",
            avg_inc,
        )

    if force_avg_incidence:
        inc = np.full((hgt_spec.length, hgt_spec.width), np.nan, dtype=np.float32)
        inc[valid_geom] = float(avg_inc)
    else:
        inc = incidence_from_u(u2, valid_geom, avg_incidence_angle=avg_inc)

    az = azimuth_from_en(e2, n2, valid_geom)
    lat, lon = build_lat_lon(hgt_spec)

    inc = fill_nan_inside_mask(inc, valid_geom, fallback_value=avg_inc)
    az = fill_nan_inside_mask(az, valid_geom, fallback_value=float(meta_txt.get("heading", 0.0)))
    hgt = fill_nan_inside_mask(hgt, valid_geom, fallback_value=float(meta_txt.get("avg_height", 0.0)))

    los_east = apply_mask_keep_nan(e2, valid_geom)
    los_north = apply_mask_keep_nan(n2, valid_geom)
    los_up = apply_mask_keep_nan(u2, valid_geom)

    los_norm = np.sqrt(los_east**2 + los_north**2 + los_up**2)
    logging.info(
        "Final LOS norm on unified valid pixels: min=%.4f, median=%.4f, max=%.4f",
        float(np.nanmin(los_norm[valid_geom])),
        float(np.nanmedian(los_norm[valid_geom])),
        float(np.nanmax(los_norm[valid_geom])),
    )
    logging.info(
        "Final incidenceAngle on unified valid pixels: min=%.2f, median=%.2f, max=%.2f degree",
        float(np.nanmin(inc[valid_geom])),
        float(np.nanmedian(inc[valid_geom])),
        float(np.nanmax(inc[valid_geom])),
    )

    out = {
        "height": hgt.astype(np.float32),
        "incidenceAngle": inc.astype(np.float32),
        "azimuthAngle": az.astype(np.float32),
        "latitude": lat.astype(np.float32),
        "longitude": lon.astype(np.float32),
        # True = trusted land / valid geometry pixel.
        "waterMask": valid_geom.astype(np.bool_),
        "los_east": los_east.astype(np.float32),
        "los_north": los_north.astype(np.float32),
        "los_up": los_up.astype(np.float32),
    }

    return out, hgt_spec


# ============================================================
# Interferogram stack QA and loop closure
# ============================================================

def _meta_scale_value(meta: Dict[str, str]) -> Optional[float]:
    """Try to infer a coherence divisor / scale factor from metadata keys."""
    divisor = _get_meta_float(
        meta,
        [
            "COHERENCE_MAX", "coherence_max", "CC_MAX", "cc_max", "VALID_MAX", "valid_max",
            "RANGE_MAX", "range_max", "MAX_VALUE", "max_value",
        ],
        None,
    )
    if divisor is not None and np.isfinite(divisor) and divisor > 1.5:
        return float(divisor)

    scale_factor = _get_meta_float(
        meta,
        ["SCALE_FACTOR", "scale_factor", "Scale", "scale"],
        None,
    )
    if scale_factor is not None and np.isfinite(scale_factor) and 0.0 < scale_factor < 1.0:
        # Return negative value to mark multiplicative scaling by abs(value).
        return -float(scale_factor)
    return None


def scale_coherence(
    cc_raw: np.ndarray,
    cc_meta: Dict[str, str],
    fallback_cc_max: float = 255.0,
    source_dtype: Optional[np.dtype] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Scale LiCSAR coherence to 0-1 using dtype / metadata rules.

    Priority:
      1) uint8 -> divide by 255;
      2) metadata divisor such as COHERENCE_MAX / CC_MAX / VALID_MAX -> divide;
      3) metadata scale_factor in (0, 1) -> multiply;
      4) floating dtype with max <= 1.5 -> already normalized;
      5) integer dtype with max <= 255.5 -> divide by 255;
      6) fallback --cc-max divisor.
    """
    raw_dtype = np.dtype(source_dtype if source_dtype is not None else cc_raw.dtype)
    cc = cc_raw.astype(np.float32)
    finite = np.isfinite(cc)
    if not np.any(finite):
        return np.zeros_like(cc, dtype=np.float32), {
            "coherence_dtype": str(raw_dtype),
            "coherence_scale_method": "all_invalid",
            "coherence_raw_min": None,
            "coherence_raw_max": None,
            "coherence_divisor": None,
        }

    vmin = float(np.nanmin(cc[finite]))
    vmax = float(np.nanmax(cc[finite]))
    meta_scale = _meta_scale_value(cc_meta)

    divisor: Optional[float] = None
    method = ""

    if np.issubdtype(raw_dtype, np.uint8):
        divisor = 255.0
        method = "uint8_div_255"
        coh = cc / divisor
    elif meta_scale is not None and meta_scale > 1.5:
        divisor = float(meta_scale)
        method = "metadata_divisor"
        coh = cc / divisor
    elif meta_scale is not None and meta_scale < 0.0:
        divisor = None
        method = "metadata_scale_factor_multiply"
        coh = cc * abs(float(meta_scale))
    elif np.issubdtype(raw_dtype, np.floating) and vmax <= 1.5:
        divisor = 1.0
        method = "float_already_normalized"
        coh = cc
    elif np.issubdtype(raw_dtype, np.integer) and vmax <= 255.5:
        divisor = 255.0
        method = "integer_max_le_255_div_255"
        coh = cc / divisor
    else:
        divisor = float(fallback_cc_max)
        method = "fallback_divisor"
        coh = cc / divisor

    coh = np.clip(coh, 0.0, 1.0)
    coh[~np.isfinite(coh)] = 0.0
    info = {
        "coherence_dtype": str(raw_dtype),
        "coherence_scale_method": method,
        "coherence_raw_min": vmin,
        "coherence_raw_max": vmax,
        "coherence_divisor": divisor,
    }
    return coh.astype(np.float32), info


def calc_pair_bperp(date1: str, date2: str, bperp_by_date: Dict[str, float]) -> float:
    if date1 not in bperp_by_date:
        raise KeyError(f"{date1} not found in baselines file")
    if date2 not in bperp_by_date:
        raise KeyError(f"{date2} not found in baselines file")
    return float(bperp_by_date[date2] - bperp_by_date[date1])


def make_reject_reason(reasons: List[str]) -> str:
    return ";".join(reasons) if reasons else ""


def load_pair_arrays(
    pair: PairRecord,
    ref_grid: GridSpec,
    fallback_cc_max: float,
    min_unw_valid_ratio: float,
    min_mean_coh: float,
    valid_mask: np.ndarray,
    unw_zero_eps: float,
    keep_zero_unw_valid: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, Dict[str, object]]:
    """
    Load one LiCSAR interferogram pair, apply the trusted geometry mask, and
    perform initial coverage/coherence screening.
    """
    unw_raw, unw_spec = read_tif(pair.unw_file)
    cc_raw, cc_spec, cc_meta = read_tif_with_metadata(pair.cc_file)

    if valid_mask.shape != (ref_grid.length, ref_grid.width):
        raise ValueError(
            f"valid_mask shape mismatch for {pair.pair_dir.name}: "
            f"expected {(ref_grid.length, ref_grid.width)}, got {valid_mask.shape}"
        )

    valid_mask = valid_mask.astype(bool)
    mask_pixel_count = int(np.count_nonzero(valid_mask))
    if mask_pixel_count <= 0:
        raise RuntimeError("The interferogram valid mask is empty. Check geometry waterMask / landmask.")

    # Clean unwrapPhase before resampling. By default, LiCSAR unw=0 background
    # is treated as invalid and excluded from bilinear interpolation, so zero
    # background pixels do not contaminate valid neighboring phase values.
    unw_src = clean_invalid(unw_raw, unw_spec.nodata)
    if not keep_zero_unw_valid:
        unw_src = unw_src.copy()
        unw_src[np.isfinite(unw_src) & (np.abs(unw_src) <= float(unw_zero_eps))] = np.nan

    unw, unw_resampled, unw_grid_info = align_array_to_reference_grid(
        arr=unw_src,
        src=unw_spec,
        ref=ref_grid,
        path=pair.unw_file,
        layer_name="unwrapPhase",
        resampling="bilinear",
        nodata=None,
        fill_value=np.nan,
    )

    # Scale coherence on the source grid first, then resample normalized
    # coherence to the hgt reference grid. This preserves the original dtype /
    # metadata-based scale decision while ensuring spatial alignment.
    cc_raw_for_scale = cc_raw.astype(np.float32, copy=True)
    cc_nodata_mask = np.zeros(cc_raw_for_scale.shape, dtype=bool)
    if cc_spec.nodata is not None and np.isfinite(cc_spec.nodata):
        cc_nodata_mask = cc_raw_for_scale == cc_spec.nodata
        cc_raw_for_scale[cc_nodata_mask] = np.nan
    coh_src, coh_scale_info = scale_coherence(
        cc_raw_for_scale,
        cc_meta,
        fallback_cc_max=fallback_cc_max,
        source_dtype=cc_raw.dtype,
    )
    if np.any(cc_nodata_mask):
        coh_src[cc_nodata_mask] = np.nan

    coh, coh_resampled, coh_grid_info = align_array_to_reference_grid(
        arr=coh_src,
        src=cc_spec,
        ref=ref_grid,
        path=pair.cc_file,
        layer_name="coherence",
        resampling="bilinear",
        nodata=None,
        fill_value=np.nan,
    )
    coh = np.clip(coh, 0.0, 1.0)
    coh[~np.isfinite(coh)] = 0.0

    if keep_zero_unw_valid:
        valid_unw = np.isfinite(unw) & valid_mask
    else:
        valid_unw = np.isfinite(unw) & (np.abs(unw) > float(unw_zero_eps)) & valid_mask

    valid_coh = np.isfinite(coh) & (coh > 0.0) & valid_mask
    valid_final = valid_unw & valid_coh

    unw_out = np.zeros_like(unw, dtype=np.float32)
    coh_out = np.zeros_like(coh, dtype=np.float32)
    unw_out[valid_final] = unw[valid_final].astype(np.float32)
    coh_out[valid_final] = coh[valid_final].astype(np.float32)

    valid_unw_pixel_count = int(np.count_nonzero(valid_unw))
    valid_final_pixel_count = int(np.count_nonzero(valid_final))
    valid_unw_ratio = float(valid_unw_pixel_count) / float(mask_pixel_count)
    valid_final_ratio = float(valid_final_pixel_count) / float(mask_pixel_count)
    scene_valid_unw_ratio = float(valid_unw_pixel_count) / float(unw.size)

    mean_coh_valid_unw = float(np.mean(coh[valid_unw & np.isfinite(coh)])) if np.any(valid_unw & np.isfinite(coh)) else 0.0
    mean_coh_final = float(np.mean(coh_out[valid_final])) if valid_final_pixel_count > 0 else 0.0
    median_coh_final = float(np.median(coh_out[valid_final])) if valid_final_pixel_count > 0 else 0.0

    reasons: List[str] = []
    if valid_unw_ratio < min_unw_valid_ratio:
        reasons.append("low_unwrapped_valid_percentage")
    if mean_coh_valid_unw < min_mean_coh:
        reasons.append("low_average_coherence")
    initial_keep = len(reasons) == 0

    qa: Dict[str, object] = {
        "mask_pixel_count": mask_pixel_count,
        "valid_unw_pixel_count": valid_unw_pixel_count,
        "valid_final_pixel_count": valid_final_pixel_count,
        "valid_unw_ratio": valid_unw_ratio,
        "valid_final_ratio": valid_final_ratio,
        "scene_valid_unw_ratio": scene_valid_unw_ratio,
        "mean_coh_valid_unw": mean_coh_valid_unw,
        "mean_coh_final": mean_coh_final,
        "median_coh_final": median_coh_final,
        "initial_keep": bool(initial_keep),
        "initial_reject_reason": make_reject_reason(reasons),
        "unwrap_resampled": bool(unw_resampled),
        "unwrap_resampling_method": str(unw_grid_info.get("resampling_method", "none")),
        "unwrap_source_shape": unw_grid_info.get("source_shape"),
        "unwrap_source_center0": unw_grid_info.get("source_center0"),
        "unwrap_target_center0": unw_grid_info.get("target_center0"),
        "coherence_resampled": bool(coh_resampled),
        "coherence_resampling_method": str(coh_grid_info.get("resampling_method", "none")),
        "coherence_source_shape": coh_grid_info.get("source_shape"),
        "coherence_source_center0": coh_grid_info.get("source_center0"),
        "coherence_target_center0": coh_grid_info.get("target_center0"),
        **coh_scale_info,
    }
    return unw_out, coh_out, valid_final.astype(np.bool_), bool(initial_keep), qa


def run_loop_closure_screening(
    records: List[Dict[str, object]],
    valid_mask: np.ndarray,
    loop_min_valid_ratio: float,
    loop_phase_threshold: float,
    loop_bad_pixel_ratio: float,
    loop_rms_threshold: float,
    max_loop_count: int = 0,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    """
    Score triangular closure loops on initially retained IFGs.

    For dates A < B < C, use:
        closure = unw(A_B) + unw(B_C) - unw(A_C)

    Direct closure phase is used instead of wrapped closure phase, because an
    unwrapping error appears as an integer multiple of 2*pi and would disappear
    if blindly wrapped to [-pi, pi].
    """
    mask_pixel_count = int(np.count_nonzero(valid_mask))
    if mask_pixel_count <= 0:
        raise RuntimeError("valid_mask is empty; cannot run loop closure screening")

    kept_records = [r for r in records if bool(r["initial_keep"])]
    by_dates: Dict[Tuple[str, str], Dict[str, object]] = {
        (str(r["date1"]), str(r["date2"])): r for r in kept_records
    }

    dates = sorted({str(r["date1"]) for r in kept_records} | {str(r["date2"]) for r in kept_records})
    pair_stats: Dict[str, Dict[str, object]] = {
        str(r["pair"]): {
            "n_tested_loops": 0,
            "n_bad_loops": 0,
            "bad_loop_fraction": 0.0,
            "worst_loop_rms_rad": 0.0,
            "worst_loop_bad_pixel_ratio": 0.0,
            "worst_loop_valid_ratio": 0.0,
        }
        for r in kept_records
    }
    loop_rows: List[Dict[str, object]] = []

    loop_count = 0
    cap_reached = False
    for ia in range(len(dates) - 2):
        if cap_reached:
            break
        for ib in range(ia + 1, len(dates) - 1):
            if cap_reached:
                break
            for ic in range(ib + 1, len(dates)):
                a, b, c = dates[ia], dates[ib], dates[ic]
                rec_ab = by_dates.get((a, b))
                rec_bc = by_dates.get((b, c))
                rec_ac = by_dates.get((a, c))
                if rec_ab is None or rec_bc is None or rec_ac is None:
                    continue

                if max_loop_count > 0 and loop_count >= max_loop_count:
                    cap_reached = True
                    break

                loop_count += 1
                pair_names = [str(rec_ab["pair"]), str(rec_bc["pair"]), str(rec_ac["pair"])]

                common_valid = (
                    rec_ab["valid_data"].astype(bool)
                    & rec_bc["valid_data"].astype(bool)
                    & rec_ac["valid_data"].astype(bool)
                    & valid_mask.astype(bool)
                )
                valid_pixel_count = int(np.count_nonzero(common_valid))
                valid_ratio = float(valid_pixel_count) / float(mask_pixel_count)

                row: Dict[str, object] = {
                    "loop": f"{a}_{b}_{c}",
                    "date_a": a,
                    "date_b": b,
                    "date_c": c,
                    "pair_ab": pair_names[0],
                    "pair_bc": pair_names[1],
                    "pair_ac": pair_names[2],
                    "valid_pixel_count": valid_pixel_count,
                    "valid_ratio": valid_ratio,
                    "used_for_scoring": False,
                    "loop_rms_rad": None,
                    "loop_median_abs_rad": None,
                    "loop_bad_pixel_ratio": None,
                    "is_bad_loop": False,
                    "bad_reason": "insufficient_common_valid_pixels" if valid_ratio < loop_min_valid_ratio else "",
                }

                if valid_ratio >= loop_min_valid_ratio and valid_pixel_count > 0:
                    closure = (
                        rec_ab["unwrap"][common_valid]
                        + rec_bc["unwrap"][common_valid]
                        - rec_ac["unwrap"][common_valid]
                    ).astype(np.float32)
                    closure = closure[np.isfinite(closure)]

                    if closure.size > 0:
                        abs_closure = np.abs(closure)
                        loop_rms = float(np.sqrt(np.mean(closure.astype(np.float64) ** 2)))
                        loop_median_abs = float(np.median(abs_closure))
                        bad_pixel_ratio = float(np.mean(abs_closure > float(loop_phase_threshold)))
                        is_bad_loop = (bad_pixel_ratio >= loop_bad_pixel_ratio) or (loop_rms >= loop_rms_threshold)
                        reasons = []
                        if bad_pixel_ratio >= loop_bad_pixel_ratio:
                            reasons.append("high_bad_pixel_ratio")
                        if loop_rms >= loop_rms_threshold:
                            reasons.append("high_loop_rms")

                        row.update({
                            "used_for_scoring": True,
                            "loop_rms_rad": loop_rms,
                            "loop_median_abs_rad": loop_median_abs,
                            "loop_bad_pixel_ratio": bad_pixel_ratio,
                            "is_bad_loop": bool(is_bad_loop),
                            "bad_reason": make_reject_reason(reasons),
                        })

                        for pair_name in pair_names:
                            st = pair_stats[pair_name]
                            st["n_tested_loops"] = int(st["n_tested_loops"]) + 1
                            if is_bad_loop:
                                st["n_bad_loops"] = int(st["n_bad_loops"]) + 1
                            st["worst_loop_rms_rad"] = max(float(st["worst_loop_rms_rad"]), loop_rms)
                            st["worst_loop_bad_pixel_ratio"] = max(
                                float(st["worst_loop_bad_pixel_ratio"]), bad_pixel_ratio
                            )
                            st["worst_loop_valid_ratio"] = max(float(st["worst_loop_valid_ratio"]), valid_ratio)
                    else:
                        row["bad_reason"] = "no_finite_closure_pixels"

                loop_rows.append(row)

    if cap_reached:
        logging.warning(
            "Loop closure scoring stopped at --max-loop-count=%d. Increase it or set 0 for all loops.",
            max_loop_count,
        )

    for st in pair_stats.values():
        n_tested = int(st["n_tested_loops"])
        n_bad = int(st["n_bad_loops"])
        st["bad_loop_fraction"] = float(n_bad) / float(n_tested) if n_tested > 0 else 0.0

    logging.info(
        "Loop closure summary: candidate IFGs=%d, scored_loops=%d, bad_loops=%d",
        len(kept_records),
        int(sum(1 for r in loop_rows if r.get("used_for_scoring"))),
        int(sum(1 for r in loop_rows if r.get("is_bad_loop"))),
    )
    return loop_rows, pair_stats


def build_ifgram_stack(
    pairs: List[PairRecord],
    ref_grid: GridSpec,
    bperp_by_date: Dict[str, float],
    fallback_cc_max: float,
    min_unw_valid_ratio: float,
    min_mean_coh: float,
    valid_mask: np.ndarray,
    unw_zero_eps: float,
    keep_zero_unw_valid: bool,
    enable_loop_closure: bool,
    loop_min_valid_ratio: float,
    loop_phase_threshold: float,
    loop_bad_pixel_ratio: float,
    loop_rms_threshold: float,
    loop_min_tested_loops: int,
    loop_min_bad_loops: int,
    loop_bad_loop_fraction: float,
    max_loop_count: int,
) -> Dict[str, object]:
    """
    Build MintPy ifgramStack datasets using the same valid geometry mask as
    geometryGeo.h5, with initial QA and conservative loop-closure rejection.
    """
    L, W = ref_grid.length, ref_grid.width

    if valid_mask.shape != (L, W):
        raise ValueError(f"valid_mask shape mismatch: expected {(L, W)}, got {valid_mask.shape}")
    valid_mask = valid_mask.astype(bool)
    mask_pixel_count = int(np.count_nonzero(valid_mask))
    if mask_pixel_count <= 0:
        raise RuntimeError("valid_mask is empty; cannot build ifgramStack.h5")

    logging.info(
        "Use unified valid geometry mask for ifgram stack: %d / %d pixels (ratio=%.3f)",
        mask_pixel_count,
        valid_mask.size,
        float(mask_pixel_count) / float(valid_mask.size),
    )

    records: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []

    for pair in pairs:
        pair_name = f"{pair.date1}_{pair.date2}"
        try:
            unw, coh, valid_data, initial_keep, qa = load_pair_arrays(
                pair=pair,
                ref_grid=ref_grid,
                fallback_cc_max=fallback_cc_max,
                min_unw_valid_ratio=min_unw_valid_ratio,
                min_mean_coh=min_mean_coh,
                valid_mask=valid_mask,
                unw_zero_eps=unw_zero_eps,
                keep_zero_unw_valid=keep_zero_unw_valid,
            )
            pair_bperp = calc_pair_bperp(pair.date1, pair.date2, bperp_by_date)
        except GridCompatibilityError as e:
            logging.warning("Skip pair %s because grid is incompatible: %s", pair_name, e)
            skipped_rows.append({
                "pair": pair_name,
                "date1": pair.date1,
                "date2": pair.date2,
                "reason": "grid_incompatible",
                "message": str(e),
            })
            continue
        except Exception as e:
            logging.warning("Skip pair %s because it cannot be loaded/scored: %s", pair_name, e)
            skipped_rows.append({
                "pair": pair_name,
                "date1": pair.date1,
                "date2": pair.date2,
                "reason": "load_or_baseline_error",
                "message": str(e),
            })
            continue

        rec: Dict[str, object] = {
            "pair": pair_name,
            "date1": pair.date1,
            "date2": pair.date2,
            "bperp_m": float(pair_bperp),
            "unwrap": unw.astype(np.float32, copy=False),
            "coherence": coh.astype(np.float32, copy=False),
            "valid_data": valid_data.astype(np.bool_, copy=False),
            "initial_keep": bool(initial_keep),
            "loop_rejected": False,
            "keep": bool(initial_keep),
            "reject_reason": str(qa.get("initial_reject_reason", "")),
            "qa": qa,
        }
        records.append(rec)

        logging.info(
            "Initial QA %-17s bperp=%8.2f m, unw_valid=%.3f, final_valid=%.3f, "
            "mean_coh=%.3f, initial_keep=%s, scale=%s",
            pair_name,
            pair_bperp,
            qa["valid_unw_ratio"],
            qa["valid_final_ratio"],
            qa["mean_coh_valid_unw"],
            initial_keep,
            qa["coherence_scale_method"],
        )

    if not records:
        raise RuntimeError(
            "No grid-compatible interferogram pairs remain after loading. "
            "Check the LiCSAR products or resample incompatible pairs to the reference geometry grid."
        )

    loop_rows: List[Dict[str, object]] = []
    loop_pair_stats: Dict[str, Dict[str, object]] = {}

    if enable_loop_closure:
        loop_rows, loop_pair_stats = run_loop_closure_screening(
            records=records,
            valid_mask=valid_mask,
            loop_min_valid_ratio=loop_min_valid_ratio,
            loop_phase_threshold=loop_phase_threshold,
            loop_bad_pixel_ratio=loop_bad_pixel_ratio,
            loop_rms_threshold=loop_rms_threshold,
            max_loop_count=max_loop_count,
        )
    else:
        logging.warning("Loop closure screening is disabled by user option.")

    for rec in records:
        pair_name = str(rec["pair"])
        st = loop_pair_stats.get(pair_name, {
            "n_tested_loops": 0,
            "n_bad_loops": 0,
            "bad_loop_fraction": 0.0,
            "worst_loop_rms_rad": 0.0,
            "worst_loop_bad_pixel_ratio": 0.0,
            "worst_loop_valid_ratio": 0.0,
        })
        rec["loop_stats"] = st

        loop_rejected = (
            bool(rec["initial_keep"])
            and int(st["n_tested_loops"]) >= int(loop_min_tested_loops)
            and int(st["n_bad_loops"]) >= int(loop_min_bad_loops)
            and float(st["bad_loop_fraction"]) >= float(loop_bad_loop_fraction)
        )
        rec["loop_rejected"] = bool(loop_rejected)

        if not bool(rec["initial_keep"]):
            rec["keep"] = False
            rec["reject_reason"] = str(rec["qa"].get("initial_reject_reason", "initial_quality_reject"))
        elif loop_rejected:
            rec["keep"] = False
            rec["reject_reason"] = "loop_closure_unwrapping_error"
        else:
            rec["keep"] = True
            rec["reject_reason"] = ""

    unwrap_phase = np.stack([r["unwrap"] for r in records], axis=0).astype(np.float32, copy=False)
    coherence = np.stack([r["coherence"] for r in records], axis=0).astype(np.float32, copy=False)
    date = np.asarray([(str(r["date1"]).encode("ascii"), str(r["date2"]).encode("ascii")) for r in records], dtype="S8")
    bperp = np.asarray([float(r["bperp_m"]) for r in records], dtype=np.float32)
    drop_ifgram = np.asarray([bool(r["keep"]) for r in records], dtype=np.bool_)

    if not np.any(drop_ifgram):
        logging.warning("All loaded interferograms are marked as drop. Consider relaxing QA thresholds.")

    qa_rows: List[Dict[str, object]] = []
    bad_ifgram_rows: List[Dict[str, object]] = []
    for rec in records:
        qa = rec["qa"]
        st = rec.get("loop_stats", {})
        row = {
            "pair": rec["pair"],
            "date1": rec["date1"],
            "date2": rec["date2"],
            "bperp_m": rec["bperp_m"],
            "mask_pixel_count": qa["mask_pixel_count"],
            "valid_unw_pixel_count": qa["valid_unw_pixel_count"],
            "valid_final_pixel_count": qa["valid_final_pixel_count"],
            "valid_unw_ratio": qa["valid_unw_ratio"],
            "valid_final_ratio": qa["valid_final_ratio"],
            "scene_valid_unw_ratio": qa["scene_valid_unw_ratio"],
            "mean_coh_valid_unw": qa["mean_coh_valid_unw"],
            "mean_coh_final": qa["mean_coh_final"],
            "median_coh_final": qa["median_coh_final"],
            "coherence_dtype": qa["coherence_dtype"],
            "coherence_raw_min": qa["coherence_raw_min"],
            "coherence_raw_max": qa["coherence_raw_max"],
            "coherence_scale_method": qa["coherence_scale_method"],
            "coherence_divisor": qa["coherence_divisor"],
            "unwrap_resampled": qa.get("unwrap_resampled", False),
            "unwrap_resampling_method": qa.get("unwrap_resampling_method", "none"),
            "unwrap_source_shape": qa.get("unwrap_source_shape", ""),
            "unwrap_source_center0": qa.get("unwrap_source_center0", ""),
            "unwrap_target_center0": qa.get("unwrap_target_center0", ""),
            "coherence_resampled": qa.get("coherence_resampled", False),
            "coherence_resampling_method": qa.get("coherence_resampling_method", "none"),
            "coherence_source_shape": qa.get("coherence_source_shape", ""),
            "coherence_source_center0": qa.get("coherence_source_center0", ""),
            "coherence_target_center0": qa.get("coherence_target_center0", ""),
            "initial_keep": rec["initial_keep"],
            "initial_reject_reason": qa["initial_reject_reason"],
            "n_tested_loops": st.get("n_tested_loops", 0),
            "n_bad_loops": st.get("n_bad_loops", 0),
            "bad_loop_fraction": st.get("bad_loop_fraction", 0.0),
            "worst_loop_rms_rad": st.get("worst_loop_rms_rad", 0.0),
            "worst_loop_bad_pixel_ratio": st.get("worst_loop_bad_pixel_ratio", 0.0),
            "worst_loop_valid_ratio": st.get("worst_loop_valid_ratio", 0.0),
            "loop_rejected": rec["loop_rejected"],
            "keep": rec["keep"],
            "reject_reason": rec["reject_reason"],
        }
        qa_rows.append(row)
        if not bool(rec["keep"]):
            bad_ifgram_rows.append({
                "pair": rec["pair"],
                "date1": rec["date1"],
                "date2": rec["date2"],
                "reject_reason": rec["reject_reason"],
                "initial_keep": rec["initial_keep"],
                "loop_rejected": rec["loop_rejected"],
                "valid_unw_ratio": qa["valid_unw_ratio"],
                "mean_coh_valid_unw": qa["mean_coh_valid_unw"],
                "n_tested_loops": st.get("n_tested_loops", 0),
                "n_bad_loops": st.get("n_bad_loops", 0),
                "bad_loop_fraction": st.get("bad_loop_fraction", 0.0),
                "worst_loop_rms_rad": st.get("worst_loop_rms_rad", 0.0),
                "worst_loop_bad_pixel_ratio": st.get("worst_loop_bad_pixel_ratio", 0.0),
            })

    logging.info(
        "Interferogram stack summary: loaded=%d, kept=%d, initial_rejected=%d, loop_rejected=%d, skipped=%d",
        len(records),
        int(np.count_nonzero(drop_ifgram)),
        int(sum(1 for r in records if not bool(r["initial_keep"]))),
        int(sum(1 for r in records if bool(r["loop_rejected"]))),
        len(skipped_rows),
    )

    return {
        "unwrapPhase": unwrap_phase,
        "coherence": coherence,
        "date": date,
        "bperp": bperp,
        "dropIfgram": drop_ifgram,
        "qa_rows": qa_rows,
        "skipped_rows": skipped_rows,
        "bad_ifgram_rows": bad_ifgram_rows,
        "loop_rows": loop_rows,
        "ifgram_valid_mask": valid_mask.astype(np.bool_),
    }


# ============================================================
# HDF5 writing
# ============================================================

def common_attrs_from_grid(
    spec: GridSpec,
    frame_dir: Path,
    meta_txt: Dict[str, object],
    wavelength: float,
    orbit_direction: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    # Infer orbit direction if user did not explicitly provide it.
    orbit_dir = orbit_direction
    if orbit_dir == "UNKNOWN":
        heading = meta_txt.get("heading", None)
        if heading is not None:
            try:
                heading_f = float(heading)
                orbit_dir = "ASCENDING" if heading_f < 0 else "DESCENDING"
            except Exception:
                orbit_dir = "UNKNOWN"

    alooks, rlooks, looks_info = get_looks_from_metadata(meta_txt)

    attrs: Dict[str, object] = {
        "FILE_LENGTH": spec.length,
        "LENGTH": spec.length,
        "WIDTH": spec.width,
        "X_FIRST": spec.x_first,
        "Y_FIRST": spec.y_first,
        "X_STEP": spec.x_step,
        "Y_STEP": spec.y_step,
        "X_UNIT": spec.x_unit,
        "Y_UNIT": spec.y_unit,
        "PROCESSOR": "LiCSAR",
        "FRAME_ID": frame_dir.name,
        "PLATFORM": "Sentinel-1",
        "WAVELENGTH": float(wavelength),
        "ORBIT_DIRECTION": orbit_dir,
    }

    if alooks is not None:
        attrs["ALOOKS"] = int(alooks)
    if rlooks is not None:
        attrs["RLOOKS"] = int(rlooks)

    if "heading" in meta_txt:
        attrs["HEADING"] = float(meta_txt["heading"])
    if "center_time" in meta_txt:
        try:
            attrs["CENTER_LINE_UTC"] = float(hms_to_seconds(str(meta_txt["center_time"])))
        except Exception:
            pass
    if "avg_incidence_angle" in meta_txt:
        attrs["AVG_INCIDENCE_ANGLE"] = float(meta_txt["avg_incidence_angle"])
    if "azimuth_resolution" in meta_txt:
        attrs["AZIMUTH_RESOLUTION"] = float(meta_txt["azimuth_resolution"])
    if "range_resolution" in meta_txt:
        attrs["RANGE_RESOLUTION"] = float(meta_txt["range_resolution"])
    if "centre_range_m" in meta_txt:
        attrs["CENTER_RANGE"] = float(meta_txt["centre_range_m"])
    if "centre_range_ok_m" in meta_txt:
        attrs["CENTER_RANGE_OK"] = float(meta_txt["centre_range_ok_m"])
    if "applied_DEM" in meta_txt:
        attrs["DEM_SOURCE"] = str(meta_txt["applied_DEM"])
    if "master" in meta_txt:
        attrs["LICSAR_MASTER_DATE"] = str(meta_txt["master"])

    return attrs, looks_info


def _mintpy_attr_dict(attrs: Dict[str, object]) -> Dict[str, object]:
    """Normalize attributes before passing them to mintpy.utils.writefile.write()."""
    out: Dict[str, object] = {}
    for k, v in attrs.items():
        if isinstance(v, np.generic):
            v = v.item()
        out[str(k)] = v
    return out


def write_geometry_h5(
    out_file: Path,
    geom: Dict[str, np.ndarray],
    common_attrs: Dict[str, object],
) -> None:
    """Write geometryGeo.h5 using MintPy's writefile.write()."""
    out_file.parent.mkdir(parents=True, exist_ok=True)

    attrs = dict(common_attrs)
    attrs["FILE_TYPE"] = "geometry"
    attrs = _mintpy_attr_dict(attrs)

    ds_dict = {
        "height": geom["height"].astype(np.float32),
        "latitude": geom["latitude"].astype(np.float32),
        "longitude": geom["longitude"].astype(np.float32),
        "incidenceAngle": geom["incidenceAngle"].astype(np.float32),
        "azimuthAngle": geom["azimuthAngle"].astype(np.float32),
        "waterMask": geom["waterMask"].astype(np.bool_),
        "los_east": geom["los_east"].astype(np.float32),
        "los_north": geom["los_north"].astype(np.float32),
        "los_up": geom["los_up"].astype(np.float32),
    }

    ds_unit_dict = {
        "height": "m",
        "latitude": "degree",
        "longitude": "degree",
        "incidenceAngle": "degree",
        "azimuthAngle": "degree",
        "waterMask": "1",
        "los_east": "1",
        "los_north": "1",
        "los_up": "1",
    }

    writefile.write(
        datasetDict=ds_dict,
        out_file=str(out_file),
        metadata=attrs,
        compression=LZF,
        ds_unit_dict=ds_unit_dict,
        print_msg=True,
    )


def write_ifgram_h5(
    out_file: Path,
    stack: Dict[str, object],
    common_attrs: Dict[str, object],
) -> None:
    """Write ifgramStack.h5 using MintPy's writefile.write(). No fake connectComponent is written."""
    out_file.parent.mkdir(parents=True, exist_ok=True)

    attrs = dict(common_attrs)
    attrs["FILE_TYPE"] = "ifgramStack"
    attrs["UNIT"] = "radian"
    attrs = _mintpy_attr_dict(attrs)

    ds_dict = {
        "date": stack["date"].astype("S8"),
        "bperp": stack["bperp"].astype(np.float32),
        # MintPy convention: True = keep, False = drop.
        "dropIfgram": stack["dropIfgram"].astype(np.bool_),
        "unwrapPhase": stack["unwrapPhase"].astype(np.float32),
        "coherence": stack["coherence"].astype(np.float32),
    }

    ds_unit_dict = {
        "date": None,
        "bperp": "m",
        "dropIfgram": "1",
        "unwrapPhase": "radian",
        "coherence": "1",
    }

    writefile.write(
        datasetDict=ds_dict,
        out_file=str(out_file),
        metadata=attrs,
        compression=LZF,
        ds_unit_dict=ds_unit_dict,
        print_msg=True,
    )


# ============================================================
# QA / config
# ============================================================

def _csv_escape(v: object) -> str:
    if v is None:
        return ""
    val = str(v).replace("\n", " ").replace("\r", " ")
    if "," in val or '"' in val:
        val = '"' + val.replace('"', '""') + '"'
    return val


def write_rows_csv(path: Path, rows: List[Dict[str, object]], header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(_csv_escape(r.get(h, "")) for h in header) + "\n")


def write_pair_table(path: Path, qa_rows: List[Dict[str, object]]) -> None:
    header = [
        "pair", "date1", "date2", "bperp_m",
        "mask_pixel_count", "valid_unw_pixel_count", "valid_final_pixel_count",
        "valid_unw_ratio", "valid_final_ratio", "scene_valid_unw_ratio",
        "mean_coh_valid_unw", "mean_coh_final", "median_coh_final",
        "coherence_dtype", "coherence_raw_min", "coherence_raw_max",
        "coherence_scale_method", "coherence_divisor",
        "unwrap_resampled", "unwrap_resampling_method", "unwrap_source_shape",
        "unwrap_source_center0", "unwrap_target_center0",
        "coherence_resampled", "coherence_resampling_method", "coherence_source_shape",
        "coherence_source_center0", "coherence_target_center0",
        "initial_keep", "initial_reject_reason",
        "n_tested_loops", "n_bad_loops", "bad_loop_fraction",
        "worst_loop_rms_rad", "worst_loop_bad_pixel_ratio", "worst_loop_valid_ratio",
        "loop_rejected", "keep", "reject_reason",
    ]
    write_rows_csv(path, qa_rows, header)


def write_bad_ifgram_table(path: Path, bad_rows: List[Dict[str, object]]) -> None:
    header = [
        "pair", "date1", "date2", "reject_reason", "initial_keep", "loop_rejected",
        "valid_unw_ratio", "mean_coh_valid_unw",
        "n_tested_loops", "n_bad_loops", "bad_loop_fraction",
        "worst_loop_rms_rad", "worst_loop_bad_pixel_ratio",
    ]
    write_rows_csv(path, bad_rows, header)


def write_loop_closure_table(path: Path, loop_rows: List[Dict[str, object]]) -> None:
    header = [
        "loop", "date_a", "date_b", "date_c",
        "pair_ab", "pair_bc", "pair_ac",
        "valid_pixel_count", "valid_ratio", "used_for_scoring",
        "loop_rms_rad", "loop_median_abs_rad", "loop_bad_pixel_ratio",
        "is_bad_loop", "bad_reason",
    ]
    write_rows_csv(path, loop_rows, header)


def write_skipped_pair_table(path: Path, skipped_rows: List[Dict[str, object]]) -> None:
    header = ["pair", "date1", "date2", "reason", "message"]
    write_rows_csv(path, skipped_rows, header)


def write_summary_json(path: Path, summary: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def write_default_cfg(out_cfg: Path) -> None:
    out_cfg.parent.mkdir(parents=True, exist_ok=True)

    cfg = """# MintPy config for LiCSAR-prepared HDF5
mintpy.load.processor                = auto
mintpy.unwrapError.method            = no
mintpy.network.coherenceBased        = yes
mintpy.network.minCoherence          = 0.2
mintpy.reference.minCoherence        = 0.3
mintpy.troposphericDelay.method      = no
mintpy.deramp                        = linear
mintpy.save.hdfEos5                  = no

# Notes:
# 1) inputs/ifgramStack.h5 and inputs/geometryGeo.h5 are already prepared.
# 2) No fake/dummy connectComponent is written by this script.
# 3) Keep unwrap-error correction OFF unless real connected components or a
#    validated unwrap-error correction workflow is available.
# 4) Bad IFGs rejected by coverage/coherence screening or loop closure are
#    recorded in inputs/ifgramStack.h5:/dropIfgram as False.
"""
    out_cfg.write_text(cfg, encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main():
    args = build_parser().parse_args()
    setup_logging(args.verbose)

    if getattr(args, "write_dummy_conncomp", False):
        logging.warning(
            "--write-dummy-conncomp is deprecated and ignored. "
            "This script never writes fake connectComponent."
        )

    frame_dir = args.frame_dir.resolve()
    outdir = (args.outdir or (frame_dir / "mintpy")).resolve()
    inputs_dir = outdir / "inputs"
    qa_dir = outdir / "qa"
    cfg_dir = outdir / "config"

    logging.info("Frame dir: %s", frame_dir)
    logging.info("Output dir: %s", outdir)

    # 1) metadata
    meta_files = scan_metadata_files(frame_dir)
    meta_txt = parse_metadata_txt(meta_files["metadata_txt"])
    baseline_info = parse_baselines(meta_files["baselines"])

    logging.info("metadata.txt master=%s", meta_txt.get("master", "N/A"))
    logging.info("metadata.txt heading=%s", meta_txt.get("heading", "N/A"))
    logging.info("metadata.txt avg_incidence_angle=%s", meta_txt.get("avg_incidence_angle", "N/A"))
    logging.info("baselines master=%s", baseline_info["master_date"])

    # 2) geometry
    geom, grid = build_geometry(
        frame_dir=frame_dir,
        meta_files=meta_files,
        meta_txt=meta_txt,
        force_avg_incidence=args.force_avg_incidence,
    )

    # 3) interferograms
    pairs = scan_interferograms(frame_dir)
    logging.info("Found %d interferogram pairs.", len(pairs))

    # 4) ifgram stack with initial QA + loop closure screening
    valid_ifgram_mask = geom["waterMask"].astype(bool)
    stack = build_ifgram_stack(
        pairs=pairs,
        ref_grid=grid,
        bperp_by_date=baseline_info["bperp_by_date"],
        fallback_cc_max=args.cc_max,
        min_unw_valid_ratio=args.min_unw_valid_ratio,
        min_mean_coh=args.min_mean_coh,
        valid_mask=valid_ifgram_mask,
        unw_zero_eps=args.unw_zero_eps,
        keep_zero_unw_valid=args.keep_zero_unw_valid,
        enable_loop_closure=not args.disable_loop_closure,
        loop_min_valid_ratio=args.loop_min_valid_ratio,
        loop_phase_threshold=args.loop_phase_threshold,
        loop_bad_pixel_ratio=args.loop_bad_pixel_ratio,
        loop_rms_threshold=args.loop_rms_threshold,
        loop_min_tested_loops=args.loop_min_tested_loops,
        loop_min_bad_loops=args.loop_min_bad_loops,
        loop_bad_loop_fraction=args.loop_bad_loop_fraction,
        max_loop_count=args.max_loop_count,
    )

    # 5) common attrs
    common_attrs, looks_info = common_attrs_from_grid(
        spec=grid,
        frame_dir=frame_dir,
        meta_txt=meta_txt,
        wavelength=args.wavelength,
        orbit_direction=args.orbit_direction,
    )

    # 6) write HDF5
    ifgram_h5 = inputs_dir / "ifgramStack.h5"
    geom_h5 = inputs_dir / "geometryGeo.h5"

    write_ifgram_h5(out_file=ifgram_h5, stack=stack, common_attrs=common_attrs)
    write_geometry_h5(out_file=geom_h5, geom=geom, common_attrs=common_attrs)

    # 7) QA outputs
    write_pair_table(qa_dir / "pair_table.csv", stack["qa_rows"])
    write_bad_ifgram_table(qa_dir / "bad_ifgrams.csv", stack.get("bad_ifgram_rows", []))
    write_loop_closure_table(qa_dir / "loop_closure_table.csv", stack.get("loop_rows", []))
    write_skipped_pair_table(qa_dir / "skipped_pairs.csv", stack.get("skipped_rows", []))

    n_loaded = int(stack["unwrapPhase"].shape[0])
    n_kept = int(np.count_nonzero(stack["dropIfgram"]))
    n_initial_rejected = int(sum(1 for r in stack["qa_rows"] if not bool(r.get("initial_keep", False))))
    n_loop_rejected = int(sum(1 for r in stack["qa_rows"] if bool(r.get("loop_rejected", False))))

    write_summary_json(
        qa_dir / "summary.json",
        {
            "frame_id": frame_dir.name,
            "n_ifg_scanned": len(pairs),
            "n_ifg_loaded": n_loaded,
            "n_ifg_kept": n_kept,
            "n_ifg_initial_rejected": n_initial_rejected,
            "n_ifg_loop_rejected": n_loop_rejected,
            "n_ifg_skipped": len(stack.get("skipped_rows", [])),
            "grid": {
                "length": grid.length,
                "width": grid.width,
                "x_first": grid.x_first,
                "y_first": grid.y_first,
                "x_step": grid.x_step,
                "y_step": grid.y_step,
                "x_unit": grid.x_unit,
                "y_unit": grid.y_unit,
                "crs_key": grid.crs_key,
            },
            "qa_thresholds": {
                "min_unw_valid_ratio": args.min_unw_valid_ratio,
                "min_mean_coherence": args.min_mean_coh,
                "unw_zero_eps": args.unw_zero_eps,
                "keep_zero_unw_valid": args.keep_zero_unw_valid,
                "loop_closure_enabled": not args.disable_loop_closure,
                "loop_min_valid_ratio": args.loop_min_valid_ratio,
                "loop_phase_threshold": args.loop_phase_threshold,
                "loop_bad_pixel_ratio": args.loop_bad_pixel_ratio,
                "loop_rms_threshold": args.loop_rms_threshold,
                "loop_min_tested_loops": args.loop_min_tested_loops,
                "loop_min_bad_loops": args.loop_min_bad_loops,
                "loop_bad_loop_fraction": args.loop_bad_loop_fraction,
                "max_loop_count": args.max_loop_count,
            },
            "geometry_valid_ratio": float(np.count_nonzero(geom["waterMask"])) / float(geom["waterMask"].size),
            "ifgram_mask_pixel_count": int(np.count_nonzero(valid_ifgram_mask)),
            "ifgram_mask_total_pixel_count": int(valid_ifgram_mask.size),
            "ifgram_mask_valid_ratio": float(np.count_nonzero(valid_ifgram_mask)) / float(valid_ifgram_mask.size),
            "metadata_txt": meta_txt,
            "looks_info": looks_info,
            "baseline_master_date": baseline_info["master_date"],
        },
    )

    # 8) MintPy config
    write_default_cfg(out_cfg=cfg_dir / "mintpy_licsar.cfg")

    logging.info("Done.")
    logging.info("Generated:")
    logging.info("  %s", ifgram_h5)
    logging.info("  %s", geom_h5)
    logging.info("  %s", cfg_dir / "mintpy_licsar.cfg")
    logging.info("  %s", qa_dir / "pair_table.csv")
    logging.info("  %s", qa_dir / "bad_ifgrams.csv")
    logging.info("  %s", qa_dir / "loop_closure_table.csv")
    logging.info("Next: inspect HDF5 with info.py / view.py, then run smallbaselineApp.py.")


if __name__ == "__main__":
    main()
