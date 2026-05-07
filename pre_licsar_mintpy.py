#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prep_licsar.py

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

Notes
-----
1) This script writes MintPy-style HDF5 directly.
2) LiCSAR coherence is commonly 0-255; this script rescales it to 0-1.
3) No real connected components are provided here.
4) For the first run, keep unwrap-error correction OFF in MintPy config.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from mintpy.utils import readfile


# ============================================================
# Constants
# ============================================================

DEFAULT_WAVELENGTH = 0.05546576   # Sentinel-1 wavelength [m]

# LiCSAR processor default multilooking factors for interferogram generation
DEFAULT_RLOOKS = 20
DEFAULT_ALOOKS = 4

LZF = "lzf"


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


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare LiCSAR single-frame data for MintPy."
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
        help="Coherence max value for rescaling to 0-1 (default: 255)",
    )
    p.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.05,
        help="Minimum valid-pixel ratio to keep an interferogram",
    )
    p.add_argument(
        "--min-mean-coh",
        type=float,
        default=0.05,
        help="Minimum mean coherence to keep an interferogram",
    )
    p.add_argument(
        "--write-dummy-conncomp",
        action="store_true",
        help="Write a fake connectComponent dataset (1 valid / 0 invalid). "
             "Only for compatibility testing; not for unwrap-error correction.",
    )
    p.add_argument(
        "--force-avg-incidence",
        action="store_true",
        help="Ignore U-derived incidence angle and fill a constant "
             "avg_incidence_angle from metadata.txt",
    )
    p.add_argument(
        "--orbit-direction",
        choices=["ASCENDING", "DESCENDING", "UNKNOWN"],
        default="UNKNOWN",
        help="Optional manual orbit direction metadata",
    )
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


def _build_crs_key(meta: Dict[str, str]) -> str:
    """Build a lightweight CRS signature from whatever MintPy/GDAL metadata provides."""
    parts = [
        _get_meta_str(meta, ["EPSG"], ""),
        _get_meta_str(meta, ["UTM_ZONE"], ""),
        _get_meta_str(meta, ["PROJECTION"], ""),
        _get_meta_str(meta, ["GEOCOOR_REF"], ""),
        _get_meta_str(meta, ["X_UNIT"], ""),
        _get_meta_str(meta, ["Y_UNIT"], ""),
    ]
    return "|".join(parts)


def read_tif(path: Path) -> Tuple[np.ndarray, GridSpec]:
    """Read GeoTIFF via MintPy's readfile.read()."""
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
    return arr, spec


def assert_same_grid(ref: GridSpec, cur: GridSpec, path: Path) -> None:
    if ref.width != cur.width or ref.length != cur.length:
        raise ValueError(f"Grid size mismatch in {path}")

    if ref.crs_key != cur.crs_key:
        # keep this strict if MintPy metadata provides enough CRS hints
        # if both are empty, do not fail on CRS alone
        if ref.crs_key or cur.crs_key:
            raise ValueError(f"CRS mismatch in {path}")

    vals_ref = np.array([ref.x_first, ref.y_first, ref.x_step, ref.y_step], dtype=np.float64)
    vals_cur = np.array([cur.x_first, cur.y_first, cur.x_step, cur.y_step], dtype=np.float64)
    if not np.allclose(vals_ref, vals_cur, atol=1e-12):
        raise ValueError(f"Geo grid mismatch in {path}")


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

    We use abs(u) to reduce sign-convention ambiguity in LiCSAR LOS vectors.
    """
    inc = np.full_like(u, np.nan, dtype=np.float32)
    uu = np.clip(np.abs(u[valid]), 0.0, 1.0)
    inc[valid] = np.degrees(np.arccos(uu)).astype(np.float32)

    if np.any(valid):
        med = float(np.nanmedian(inc))
        if avg_incidence_angle is not None and abs(med - avg_incidence_angle) > 15.0:
            logging.warning(
                "Derived incidence median=%.2f differs strongly from metadata avg_incidence_angle=%.2f",
                med, avg_incidence_angle
            )
    return inc


def azimuth_from_en(e: np.ndarray, n: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    MintPy azimuthAngle:
        angle measured from North, anti-clockwise positive.

    Formula:
        az = atan2(-E, N)
    """
    az = np.full_like(e, np.nan, dtype=np.float32)
    az[valid] = np.degrees(np.arctan2(-e[valid], n[valid])).astype(np.float32)
    return az


def infer_landmask_to_watermask(landmask: np.ndarray) -> np.ndarray:
    """
    MintPy /waterMask:
        non-zero value for pixels on land.
    Your LiCSAR file is named landmask, so the likely mapping is:
        land != 0  -> True
    """
    unique_vals = np.unique(landmask[np.isfinite(landmask)])
    logging.info("landmask unique values (sample): %s", unique_vals[:20])
    return (landmask != 0)


def fill_nan_with_median(arr: np.ndarray, valid: np.ndarray, fallback_value: Optional[float] = None) -> np.ndarray:
    out = arr.astype(np.float32).copy()
    if np.any(valid):
        med = float(np.nanmedian(out[valid]))
    elif fallback_value is not None:
        med = float(fallback_value)
    else:
        med = 0.0

    out[~valid] = med
    out[~np.isfinite(out)] = med
    return out


def build_geometry(
    frame_dir: Path,
    meta_files: Dict[str, Path],
    meta_txt: Dict[str, object],
    force_avg_incidence: bool = False,
) -> Tuple[Dict[str, np.ndarray], GridSpec]:
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
            "ENU norm statistics on valid pixels: min=%.4f, median=%.4f, max=%.4f",
            float(np.nanmin(norm[valid_vec])),
            float(np.nanmedian(norm[valid_vec])),
            float(np.nanmax(norm[valid_vec]))
        )
        logging.info(
            "Valid ENU pixel ratio: %.3f",
            float(np.count_nonzero(valid_vec)) / float(valid_vec.size)
        )
    else:
        raise RuntimeError("No valid ENU pixels found in E/N/U geometry files.")

    avg_inc = float(meta_txt["avg_incidence_angle"]) if "avg_incidence_angle" in meta_txt else None

    if force_avg_incidence:
        if avg_inc is None:
            raise ValueError("--force-avg-incidence requested but avg_incidence_angle missing in metadata.txt")
        inc = np.full((hgt_spec.length, hgt_spec.width), avg_inc, dtype=np.float32)
    else:
        inc = incidence_from_u(u2, valid_vec, avg_incidence_angle=avg_inc)

    az = azimuth_from_en(e2, n2, valid_vec)
    lat, lon = build_lat_lon(hgt_spec)

    # Fill background NaN to avoid downstream MintPy issues in kept geometry pixels
    if avg_inc is None:
        avg_inc = 39.0

    inc = fill_nan_with_median(inc, np.isfinite(inc), fallback_value=avg_inc)
    az = fill_nan_with_median(az, np.isfinite(az), fallback_value=float(meta_txt.get("heading", 0.0)))
    hgt = fill_nan_with_median(hgt, np.isfinite(hgt), fallback_value=float(meta_txt.get("avg_height", 0.0)))

    # Keep auxiliary LOS vectors as float; fill NaN with 0 outside valid region
    los_east = np.nan_to_num(e2, nan=0.0).astype(np.float32)
    los_north = np.nan_to_num(n2, nan=0.0).astype(np.float32)
    los_up = np.nan_to_num(u2, nan=0.0).astype(np.float32)

    out = {
        "height": hgt.astype(np.float32),
        "incidenceAngle": inc.astype(np.float32),
        "azimuthAngle": az.astype(np.float32),
        "latitude": lat.astype(np.float32),
        "longitude": lon.astype(np.float32),
        "los_east": los_east,
        "los_north": los_north,
        "los_up": los_up,
    }

    if meta_files["landmask"] is not None:
        landmask_raw, lm_spec = read_tif(meta_files["landmask"])
        assert_same_grid(hgt_spec, lm_spec, meta_files["landmask"])
        landmask = clean_invalid(landmask_raw, lm_spec.nodata)
        out["waterMask"] = infer_landmask_to_watermask(landmask)

    return out, hgt_spec


# ============================================================
# Interferogram stack
# ============================================================

def scale_coherence(cc: np.ndarray, cc_max: float) -> np.ndarray:
    cc = cc.astype(np.float32)
    finite = np.isfinite(cc)
    if not np.any(finite):
        return np.zeros_like(cc, dtype=np.float32)

    vmax = float(np.nanmax(cc[finite]))
    if vmax <= 1.5:
        coh = cc
    else:
        coh = cc / float(cc_max)

    coh = np.clip(coh, 0.0, 1.0)
    coh[~np.isfinite(coh)] = 0.0
    return coh.astype(np.float32)


def calc_pair_bperp(date1: str, date2: str, bperp_by_date: Dict[str, float]) -> float:
    if date1 not in bperp_by_date:
        raise KeyError(f"{date1} not found in baselines file")
    if date2 not in bperp_by_date:
        raise KeyError(f"{date2} not found in baselines file")
    return float(bperp_by_date[date2] - bperp_by_date[date1])


def load_pair_arrays(
    pair: PairRecord,
    ref_grid: GridSpec,
    cc_max: float,
    min_valid_ratio: float,
    min_mean_coh: float,
) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, float]]:
    unw_raw, unw_spec = read_tif(pair.unw_file)
    cc_raw, cc_spec = read_tif(pair.cc_file)

    assert_same_grid(ref_grid, unw_spec, pair.unw_file)
    assert_same_grid(ref_grid, cc_spec, pair.cc_file)

    unw = clean_invalid(unw_raw, unw_spec.nodata)
    coh = scale_coherence(cc_raw, cc_max=cc_max)

    # LiCSAR often has unw=0 in invalid regions. Use coherence to build validity.
    valid = np.isfinite(unw) & np.isfinite(coh) & (coh > 0)

    # Avoid NaN inside kept interferograms, per MintPy plotting/network behavior.
    unw_out = np.zeros_like(unw, dtype=np.float32)
    coh_out = np.zeros_like(coh, dtype=np.float32)
    unw_out[valid] = unw[valid].astype(np.float32)
    coh_out[valid] = coh[valid].astype(np.float32)

    valid_ratio = float(np.count_nonzero(valid)) / float(unw.size)
    mean_coh = float(np.mean(coh_out[valid])) if np.any(valid) else 0.0
    keep = (valid_ratio >= min_valid_ratio) and (mean_coh >= min_mean_coh)

    qa = {
        "valid_ratio": valid_ratio,
        "mean_coh": mean_coh,
    }
    return unw_out, coh_out, keep, qa


def build_ifgram_stack(
    pairs: List[PairRecord],
    ref_grid: GridSpec,
    bperp_by_date: Dict[str, float],
    cc_max: float,
    min_valid_ratio: float,
    min_mean_coh: float,
) -> Dict[str, np.ndarray]:
    n_ifg = len(pairs)
    L, W = ref_grid.length, ref_grid.width

    unwrap_phase = np.zeros((n_ifg, L, W), dtype=np.float32)
    coherence = np.zeros((n_ifg, L, W), dtype=np.float32)
    date = np.empty((n_ifg, 2), dtype="S8")
    bperp = np.zeros((n_ifg,), dtype=np.float32)

    # MintPy convention: False=drop, True=keep
    drop_ifgram = np.zeros((n_ifg,), dtype=np.bool_)
    qa_rows = []

    for i, pair in enumerate(pairs):
        unw, coh, keep, qa = load_pair_arrays(
            pair=pair,
            ref_grid=ref_grid,
            cc_max=cc_max,
            min_valid_ratio=min_valid_ratio,
            min_mean_coh=min_mean_coh,
        )

        unwrap_phase[i] = unw
        coherence[i] = coh
        date[i, 0] = pair.date1.encode("ascii")
        date[i, 1] = pair.date2.encode("ascii")
        bperp[i] = calc_pair_bperp(pair.date1, pair.date2, bperp_by_date)
        drop_ifgram[i] = bool(keep)

        qa_rows.append({
            "pair": f"{pair.date1}_{pair.date2}",
            "date1": pair.date1,
            "date2": pair.date2,
            "bperp_m": float(bperp[i]),
            "valid_ratio": qa["valid_ratio"],
            "mean_coh": qa["mean_coh"],
            "keep": bool(keep),
        })

        logging.info(
            "Pair %-17s bperp=%8.2f m, valid_ratio=%.3f, mean_coh=%.3f, keep=%s",
            f"{pair.date1}_{pair.date2}",
            bperp[i],
            qa["valid_ratio"],
            qa["mean_coh"],
            keep,
        )

    if not np.any(drop_ifgram):
        logging.warning("All interferograms are marked as drop. Consider relaxing thresholds.")
    return {
        "unwrapPhase": unwrap_phase,
        "coherence": coherence,
        "date": date,
        "bperp": bperp,
        "dropIfgram": drop_ifgram,
        "qa_rows": qa_rows,
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
) -> Dict[str, object]:
    # infer orbit direction if user did not explicitly provide it
    orbit_dir = orbit_direction
    if orbit_dir == "UNKNOWN":
        heading = meta_txt.get("heading", None)
        if heading is not None:
            try:
                heading = float(heading)
                orbit_dir = "ASCENDING" if heading < 0 else "DESCENDING"
            except Exception:
                orbit_dir = "UNKNOWN"

    attrs = {
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
        "RLOOKS": int(DEFAULT_RLOOKS),
        "ALOOKS": int(DEFAULT_ALOOKS),
    }

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
    if "applied_DEM" in meta_txt:
        attrs["DEM_SOURCE"] = str(meta_txt["applied_DEM"])
    if "master" in meta_txt:
        attrs["LICSAR_MASTER_DATE"] = str(meta_txt["master"])

    return attrs


def set_h5_attrs(h5obj, attrs: Dict[str, object]) -> None:
    for k, v in attrs.items():
        h5obj.attrs[k] = v


def write_geometry_h5(
    out_file: Path,
    geom: Dict[str, np.ndarray],
    common_attrs: Dict[str, object],
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_file, "w") as f:
        attrs = dict(common_attrs)
        attrs["FILE_TYPE"] = "geometry"
        set_h5_attrs(f, attrs)

        f.create_dataset("height", data=geom["height"], dtype=np.float32, compression=LZF)
        f.create_dataset("latitude", data=geom["latitude"], dtype=np.float32, compression=LZF)
        f.create_dataset("longitude", data=geom["longitude"], dtype=np.float32, compression=LZF)
        f.create_dataset("incidenceAngle", data=geom["incidenceAngle"], dtype=np.float32, compression=LZF)
        f.create_dataset("azimuthAngle", data=geom["azimuthAngle"], dtype=np.float32, compression=LZF)

        if "waterMask" in geom:
            f.create_dataset(
                "waterMask",
                data=geom["waterMask"].astype(np.bool_),
                dtype=np.bool_,
                compression=LZF,
            )

        # Auxiliary LiCSAR-specific geometry layers
        f.create_dataset("los_east", data=geom["los_east"], dtype=np.float32, compression=LZF)
        f.create_dataset("los_north", data=geom["los_north"], dtype=np.float32, compression=LZF)
        f.create_dataset("los_up", data=geom["los_up"], dtype=np.float32, compression=LZF)


def write_ifgram_h5(
    out_file: Path,
    stack: Dict[str, np.ndarray],
    common_attrs: Dict[str, object],
    write_dummy_conncomp: bool = False,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_file, "w") as f:
        attrs = dict(common_attrs)
        attrs["FILE_TYPE"] = "ifgramStack"
        attrs["UNIT"] = "radian"
        set_h5_attrs(f, attrs)

        n_ifg, L, W = stack["unwrapPhase"].shape
        str8 = h5py.string_dtype(encoding="ascii", length=8)

        f.create_dataset("date", data=stack["date"], dtype=str8)
        f.create_dataset("bperp", data=stack["bperp"], dtype=np.float32)
        f.create_dataset("dropIfgram", data=stack["dropIfgram"], dtype=np.bool_)
        f.create_dataset(
            "unwrapPhase",
            data=stack["unwrapPhase"],
            dtype=np.float32,
            compression=LZF,
            chunks=(1, L, W),
        )
        f.create_dataset(
            "coherence",
            data=stack["coherence"],
            dtype=np.float32,
            compression=LZF,
            chunks=(1, L, W),
        )

        if write_dummy_conncomp:
            conn = np.where(stack["coherence"] > 0, 1, 0).astype(np.int16)
            f.create_dataset(
                "connectComponent",
                data=conn,
                dtype=np.int16,
                compression=LZF,
                chunks=(1, L, W),
            )


# ============================================================
# QA / config
# ============================================================

def write_pair_table(path: Path, qa_rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["pair", "date1", "date2", "bperp_m", "valid_ratio", "mean_coh", "keep"]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in qa_rows:
            vals = [str(r[h]) for h in header]
            f.write(",".join(vals) + "\n")


def write_summary_json(path: Path, summary: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def write_default_cfg(out_cfg: Path, write_dummy_conncomp: bool) -> None:
    out_cfg.parent.mkdir(parents=True, exist_ok=True)

    cfg = f"""# MintPy config for LiCSAR-prepared HDF5
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
# 2) connectComponent is {"dummy-only" if write_dummy_conncomp else "not provided"}.
# 3) Keep unwrap-error correction OFF for the first run.
"""
    out_cfg.write_text(cfg, encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main():
    args = build_parser().parse_args()
    setup_logging(args.verbose)

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

    # 4) ifgram stack
    stack = build_ifgram_stack(
        pairs=pairs,
        ref_grid=grid,
        bperp_by_date=baseline_info["bperp_by_date"],
        cc_max=args.cc_max,
        min_valid_ratio=args.min_valid_ratio,
        min_mean_coh=args.min_mean_coh,
    )

    # 5) common attrs
    common_attrs = common_attrs_from_grid(
        spec=grid,
        frame_dir=frame_dir,
        meta_txt=meta_txt,
        wavelength=args.wavelength,
        orbit_direction=args.orbit_direction,
    )

    # 6) write HDF5
    ifgram_h5 = inputs_dir / "ifgramStack.h5"
    geom_h5 = inputs_dir / "geometryGeo.h5"

    write_ifgram_h5(
        out_file=ifgram_h5,
        stack=stack,
        common_attrs=common_attrs,
        write_dummy_conncomp=args.write_dummy_conncomp,
    )
    write_geometry_h5(
        out_file=geom_h5,
        geom=geom,
        common_attrs=common_attrs,
    )

    # 7) QA outputs
    write_pair_table(qa_dir / "pair_table.csv", stack["qa_rows"])
    write_summary_json(
        qa_dir / "summary.json",
        {
            "frame_id": frame_dir.name,
            "n_ifg": len(pairs),
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
            "metadata_txt": meta_txt,
            "baseline_master_date": baseline_info["master_date"],
        },
    )

    # 8) MintPy config
    write_default_cfg(
        out_cfg=cfg_dir / "mintpy_licsar.cfg",
        write_dummy_conncomp=args.write_dummy_conncomp,
    )

    logging.info("Done.")
    logging.info("Generated:")
    logging.info("  %s", ifgram_h5)
    logging.info("  %s", geom_h5)
    logging.info("  %s", cfg_dir / "mintpy_licsar.cfg")
    logging.info("Next: inspect HDF5 with info.py / view.py, then run smallbaselineApp.py.")


if __name__ == "__main__":
    main()