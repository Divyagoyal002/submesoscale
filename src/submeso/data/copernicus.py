"""Copernicus Marine Service retrieval (FR-1, FR-2).

Authentication: run ``copernicusmarine login`` once, or set the environment
variables COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD.
Requests are split by calendar year so they stay small and resumable: files that
already exist are skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from submeso.config import Config, ProductSpec
from submeso.data.grid import make_target_grid, regrid_dataarray, standardize_coords

log = logging.getLogger(__name__)


def _year_chunks(start: str, end: str) -> list[tuple[str, str]]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    chunks = []
    for y in range(s.year, e.year + 1):
        a = max(s, pd.Timestamp(f"{y}-01-01"))
        b = min(e, pd.Timestamp(f"{y}-12-31"))
        chunks.append((a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")))
    return chunks


def download_product(
    cfg: Config, key: str, start: str, end: str, pad_deg: float = 0.5
) -> list[Path]:
    """Download ``cfg.data.<key>`` for [start, end]; returns the yearly NetCDF files."""
    import copernicusmarine  # optional dependency: pip install -e ".[data]"

    spec: ProductSpec = getattr(cfg.data, key)
    if not spec.dataset_id:
        raise ValueError(f"data.{key}.dataset_id is not set in the config")
    out_dir = cfg.data_root / "raw" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    r = cfg.region
    files = []
    for a, b in _year_chunks(start, end):
        path = out_dir / f"{key}_{a}_{b}.nc"
        files.append(path)
        if path.exists():
            log.info("exists, skipping: %s", path)
            continue
        log.info("Downloading %s %s..%s", spec.dataset_id, a, b)
        kwargs = {}
        if spec.depth is not None:
            kwargs.update(minimum_depth=0.0, maximum_depth=spec.depth)
        copernicusmarine.subset(
            dataset_id=spec.dataset_id,
            variables=[spec.variable],
            minimum_longitude=r.lon_min - pad_deg,
            maximum_longitude=r.lon_max + pad_deg,
            minimum_latitude=r.lat_min - pad_deg,
            maximum_latitude=r.lat_max + pad_deg,
            start_datetime=f"{a}T00:00:00",
            end_datetime=f"{b}T23:59:59",
            output_directory=str(out_dir),
            output_filename=path.name,
            **kwargs,
        )
    return files


def download_all(cfg: Config) -> None:
    """Model truth for train/val/test, satellite L4 for test + reconstruction period."""
    p = cfg.period
    download_product(cfg, "hr_adt", p.train[0], p.test[1])
    download_product(cfg, "hr_sst", p.train[0], p.test[1])
    l4_start = min(p.test[0], p.reconstruct[0])
    l4_end = max(p.test[1], p.reconstruct[1])
    download_product(cfg, "adt_l4", l4_start, l4_end)
    download_product(cfg, "sst_l4", l4_start, l4_end)


def open_product(cfg: Config, key: str) -> xr.DataArray:
    """Open the downloaded yearly files of one product as a (time, lat, lon) DataArray."""
    spec: ProductSpec = getattr(cfg.data, key)
    files = sorted((cfg.data_root / "raw" / key).glob(f"{key}_*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No files for '{key}' in {cfg.data_root / 'raw' / key}; run `submeso download`"
        )
    da = xr.open_mfdataset(files, combine="by_coords")[spec.variable]
    if "depth" in da.dims:
        da = da.isel(depth=0)
    return standardize_coords(da)


def load_hr_truth(cfg: Config) -> xr.Dataset:
    """High-resolution model 'truth' regridded to the target grid.

    Model sea-surface height (``zos``) differs from satellite ADT only by a
    constant reference offset, which is irrelevant here because the per-patch mean
    is removed during normalisation.
    """
    r = cfg.region
    lat, lon = make_target_grid(
        r.lon_min, r.lon_max, r.lat_min, r.lat_max, cfg.osse.target_resolution_deg
    )
    period = slice(cfg.period.train[0], cfg.period.test[1])
    fields = {
        "adt": open_product(cfg, "hr_adt").sel(time=period),
        "sst": open_product(cfg, "hr_sst").sel(time=period),
    }
    times = np.intersect1d(fields["adt"].time.values, fields["sst"].time.values)
    first = fields["adt"].sel(time=times[0]).values
    # nearest-neighbour regrid of the raw land mask; values themselves use linear interpolation
    land = _nearest_mask(
        ~np.isfinite(first), fields["adt"].lat.values, fields["adt"].lon.values, lat, lon
    )
    out = {}
    for var, da in fields.items():
        log.info("Regridding model %s to the target grid", var)
        arr = regrid_dataarray(da.sel(time=times).load(), lat, lon)
        if var == "sst" and np.nanmean(arr) > 200:
            arr = arr - 273.15
        arr[:, land] = np.nan
        out[var] = (("time", "lat", "lon"), arr.astype(np.float32))
    return xr.Dataset(out, coords={"time": times, "lat": lat, "lon": lon})


def _nearest_mask(src: np.ndarray, src_lat, src_lon, lat, lon) -> np.ndarray:
    iy = np.abs(src_lat[:, None] - lat[None, :]).argmin(axis=0)
    ix = np.abs(src_lon[:, None] - lon[None, :]).argmin(axis=0)
    return src[np.ix_(iy, ix)]
