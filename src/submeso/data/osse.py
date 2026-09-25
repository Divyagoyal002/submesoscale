"""Observing System Simulation Experiment (OSSE): build paired LR/HR training data.

A high-resolution 'truth' (ocean-model output or the synthetic nature run) is
degraded to look like a gridded satellite L4 product:

    ADT: Gaussian smoothing (mimics optimal-interpolation mapping)
         -> sample onto the coarse L4 product grid (e.g. 1/16 deg)
         -> additive measurement/mapping noise
         -> interpolate back onto the target grid (same path as real L4 data)
    SST: light smoothing + noise on the target grid (L4 SST is already fine-scale)

The resulting pairs are stored in one NetCDF file with a time dimension so the
train/val/test split can be done by date.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from submeso.config import Config
from submeso.data.grid import (
    km_to_grid_sigma,
    make_target_grid,
    nan_gaussian_filter,
    regrid_2d,
    regrid_dataarray,
    standardize_coords,
)

log = logging.getLogger(__name__)


def degrade_adt_to_coarse(
    adt: np.ndarray, lat: np.ndarray, lon: np.ndarray, cfg: Config, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HR ADT (T, ny, nx) -> synthetic L4 ADT on the coarse product grid.

    The field is smoothed first, so sampling it onto the coarse grid does not alias.
    """
    o, r = cfg.osse, cfg.region
    lat_c, lon_c = make_target_grid(r.lon_min, r.lon_max, r.lat_min, r.lat_max, o.lr_resolution_deg)
    smooth = nan_gaussian_filter(adt, km_to_grid_sigma(o.adt_smoothing_km, lat, lon))
    ocean = np.isfinite(adt[0])
    coarse = np.stack([regrid_2d(s, lat, lon, lat_c, lon_c) for s in smooth])
    coarse_ocean = regrid_2d(ocean.astype(float), lat, lon, lat_c, lon_c) > 0.5
    coarse = coarse + o.adt_noise_m * rng.standard_normal(coarse.shape)
    coarse[:, ~coarse_ocean] = np.nan
    return coarse, lat_c, lon_c


def degrade_sst(
    sst: np.ndarray, lat: np.ndarray, lon: np.ndarray, cfg: Config, rng: np.random.Generator
) -> np.ndarray:
    o = cfg.osse
    out = nan_gaussian_filter(sst, km_to_grid_sigma(o.sst_smoothing_km, lat, lon))
    return out + o.sst_noise_k * rng.standard_normal(out.shape)


def coarse_to_target(
    coarse: np.ndarray,
    lat_c: np.ndarray,
    lon_c: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Cubic interpolation of a (T, ny_c, nx_c) field to the target grid, then mask.

    Cubic (not bilinear) so that derived vorticity has no grid-cell artefacts; the
    same function is used for real L4 ADT in :func:`prepare_real_inputs`.
    """
    out = np.stack([regrid_2d(c, lat_c, lon_c, lat, lon, method="cubic") for c in coarse])
    out[:, ~mask] = np.nan
    return out.astype(np.float32)


def build_osse_pairs(cfg: Config, truth: xr.Dataset) -> xr.Dataset:
    """Create the paired dataset (adt_lr, sst_in -> adt_hr) on the target grid."""
    rng = np.random.default_rng(cfg.seed + 1)
    truth = standardize_coords(truth)
    lat, lon = truth.lat.values, truth.lon.values
    adt_hr = truth.adt.values.astype(np.float64)
    mask = np.all(np.isfinite(adt_hr), axis=0)
    log.info("Degrading ADT (%d snapshots on %dx%d grid)", *adt_hr.shape)
    coarse, lat_c, lon_c = degrade_adt_to_coarse(adt_hr, lat, lon, cfg, rng)
    adt_lr = coarse_to_target(coarse, lat_c, lon_c, lat, lon, mask)
    sst_in = degrade_sst(truth.sst.values.astype(np.float64), lat, lon, cfg, rng).astype(np.float32)
    sst_in[:, ~mask] = np.nan
    adt_hr = adt_hr.astype(np.float32)
    adt_hr[:, ~mask] = np.nan
    dims = ("time", "lat", "lon")
    return xr.Dataset(
        {
            "adt_hr": (dims, adt_hr, {"units": "m"}),
            "adt_lr": (dims, adt_lr, {"units": "m"}),
            "sst_in": (dims, sst_in, {"units": "degC"}),
            "mask": (("lat", "lon"), mask.astype(np.uint8)),
        },
        coords={"time": truth.time.values, "lat": lat, "lon": lon},
        attrs={
            "description": "OSSE pairs: degraded (adt_lr, sst_in) -> truth adt_hr",
            "lr_resolution_deg": cfg.osse.lr_resolution_deg,
            "adt_smoothing_km": cfg.osse.adt_smoothing_km,
            "adt_noise_m": cfg.osse.adt_noise_m,
        },
    )


def prepare_real_inputs(
    cfg: Config,
    adt_l4: xr.DataArray,
    sst_l4: xr.DataArray,
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray,
) -> xr.Dataset:
    """Regrid real L4 ADT/SST products onto the target grid (same path as the OSSE)."""
    adt_l4 = standardize_coords(adt_l4)
    sst_l4 = standardize_coords(sst_l4)
    common = np.intersect1d(
        adt_l4.time.values.astype("datetime64[D]"), sst_l4.time.values.astype("datetime64[D]")
    )
    if len(common) == 0:
        raise ValueError("ADT and SST products share no dates")
    adt_l4 = adt_l4.assign_coords(time=adt_l4.time.values.astype("datetime64[D]")).sel(time=common)
    sst_l4 = sst_l4.assign_coords(time=sst_l4.time.values.astype("datetime64[D]")).sel(time=common)
    adt = regrid_dataarray(adt_l4, lat, lon, method="cubic").astype(np.float32)
    sst = regrid_dataarray(sst_l4, lat, lon).astype(np.float32)
    if np.nanmean(sst) > 200:  # Kelvin -> Celsius
        sst -= 273.15
    adt[:, ~mask] = np.nan
    sst[:, ~mask] = np.nan
    dims = ("time", "lat", "lon")
    return xr.Dataset(
        {
            "adt_lr": (dims, adt),
            "sst_in": (dims, sst),
            "mask": (("lat", "lon"), mask.astype(np.uint8)),
        },
        coords={"time": common.astype("datetime64[ns]"), "lat": lat, "lon": lon},
    )
