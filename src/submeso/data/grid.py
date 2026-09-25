"""Grid utilities: target grids, NaN-aware smoothing, coarsening and regridding.

The same regridding path is used for (a) OSSE low-resolution inputs and (b) real
satellite L4 products, so that the network sees statistically identical inputs at
training and inference time.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

from submeso.physics import R_EARTH


def make_target_grid(
    lon_min: float, lon_max: float, lat_min: float, lat_max: float, resolution_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Regular, ascending lat/lon cell-centre coordinates covering the box."""
    lat = np.arange(lat_min + resolution_deg / 2, lat_max, resolution_deg)
    lon = np.arange(lon_min + resolution_deg / 2, lon_max, resolution_deg)
    return lat.round(6), lon.round(6)


def km_to_grid_sigma(sigma_km: float, lat: np.ndarray, lon: np.ndarray) -> tuple[float, float]:
    """Convert an isotropic Gaussian sigma in km to (sigma_y, sigma_x) in grid points."""
    dlat = np.deg2rad(abs(np.mean(np.diff(lat))))
    dlon = np.deg2rad(abs(np.mean(np.diff(lon))))
    mid_lat = np.deg2rad(np.mean(lat))
    dy_km = R_EARTH * dlat / 1e3
    dx_km = R_EARTH * np.cos(mid_lat) * dlon / 1e3
    return sigma_km / dy_km, sigma_km / dx_km


def nan_gaussian_filter(field: np.ndarray, sigma: tuple[float, float]) -> np.ndarray:
    """Gaussian filter that ignores NaNs (normalised convolution) on the last two axes."""
    if sigma[0] <= 0 and sigma[1] <= 0:
        return field.copy()
    full_sigma = (0,) * (field.ndim - 2) + tuple(sigma)
    valid = np.isfinite(field)
    filled = np.where(valid, field, 0.0)
    num = ndimage.gaussian_filter(filled, full_sigma, mode="nearest")
    den = ndimage.gaussian_filter(valid.astype(np.float64), full_sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(valid, out, np.nan)


def fill_nearest(field: np.ndarray) -> np.ndarray:
    """Fill NaNs in a 2-D field with the nearest valid value (extrapolation over land)."""
    invalid = ~np.isfinite(field)
    if not invalid.any():
        return field
    if invalid.all():
        return np.zeros_like(field)
    idx = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return field[tuple(idx)]


def regrid_2d(
    field: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate a 2-D field to a new regular grid.

    Land/NaN points are first filled by nearest-neighbour so that coastal target
    points get a value; callers re-apply the target land mask afterwards.
    """
    src_lat = np.asarray(src_lat)
    src_lon = np.asarray(src_lon)
    f = fill_nearest(np.asarray(field, dtype=np.float64))
    if src_lat[0] > src_lat[-1]:
        src_lat, f = src_lat[::-1], f[::-1]
    if src_lon[0] > src_lon[-1]:
        src_lon, f = src_lon[::-1], f[:, ::-1]
    interp = RegularGridInterpolator(
        (src_lat, src_lon), f, method=method, bounds_error=False, fill_value=None
    )
    yy, xx = np.meshgrid(dst_lat, dst_lon, indexing="ij")
    return interp(np.stack([yy.ravel(), xx.ravel()], axis=-1)).reshape(yy.shape)


def regrid_dataarray(
    da: xr.DataArray, dst_lat: np.ndarray, dst_lon: np.ndarray, method: str = "linear"
) -> np.ndarray:
    """Regrid a (time, lat, lon) or (lat, lon) DataArray; returns a NumPy array."""
    lat_name = _find_coord(da, ("latitude", "lat", "nav_lat"))
    lon_name = _find_coord(da, ("longitude", "lon", "nav_lon"))
    src_lat, src_lon = da[lat_name].values, da[lon_name].values
    arr = da.transpose(..., lat_name, lon_name).values
    if arr.ndim == 2:
        return regrid_2d(arr, src_lat, src_lon, dst_lat, dst_lon, method)
    return np.stack([regrid_2d(a, src_lat, src_lon, dst_lat, dst_lon, method) for a in arr])


def standardize_coords(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Rename common coordinate aliases to (time, lat, lon) and sort ascending."""
    rename = {}
    for alias, std in (
        ("latitude", "lat"),
        ("longitude", "lon"),
        ("nav_lat", "lat"),
        ("nav_lon", "lon"),
    ):
        if alias in ds.dims or alias in ds.coords:
            rename[alias] = std
    ds = ds.rename(rename)
    for c in ("lat", "lon"):
        if c in ds.dims:
            ds = ds.sortby(c)
    return ds


def _find_coord(da: xr.DataArray, names: tuple[str, ...]) -> str:
    for n in names:
        if n in da.dims:
            return n
    raise KeyError(f"None of {names} found in dims {da.dims}")
