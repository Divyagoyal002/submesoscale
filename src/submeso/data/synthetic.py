"""Synthetic SQG-like 'nature run' for demos, CI and method development.

Real OSSE training needs a high-resolution ocean model (e.g. a Copernicus
reanalysis). To make the full pipeline runnable without credentials, this module
generates a physically-motivated stand-in:

* SSH anomalies with a power-law wavenumber spectrum (k^-11/3 by default, the
  surface-quasi-geostrophic regime typical of the submesoscale-rich upper ocean),
  evolving in time through scale-dependent random phase drift.
* A cyclonic basin-scale 'rim current' in a semi-enclosed basin (Black Sea-like).
* SST following SQG theory: surface buoyancy b_hat = N |k| psi_hat, so SST
  anomalies are the SSH field 'whitened' by |k|. This is exactly why SST carries
  small-scale information that helps super-resolve ADT.
* Lagrangian drifters advected by the true geostrophic currents, for exercising the
  in-situ validation code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

from submeso.config import Config
from submeso.data.grid import make_target_grid
from submeso.physics import R_EARTH, geostrophic_velocity


def _basin_mask(lat: np.ndarray, lon: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Superellipse basin with a wiggly coastline and one island. True = ocean."""
    yy, xx = np.meshgrid(np.linspace(-1, 1, len(lat)), np.linspace(-1, 1, len(lon)), indexing="ij")
    wiggle = ndimage.gaussian_filter(rng.standard_normal(yy.shape), sigma=len(lat) / 10)
    wiggle = 0.12 * wiggle / (np.abs(wiggle).max() + 1e-12)
    ocean = (np.abs(xx / 0.95) ** 4 + np.abs(yy / 0.9) ** 4) < 1.0 + wiggle
    island = ((xx - 0.35) / 0.08) ** 2 + ((yy + 0.2) / 0.12) ** 2 < 1.0
    return ocean & ~island


def generate_nature_run(cfg: Config) -> xr.Dataset:
    """Generate a daily (time, lat, lon) dataset with ``adt`` [m] and ``sst`` [degC]."""
    spec = cfg.data.synthetic
    rng = np.random.default_rng(cfg.seed)
    r = cfg.region
    lat, lon = make_target_grid(
        r.lon_min, r.lon_max, r.lat_min, r.lat_max, cfg.osse.target_resolution_deg
    )
    ny, nx = len(lat), len(lon)
    t0 = pd.Timestamp(cfg.period.train[0])
    t1 = pd.Timestamp(cfg.period.reconstruct[1])
    times = pd.date_range(t0, t1, freq="D")

    # Wavenumbers in cycles/km using the mid-latitude grid spacing.
    res = np.deg2rad(cfg.osse.target_resolution_deg)
    dy_km = R_EARTH * res / 1e3
    dx_km = R_EARTH * np.cos(np.deg2rad(lat.mean())) * res / 1e3
    ky = np.fft.fftfreq(ny, d=dy_km)[:, None]
    kx = np.fft.rfftfreq(nx, d=dx_km)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k0 = 1.0 / 250.0  # roll-off at basin scale (250 km)
    amp = (k**2 + k0**2) ** (spec.spectral_slope / 4.0)
    amp *= np.exp(-((k * spec.dissipation_km) ** 2))  # viscous roll-off, like a real model
    amp[0, 0] = 0.0

    phase0 = rng.uniform(0, 2 * np.pi, k.shape)
    # Small scales decorrelate faster: omega ~ k^(2/3) (turbulent eddy turnover).
    kref = 1.0 / 50.0
    omega = (2 * np.pi / spec.decorrelation_days) * (np.maximum(k, k0) / kref) ** (2 / 3)
    omega *= rng.normal(1.0, 0.3, k.shape) * rng.choice([-1, 1], k.shape)

    ocean = _basin_mask(lat, lon, rng) if spec.basin else np.ones((ny, nx), bool)

    # Mean dynamic topography: cyclonic rim current (low sea level in the basin centre).
    yy, xx = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), indexing="ij")
    mdt = -0.15 * np.exp(-((xx / 0.6) ** 2 + (yy / 0.55) ** 2))

    lat2d = np.broadcast_to(lat[:, None], (ny, nx))
    adt = np.empty((len(times), ny, nx), np.float32)
    sst = np.empty_like(adt)
    for i, t in enumerate(times):
        coeff = amp * np.exp(1j * (phase0 + omega * i))
        eta = np.fft.irfft2(coeff, s=(ny, nx))
        theta = np.fft.irfft2(coeff * k / kref, s=(ny, nx))  # SQG: b_hat ~ |k| psi_hat
        if i == 0:
            eta_scale = spec.ssh_rms_m / eta.std()
            theta_scale = 0.6 * spec.sst_sqg_coupling / theta.std()
        doy = t.dayofyear
        seasonal = 16.0 + 8.0 * np.sin(2 * np.pi * (doy - 110) / 365.25)
        steric = 0.05 * np.sin(2 * np.pi * (doy - 150) / 365.25)
        adt[i] = mdt + steric + eta_scale * eta
        sst[i] = (
            seasonal
            - 0.6 * (lat2d - lat.mean())
            + theta_scale * theta
            + spec.sst_noise_k * rng.standard_normal((ny, nx))
        )

    adt[:, ~ocean] = np.nan
    sst[:, ~ocean] = np.nan
    ds = xr.Dataset(
        {
            "adt": (
                ("time", "lat", "lon"),
                adt,
                {"units": "m", "long_name": "absolute dynamic topography"},
            ),
            "sst": (
                ("time", "lat", "lon"),
                sst,
                {"units": "degC", "long_name": "sea surface temperature"},
            ),
        },
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={"source": "submeso synthetic SQG nature run", "seed": cfg.seed},
    )
    return ds


def simulate_drifters(
    truth: xr.Dataset, n_drifters: int = 60, seed: int = 0, noise_ms: float = 0.03
) -> pd.DataFrame:
    """Advect drifters (RK2, 6-hourly) in the true geostrophic field.

    Returns a DataFrame with the same columns as the GDP loader:
    ``id, time, lat, lon, u, v`` (u/v = observed velocity incl. ageostrophic noise).
    """
    rng = np.random.default_rng(seed)
    lat, lon = truth.lat.values, truth.lon.values
    u, v = geostrophic_velocity(truth.adt.values, lat, lon)
    u, v = np.nan_to_num(u), np.nan_to_num(v)
    ocean = np.isfinite(truth.adt.values[0])
    t_days = (truth.time.values - truth.time.values[0]) / np.timedelta64(1, "D")

    iu = RegularGridInterpolator((t_days, lat, lon), u, bounds_error=False, fill_value=0.0)
    iv = RegularGridInterpolator((t_days, lat, lon), v, bounds_error=False, fill_value=0.0)
    iocean = RegularGridInterpolator(
        (lat, lon), ocean.astype(float), bounds_error=False, fill_value=0.0
    )

    oy, ox = np.nonzero(ndimage.binary_erosion(ocean, iterations=4))
    pick = rng.choice(len(oy), n_drifters, replace=False)
    pos = np.stack([lat[oy[pick]], lon[ox[pick]]], axis=-1)
    alive = np.ones(n_drifters, bool)
    dt_days = 0.25
    m_per_deg = R_EARTH * np.pi / 180.0

    def vel(t, p):
        q = np.column_stack([np.full(len(p), t), p])
        return iu(q), iv(q)

    rows = []
    for t in np.arange(0, t_days[-1], dt_days):
        ut, vt = vel(t, pos)
        obs_u = ut + noise_ms * rng.standard_normal(n_drifters)
        obs_v = vt + noise_ms * rng.standard_normal(n_drifters)
        stamp = truth.time.values[0] + np.timedelta64(int(t * 86400), "s")
        for d in np.nonzero(alive)[0]:
            rows.append((d, stamp, pos[d, 0], pos[d, 1], obs_u[d], obs_v[d]))
        # midpoint (RK2) step
        dlat = vt * dt_days * 86400 / m_per_deg
        dlon = ut * dt_days * 86400 / (m_per_deg * np.cos(np.deg2rad(pos[:, 0])))
        mid = pos + 0.5 * np.stack([dlat, dlon], -1)
        um, vm = vel(t + dt_days / 2, mid)
        dlat = vm * dt_days * 86400 / m_per_deg
        dlon = um * dt_days * 86400 / (m_per_deg * np.cos(np.deg2rad(mid[:, 0])))
        pos = pos + np.stack([dlat, dlon], -1)
        alive &= iocean(pos) > 0.99  # drifters that beach are removed
    return pd.DataFrame(rows, columns=["id", "time", "lat", "lon", "u", "v"])
