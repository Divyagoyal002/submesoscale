"""OSSE test-set metrics: where the true high-resolution field is known."""

from __future__ import annotations

import numpy as np
import xarray as xr

from submeso.physics import coriolis, geostrophic_velocity, relative_vorticity
from submeso.validation.spectra import (
    band_energy_ratio,
    effective_resolution,
    grid_km,
    isotropic_spectrum,
    largest_ocean_square,
    spectral_slope,
)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.nanmean(d**2)))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def osse_metrics(
    truth: np.ndarray,
    fields: dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Compare each estimate in ``fields`` (name -> (T, ny, nx) ADT) against ``truth``.

    Reports ADT / velocity / speed RMSE, Rossby-number correlation, spectral slopes
    (10-100 km) and effective resolution over the largest land-free square.
    """
    f = coriolis(lat)[:, None]
    ut, vt = geostrophic_velocity(truth, lat, lon)
    rot = relative_vorticity(ut, vt, lat, lon) / f
    box = largest_ocean_square(mask)
    dy, dx = grid_km(lat, lon)
    t_box = truth[(slice(None), *box)]
    k, e_true = isotropic_spectrum(t_box, dy, dx)
    out: dict = {
        "truth": {
            "ssh_slope_10_100km": spectral_slope(k, e_true, 10, 100),
            "speed_rms_cms": float(np.sqrt(np.nanmean(ut**2 + vt**2)) * 100),
        },
        "box_size_px": int(box[0].stop - box[0].start),
    }
    for name, est in fields.items():
        u, v = geostrophic_velocity(est, lat, lon)
        ro = relative_vorticity(u, v, lat, lon) / f
        e_box = est[(slice(None), *box)]
        _, e_est = isotropic_spectrum(e_box, dy, dx)
        eff_res, _, _ = effective_resolution(t_box, e_box, dy, dx)
        out[name] = {
            "rmse_adt_cm": _rmse(est, truth) * 100,
            "rmse_u_cms": _rmse(u, ut) * 100,
            "rmse_v_cms": _rmse(v, vt) * 100,
            "rmse_speed_cms": _rmse(np.hypot(u, v), np.hypot(ut, vt)) * 100,
            "corr_speed": _corr(np.hypot(u, v), np.hypot(ut, vt)),
            "corr_rossby": _corr(ro, rot),
            "ssh_slope_10_100km": spectral_slope(k, e_est, 10, 100),
            "energy_ratio_10_50km": band_energy_ratio(k, e_est, e_true, 10, 50),
            "effective_resolution_km": eff_res,
        }
    return out


def evaluate_osse_test(recon: xr.Dataset, truth: xr.DataArray) -> dict:
    """Metrics for the super-resolved and LR fields of a reconstruction dataset."""
    lat, lon = recon.lat.values, recon.lon.values
    mask = np.all(np.isfinite(truth.values), axis=0)
    fields = {"lr": recon.adt_lr.values, "sr": recon.adt_sr.values}
    m = osse_metrics(truth.values, fields, lat, lon, mask)
    m["improvement_pct"] = {
        key: 100 * (1 - m["sr"][key] / m["lr"][key])
        for key in (
            "rmse_adt_cm",
            "rmse_u_cms",
            "rmse_v_cms",
            "rmse_speed_cms",
            "effective_resolution_km",
        )
    }
    return m
