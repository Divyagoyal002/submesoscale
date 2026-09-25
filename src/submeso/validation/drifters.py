"""Validation against in-situ drifting buoys (FR-6).

Drifters measure the *total* near-surface current (geostrophic + Ekman + inertial
+ tidal + wind slip), so observations are first:

1. restricted to drogued periods (drogue still attached => ~15 m currents, little
   direct wind slip);
2. averaged over a window (default 24 h) per drifter to suppress inertial and
   tidal oscillations.

Absolute errors therefore include an irreducible ageostrophic part; the
meaningful number is the *relative* change between the L4 and super-resolved
currents at the same points.
"""

from __future__ import annotations

import io
import logging
import urllib.parse

import numpy as np
import pandas as pd
import requests
import xarray as xr

log = logging.getLogger(__name__)

ERDDAP_VARS = ["ID", "time", "latitude", "longitude", "ve", "vn", "drogue_lost_date"]


def download_gdp(
    erddap_url: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download Global Drifter Program data from NOAA AOML ERDDAP as a DataFrame."""
    constraints = (
        f"{','.join(ERDDAP_VARS)}"
        f"&time>={start}T00:00:00Z&time<={end}T23:59:59Z"
        f"&latitude>={lat_min}&latitude<={lat_max}&longitude>={lon_min}&longitude<={lon_max}"
    )
    query = f"{erddap_url}.csv?{urllib.parse.quote(constraints, safe='&=,')}"
    log.info("Requesting drifters: %s", query)
    r = requests.get(query, timeout=600)
    if r.status_code == 404 and "nRows = 0" in r.text:
        return pd.DataFrame(columns=["id", "time", "lat", "lon", "u", "v"])
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), skiprows=[1])  # row 1 holds units
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    lost = pd.to_datetime(df["drogue_lost_date"], utc=True, errors="coerce").dt.tz_localize(None)
    drogued = lost.isna() | (df["time"] < lost)
    log.info("Kept %d / %d drogued observations", int(drogued.sum()), len(df))
    df = df[drogued].rename(
        columns={"ID": "id", "latitude": "lat", "longitude": "lon", "ve": "u", "vn": "v"}
    )
    df = df[["id", "time", "lat", "lon", "u", "v"]].dropna()
    return df[(df.u.abs() < 3) & (df.v.abs() < 3)].reset_index(drop=True)


def average_drifters(df: pd.DataFrame, window_hours: int = 24, min_obs: int = 3) -> pd.DataFrame:
    """Per-drifter block averages; windows with fewer than ``min_obs`` fixes are dropped."""
    if df.empty:
        return df
    d = df.copy()
    d["bin"] = d["time"].dt.floor(f"{window_hours}h")
    g = d.groupby(["id", "bin"])
    out = g.agg(
        lat=("lat", "mean"), lon=("lon", "mean"), u=("u", "mean"), v=("v", "mean"), n=("u", "size")
    )
    out = out.reset_index().rename(columns={"bin": "time"})
    out["time"] = out["time"] + pd.Timedelta(hours=window_hours / 2)
    return out[out["n"] >= min_obs].reset_index(drop=True)


def _days(t) -> np.ndarray:
    """Datetimes -> float days since 1970, independent of the datetime64 unit (ns/us/s)."""
    return (np.asarray(t).astype("datetime64[s]") - np.datetime64(0, "s")) / np.timedelta64(1, "D")


def sample_field(ds: xr.Dataset, var: str, obs: pd.DataFrame) -> np.ndarray:
    """Linearly interpolate ``ds[var]`` (time, lat, lon) at drifter positions.

    Time is interpolated on a numeric axis: mixing datetime64 units (pandas 3 uses
    microseconds by default) otherwise makes xarray return NaN silently.
    """
    da = ds[var].assign_coords(time=_days(ds.time.values))
    pts = {
        "time": xr.DataArray(_days(obs["time"].values), dims="obs"),
        "lat": xr.DataArray(obs["lat"].values, dims="obs"),
        "lon": xr.DataArray(obs["lon"].values, dims="obs"),
    }
    return da.interp(pts, method="linear").values


def vector_correlation(u1, v1, u2, v2) -> tuple[float, float]:
    """Complex correlation (Kundu 1976): magnitude and mean veering angle [deg]."""
    w1 = u1 + 1j * v1
    w2 = u2 + 1j * v2
    w1 = w1 - w1.mean()
    w2 = w2 - w2.mean()
    rho = (np.conj(w1) * w2).mean() / np.sqrt((np.abs(w1) ** 2).mean() * (np.abs(w2) ** 2).mean())
    return float(np.abs(rho)), float(np.degrees(np.angle(rho)))


def compare_with_drifters(
    recon: xr.Dataset, obs: pd.DataFrame, tags: tuple[str, ...] = ("lr", "sr")
) -> tuple[dict, pd.DataFrame]:
    """Metrics of each reconstructed current field against drifter velocities.

    Returns (metrics dict, matched DataFrame with model velocities per tag).
    """
    t0, t1 = recon.time.values[0], recon.time.values[-1]
    obs = obs[(obs.time >= t0) & (obs.time <= t1)].copy()
    for tag in tags:
        obs[f"u_{tag}"] = sample_field(recon, f"u_{tag}", obs)
        obs[f"v_{tag}"] = sample_field(recon, f"v_{tag}", obs)
    cols = [f"{c}_{t}" for t in tags for c in ("u", "v")]
    obs = obs.dropna(subset=cols).reset_index(drop=True)
    metrics: dict = {"n_obs": len(obs), "n_drifters": int(obs["id"].nunique()) if len(obs) else 0}
    if len(obs) < 3:
        log.warning("Only %d collocated drifter observations; metrics skipped", len(obs))
        return metrics, obs
    for tag in tags:
        du = obs[f"u_{tag}"] - obs.u
        dv = obs[f"v_{tag}"] - obs.v
        rho, angle = vector_correlation(
            obs.u.values, obs.v.values, obs[f"u_{tag}"].values, obs[f"v_{tag}"].values
        )
        ang_err = np.degrees(
            np.angle(
                np.exp(
                    1j * (np.arctan2(obs[f"v_{tag}"], obs[f"u_{tag}"]) - np.arctan2(obs.v, obs.u))
                )
            )
        )
        metrics[tag] = {
            "rmse_u_cms": float(np.sqrt((du**2).mean()) * 100),
            "rmse_v_cms": float(np.sqrt((dv**2).mean()) * 100),
            "rmse_vector_cms": float(np.sqrt((du**2 + dv**2).mean()) * 100),
            "bias_u_cms": float(du.mean() * 100),
            "bias_v_cms": float(dv.mean() * 100),
            "corr_u": float(np.corrcoef(obs.u, obs[f"u_{tag}"])[0, 1]),
            "corr_v": float(np.corrcoef(obs.v, obs[f"v_{tag}"])[0, 1]),
            "vector_corr": rho,
            "veering_deg": angle,
            "median_abs_direction_error_deg": float(np.median(np.abs(ang_err))),
        }
    if "lr" in tags and "sr" in tags:
        metrics["improvement_pct"] = {
            k: 100 * (1 - metrics["sr"][k] / metrics["lr"][k])
            for k in ("rmse_u_cms", "rmse_v_cms", "rmse_vector_cms")
        }
    return metrics, obs
