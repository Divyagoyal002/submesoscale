"""Geophysical fluid dynamics helpers: Coriolis, geostrophic currents, vorticity.

All functions work on regular latitude/longitude grids. NumPy versions are used for
analysis; the ``*_torch`` versions are differentiable and are used inside the
physics-informed loss.

Geostrophic balance on the sea surface:

    u_g = -(g / f) * d(eta)/dy
    v_g =  (g / f) * d(eta)/dx

with f = 2 * Omega * sin(lat), dx = R * cos(lat) * dlon, dy = R * dlat.
"""

from __future__ import annotations

import numpy as np
import torch

G = 9.81  # gravitational acceleration [m s-2]
OMEGA = 7.2921159e-5  # Earth's rotation rate [rad s-1]
R_EARTH = 6_371_000.0  # mean Earth radius [m]
MIN_ABS_LAT = 2.0  # geostrophy is singular at the equator; mask |lat| below this [deg]


def coriolis(lat_deg: np.ndarray | float) -> np.ndarray:
    """Coriolis parameter f [s-1] for latitude(s) in degrees."""
    return 2.0 * OMEGA * np.sin(np.deg2rad(lat_deg))


def grid_spacing(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (dx per latitude row [m], dy [m]) for a regular lat/lon grid.

    ``dx`` has shape (ny,) because it shrinks with cos(lat); ``dy`` is a scalar and
    carries the sign of the latitude ordering (negative if lat is descending).
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    dlon = np.deg2rad(np.mean(np.diff(lon)))
    dlat = np.deg2rad(np.mean(np.diff(lat)))
    dx = R_EARTH * np.cos(np.deg2rad(lat)) * dlon
    dy = R_EARTH * dlat
    return dx, float(dy)


def geostrophic_velocity(
    adt: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Surface geostrophic velocity (u, v) [m s-1] from ADT [m].

    ``adt`` has shape (..., ny, nx). NaNs (land) propagate to neighbouring points,
    which is the desired behaviour: no velocities are invented at the coast.
    """
    dx, dy = grid_spacing(lat, lon)
    deta_dy = np.gradient(adt, axis=-2) / dy
    deta_dx = np.gradient(adt, axis=-1) / dx[:, None]
    f = coriolis(lat)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = -G / f * deta_dy
        v = G / f * deta_dx
    equatorial = np.abs(np.asarray(lat))[:, None] < MIN_ABS_LAT
    u = np.where(equatorial, np.nan, u)
    v = np.where(equatorial, np.nan, v)
    return u, v


def relative_vorticity(
    u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    """Relative vorticity zeta = dv/dx - du/dy [s-1]."""
    dx, dy = grid_spacing(lat, lon)
    return np.gradient(v, axis=-1) / dx[:, None] - np.gradient(u, axis=-2) / dy


def rossby_number(adt: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Local Rossby number zeta / f computed from ADT. |Ro| ~ O(1) marks submesoscale."""
    u, v = geostrophic_velocity(adt, lat, lon)
    return relative_vorticity(u, v, lat, lon) / coriolis(lat)[:, None]


def rossby_deformation_radius(
    lat_deg: float, n_buoyancy: float = 3e-3, depth: float = 500.0
) -> float:
    """First baroclinic Rossby radius estimate L_d = N H / (pi |f|) [m]."""
    return n_buoyancy * depth / (np.pi * abs(float(coriolis(lat_deg))))


# ---------------------------------------------------------------------------
# Differentiable (PyTorch) versions used in the loss
# ---------------------------------------------------------------------------


def _central_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Central difference with one-sided differences at the edges (like np.gradient)."""
    n = x.shape[dim]
    front = x.narrow(dim, 1, 1) - x.narrow(dim, 0, 1)
    back = x.narrow(dim, n - 1, 1) - x.narrow(dim, n - 2, 1)
    inner = (x.narrow(dim, 2, n - 2) - x.narrow(dim, 0, n - 2)) / 2.0
    return torch.cat([front, inner, back], dim=dim)


def geostrophic_velocity_torch(
    adt: torch.Tensor, lat: torch.Tensor, dlat_deg: float, dlon_deg: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable geostrophic velocity.

    Args:
        adt: (B, 1, H, W) ADT in metres.
        lat: (B, H) latitude of each row in degrees.
        dlat_deg, dlon_deg: grid spacing in degrees (signed).
    Returns:
        u, v each (B, 1, H, W) in m/s.
    """
    lat_r = torch.deg2rad(lat)[:, None, :, None]
    f = 2.0 * OMEGA * torch.sin(lat_r)
    f = torch.where(
        f.abs() < 2.0 * OMEGA * np.sin(np.deg2rad(MIN_ABS_LAT)), torch.full_like(f, float("nan")), f
    )
    dx = R_EARTH * torch.cos(lat_r) * np.deg2rad(dlon_deg)
    dy = R_EARTH * np.deg2rad(dlat_deg)
    u = -G / f * _central_diff(adt, dim=-2) / dy
    v = G / f * _central_diff(adt, dim=-1) / dx
    return u, v


def relative_vorticity_torch(
    u: torch.Tensor, v: torch.Tensor, lat: torch.Tensor, dlat_deg: float, dlon_deg: float
) -> torch.Tensor:
    """Differentiable relative vorticity (B, 1, H, W) [s-1]."""
    lat_r = torch.deg2rad(lat)[:, None, :, None]
    dx = R_EARTH * torch.cos(lat_r) * np.deg2rad(dlon_deg)
    dy = R_EARTH * np.deg2rad(dlat_deg)
    return _central_diff(v, dim=-1) / dx - _central_diff(u, dim=-2) / dy


def coriolis_torch(lat: torch.Tensor) -> torch.Tensor:
    """Coriolis parameter broadcastable to (B, 1, H, W)."""
    return 2.0 * OMEGA * torch.sin(torch.deg2rad(lat))[:, None, :, None]
