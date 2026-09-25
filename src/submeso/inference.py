"""Apply a trained model to full-domain fields with overlapping, blended tiles.

Tiles have the training patch size and use the same per-tile normalisation as
training, so inference sees exactly the input distribution the model was trained
on. Overlaps are blended with a tapered window to avoid seams.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

from submeso.data.dataset import NormStats, normalize_inputs
from submeso.physics import coriolis, geostrophic_velocity, relative_vorticity

log = logging.getLogger(__name__)


def _starts(n: int, tile: int, step: int) -> list[int]:
    if n <= tile:
        return [0]
    s = list(range(0, n - tile + 1, step))
    if s[-1] != n - tile:
        s.append(n - tile)
    return s


def _taper(ty: int, tx: int) -> np.ndarray:
    wy = np.hanning(ty + 2)[1:-1]
    wx = np.hanning(tx + 2)[1:-1]
    return (np.outer(wy, wx) + 1e-3).astype(np.float32)


@torch.no_grad()
def super_resolve_snapshot(
    model: torch.nn.Module,
    stats: NormStats,
    adt_lr: np.ndarray,
    sst: np.ndarray,
    mask: np.ndarray,
    tile: int,
    overlap: int,
    device: torch.device,
) -> np.ndarray:
    """Super-resolve one (ny, nx) snapshot; returns ADT [m] with NaN over land."""
    ny, nx = adt_lr.shape
    ty, tx = min(tile, ny), min(tile, nx)
    step_y, step_x = max(1, ty - overlap), max(1, tx - overlap)
    window = _taper(ty, tx)
    acc = np.zeros((ny, nx), np.float64)
    wsum = np.zeros((ny, nx), np.float64)
    xs, idx = [], []
    for y in _starts(ny, ty, step_y):
        for x in _starts(nx, tx, step_x):
            sl = (slice(y, y + ty), slice(x, x + tx))
            if not mask[sl].any():
                continue
            inp, _, _ = normalize_inputs(adt_lr[sl], sst[sl], mask[sl], stats)
            xs.append(inp)
            idx.append(sl)
    for i in range(0, len(xs), 64):
        batch = torch.from_numpy(np.stack(xs[i : i + 64])).to(device)
        res = model(batch).float().cpu().numpy()[:, 0] * stats.residual_scale
        for r, sl in zip(res, idx[i : i + 64], strict=True):
            acc[sl] += r * window
            wsum[sl] += window
    residual = np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), 0.0)
    out = adt_lr + residual
    out[~mask] = np.nan
    return out.astype(np.float32)


def reconstruct(
    model: torch.nn.Module,
    stats: NormStats,
    inputs: xr.Dataset,
    tile: int,
    overlap: int | None = None,
    device: torch.device | str = "cpu",
) -> xr.Dataset:
    """Super-resolve every snapshot of ``inputs`` (adt_lr, sst_in, mask).

    Returns a dataset with ADT, geostrophic currents and Rossby number for both the
    super-resolved (``*_sr``) and the input L4 (``*_lr``) fields, plus the SST input
    (kept so the web app can re-run the model live).
    """
    device = torch.device(device)
    model = model.to(device).eval()
    overlap = tile // 4 if overlap is None else overlap
    mask = inputs.mask.values.astype(bool)
    lat, lon = inputs.lat.values, inputs.lon.values
    adt_lr = inputs.adt_lr.values
    sst = inputs.sst_in.values
    adt_sr = np.stack(
        [
            super_resolve_snapshot(model, stats, adt_lr[t], sst[t], mask, tile, overlap, device)
            for t in tqdm(range(adt_lr.shape[0]), desc="super-resolving", leave=False)
        ]
    )
    return add_derived_fields(
        xr.Dataset(
            {
                "adt_sr": (
                    ("time", "lat", "lon"),
                    adt_sr,
                    {"units": "m", "long_name": "super-resolved ADT"},
                ),
                "adt_lr": (
                    ("time", "lat", "lon"),
                    adt_lr.astype(np.float32),
                    {"units": "m", "long_name": "input L4 ADT"},
                ),
                "sst_in": (
                    ("time", "lat", "lon"),
                    sst.astype(np.float32),
                    {"units": "degC", "long_name": "input SST"},
                ),
            },
            coords={"time": inputs.time.values, "lat": lat, "lon": lon},
        )
    )


def add_derived_fields(ds: xr.Dataset) -> xr.Dataset:
    """Add u, v, speed and Rossby number for every ``adt_*`` variable."""
    lat, lon = ds.lat.values, ds.lon.values
    f = coriolis(lat)[:, None]
    dims = ("time", "lat", "lon")
    for name in [v for v in ds.data_vars if v.startswith("adt_")]:
        tag = name.removeprefix("adt_")
        u, v = geostrophic_velocity(ds[name].values, lat, lon)
        ro = relative_vorticity(u, v, lat, lon) / f
        ds[f"u_{tag}"] = (
            dims,
            u.astype(np.float32),
            {"units": "m s-1", "long_name": f"geostrophic eastward velocity ({tag})"},
        )
        ds[f"v_{tag}"] = (
            dims,
            v.astype(np.float32),
            {"units": "m s-1", "long_name": f"geostrophic northward velocity ({tag})"},
        )
        ds[f"speed_{tag}"] = (dims, np.hypot(u, v).astype(np.float32), {"units": "m s-1"})
        ds[f"ro_{tag}"] = (
            dims,
            ro.astype(np.float32),
            {"units": "1", "long_name": f"Rossby number zeta/f ({tag})"},
        )
    return ds
