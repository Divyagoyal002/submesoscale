"""PyTorch dataset of random OSSE patches, plus normalisation statistics.

Normalisation is per-patch for the absolute level (the spatial mean of the LR ADT
and SST are removed, because the basin-mean sea level and seasonal SST carry no
fine-scale information) and global for the scale (std computed on the training
split only).

The network predicts a *residual* added to the LR ADT:

    adt_pred = adt_lr + residual_scale * net(inputs)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

N_INPUT_CHANNELS = 3  # adt_lr anomaly, sst anomaly, ocean mask


@dataclass
class NormStats:
    adt_scale: float
    sst_scale: float
    residual_scale: float
    dlat: float
    dlon: float

    def to_dict(self) -> dict:
        return asdict(self)


def select_period(ds: xr.Dataset, period: tuple[str, str]) -> xr.Dataset:
    return ds.sel(time=slice(period[0], period[1]))


def compute_stats(train: xr.Dataset) -> NormStats:
    """Scale statistics from the training split (subsampled in time for speed)."""
    sub = train.isel(time=slice(None, None, max(1, train.sizes["time"] // 200)))
    adt_lr = sub.adt_lr.values
    sst = sub.sst_in.values
    adt_anom = adt_lr - np.nanmean(adt_lr, axis=(1, 2), keepdims=True)
    sst_anom = sst - np.nanmean(sst, axis=(1, 2), keepdims=True)
    lat, lon = train.lat.values, train.lon.values
    return NormStats(
        adt_scale=float(np.nanstd(adt_anom)),
        sst_scale=float(np.nanstd(sst_anom)),
        residual_scale=float(np.nanstd(sub.adt_hr.values - adt_lr)),
        dlat=float(np.mean(np.diff(lat))),
        dlon=float(np.mean(np.diff(lon))),
    )


def valid_patch_origins(
    mask: np.ndarray, patch: int, min_ocean: float, stride: int = 4
) -> np.ndarray:
    """Top-left corners (iy, ix) of patches whose ocean fraction is >= ``min_ocean``."""
    ny, nx = mask.shape
    if ny < patch or nx < patch:
        raise ValueError(f"Domain {mask.shape} smaller than patch size {patch}")
    # Integral image: ocean count of mask[y:y+patch, x:x+patch] in O(1) per origin.
    s = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    iy, ix = np.meshgrid(
        np.arange(0, ny - patch + 1, stride), np.arange(0, nx - patch + 1, stride), indexing="ij"
    )
    count = s[iy + patch, ix + patch] - s[iy, ix + patch] - s[iy + patch, ix] + s[iy, ix]
    ok = count / patch**2 >= min_ocean
    origins = np.stack([iy[ok], ix[ok]], axis=-1)
    if len(origins) == 0:
        raise ValueError("No patch satisfies min_ocean_fraction; lower it or the patch size")
    return origins


def normalize_inputs(
    adt_lr: np.ndarray, sst: np.ndarray, mask: np.ndarray, stats: NormStats
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the (3, H, W) network input from one tile.

    Returns (inputs, lr_anomaly_filled [m], adt_offset [m]).
    """
    ocean = mask.astype(bool) & np.isfinite(adt_lr) & np.isfinite(sst)
    if not ocean.any():
        zeros = np.zeros_like(adt_lr, dtype=np.float32)
        return np.stack([zeros, zeros, zeros]), zeros, 0.0
    adt_off = float(adt_lr[ocean].mean())
    sst_off = float(sst[ocean].mean())
    a = np.where(ocean, adt_lr - adt_off, 0.0)
    s = np.where(ocean, sst - sst_off, 0.0)
    x = np.stack([a / stats.adt_scale, s / stats.sst_scale, ocean.astype(np.float64)]).astype(
        np.float32
    )
    return x, a.astype(np.float32), adt_off


class PatchDataset(Dataset):
    """Random (time, y, x) patches from an OSSE pairs dataset.

    Call :meth:`resample` at the start of each epoch for fresh training patches;
    validation datasets keep a fixed sample list for comparable scores.
    """

    def __init__(
        self,
        pairs: xr.Dataset,
        stats: NormStats,
        patch_size: int,
        n_samples: int,
        min_ocean_fraction: float,
        seed: int = 0,
    ):
        self.adt_lr = pairs.adt_lr.values.astype(np.float32)
        self.adt_hr = pairs.adt_hr.values.astype(np.float32)
        self.sst = pairs.sst_in.values.astype(np.float32)
        self.mask = pairs.mask.values.astype(bool)
        self.lat = pairs.lat.values.astype(np.float32)
        self.stats = stats
        self.patch = patch_size
        self.n_samples = n_samples
        self.origins = valid_patch_origins(self.mask, patch_size, min_ocean_fraction)
        self.seed = seed
        self.resample(0)

    def resample(self, epoch: int) -> None:
        rng = np.random.default_rng((self.seed, epoch))
        self.samples = np.column_stack(
            [
                rng.integers(0, self.adt_lr.shape[0], self.n_samples),
                self.origins[rng.integers(0, len(self.origins), self.n_samples)],
            ]
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        t, y, x = self.samples[i]
        sl = (slice(y, y + self.patch), slice(x, x + self.patch))
        m = self.mask[sl]
        inputs, lr, off = normalize_inputs(self.adt_lr[t][sl], self.sst[t][sl], m, self.stats)
        ocean = inputs[2].astype(bool)
        hr = np.where(ocean, self.adt_hr[t][sl] - off, 0.0).astype(np.float32)
        return {
            "x": torch.from_numpy(inputs),
            "lr": torch.from_numpy(lr[None]),
            "hr": torch.from_numpy(hr[None]),
            "mask": torch.from_numpy(ocean[None].astype(np.float32)),
            "lat": torch.from_numpy(self.lat[y : y + self.patch].copy()),
        }
