"""Physics-informed loss for ADT super-resolution.

    L = w_adt  * ||eta_pred - eta_true||^2 / sigma_res^2
      + w_geo  * ||u_g(eta_pred) - u_g(eta_true)||^2 / U^2        (geostrophic currents)
      + w_vort * ||zeta/f(eta_pred) - zeta/f(eta_true)||^2          (Rossby number)
      + w_spec * mean_k |log E_pred(k) - log E_true(k)|             (isotropic spectrum)

* The geostrophic term constrains the ADT *gradients*, i.e. the currents we care
  about, rather than just the height field (reference study, Sect. 2).
* The Rossby-number term is a second-derivative constraint normalised by f; it is
  dominated by the submesoscale (|Ro| ~ O(1)) and is this project's extension.
* The spectral term penalises both over-smoothing (missing energy) and spurious
  small-scale noise (excess energy) - the failure modes that "sharper is not
  more accurate" warns about.

Derivatives are only scored where the finite-difference stencil is fully in the
ocean (mask eroded by one pixel per derivative order).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from submeso.config import LossConfig
from submeso.data.dataset import NormStats
from submeso.physics import coriolis_torch, geostrophic_velocity_torch, relative_vorticity_torch


def erode(mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    for _ in range(iterations):
        mask = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    return mask


def masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = torch.nan_to_num(a - b) * mask
    return diff.pow(2).sum() / mask.sum().clamp_min(1.0)


class RadialSpectrum(nn.Module):
    """Isotropic (radially averaged) power spectrum of (B, 1, H, W) fields."""

    def __init__(self, size: int):
        super().__init__()
        ky = torch.fft.fftfreq(size)[:, None]
        kx = torch.fft.rfftfreq(size)[None, :]
        k = torch.sqrt(kx**2 + ky**2)
        bins = torch.clamp((k * size).round().long(), max=size // 2)
        self.n_bins = size // 2 + 1
        self.register_buffer("bins", bins.flatten(), persistent=False)
        self.register_buffer(
            "counts",
            torch.bincount(bins.flatten(), minlength=self.n_bins).float(),
            persistent=False,
        )
        win = torch.hann_window(size, periodic=False)
        self.register_buffer("window", (win[:, None] * win[None, :]), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=(-2, -1), keepdim=True)
        p = torch.fft.rfft2(x * self.window).abs().pow(2).flatten(-2)  # (B, 1, K)
        e = torch.zeros(*p.shape[:-1], self.n_bins, device=x.device, dtype=p.dtype)
        e.index_add_(-1, self.bins, p)
        return (e / self.counts)[..., 1:]  # drop the mean (k=0)


class PhysicsInformedLoss(nn.Module):
    def __init__(self, cfg: LossConfig, stats: NormStats, patch_size: int):
        super().__init__()
        self.cfg = cfg
        self.stats = stats
        self.spectrum = RadialSpectrum(patch_size)

    def forward(
        self, pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor, lat: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        c, s = self.cfg, self.stats
        pred, true = pred.float(), true.float()
        terms: dict[str, torch.Tensor] = {}
        terms["adt"] = masked_mse(pred, true, mask) / s.residual_scale**2

        if c.w_geo > 0 or c.w_vort > 0:
            m1 = erode(mask, 1)
            up, vp = geostrophic_velocity_torch(pred, lat, s.dlat, s.dlon)
            ut, vt = geostrophic_velocity_torch(true, lat, s.dlat, s.dlon)
            if c.w_geo > 0:
                terms["geo"] = (
                    masked_mse(up, ut, m1) + masked_mse(vp, vt, m1)
                ) / c.velocity_scale**2
            if c.w_vort > 0:
                m2 = erode(mask, 2)
                f = coriolis_torch(lat)
                rop = (
                    relative_vorticity_torch(
                        torch.nan_to_num(up), torch.nan_to_num(vp), lat, s.dlat, s.dlon
                    )
                    / f
                )
                rot = (
                    relative_vorticity_torch(
                        torch.nan_to_num(ut), torch.nan_to_num(vt), lat, s.dlat, s.dlon
                    )
                    / f
                )
                terms["vort"] = masked_mse(rop, rot, m2)

        if c.w_spec > 0:
            ep = self.spectrum(pred * mask)
            et = self.spectrum(true * mask)
            eps = 1e-6 * et.mean(dim=-1, keepdim=True).detach() + 1e-20
            terms["spec"] = (torch.log(ep + eps) - torch.log(et + eps)).abs().mean()

        weights = {"adt": c.w_adt, "geo": c.w_geo, "vort": c.w_vort, "spec": c.w_spec}
        total = sum(weights[k] * v for k, v in terms.items())
        return total, {k: float(v.detach()) for k, v in terms.items()}
