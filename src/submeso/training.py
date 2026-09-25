"""Training loop: AMP, cosine LR schedule, gradient clipping, early stopping, checkpoints."""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from submeso.config import Config, config_from_dict
from submeso.data.dataset import NormStats, PatchDataset, compute_stats, select_period
from submeso.models.losses import PhysicsInformedLoss, erode, masked_mse
from submeso.models.networks import build_model, count_parameters
from submeso.physics import geostrophic_velocity_torch

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds legacy global RNG used by third-party code
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_checkpoint(
    path: Path, model: torch.nn.Module, cfg: Config, stats: NormStats, extra: dict
) -> None:
    torch.save(
        {"model": model.state_dict(), "config": cfg.to_dict(), "stats": stats.to_dict(), **extra},
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu"):
    """Return (model, stats, config_dict) from a checkpoint written by :func:`train`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(Config, ckpt["config"])
    model = build_model(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, NormStats(**ckpt["stats"]), cfg


@torch.no_grad()
def _batch_metrics(pred, hr, lr, mask, lat, stats: NormStats) -> dict[str, float]:
    """RMSE of ADT [cm] and geostrophic speed error [cm/s] for prediction and LR baseline."""
    out = {}
    m1 = erode(mask, 1)
    ut, vt = geostrophic_velocity_torch(hr, lat, stats.dlat, stats.dlon)
    for name, field in (("sr", pred), ("lr", lr)):
        out[f"rmse_adt_cm_{name}"] = math.sqrt(float(masked_mse(field, hr, mask))) * 100
        u, v = geostrophic_velocity_torch(field, lat, stats.dlat, stats.dlon)
        out[f"rmse_vel_cms_{name}"] = (
            math.sqrt(float(masked_mse(u, ut, m1) + masked_mse(v, vt, m1))) * 100
        )
    return out


def _run_epoch(
    model, loader, loss_fn, stats, device, optimizer=None, scaler=None, grad_clip=None, amp=False
):
    training = optimizer is not None
    model.train(training)
    sums: dict[str, float] = {}
    n = 0
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with (
            torch.set_grad_enabled(training),
            torch.autocast(device.type, enabled=amp and device.type == "cuda"),
        ):
            residual = model(b["x"])
        pred = b["lr"] + stats.residual_scale * residual.float()
        with torch.set_grad_enabled(training):
            loss, terms = loss_fn(pred, b["hr"], b["mask"], b["lat"])
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            terms.update(_batch_metrics(pred, b["hr"], b["lr"], b["mask"], b["lat"], stats))
        terms["loss"] = float(loss.detach())
        for k, v in terms.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


def train(cfg: Config) -> Path:
    """Train on the OSSE pairs; returns the path of the best checkpoint."""
    seed_everything(cfg.seed)
    device = resolve_device(cfg.train.device)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")

    pairs = xr.open_dataset(cfg.pairs_path).load()
    train_ds = select_period(pairs, cfg.period.train)
    val_ds = select_period(pairs, cfg.period.val)
    if train_ds.sizes["time"] == 0 or val_ds.sizes["time"] == 0:
        raise ValueError("Empty train or val split - check config.period against the pairs file")
    stats = compute_stats(train_ds)
    (run_dir / "norm_stats.json").write_text(json.dumps(stats.to_dict(), indent=2))
    log.info(
        "Train %d days, val %d days, stats %s", train_ds.sizes["time"], val_ds.sizes["time"], stats
    )

    t = cfg.train
    train_data = PatchDataset(
        train_ds, stats, t.patch_size, t.patches_per_epoch, t.min_ocean_fraction, cfg.seed
    )
    val_data = PatchDataset(
        val_ds, stats, t.patch_size, t.val_patches, t.min_ocean_fraction, cfg.seed + 999
    )
    loader_kw = {
        "batch_size": t.batch_size,
        "num_workers": t.num_workers,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(val_data, shuffle=False, **loader_kw)

    model = build_model(cfg.model).to(device)
    log.info(
        "Model %s with %s parameters on %s", cfg.model.name, f"{count_parameters(model):,}", device
    )
    loss_fn = PhysicsInformedLoss(cfg.loss, stats, t.patch_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t.epochs, eta_min=t.lr * 0.02
    )
    scaler = torch.amp.GradScaler(device.type, enabled=t.amp and device.type == "cuda")

    history_path = run_dir / "history.csv"
    best_path = run_dir / "best.pt"
    best, patience = float("inf"), 0
    with open(history_path, "w", newline="") as fh:
        writer = None
        for epoch in range(1, t.epochs + 1):
            start = time.time()
            train_data.resample(epoch)
            train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_kw)
            tr = _run_epoch(
                model, train_loader, loss_fn, stats, device, optimizer, scaler, t.grad_clip, t.amp
            )
            va = _run_epoch(model, val_loader, loss_fn, stats, device)
            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "time_s": round(time.time() - start, 1),
            }
            row.update({f"train_{k}": v for k, v in tr.items()})
            row.update({f"val_{k}": v for k, v in va.items()})
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            fh.flush()
            log.info(
                "epoch %3d | train %.4f | val %.4f | ADT rmse %.2f cm (LR %.2f) | vel rmse %.2f cm/s (LR %.2f)",
                epoch, tr["loss"], va["loss"], va["rmse_adt_cm_sr"], va["rmse_adt_cm_lr"],
                va["rmse_vel_cms_sr"], va["rmse_vel_cms_lr"],
            )  # fmt: skip
            if va["loss"] < best:
                best, patience = va["loss"], 0
                save_checkpoint(best_path, model, cfg, stats, {"epoch": epoch, "val": va})
            else:
                patience += 1
                if patience >= t.early_stopping_patience:
                    log.info("Early stopping at epoch %d (best val loss %.4f)", epoch, best)
                    break
    save_checkpoint(run_dir / "last.pt", model, cfg, stats, {"epoch": epoch})
    return best_path
