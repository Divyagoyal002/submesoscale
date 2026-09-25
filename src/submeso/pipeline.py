"""End-to-end pipeline stages. Each stage reads/writes files so it can be re-run alone.

prepare   -> data/<region>/hr_truth.nc            (model truth on the target grid)
pairs     -> data/<region>/osse_pairs.nc          (degraded inputs + truth)
train     -> runs/<exp>/best.pt, history.csv
evaluate  -> runs/<exp>/metrics_osse.json, figures/   (OSSE test year, truth known)
reconstruct -> runs/<exp>/reconstruction.nc       (real L4 satellite inputs)
validate  -> runs/<exp>/metrics_drifters.json, metrics_spectra.json, figures/
visualize -> runs/<exp>/figures/animation.*, snapshots
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from submeso.config import Config
from submeso.data import copernicus, osse, synthetic
from submeso.data.dataset import select_period
from submeso.inference import reconstruct
from submeso.training import load_checkpoint, resolve_device, train
from submeso.validation import drifters as drift
from submeso.validation.metrics import evaluate_osse_test
from submeso.validation.spectra import (
    grid_km,
    isotropic_spectrum,
    largest_ocean_square,
    spectral_slope,
)
from submeso.viz import plots

log = logging.getLogger(__name__)


def _truth_path(cfg: Config) -> Path:
    return cfg.data_root / "hr_truth.nc"


def _drifter_path(cfg: Config) -> Path:
    return cfg.data_root / "drifters.csv"


def _figures(cfg: Config) -> Path:
    d = cfg.run_dir / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
    log.info("wrote %s", path)


def _best_ckpt(cfg: Config, ckpt: str | None) -> Path:
    return Path(ckpt) if ckpt else cfg.run_dir / "best.pt"


# --------------------------------------------------------------------------- stages


def download(cfg: Config) -> None:
    if cfg.data.source == "synthetic":
        log.info("Synthetic source: nothing to download")
        return
    copernicus.download_all(cfg)
    p = cfg.period
    r = cfg.region
    df = drift.download_gdp(
        cfg.data.drifter_erddap_url, r.lon_min, r.lon_max, r.lat_min, r.lat_max,
        min(p.test[0], p.reconstruct[0]), max(p.test[1], p.reconstruct[1]),
    )  # fmt: skip
    df.to_csv(_drifter_path(cfg), index=False)
    log.info("Saved %d drifter observations", len(df))


def prepare(cfg: Config) -> Path:
    """Write the high-resolution truth on the target grid."""
    cfg.data_root.mkdir(parents=True, exist_ok=True)
    if cfg.data.source == "synthetic":
        truth = synthetic.generate_nature_run(cfg)
        _write_synthetic_observations(cfg, truth)
    else:
        truth = copernicus.load_hr_truth(cfg)
    truth.to_netcdf(_truth_path(cfg))
    log.info("Truth: %s", dict(truth.sizes))
    return _truth_path(cfg)


def _write_synthetic_observations(cfg: Config, truth: xr.Dataset) -> None:
    """Emulate real L4 products + drifters for the reconstruction period.

    L4 files are written on the coarse grid in the same layout as Copernicus
    products (SST in Kelvin as ``analysed_sst``) so that the real-data code path is
    exercised end to end.
    """
    rec = select_period(truth, cfg.period.reconstruct)
    rng = np.random.default_rng(cfg.seed + 7)
    lat, lon = rec.lat.values, rec.lon.values
    coarse, lat_c, lon_c = osse.degrade_adt_to_coarse(
        rec.adt.values.astype(np.float64), lat, lon, cfg, rng
    )
    sst = osse.degrade_sst(rec.sst.values.astype(np.float64), lat, lon, cfg, rng)
    for key, ds in (
        ("adt_l4", xr.Dataset({"adt": (("time", "latitude", "longitude"), coarse.astype(np.float32))},
                              coords={"time": rec.time.values, "latitude": lat_c, "longitude": lon_c})),
        ("sst_l4", xr.Dataset({"analysed_sst": (("time", "latitude", "longitude"), (sst + 273.15).astype(np.float32))},
                              coords={"time": rec.time.values, "latitude": lat, "longitude": lon})),
    ):  # fmt: skip
        out = cfg.data_root / "raw" / key / f"{key}_synthetic.nc"
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out)
    df = synthetic.simulate_drifters(rec, n_drifters=60, seed=cfg.seed)
    df.to_csv(_drifter_path(cfg), index=False)


def build_pairs(cfg: Config) -> Path:
    truth = xr.open_dataset(_truth_path(cfg))
    truth = truth.sel(time=slice(cfg.period.train[0], cfg.period.test[1])).load()
    pairs = osse.build_osse_pairs(cfg, truth)
    pairs.to_netcdf(cfg.pairs_path)
    log.info("Pairs: %s -> %s", dict(pairs.sizes), cfg.pairs_path)
    return cfg.pairs_path


def evaluate(cfg: Config, ckpt: str | None = None) -> dict:
    """OSSE test-period evaluation where the true fine-scale field is known."""
    device = resolve_device(cfg.train.device)
    model, stats, _ = load_checkpoint(_best_ckpt(cfg, ckpt), device)
    test = select_period(xr.open_dataset(cfg.pairs_path), cfg.period.test).load()
    recon = reconstruct(model, stats, test, cfg.train.patch_size, device=device)
    recon.to_netcdf(cfg.run_dir / "osse_test_reconstruction.nc")
    metrics = evaluate_osse_test(recon, test.adt_hr)
    _write_json(cfg.run_dir / "metrics_osse.json", metrics)

    fig = _figures(cfg)
    box = largest_ocean_square(test.mask.values.astype(bool))
    dy, dx = grid_km(test.lat.values, test.lon.values)
    spectra = {
        "truth": isotropic_spectrum(test.adt_hr.values[(slice(None), *box)], dy, dx),
        "lr": isotropic_spectrum(recon.adt_lr.values[(slice(None), *box)], dy, dx),
        "sr": isotropic_spectrum(recon.adt_sr.values[(slice(None), *box)], dy, dx),
    }
    plots.plot_spectra(spectra, fig / "osse_spectra.png", "OSSE test period")
    plots.plot_snapshot(
        recon, recon.sizes["time"] // 2, fig / "osse_snapshot.png", truth=test.adt_hr
    )
    if (cfg.run_dir / "history.csv").exists():
        plots.plot_history(cfg.run_dir / "history.csv", fig / "training_history.png")
    return metrics


def reconstruct_real(cfg: Config, ckpt: str | None = None) -> Path:
    """Apply the trained model to real (or emulated) L4 satellite ADT + SST."""
    device = resolve_device(cfg.train.device)
    model, stats, _ = load_checkpoint(_best_ckpt(cfg, ckpt), device)
    pairs = xr.open_dataset(cfg.pairs_path)
    mask = pairs.mask.values.astype(bool)
    start, end = cfg.period.reconstruct
    adt = copernicus.open_product(cfg, "adt_l4").sel(time=slice(start, end)).load()
    sst = copernicus.open_product(cfg, "sst_l4").sel(time=slice(start, end)).load()
    inputs = osse.prepare_real_inputs(cfg, adt, sst, pairs.lat.values, pairs.lon.values, mask)
    recon = reconstruct(model, stats, inputs, cfg.train.patch_size, device=device)
    recon.attrs.update(
        {
            "title": f"Super-resolved geostrophic currents - {cfg.region.name}",
            "model": cfg.model.name,
        }
    )
    out = cfg.run_dir / "reconstruction.nc"
    recon.to_netcdf(out)
    log.info("Reconstruction: %s -> %s", dict(recon.sizes), out)
    return out


def validate(cfg: Config) -> dict:
    """Drifter comparison (FR-6) and wavenumber spectra (FR-7) on the real reconstruction."""
    recon = xr.open_dataset(cfg.run_dir / "reconstruction.nc").load()
    fig = _figures(cfg)
    results: dict = {}
    if _drifter_path(cfg).exists():
        raw = pd.read_csv(_drifter_path(cfg), parse_dates=["time"])
        obs = drift.average_drifters(raw, window_hours=24)
        metrics, matched = drift.compare_with_drifters(recon, obs)
        results["drifters"] = metrics
        if metrics.get("n_obs", 0) >= 3:
            plots.plot_drifter_scatter(matched, fig / "drifter_scatter.png")
            matched.to_csv(cfg.run_dir / "drifter_matchups.csv", index=False)
    else:
        log.warning("No drifter file at %s; skipping in-situ validation", _drifter_path(cfg))

    mask = np.all(np.isfinite(recon.adt_lr.values), axis=0)
    box = largest_ocean_square(mask)
    dy, dx = grid_km(recon.lat.values, recon.lon.values)
    spectra = {
        tag: isotropic_spectrum(recon[f"adt_{tag}"].values[(slice(None), *box)], dy, dx)
        for tag in ("lr", "sr")
    }
    results["spectra"] = {
        "box_size_km": float((box[0].stop - box[0].start) * dy),
        **{
            f"slope_10_100km_{tag}": spectral_slope(k, e, 10, 100)
            for tag, (k, e) in spectra.items()
        },
    }
    plots.plot_spectra(spectra, fig / "real_spectra.png", "Satellite period")
    _write_json(cfg.run_dir / "metrics_validation.json", results)
    return results


def visualize(cfg: Config) -> list[Path]:
    recon = xr.open_dataset(cfg.run_dir / "reconstruction.nc").load()
    fig = _figures(cfg)
    outs = [
        plots.plot_snapshot(recon, 0, fig / "real_snapshot_first.png"),
        plots.plot_snapshot(recon, recon.sizes["time"] - 1, fig / "real_snapshot_last.png"),
        plots.animate(recon, fig / "animation_sr.mp4", tag="sr"),
    ]
    for o in outs:
        log.info("wrote %s", o)
    return outs


def run_all(cfg: Config) -> None:
    download(cfg)
    prepare(cfg)
    build_pairs(cfg)
    train(cfg)
    evaluate(cfg)
    reconstruct_real(cfg)
    validate(cfg)
    visualize(cfg)
