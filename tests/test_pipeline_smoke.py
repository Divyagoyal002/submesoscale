"""End-to-end smoke test on a tiny synthetic configuration (CPU, < 1 min)."""

import json
from pathlib import Path

import pytest
import xarray as xr

from submeso import pipeline
from submeso.config import load_config
from submeso.training import train

DEMO = Path(__file__).parents[1] / "configs" / "demo_synthetic.yaml"


@pytest.fixture
def tiny_cfg(tmp_path):
    return load_config(
        DEMO,
        [
            f"data.root={tmp_path / 'data'}",
            f"output_dir={tmp_path / 'run'}",
            "region.lon_min=3.0", "region.lon_max=7.0", "region.lat_min=38.0", "region.lat_max=41.0",
            "period.train=[2020-01-01, 2020-01-20]",
            "period.val=[2020-01-21, 2020-01-25]",
            "period.test=[2020-01-26, 2020-01-30]",
            "period.reconstruct=[2020-01-31, 2020-02-06]",
            "model.channels=8", "model.n_blocks=2",
            "train.epochs=2", "train.patches_per_epoch=64", "train.val_patches=32",
            "train.batch_size=16", "train.patch_size=32", "train.num_workers=0",
            "train.device=cpu", "train.min_ocean_fraction=0.5",
        ],
    )  # fmt: skip


def test_full_pipeline_runs(tiny_cfg):
    pipeline.prepare(tiny_cfg)
    pipeline.build_pairs(tiny_cfg)
    pairs = xr.open_dataset(tiny_cfg.pairs_path)
    assert {"adt_hr", "adt_lr", "sst_in", "mask"} <= set(pairs.data_vars)

    ckpt = train(tiny_cfg)
    assert ckpt.exists()

    metrics = pipeline.evaluate(tiny_cfg)
    assert metrics["sr"]["rmse_adt_cm"] > 0

    out = pipeline.reconstruct_real(tiny_cfg)
    recon = xr.open_dataset(out)
    assert {"adt_sr", "u_sr", "v_sr", "ro_sr", "adt_lr"} <= set(recon.data_vars)
    assert recon.sizes["time"] == 7

    results = pipeline.validate(tiny_cfg)
    assert "spectra" in results
    saved = json.loads((tiny_cfg.run_dir / "metrics_validation.json").read_text())
    assert saved.keys() == results.keys()

    outs = pipeline.visualize(tiny_cfg)
    assert all(Path(o).exists() for o in outs)
