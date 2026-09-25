from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from submeso.config import Config, load_config
from submeso.validation.drifters import average_drifters, compare_with_drifters, vector_correlation

CONFIGS = sorted(Path(__file__).parents[1].joinpath("configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_shipped_configs_load(path):
    cfg = load_config(path)
    assert isinstance(cfg, Config)
    p = cfg.period
    assert p.train[1] < p.val[0] <= p.val[1] < p.test[0], (
        "splits must be disjoint and ordered in time"
    )
    if cfg.data.source == "copernicus":
        for key in ("hr_adt", "hr_sst", "adt_l4", "sst_l4"):
            spec = getattr(cfg.data, key)
            assert spec.dataset_id and spec.variable


def test_unknown_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("train:\n  epochz: 3\n")
    with pytest.raises(ValueError, match="epochz"):
        load_config(bad)


def test_overrides(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("train:\n  epochs: 3\n")
    cfg = load_config(
        f, ["train.epochs=7", "model.name=srcnn", "period.test=[2001-01-01, 2001-12-31]"]
    )
    assert cfg.train.epochs == 7 and cfg.model.name == "srcnn"
    assert cfg.period.test == ("2001-01-01", "2001-12-31")


def test_vector_correlation_bounds():
    rng = np.random.default_rng(0)
    u, v = rng.normal(size=200), rng.normal(size=200)
    rho, ang = vector_correlation(u, v, u, v)
    assert rho == pytest.approx(1.0) and ang == pytest.approx(0.0, abs=1e-9)
    # rotating the second field by 90 degrees keeps |rho| = 1 but veers by 90
    rho, ang = vector_correlation(u, v, -v, u)
    assert rho == pytest.approx(1.0) and ang == pytest.approx(90.0)


def _uniform_recon(u0: float, v0: float) -> xr.Dataset:
    t = pd.date_range("2022-01-01", periods=5)
    lat, lon = np.linspace(36, 40, 9), np.linspace(0, 4, 9)
    shape = (5, 9, 9)
    data = {}
    for tag, du in (("lr", 0.1), ("sr", 0.0)):
        data[f"u_{tag}"] = (("time", "lat", "lon"), np.full(shape, u0 + du))
        data[f"v_{tag}"] = (("time", "lat", "lon"), np.full(shape, v0))
    return xr.Dataset(data, coords={"time": t, "lat": lat, "lon": lon})


def test_drifter_comparison_scores_better_field_higher():
    rng = np.random.default_rng(0)
    n = 40
    obs = pd.DataFrame(
        {
            "id": rng.integers(0, 4, n),
            "time": pd.Timestamp("2022-01-02") + pd.to_timedelta(rng.uniform(0, 48, n), "h"),
            "lat": rng.uniform(37, 39, n),
            "lon": rng.uniform(1, 3, n),
            "u": 0.2 + 0.01 * rng.normal(size=n),
            "v": -0.1 + 0.01 * rng.normal(size=n),
        }
    )
    metrics, matched = compare_with_drifters(_uniform_recon(0.2, -0.1), obs)
    assert metrics["n_obs"] == n
    assert metrics["sr"]["rmse_u_cms"] < metrics["lr"]["rmse_u_cms"]
    assert metrics["improvement_pct"]["rmse_vector_cms"] > 50


def test_average_drifters_daily_bins():
    times = pd.date_range("2022-01-01", periods=8, freq="6h")
    df = pd.DataFrame(
        {"id": 1, "time": times, "lat": 38.0, "lon": 2.0, "u": np.arange(8.0), "v": 0.0}
    )
    daily = average_drifters(df, 24)
    assert len(daily) == 2
    np.testing.assert_allclose(daily.u, [1.5, 5.5])
