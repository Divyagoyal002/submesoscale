import numpy as np
import pytest
import torch

from submeso.config import LossConfig, ModelConfig
from submeso.data.dataset import N_INPUT_CHANNELS, NormStats
from submeso.models.losses import PhysicsInformedLoss, RadialSpectrum, erode
from submeso.models.networks import ResidualSR, build_model

STATS = NormStats(adt_scale=0.05, sst_scale=0.5, residual_scale=0.01, dlat=1 / 24, dlon=1 / 24)


def _batch(b=2, n=32):
    torch.manual_seed(0)
    lat = torch.linspace(38, 39.3, n).repeat(b, 1)
    true = 0.02 * torch.randn(b, 1, n, n)
    mask = torch.ones(b, 1, n, n)
    mask[:, :, :5, :5] = 0
    return true, mask, lat


@pytest.mark.parametrize("name", ["resnet", "srcnn"])
def test_models_preserve_spatial_shape(name):
    model = build_model(ModelConfig(name=name, channels=16, n_blocks=2))
    x = torch.randn(2, N_INPUT_CHANNELS, 40, 56)
    assert model(x).shape == (2, 1, 40, 56)


def test_residual_network_starts_as_identity():
    """Zero-initialised tail => prediction equals the LR input before training."""
    model = ResidualSR(channels=16, n_blocks=2)
    assert torch.count_nonzero(model(torch.randn(1, N_INPUT_CHANNELS, 32, 32))) == 0


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model(ModelConfig(name="nope"))


def test_loss_is_zero_for_perfect_prediction_and_positive_otherwise():
    true, mask, lat = _batch()
    loss_fn = PhysicsInformedLoss(LossConfig(), STATS, 32)
    zero, terms = loss_fn(true.clone(), true, mask, lat)
    assert float(zero) == pytest.approx(0.0, abs=1e-6)
    assert set(terms) == {"adt", "geo", "vort", "spec"}
    pos, _ = loss_fn(true + 0.01 * torch.randn_like(true), true, mask, lat)
    assert float(pos) > 0


def test_loss_ignores_land_and_backpropagates():
    true, mask, lat = _batch()
    loss_fn = PhysicsInformedLoss(LossConfig(w_spec=0.0), STATS, 32)
    pred = true.clone()
    pred[:, :, :2, :2] += 5.0  # huge error, but deep inside land (beyond derivative stencils)
    loss, _ = loss_fn(pred, true, mask, lat)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    pred = (true + 0.01 * torch.randn_like(true)).requires_grad_()
    loss, _ = loss_fn(pred, true, mask, lat)
    loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


def test_geostrophic_term_penalises_gradient_errors_more_than_offsets():
    """A constant offset has no currents; a small-scale ripple does."""
    true, mask, lat = _batch()
    cfg = LossConfig(w_adt=0.0, w_geo=1.0, w_vort=0.0, w_spec=0.0)
    loss_fn = PhysicsInformedLoss(cfg, STATS, 32)
    offset, _ = loss_fn(true + 0.01, true, mask, lat)
    ripple = 0.01 * torch.sin(torch.arange(32.0) * 2.0)[None, None, None, :]
    wiggly, _ = loss_fn(true + ripple, true, mask, lat)
    assert float(offset) < 1e-8 < float(wiggly)


def test_radial_spectrum_conserves_variance_ordering():
    spec = RadialSpectrum(32)
    smooth = torch.from_numpy(np.outer(np.sin(np.linspace(0, 2 * np.pi, 32)), np.ones(32))).float()[
        None, None
    ]
    rough = torch.randn(1, 1, 32, 32)
    es, er = spec(smooth), spec(rough)
    # smooth field: energy concentrated at low k; noise: much flatter
    assert es[..., 0] / es[..., -1] > er[..., 0] / er[..., -1]


def test_erode_shrinks_mask():
    m = torch.zeros(1, 1, 10, 10)
    m[..., 2:8, 2:8] = 1
    assert erode(m, 1).sum() == 16
