import numpy as np
import pytest

from submeso.data.dataset import valid_patch_origins
from submeso.data.grid import fill_nearest, make_target_grid, nan_gaussian_filter, regrid_2d
from submeso.validation.spectra import (
    band_energy_ratio,
    effective_resolution,
    isotropic_spectrum,
    largest_ocean_square,
    spectral_slope,
)


def test_target_grid_is_regular_and_inside_box():
    lat, lon = make_target_grid(-5.5, 9.0, 35.0, 42.5, 1 / 24)
    assert lat.min() > 35.0 and lat.max() < 42.5
    np.testing.assert_allclose(np.diff(lon), 1 / 24, atol=1e-5)


def test_nan_gaussian_filter_preserves_constant_and_nans():
    f = np.full((30, 40), 3.0)
    f[10:15, 10:15] = np.nan
    out = nan_gaussian_filter(f, (3.0, 3.0))
    assert np.isnan(out[12, 12])
    np.testing.assert_allclose(out[np.isfinite(out)], 3.0)


def test_fill_nearest_and_regrid_identity():
    lat = np.linspace(0, 1, 20)
    lon = np.linspace(0, 2, 30)
    f = np.add.outer(lat, lon)
    np.testing.assert_allclose(regrid_2d(f, lat, lon, lat, lon), f, atol=1e-10)
    g = f.copy()
    g[5, 5] = np.nan
    assert np.isfinite(fill_nearest(g)).all()


def test_patch_origins_respect_ocean_fraction():
    mask = np.zeros((100, 100), bool)
    mask[:, :60] = True
    origins = valid_patch_origins(mask, 32, 1.0)
    assert (origins[:, 1] + 32 <= 60).all()
    with pytest.raises(ValueError):
        valid_patch_origins(np.zeros((100, 100), bool), 32, 0.5)


def _power_law_field(n=256, slope=-11 / 3, seed=0):
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.fftfreq(n)[None, :]
    k = np.hypot(kx, ky)
    k[0, 0] = 1
    amp = k ** (slope / 2 - 0.5)  # 2-D amplitude for a 1-D isotropic slope
    amp[0, 0] = 0
    return np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((n, n)))))


def test_spectral_slope_recovers_known_power_law():
    fields = np.stack([_power_law_field(seed=s) for s in range(4)])
    k, e = isotropic_spectrum(fields, 2.0, 2.0)
    assert spectral_slope(k, e, 10, 100) == pytest.approx(-11 / 3, abs=0.35)


def test_effective_resolution_and_energy_ratio():
    truth = np.stack([_power_law_field(seed=s) for s in range(3)])
    dy = dx = 2.0
    # perfect estimate: resolved down to the grid scale
    res_perfect, _, _ = effective_resolution(truth, truth.copy(), dy, dx)
    assert res_perfect <= 4.5
    # heavily smoothed estimate: much coarser effective resolution, missing energy
    from scipy.ndimage import gaussian_filter

    smooth = gaussian_filter(truth, (0, 6, 6))
    res_smooth, _, _ = effective_resolution(truth, smooth, dy, dx)
    assert res_smooth > 5 * res_perfect
    k, e_t = isotropic_spectrum(truth, dy, dx)
    _, e_s = isotropic_spectrum(smooth, dy, dx)
    assert band_energy_ratio(k, e_s, e_t, 10, 50) < 0.5
    assert band_energy_ratio(k, e_t, e_t, 10, 50) == pytest.approx(1.0)


def test_largest_ocean_square():
    mask = np.zeros((50, 60), bool)
    mask[5:25, 10:40] = True
    sy, sx = largest_ocean_square(mask)
    assert sy.stop - sy.start == 20
    assert mask[sy, sx].all()
