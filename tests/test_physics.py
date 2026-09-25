import numpy as np
import pytest
import torch

from submeso.physics import (
    R_EARTH,
    G,
    coriolis,
    geostrophic_velocity,
    geostrophic_velocity_torch,
    relative_vorticity,
    relative_vorticity_torch,
    rossby_deformation_radius,
)

LAT = np.linspace(38.0, 42.0, 97)
LON = np.linspace(2.0, 8.0, 145)


def test_coriolis_sign_and_magnitude():
    assert coriolis(0.0) == pytest.approx(0.0, abs=1e-12)
    assert coriolis(45.0) == pytest.approx(1.0313e-4, rel=1e-3)
    assert coriolis(-30.0) < 0


def test_geostrophic_velocity_matches_analytic_meridional_wave():
    """eta = A sin(2 pi y / L)  =>  u = -(g/f) d(eta)/dy,  v = 0."""
    amp, wavelength = 0.1, 200e3
    y = R_EARTH * np.deg2rad(LAT - LAT[0])
    eta = amp * np.sin(2 * np.pi * y / wavelength)[:, None] * np.ones((1, LON.size))
    u, v = geostrophic_velocity(eta, LAT, LON)
    expected_u = (
        -G
        / coriolis(LAT)[:, None]
        * amp
        * (2 * np.pi / wavelength)
        * np.cos(2 * np.pi * y / wavelength)[:, None]
    )
    # central differences: relative truncation error ~ (k dy)^2 / 6 ~ 0.35 % here
    np.testing.assert_allclose(
        u[2:-2], np.broadcast_to(expected_u, u.shape)[2:-2], rtol=6e-3, atol=1e-4
    )
    np.testing.assert_allclose(v, 0.0, atol=1e-10)


def test_gaussian_eddy_is_geostrophically_consistent():
    """A high-pressure (warm) eddy rotates anticyclonically: negative vorticity in the NH."""
    yy, xx = np.meshgrid(LAT, LON, indexing="ij")
    eta = 0.2 * np.exp(-(((yy - 40) / 0.4) ** 2 + ((xx - 5) / 0.5) ** 2))
    u, v = geostrophic_velocity(eta, LAT, LON)
    zeta = relative_vorticity(u, v, LAT, LON)
    iy, ix = np.unravel_index(np.argmax(eta), eta.shape)
    assert zeta[iy, ix] < 0
    # north of the centre the flow is eastward (clockwise rotation)
    assert u[iy + 5, ix] > 0


def test_land_nan_does_not_leak_invented_velocities():
    eta = np.random.default_rng(0).normal(size=(LAT.size, LON.size)) * 0.01
    eta[40:50, 60:70] = np.nan
    u, v = geostrophic_velocity(eta, LAT, LON)
    assert np.isnan(u[45, 65]) and np.isnan(v[45, 65])
    assert np.isfinite(u[10, 10])


def test_torch_and_numpy_agree():
    rng = np.random.default_rng(1)
    eta = rng.normal(size=(2, LAT.size, LON.size)) * 0.02
    u_np, v_np = geostrophic_velocity(eta, LAT, LON)
    z_np = relative_vorticity(u_np, v_np, LAT, LON)
    t = torch.tensor(eta[:, None], dtype=torch.float64)
    lat_t = torch.tensor(np.stack([LAT, LAT]), dtype=torch.float64)
    dlat, dlon = float(np.diff(LAT).mean()), float(np.diff(LON).mean())
    u_t, v_t = geostrophic_velocity_torch(t, lat_t, dlat, dlon)
    z_t = relative_vorticity_torch(u_t, v_t, lat_t, dlat, dlon)
    np.testing.assert_allclose(u_t[:, 0].numpy(), u_np, rtol=1e-6)
    np.testing.assert_allclose(v_t[:, 0].numpy(), v_np, rtol=1e-6)
    np.testing.assert_allclose(z_t[:, 0].numpy(), z_np, rtol=1e-6, atol=1e-12)


def test_equator_is_masked():
    lat = np.linspace(-3, 3, 13)
    eta = np.tile(np.linspace(0, 0.1, 13)[:, None], (1, 10))
    u, _ = geostrophic_velocity(eta, lat, np.linspace(0, 1, 10))
    assert np.isnan(u[6]).all()


def test_rossby_radius_is_order_10km_in_mediterranean():
    assert 5e3 < rossby_deformation_radius(38.0) < 30e3
