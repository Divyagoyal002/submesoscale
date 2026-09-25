"""Wavenumber spectral analysis (FR-7).

Spectra are computed over the largest land-free square in the domain, detrended
and Hann-windowed, then radially averaged. Two diagnostics are provided:

* spectral slope over a wavelength band (e.g. 10-100 km); SQG-like submesoscale
  dynamics give SSH slopes near k^-11/3, while over-smoothed maps fall off much
  faster;
* effective resolution (Ballarotta et al. 2019): the wavelength at which the
  spectral score 1 - PSD(error)/PSD(truth) drops to 0.5. Scales shorter than this
  are not reliably reconstructed.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from submeso.physics import R_EARTH


def largest_ocean_square(mask: np.ndarray) -> tuple[slice, slice]:
    """Largest all-ocean square (dynamic programming)."""
    m = mask.astype(bool)
    dp = np.zeros(m.shape, np.int32)
    best, pos = 0, (0, 0)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j]:
                dp[i, j] = (
                    1 if i == 0 or j == 0 else 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
                )
                if dp[i, j] > best:
                    best, pos = dp[i, j], (i, j)
    if best == 0:
        raise ValueError("Mask contains no ocean")
    i, j = pos
    return slice(i - best + 1, i + 1), slice(j - best + 1, j + 1)


def grid_km(lat: np.ndarray, lon: np.ndarray) -> tuple[float, float]:
    dy = R_EARTH * np.deg2rad(abs(np.mean(np.diff(lat)))) / 1e3
    dx = R_EARTH * np.cos(np.deg2rad(np.mean(lat))) * np.deg2rad(abs(np.mean(np.diff(lon)))) / 1e3
    return dy, dx


def isotropic_spectrum(
    fields: np.ndarray, dy_km: float, dx_km: float
) -> tuple[np.ndarray, np.ndarray]:
    """Time-averaged isotropic spectrum of NaN-free (T, n, n) or (n, n) fields.

    Returns wavenumber [cycles/km] and spectral density [units^2 / (cycles/km)].
    """
    f = fields if fields.ndim == 3 else fields[None]
    f = signal.detrend(signal.detrend(f, axis=-1), axis=-2)
    ny, nx = f.shape[-2:]
    win = np.outer(np.hanning(ny), np.hanning(nx))
    norm = (win**2).sum()
    p = np.abs(np.fft.fft2(f * win)) ** 2 / norm * dy_km * dx_km
    p = p.mean(axis=0)
    ky = np.fft.fftfreq(ny, dy_km)[:, None]
    kx = np.fft.fftfreq(nx, dx_km)[None, :]
    k = np.hypot(ky, kx)
    dk = max(1.0 / (ny * dy_km), 1.0 / (nx * dx_km))
    edges = np.arange(dk / 2, k.max(), dk)
    which = np.digitize(k.ravel(), edges)
    centers, spec = [], []
    for b in range(1, len(edges)):
        sel = which == b
        if sel.any():
            centers.append(0.5 * (edges[b - 1] + edges[b]))
            # density: sum over annulus divided by annulus width
            spec.append(p.ravel()[sel].sum() * (1.0 / (ny * dy_km)) * (1.0 / (nx * dx_km)) / dk)
    k_out, e_out = np.array(centers), np.array(spec)
    kmax = 0.5 / max(dy_km, dx_km)  # Nyquist
    keep = k_out <= kmax
    return k_out[keep], e_out[keep]


def spectral_slope(
    k: np.ndarray, e: np.ndarray, lambda_min_km: float, lambda_max_km: float
) -> float:
    """Least-squares log-log slope over the wavelength band [lambda_min, lambda_max]."""
    sel = (k >= 1 / lambda_max_km) & (k <= 1 / lambda_min_km) & (e > 0)
    if sel.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(k[sel]), np.log(e[sel]), 1)[0])


def band_energy_ratio(
    k: np.ndarray, e_est: np.ndarray, e_ref: np.ndarray, lambda_min_km: float, lambda_max_km: float
) -> float:
    """Energy of the estimate relative to the reference within a wavelength band.

    ~1: correct fine-scale energy; <1: over-smoothed; >1: spurious small-scale energy.
    """
    sel = (k >= 1 / lambda_max_km) & (k <= 1 / lambda_min_km)
    return float(np.trapezoid(e_est[sel], k[sel]) / np.trapezoid(e_ref[sel], k[sel]))


def effective_resolution(
    truth: np.ndarray, estimate: np.ndarray, dy_km: float, dx_km: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Wavelength [km] where 1 - PSD(err)/PSD(truth) = 0.5, plus (k, score)."""
    k, e_true = isotropic_spectrum(truth, dy_km, dx_km)
    _, e_err = isotropic_spectrum(estimate - truth, dy_km, dx_km)
    score = 1.0 - e_err / e_true
    below = np.nonzero(score < 0.5)[0]
    if len(below) == 0:
        return float(1.0 / k[-1]), k, score  # resolved down to the grid scale
    i = below[0]
    if i == 0:
        return float("inf"), k, score
    # interpolate the crossing in log-wavenumber
    lk = np.interp(0.5, [score[i], score[i - 1]], [np.log(k[i]), np.log(k[i - 1])])
    return float(1.0 / np.exp(lk)), k, score
