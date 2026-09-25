"""Figures and animations (FR-8).

Colour conventions (fixed across every figure):
* truth = near-black ink, L4/low-resolution = blue (dashed), super-resolved = orange;
* ADT / speed use single-hue-family sequential maps; Rossby number uses a diverging
  map centred on a neutral zero.

Cartopy coastlines are drawn when cartopy is installed (``pip install -e ".[maps]"``);
otherwise plain lon/lat axes are used.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib import animation  # noqa: E402

log = logging.getLogger(__name__)

COLORS = {"truth": "#222222", "lr": "#2a78d6", "sr": "#eb6834"}
STYLES = {"truth": "-", "lr": "--", "sr": "-"}
LABELS = {"truth": "Model truth", "lr": "L4 altimetry (input)", "sr": "Super-resolved (CNN)"}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "font.size": 9,
        "axes.titlesize": 10,
    }
)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_CARTOPY = False


def _map_axes(fig, nrows: int, ncols: int):
    if HAS_CARTOPY:
        axes = fig.subplots(
            nrows, ncols, subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False
        )
        for ax in axes.flat:
            ax.add_feature(cfeature.LAND, facecolor="#e7e6e1", zorder=2)
            ax.coastlines(resolution="50m", linewidth=0.5, zorder=3)
        return axes
    axes = fig.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=True)
    for ax in axes.flat:
        ax.set_facecolor("#e7e6e1")  # land shows through NaNs
        ax.set_aspect("equal")
    return axes


def _quiver(ax, lon, lat, u, v, every: int, ref_speed: float):
    """Current vectors every ``every`` points; an arrow of ``ref_speed`` spans ~one spacing."""
    sl = (slice(None, None, every), slice(None, None, every))
    spacing = every * abs(float(np.mean(np.diff(lon))))
    q = ax.quiver(
        lon[::every], lat[::every], u[sl], v[sl], color="#1a1a19", zorder=4,
        angles="xy", scale_units="xy", scale=ref_speed / spacing, width=0.0025,
    )  # fmt: skip
    ax.quiverkey(
        q, 0.88, 1.02, ref_speed, f"{ref_speed:.2g} m/s", labelpos="E", fontproperties={"size": 8}
    )
    return q


def _ref_speed(*speeds: np.ndarray) -> float:
    """A round reference speed near the 90th percentile of the given fields."""
    p90 = float(np.nanpercentile(np.concatenate([np.ravel(s) for s in speeds]), 90))
    return float(
        min((0.05, 0.1, 0.2, 0.3, 0.5, 1.0), key=lambda r: abs(np.log(r / max(p90, 1e-3))))
    )


def plot_snapshot(
    recon: xr.Dataset, time_index: int, out_path: str | Path, truth: xr.DataArray | None = None
) -> Path:
    """Compare ADT+currents (top) and Rossby number (bottom) for L4, SR (and truth)."""
    from submeso.inference import add_derived_fields

    ds = recon.isel(time=time_index)
    tags = ["lr", "sr"]
    if truth is not None:
        extra = add_derived_fields(xr.Dataset({"adt_truth": truth.isel(time=[time_index])}))
        ds = ds.assign({k: extra[k].isel(time=0) for k in extra.data_vars})
        tags.append("truth")
    lat, lon = ds.lat.values, ds.lon.values
    every = max(1, len(lon) // 30)
    ref = _ref_speed(*[ds[f"speed_{t}"].values for t in tags])
    fig = plt.figure(figsize=(4.2 * len(tags), 6.2), layout="constrained")
    axes = _map_axes(fig, 2, len(tags))
    vmin, vmax = np.nanpercentile(ds.adt_sr.values, [2, 98])
    for j, tag in enumerate(tags):
        ax = axes[0, j]
        im = ax.pcolormesh(
            lon, lat, ds[f"adt_{tag}"], cmap="viridis", vmin=vmin, vmax=vmax, shading="auto"
        )
        _quiver(ax, lon, lat, ds[f"u_{tag}"].values, ds[f"v_{tag}"].values, every, ref)
        ax.set_title(LABELS[tag])
        ax2 = axes[1, j]
        ro = ax2.pcolormesh(
            lon, lat, ds[f"ro_{tag}"], cmap="RdBu_r", vmin=-0.5, vmax=0.5, shading="auto"
        )
    fig.colorbar(im, ax=axes[0, :], label="ADT [m]", shrink=0.8)
    fig.colorbar(ro, ax=axes[1, :], label="Rossby number ζ/f", shrink=0.8)
    date = pd.Timestamp(ds.time.values).strftime("%Y-%m-%d")
    fig.suptitle(f"Surface geostrophic currents, {date}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def animate(
    recon: xr.Dataset, out_path: str | Path, tag: str = "sr", fps: int = 6, max_frames: int = 120
) -> Path:
    """Time-animated map of speed + current vectors (MP4 if ffmpeg exists, else GIF)."""
    ds = recon.isel(time=slice(0, max_frames))
    lat, lon = ds.lat.values, ds.lon.values
    every = max(1, len(lon) // 30)
    speed = ds[f"speed_{tag}"].values
    u, v = ds[f"u_{tag}"].values, ds[f"v_{tag}"].values
    vmax = float(np.nanpercentile(speed, 99))
    fig = plt.figure(figsize=(8, 4.8), layout="constrained")
    ax = _map_axes(fig, 1, 1)[0, 0]
    mesh = ax.pcolormesh(lon, lat, speed[0], cmap="magma", vmin=0, vmax=vmax, shading="auto")
    q = _quiver(ax, lon, lat, u[0], v[0], every, _ref_speed(speed))
    fig.colorbar(mesh, ax=ax, label="Geostrophic speed [m/s]", shrink=0.85)
    title = ax.set_title("")
    times = pd.to_datetime(ds.time.values)

    def update(i):
        mesh.set_array(speed[i].ravel())
        sl = (slice(None, None, every), slice(None, None, every))
        q.set_UVC(np.nan_to_num(u[i][sl]), np.nan_to_num(v[i][sl]))
        title.set_text(f"{LABELS[tag]} currents - {times[i]:%Y-%m-%d}")
        return mesh, q, title

    anim = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".mp4" and animation.writers.is_available("ffmpeg"):
        anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    else:
        out = out.with_suffix(".gif")
        anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out


def plot_spectra(
    spectra: dict[str, tuple[np.ndarray, np.ndarray]], out_path: str | Path, title: str = ""
) -> Path:
    """Log-log isotropic SSH spectra with k^-5 (QG) and k^-11/3 (SQG) reference slopes."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2), layout="constrained")
    for tag, (k, e) in spectra.items():
        ax.loglog(
            k,
            e,
            STYLES.get(tag, "-"),
            color=COLORS.get(tag, "#555"),
            lw=2,
            label=LABELS.get(tag, tag),
        )
    # reference slopes, anchored a decade below the reference curve so they never overlap it
    ref_tag = "truth" if "truth" in spectra else next(iter(spectra))
    k, e = spectra[ref_tag]
    k0 = k[len(k) // 4]
    kk = np.array([k0, k[-1]])
    e0 = np.interp(k0, k, e) / 10
    for slope, ls, lab in ((-5, ":", "k$^{-5}$ (QG)"), (-11 / 3, "-.", "k$^{-11/3}$ (SQG)")):
        ax.loglog(kk, e0 * (kk / k0) ** slope, ls, color="#8a8984", lw=1, label=lab)
    ax.set_xlabel("Wavenumber [cycles/km]")
    ax.set_ylabel("SSH spectral density [m² / (cycles/km)]")
    sec = ax.secondary_xaxis(
        "top", functions=(lambda x: 1 / np.maximum(x, 1e-9), lambda x: 1 / np.maximum(x, 1e-9))
    )
    sec.set_xlabel("Wavelength [km]")
    ax.legend(frameon=False)
    ax.set_title(title)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_history(history_csv: str | Path, out_path: str | Path) -> Path:
    """Training curves: loss, then validation ADT RMSE vs the LR baseline."""
    h = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    axes[0].plot(h.epoch, h.train_loss, color=COLORS["lr"], lw=2, label="train")
    axes[0].plot(h.epoch, h.val_loss, color=COLORS["sr"], lw=2, label="validation")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_title("Physics-informed loss")
    axes[0].legend(frameon=False)
    axes[1].plot(
        h.epoch, h.val_rmse_vel_cms_lr, STYLES["lr"], color=COLORS["lr"], lw=2, label=LABELS["lr"]
    )
    axes[1].plot(h.epoch, h.val_rmse_vel_cms_sr, color=COLORS["sr"], lw=2, label=LABELS["sr"])
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Validation geostrophic velocity RMSE [cm/s]")
    axes[1].legend(frameon=False)
    out = Path(out_path)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_drifter_scatter(matched: pd.DataFrame, out_path: str | Path) -> Path:
    """Model vs drifter velocity components for L4 and SR (shared axes, 1:1 line)."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, layout="constrained")
    lim = float(np.nanpercentile(np.abs(matched[["u", "v"]].values), 99)) * 1.1
    for ax, tag in zip(axes, ("lr", "sr"), strict=True):
        obs = np.concatenate([matched.u, matched.v])
        mod = np.concatenate([matched[f"u_{tag}"], matched[f"v_{tag}"]])
        ax.scatter(obs, mod, s=8, color=COLORS[tag], alpha=0.35, linewidths=0)
        ax.plot([-lim, lim], [-lim, lim], color="#8a8984", lw=1)
        r = np.corrcoef(obs, mod)[0, 1]
        rmse = np.sqrt(np.mean((obs - mod) ** 2)) * 100
        ax.set_title(f"{LABELS[tag]}\nr = {r:.2f}, RMSE = {rmse:.1f} cm/s")
        ax.set_xlabel("Drifter velocity (u and v) [m/s]")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
    axes[0].set_ylabel("Geostrophic velocity [m/s]")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
