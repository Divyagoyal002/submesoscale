"""Typed experiment configuration loaded from YAML.

Every pipeline stage reads the same config file, so a single YAML fully describes
(and reproduces) an experiment.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Region:
    name: str = "Western Mediterranean (Alboran-Balearic)"
    lon_min: float = -5.5
    lon_max: float = 9.0
    lat_min: float = 35.0
    lat_max: float = 42.5


@dataclass
class Period:
    """Date ranges (inclusive, ISO strings). Splits are by time to avoid leakage."""

    train: tuple[str, str] = ("2013-01-01", "2019-12-31")
    val: tuple[str, str] = ("2020-01-01", "2020-12-31")
    test: tuple[str, str] = ("2021-01-01", "2021-12-31")
    reconstruct: tuple[str, str] = ("2022-01-01", "2024-12-31")


@dataclass
class ProductSpec:
    """One variable of one Copernicus Marine dataset."""

    dataset_id: str = ""
    variable: str = ""
    depth: float | None = None  # for 3-D model products: download 0..depth m, use the top level


@dataclass
class SyntheticSpec:
    """Parameters of the synthetic SQG-like 'nature run' used for demos and tests."""

    ssh_rms_m: float = 0.06
    spectral_slope: float = -11.0 / 3.0  # SSH spectrum slope (SQG regime)
    dissipation_km: float = 10.0  # e-folding wavelength of the small-scale roll-off
    sst_sqg_coupling: float = 1.0
    sst_noise_k: float = 0.05
    decorrelation_days: float = 12.0
    basin: bool = True  # semi-enclosed basin land mask


@dataclass
class DataConfig:
    root: str = "data/wmed"
    source: str = "copernicus"  # "copernicus" | "synthetic"
    hr_adt: ProductSpec = field(default_factory=ProductSpec)  # model sea-surface height (truth)
    hr_sst: ProductSpec = field(default_factory=ProductSpec)  # model surface temperature (truth)
    adt_l4: ProductSpec = field(default_factory=ProductSpec)
    sst_l4: ProductSpec = field(default_factory=ProductSpec)
    drifter_erddap_url: str = "https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/drifter_6hour_qc"
    synthetic: SyntheticSpec = field(default_factory=SyntheticSpec)


@dataclass
class OSSEConfig:
    """How the high-resolution 'truth' is degraded to mimic satellite L4 products."""

    target_resolution_deg: float = 1.0 / 24.0
    lr_resolution_deg: float = 0.0625  # grid of the real L4 ADT product (DUACS EUR: 1/16 deg)
    # Gaussian sigma mimicking the optimal-interpolation mapping. The half-power
    # wavelength is ~5.3 * sigma, so 18 km ~ the ~100 km effective resolution of DUACS.
    adt_smoothing_km: float = 18.0
    adt_noise_m: float = 0.01
    sst_smoothing_km: float = 4.0
    sst_noise_k: float = 0.05


@dataclass
class ModelConfig:
    name: str = "resnet"  # "resnet" (EDSR-style, dilated) | "srcnn"
    channels: int = 64
    n_blocks: int = 8
    dilated: bool = True
    res_scale: float = 0.1


@dataclass
class LossConfig:
    """Weights of each term of the physics-informed loss (0 disables a term)."""

    w_adt: float = 1.0
    w_geo: float = 1.0  # geostrophic velocity error
    w_vort: float = 0.2  # Rossby-number (zeta/f) error: emphasises submesoscale
    w_spec: float = 0.05  # log power-spectrum error: penalises over-smoothing / artefacts
    velocity_scale: float = 0.1  # m/s, normalises velocity errors


@dataclass
class TrainConfig:
    patch_size: int = 64
    batch_size: int = 32
    epochs: int = 60
    patches_per_epoch: int = 4096
    val_patches: int = 1024
    lr: float = 2e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    min_ocean_fraction: float = 0.7
    early_stopping_patience: int = 10
    num_workers: int = 2
    amp: bool = True
    device: str = "auto"


@dataclass
class Config:
    experiment: str = "wmed_sr"
    seed: int = 42
    output_dir: str = "runs/wmed_sr"
    region: Region = field(default_factory=Region)
    period: Period = field(default_factory=Period)
    data: DataConfig = field(default_factory=DataConfig)
    osse: OSSEConfig = field(default_factory=OSSEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ------------------------------------------------------------------ paths
    @property
    def data_root(self) -> Path:
        return Path(self.data.root)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir)

    @property
    def pairs_path(self) -> Path:
        return self.data_root / "osse_pairs.nc"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


def config_from_dict(cls: type, data: dict[str, Any] | None) -> Any:
    """Recursively build nested dataclasses, rejecting unknown keys (catches typos)."""
    if data is None:
        return cls()
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        tp = hints[name]
        if dataclasses.is_dataclass(tp) and isinstance(value, dict):
            kwargs[name] = config_from_dict(tp, value)
        elif typing.get_origin(tp) is tuple and isinstance(value, list):
            kwargs[name] = tuple(str(v) for v in value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config; ``overrides`` are ``dotted.key=value`` strings (YAML-parsed)."""
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = raw
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return config_from_dict(Config, raw)
