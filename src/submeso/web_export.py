"""Export a trained model and its fields for the in-browser web app (``site/``).

Writes, under ``<out>/``:

* ``model.onnx``   - the CNN, run live in the browser with onnxruntime-web;
* ``data/manifest.json`` - grid, dates, normalisation stats, colour ranges, metrics;
* ``data/day_XXX.bin``   - one file per day with the fields quantised to int16;
* ``data/colormaps.json`` - the matplotlib colour maps used by the Python figures.

The browser re-implements :func:`submeso.inference.super_resolve_snapshot` exactly
(same tiling, per-tile normalisation and taper), so the Python reconstruction is also
exported (``adt_ref``) and the app reports how closely its live output matches it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from submeso.config import Config
from submeso.data.dataset import N_INPUT_CHANNELS, select_period
from submeso.inference import reconstruct
from submeso.training import load_checkpoint

log = logging.getLogger(__name__)

NODATA = -32768  # int16 sentinel for land / missing


def export_onnx(model: torch.nn.Module, path: Path, tile: int) -> Path:
    """Export the CNN with a dynamic batch dimension and check it against PyTorch."""
    import onnxruntime as ort

    model = model.cpu().eval()
    dummy = torch.randn(4, N_INPUT_CHANNELS, tile, tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["x"],
        output_names=["residual"],
        dynamic_axes={"x": {0: "batch"}, "residual": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    test = torch.randn(3, N_INPUT_CHANNELS, tile, tile)
    with torch.no_grad():
        expected = model(test).numpy()
    got = session.run(None, {"x": test.numpy()})[0]
    err = float(np.abs(got - expected).max())
    if err > 1e-4:
        raise RuntimeError(f"ONNX export mismatch: max abs diff {err:.2e}")
    log.info(
        "ONNX model %s (%.2f MB), max diff vs PyTorch %.1e", path, path.stat().st_size / 1e6, err
    )
    return path


def _quantize(field: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, float, float]:
    """Map [lo, hi] linearly onto int16 (reserving NODATA); returns (q, offset, scale)."""
    scale = (hi - lo) / 65000.0 or 1.0
    offset = (hi + lo) / 2.0
    q = np.round((field - offset) / scale)
    q = np.where(np.isfinite(field), np.clip(q, -32500, 32500), NODATA).astype("<i2")
    return q, offset, scale


def _colormaps() -> dict[str, list[list[int]]]:
    from matplotlib import colormaps

    return {
        name: (np.asarray(colormaps[name](np.linspace(0, 1, 256)))[:, :3] * 255)
        .round()
        .astype(int)
        .tolist()
        for name in ("viridis", "magma", "RdBu_r")
    }


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.nanpercentile(a, q))


def export_web(
    cfg: Config,
    source: str = "test",
    out: str | Path = "site",
    max_days: int = 45,
    ckpt: str | None = None,
) -> Path:
    """Export model + data for the web app.

    ``source="test"``: the OSSE test period, where the true field is known (used for
    the synthetic demo). ``source="real"``: the real-satellite reconstruction
    (``reconstruction.nc``), which has no truth.
    """
    out = Path(out)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    model, stats, _ = load_checkpoint(ckpt or cfg.run_dir / "best.pt", "cpu")
    tile = cfg.train.patch_size
    overlap = tile // 4
    export_onnx(model, out / "model.onnx", tile)

    if source == "test":
        pairs = select_period(xr.open_dataset(cfg.pairs_path), cfg.period.test).load()
        pairs = pairs.isel(time=slice(0, max_days))
        recon = reconstruct(model, stats, pairs, tile, overlap)
        fields = {
            "adt_lr": pairs.adt_lr.values,
            "sst_in": pairs.sst_in.values,
            "adt_ref": recon.adt_sr.values,
            "adt_truth": pairs.adt_hr.values,
        }
        times = pairs.time.values
        lat, lon = pairs.lat.values, pairs.lon.values
    elif source == "real":
        recon = (
            xr.open_dataset(cfg.run_dir / "reconstruction.nc").isel(time=slice(0, max_days)).load()
        )
        if "sst_in" not in recon:
            raise ValueError(
                "reconstruction.nc has no sst_in; re-run `submeso reconstruct` with this version"
            )
        fields = {
            "adt_lr": recon.adt_lr.values,
            "sst_in": recon.sst_in.values,
            "adt_ref": recon.adt_sr.values,
        }
        times = recon.time.values
        lat, lon = recon.lat.values, recon.lon.values
    else:
        raise ValueError("source must be 'test' or 'real'")

    mask = np.all(np.isfinite(fields["adt_lr"]), axis=0)
    ranges = {}
    for name, arr in fields.items():
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        pad = 0.05 * (hi - lo) + 1e-6
        ranges[name] = (lo - pad, hi + pad)

    field_meta = []
    quantized = {}
    for name, arr in fields.items():
        q, offset, scale = _quantize(arr, *ranges[name])
        quantized[name] = q
        field_meta.append({"name": name, "offset": offset, "scale": scale})
    for old in data_dir.glob("day_*.bin"):
        old.unlink()
    for t in range(len(times)):
        with open(data_dir / f"day_{t:03d}.bin", "wb") as fh:
            for name in fields:
                fh.write(quantized[name][t].tobytes())

    from submeso.inference import add_derived_fields

    ref = add_derived_fields(
        xr.Dataset(
            {"adt_sr": (("time", "lat", "lon"), fields["adt_ref"])},
            coords={"time": times, "lat": lat, "lon": lon},
        )
    )
    adt_display = fields.get("adt_truth", fields["adt_ref"])
    metrics_path = cfg.run_dir / (
        "metrics_osse.json" if source == "test" else "metrics_validation.json"
    )
    manifest = {
        "version": 1,
        "label": "synthetic" if cfg.data.source == "synthetic" else "real",
        "source": source,
        "region": cfg.region.name,
        "experiment": cfg.experiment,
        "dates": [pd.Timestamp(t).strftime("%Y-%m-%d") for t in times],
        "grid": {
            "ny": len(lat),
            "nx": len(lon),
            "lat": [round(float(v), 6) for v in lat],
            "lon": [round(float(v), 6) for v in lon],
        },
        "fields": field_meta,
        "nodata": NODATA,
        "has_truth": "adt_truth" in fields,
        "model": {
            "file": "model.onnx",
            "tile": tile,
            "overlap": overlap,
            "input": "x",
            "output": "residual",
        },
        "stats": stats.to_dict(),
        "display": {
            "adt": [_pct(adt_display, 2), _pct(adt_display, 98)],
            "speed_max": _pct(ref.speed_sr.values, 99),
            "ro_abs": 0.5,
        },
        "metrics": json.loads(metrics_path.read_text()) if metrics_path.exists() else None,
        "ocean_fraction": float(mask.mean()),
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    (data_dir / "colormaps.json").write_text(json.dumps(_colormaps(), separators=(",", ":")) + "\n")
    size = sum(f.stat().st_size for f in data_dir.iterdir()) / 1e6
    log.info("Exported %d days (%s) to %s: %.1f MB of data", len(times), source, data_dir, size)
    return data_dir / "manifest.json"
