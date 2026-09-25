"""Command-line interface: ``submeso <stage> -c configs/<experiment>.yaml [-o key=value ...]``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from submeso import pipeline
from submeso.config import load_config
from submeso.training import train


def _export_web(cfg, args):
    from submeso.web_export import export_web  # needs onnx + onnxruntime (``.[web]`` extra)

    return export_web(cfg, source=args.source, out=args.out, max_days=args.max_days, ckpt=args.ckpt)


STAGES = {
    "download": pipeline.download,
    "prepare": pipeline.prepare,
    "pairs": pipeline.build_pairs,
    "train": train,
    "evaluate": pipeline.evaluate,
    "reconstruct": pipeline.reconstruct_real,
    "validate": pipeline.validate,
    "visualize": pipeline.visualize,
    "all": pipeline.run_all,
    "export-web": _export_web,
}

HELP = {
    "download": "fetch Copernicus ADT/SST/model data and GDP drifters",
    "prepare": "build high-resolution truth on the target grid",
    "pairs": "degrade truth into OSSE training pairs",
    "train": "train the super-resolution CNN",
    "evaluate": "score the model on the OSSE test period",
    "reconstruct": "super-resolve real L4 satellite ADT+SST",
    "validate": "drifter + spectral validation of the reconstruction",
    "visualize": "maps and animation of reconstructed currents",
    "all": "run every stage in order",
    "export-web": "export the model (ONNX) and fields for the web app in site/",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="submeso", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in STAGES:
        p = sub.add_parser(name, help=HELP[name])
        p.add_argument("-c", "--config", required=True, help="experiment YAML")
        p.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE",
                       help="override a config value, e.g. -o train.epochs=5")  # fmt: skip
        if name == "export-web":
            p.add_argument("--source", choices=["test", "real"], default="test",
                           help="'test': OSSE test period (truth known); 'real': reconstruction.nc")  # fmt: skip
            p.add_argument("--out", default="site", help="web app directory")
            p.add_argument("--max-days", type=int, default=45, help="number of days to export")
        if name in ("evaluate", "reconstruct", "export-web"):
            p.add_argument(
                "--ckpt", default=None, help="checkpoint (default: <output_dir>/best.pt)"
            )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config, args.override)
    fn = STAGES[args.stage]
    if args.stage == "export-web":
        result = fn(cfg, args)
    else:
        result = fn(cfg, args.ckpt) if getattr(args, "ckpt", None) else fn(cfg)
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
