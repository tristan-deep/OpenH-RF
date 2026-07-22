#!/usr/bin/env python3
"""Reconstruct a reference B-mode image from an OpenH-RF/zea HDF5 file.

This script is intentionally thin: it loads the submitted `pipeline.yaml`, reads
raw RF channel data from a zea file, runs the zea Pipeline, and writes a PNG.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

import zea
from zea import Config, File, Pipeline

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "ac9_15m_SK.hdf5"
DEFAULT_PIPELINE = ROOT / "pipeline.yaml"
DEFAULT_OUTPUT = ROOT / "figures" / "reference_bmode" / "ac9_15m_SK_pipeline.png"


def reconstruct_one(
    zea_file: Path, pipeline_yaml: Path, frame: int, device: str | None
) -> tuple[np.ndarray, object]:
    zea.init_device(device=device, verbose=False)
    config = Config.from_path(str(pipeline_yaml))
    pipeline = Pipeline.from_config(config)

    with File(str(zea_file)) as f:
        parameters = f.load_parameters(**config.parameters)
        raw = f.data.raw_data[frame : frame + 1]

    inputs = pipeline.prepare_parameters(parameters)
    outputs = pipeline(**{pipeline.key: raw}, **inputs, return_numpy=True)
    bmode = np.asarray(outputs[pipeline.output_key][0])
    return bmode, parameters


def save_png(image_db: np.ndarray, parameters, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extent_mm = [float(v) * 1e3 for v in parameters.extent_imshow]
    fov_ratio = abs(extent_mm[2] - extent_mm[3]) / abs(extent_mm[1] - extent_mm[0])
    fig, ax = plt.subplots(figsize=(6.2, 2.0 + 4.8 * fov_ratio), constrained_layout=True)
    im = ax.imshow(
        image_db,
        cmap="gray",
        vmin=float(parameters.dynamic_range[0]),
        vmax=float(parameters.dynamic_range[1]),
        extent=extent_mm,
        aspect="equal",
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Lateral position (mm)")
    ax.set_ylabel("Axial depth (mm)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Envelope (dB)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="Input OpenH-RF/zea HDF5 file."
    )
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE, help="Pipeline YAML.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to reconstruct.")
    parser.add_argument(
        "--device", type=str, default=None, help="Optional zea device, e.g. cpu, cuda:0, auto:0."
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not args.pipeline.exists():
        raise FileNotFoundError(args.pipeline)

    image_db, parameters = reconstruct_one(args.input, args.pipeline, args.frame, args.device)
    save_png(image_db, parameters, args.output, f"{args.input.stem}, frame {args.frame}")

    print(f"Input          : {args.input}")
    print(f"Pipeline       : {args.pipeline}")
    print(f"Output         : {args.output}")
    print(
        f"Reconstruction : shape={image_db.shape}, min={float(np.nanmin(image_db)):.2f}, max={float(np.nanmax(image_db)):.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
