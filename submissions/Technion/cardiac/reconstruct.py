# SPDX-License-Identifier: Apache-2.0
"""Reconstruct a B-mode from raw channel data using the zea.Pipeline in pipeline.yaml.

Loads the acquisition parameters and raw channel data from the zea file, builds the
delay-and-sum pipeline defined in `pipeline.yaml`, runs it on `raw_data`, and saves
the scan-converted B-mode. Mirrors the OpenH-RF template reconstruction
(examples/templates/echocardiography).

Usage:
    python reconstruct.py data/s2.hdf5 --frame 54 --out bmode_s2.png
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

import zea
from zea import Config, File, Pipeline

HERE = Path(__file__).parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("zea_file", type=Path)
    p.add_argument("--config", type=Path, default=HERE / "pipeline.yaml")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", type=Path, default=HERE / "bmode.png")
    args = p.parse_args()

    zea.init_device()
    config = Config.from_path(str(args.config))

    with File(str(args.zea_file)) as f:
        parameters = f.load_parameters(**config.parameters)
        raw = f.data.raw_data[args.frame : args.frame + 1]  # (1, n_tx, n_ax, n_el, n_ch)

    pipeline = Pipeline.from_config(config)
    inputs = pipeline.prepare_parameters(parameters)
    outputs = pipeline(data=raw, **inputs, return_numpy=True)
    recon = np.asarray(outputs[pipeline.output_key])[0]  # (grid_z, grid_x)

    zea.visualize.set_mpl_style()
    fig, ax = plt.subplots(figsize=(5.5, 6))
    im = ax.imshow(
        zea.display.to_8bit(recon),
        cmap="gray",
        extent=np.asarray(parameters.extent_imshow) * 1e3,  # m -> mm
        aspect="equal",
    )
    ax.set_xlabel("lateral [mm]")
    ax.set_ylabel("depth [mm]")
    ax.set_title(f"{args.zea_file.stem} — frame {args.frame}")
    fig.colorbar(im, ax=ax, fraction=0.046, label="a.u. (8-bit)")
    fig.tight_layout()
    fig.savefig(str(args.out), dpi=130, bbox_inches="tight")
    print(f"raw {raw.shape} -> B-mode {recon.shape}; saved {args.out}")


if __name__ == "__main__":
    main()
