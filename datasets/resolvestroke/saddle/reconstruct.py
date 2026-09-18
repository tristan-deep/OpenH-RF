# SPDX-License-Identifier: Apache-2.0
"""Example reconstruction script for the resolvestroke/saddle dataset of OpenH-RF.

Dataset link: https://huggingface.co/datasets/nvidia/OpenH-RF/tree/main/resolvestroke/saddle

B-mode reconstruction of a matrix-probe diverging-wave acquisition.

The reconstruction uses a polar (sector) grid: a fan spanning the divergence
angle in the x-z plane (y = 0), with its apex at the virtual source behind the
array. The result is scan-converted and saved as a B-mode PNG showing the
diverging cone.

Requires zea>=0.1.6 (https://github.com/tue-bmd/zea), the library that does the
ultrasound processing here, together with one of its Keras backends (JAX,
PyTorch or TensorFlow). Installation instructions are at
https://zea.readthedocs.io/en/latest/installation.html.

Usage:
    python reconstruct.py
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zea
from mpl_toolkits.axes_grid1 import make_axes_locatable
from zea import Config, File, Pipeline

HERE = Path(__file__).parent
CONFIG = HERE / "pipeline.yaml"
HF_DIR = "hf://nvidia/OpenH-RF/resolvestroke/saddle/data"

# --- Inputs -----------------------------------------------------------------
# Defaults stream straight from the published corpus. Any of the 21 files works:
# the 20 clinical acquisitions (SP01-Left-1 ... SP10-Right) or the phantom PMP01.
# Swap INPUT for a local path to run against your own copy.
INPUT = f"{HF_DIR}/PMP01.hdf5"
OUTPUT = None  # Output PNG path (default: <input-stem>_bmode.png next to this script)


def reconstruct(input_path, config, pipeline=None):
    """Beamform the single saddle frame of ``input_path`` with ``config``.

    Returns:
        (image, grid, grid_type): ``image`` is the log-compressed B-mode
        (n_radial, n_angular) for a polar grid; ``grid`` the matching Cartesian
        (x, y, z) sample positions in metres.
    """
    with File(str(input_path)) as f:
        parameters = f.load_parameters(**config.parameters)
        if parameters.grid_type == "polar":
            # For a diverging wave the polar-grid apex is the virtual source
            # (|focus_distances|); derive it unless pipeline.yaml pins it. zea takes
            # zlims as on-axis depth and adds the apex to the radii itself.
            apex = config.parameters.get("distance_to_apex")
            if apex is None:
                focus = float(np.abs(np.ravel(parameters.focus_distances)[0]))
                apex = focus if focus > 0 else 0.0
            parameters = f.load_parameters(**{**config.parameters, "distance_to_apex": apex})
        raw = f.data.raw_data[0:1]  # single frame → (1, n_tx, n_ax, n_el, n_ch)

    pipeline = pipeline or Pipeline.from_config(config)
    inputs = pipeline.prepare_parameters(parameters)
    outputs = pipeline(**{pipeline.key: raw}, **inputs, return_numpy=True)
    image = np.asarray(outputs[pipeline.output_key])[0]
    grid = np.asarray(parameters.grid)  # (..., 3), last axis (x, y, z) in metres
    return image, grid, parameters.grid_type


def main():
    out_path = Path(OUTPUT) if OUTPUT else HERE / f"{Path(INPUT).stem}_bmode.png"

    zea.init_device()
    config = Config.from_path(str(CONFIG))
    image, grid, grid_type = reconstruct(INPUT, config)
    print(f"grid           : {grid.shape}  ({grid_type})")

    # Display dynamic range from pipeline.yaml (default 40 dB).
    dr = config.parameters.get("dynamic_range", [-40, 0])
    vmin, vmax = float(dr[0]), float(dr[1])

    zea.visualize.set_mpl_style()
    fig, ax = plt.subplots(figsize=(7, 7))

    if grid_type == "polar":
        # Scan-convert: place each (radial, angular) sample at its Cartesian (x, z).
        x_mm, z_mm = grid[..., 0] * 1e3, grid[..., 2] * 1e3
        pm = ax.pcolormesh(x_mm, z_mm, image, cmap="gray", vmin=vmin, vmax=vmax, shading="auto")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        title = "Diverging-wave sector B-mode (y=0 plane)"
    else:
        # Cartesian volume: show the central elevation slice.
        iy = image.shape[2] // 2
        image = image[:, :, iy]
        x_mm, z_mm = grid[0, :, iy, 0] * 1e3, grid[:, 0, iy, 2] * 1e3
        extent = [
            float(x_mm.min()),
            float(x_mm.max()),
            float(z_mm.max()),
            float(z_mm.min()),
        ]
        pm = ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
        title = "B-mode (Cartesian, elevation slice y≈0)"

    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(pm, cax=cax, label="dB")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")

    print(f"Reconstructed  : {image.shape}")
    print(f"Saved          : {out_path}")


if __name__ == "__main__":
    main()
