# SPDX-License-Identifier: Apache-2.0
"""Reconstruct B-mode images for the USTB OpenH-RF (zea) collection.

A single, geometry-agnostic script for every acquisition in every sub-dataset
folder. How each acquisition is reconstructed lives entirely in
``parameters.yaml``: the ``pipeline`` field picks the ``zea.Pipeline`` YAML and
the rest gives the display window / dynamic range. Every acquisition uses the
same workflow — load parameters, run the pipeline, plot — with no per-file logic
here. ``pipeline`` values:

* ``scanline``        — focused linear (FI) scans (line-by-line beamforming).
* ``scanline_sector`` — steered focused scans (line-by-line, fan geometry).
* ``sector``          — phased-array focused sector scans (polar + scan convert).
* ``iq``              — baseband IQ data (no demodulation).
* ``compound``        — non-focused linear scans (plane-wave / diverging / STA).

Acquisitions with many transmit events may additionally set ``refocus: true`` in
``parameters.yaml``. For those, a second REFoCUS reconstruction (transmit-encoding
recovery, ``pipeline_refocus.yaml``) is written next to the standard one as
``<name>_zea_refocus_bmode.png`` — the same load/pipeline/plot workflow, just a
different pipeline YAML.

Usage::

    python reconstruct.py                       # every .hdf5 in all sub-folders
    python reconstruct.py A_cardiac/<file>.hdf5  # a single acquisition
"""

import os

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("KERAS_BACKEND", "jax")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import zea

HERE = Path(__file__).parent
PARAMETERS = zea.Config.from_path(str(HERE / "parameters.yaml"))
PIPELINE_YAML = {
    "scanline": "pipeline_scanline.yaml",
    "scanline_sector": "pipeline_scanline_sector.yaml",
    "sector": "pipeline_sector.yaml",
    "iq": "pipeline_iq.yaml",
    "compound": "pipeline.yaml",
    "refocus": "pipeline_refocus.yaml",
}


def reconstruct(
    path: Path, pipeline_key: str | None = None, suffix: str = "_zea_bmode.png"
) -> Path:
    """Reconstruct the first frame of a zea file and save a B-mode PNG.

    ``pipeline_key`` overrides the pipeline chosen in ``parameters.yaml`` (used to
    also render the REFoCUS variant); ``suffix`` names the output PNG.
    """
    cfg = PARAMETERS[path.stem]
    config = zea.Config.from_path(str(HERE / PIPELINE_YAML[pipeline_key or cfg["pipeline"]]))
    overrides = dict(config.parameters)
    if "dynamic_range" in cfg:
        overrides["dynamic_range"] = cfg["dynamic_range"]
    if "zlims" in cfg:
        overrides["zlims"] = tuple(v * 1e-3 for v in cfg["zlims"])
    if "xlims" in cfg:
        overrides["xlims"] = tuple(v * 1e-3 for v in cfg["xlims"])

    # Load everything we need from the file, then close it before processing.
    with zea.File(str(path)) as f:
        if config.parameters.get("grid_type") == "polar":
            angles = np.asarray(f.scan.polar_angles, dtype=float)
            overrides["polar_limits"] = (float(angles.min()), float(angles.max()))
        parameters = f.load_parameters(**overrides)
        data = f.data.raw_data[:1]  # first frame

    pipeline = zea.Pipeline.from_config(config)
    outputs = pipeline(data=data, **pipeline.prepare_parameters(parameters))
    image = np.array(
        zea.display.to_8bit(
            np.squeeze(np.array(outputs["data"])), dynamic_range=parameters.dynamic_range
        )
    )

    zea.visualize.set_mpl_style()
    fig, ax = plt.subplots(figsize=(5, 6))
    if config.parameters.get("grid_type") == "scanline":
        # Scanline geometry is defined by the beamforming grid.
        gx = np.asarray(parameters.grid[..., 0]) * 1e3
        gz = np.asarray(parameters.grid[..., 2]) * 1e3
        ax.pcolormesh(gx, gz, image, cmap="gray", shading="gouraud", vmin=0, vmax=255)
        ax.invert_yaxis()
    else:  # pixel-grid pipelines: regular imshow with the grid extent
        extent = np.array(parameters.extent_imshow) * 1e3
        ax.imshow(image, extent=extent, cmap="gray", aspect="equal")

    if "xlims" in cfg:
        ax.set_xlim(cfg["xlims"])
    if "zlims" in cfg:
        ax.set_ylim(cfg["zlims"][1], cfg["zlims"][0])
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title(path.stem, fontsize=8)

    out_path = path.with_name(path.stem + suffix)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"{path.parent.name}/{path.name} -> {out_path.name}")
    return out_path


def main():
    zea.init_device()
    paths = [Path(p) for p in sys.argv[1:]] or sorted(HERE.glob("*/*.hdf5"))
    for path in paths:
        reconstruct(path)
        # Optional second reconstruction with REFoCUS transmit-encoding recovery.
        if PARAMETERS[path.stem].get("refocus"):
            reconstruct(path, pipeline_key="refocus", suffix="_zea_refocus_bmode.png")


if __name__ == "__main__":
    main()
