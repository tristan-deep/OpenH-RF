# SPDX-License-Identifier: Apache-2.0
"""Reconstruct B-mode images for the USTB OpenH-RF (zea) collection.

A single, geometry-agnostic script for every acquisition in every sub-dataset
folder. How each acquisition is reconstructed lives entirely in
``parameters.yaml``: the ``pipeline`` field picks the ``zea.Pipeline`` YAML and
the rest gives the display window / dynamic range. Every acquisition uses the
same workflow — load parameters, run the pipeline, plot — with no per-file logic
here. ``pipeline`` values:

* ``scanline``        — focused linear (FI) scans (line-by-line beamforming).
* ``sector``          — phased-array / steered focused sector scans (polar + scan convert).
* ``iq``              — baseband IQ data (no demodulation).
* ``compound``        — non-focused linear scans (plane-wave / diverging / STA).

Acquisitions with enough transmit events (>= 8) may additionally set
``refocus: true`` in ``parameters.yaml``. For those, a second REFoCUS
reconstruction (transmit-encoding recovery, Bottenus 2018) is written next to the
standard one as ``<name>_zea_refocus_bmode.png`` — the same load/pipeline/plot
workflow, just a different pipeline YAML. Phased-array ``sector`` acquisitions use
the polar ``pipeline_refocus_sector.yaml`` (keeps the sector geometry); every
other geometry uses the linear/cartesian ``pipeline_refocus.yaml``. REFoCUS is not
enabled for single/few-transmit (plane-wave tracking, PICMUS) or synthetic
transmit aperture (already multistatic) acquisitions, where the encoding inversion
is ill-posed or undefined. The secondary REFoCUS pass is opt-in via ``--refocus``;
by default only the standard reconstruction is produced.

Usage::

    python reconstruct.py                          # standard recon for every .hdf5
    python reconstruct.py --refocus                # also emit REFoCUS where enabled
    python reconstruct.py A_cardiac/<file>.hdf5     # a single acquisition
    python reconstruct.py --refocus A_cardiac/<file>.hdf5
"""

import os

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import zea

HERE = Path(__file__).parent
PARAMETERS = zea.Config.from_path(str(HERE / "parameters.yaml"))
PIPELINE_YAML = {
    "scanline": "pipeline_scanline.yaml",
    "sector": "pipeline_sector.yaml",
    "iq": "pipeline_iq.yaml",
    "compound": "pipeline.yaml",
    "refocus": "pipeline_refocus.yaml",
    "refocus_sector": "pipeline_refocus_sector.yaml",
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
    is_scanline = bool(config.parameters.get("enable_scanline"))
    overrides = dict(config.parameters)
    if "dynamic_range" in cfg:
        overrides["dynamic_range"] = cfg["dynamic_range"]
    if "zlims" in cfg:
        overrides["zlims"] = tuple(v * 1e-3 for v in cfg["zlims"])
    if "xlims" in cfg:
        overrides["xlims"] = tuple(v * 1e-3 for v in cfg["xlims"])

    # Load everything we need from the file, then close it before processing.
    with zea.File(str(path)) as f:
        # polar_limits frames the shared (non-scanline) polar grid; scanline
        # imaging builds its own per-transmit rays and ignores it.
        if config.parameters.get("grid_type") == "polar" and not is_scanline:
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
    if is_scanline:
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
    parser = argparse.ArgumentParser(
        description="Reconstruct B-mode images for the USTB OpenH-RF (zea) collection."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Specific .hdf5 files to reconstruct (default: every .hdf5 in all sub-folders).",
    )
    parser.add_argument(
        "--refocus",
        action="store_true",
        help=(
            "Also emit the secondary REFoCUS reconstruction for acquisitions flagged "
            "`refocus: true` in parameters.yaml. Off by default."
        ),
    )
    args = parser.parse_args()

    zea.init_device()
    paths = [Path(p) for p in args.paths] or sorted(HERE.glob("*/*.hdf5"))
    for path in paths:
        reconstruct(path)
        # Optional second reconstruction with REFoCUS transmit-encoding recovery
        # (opt-in via --refocus). Sector (phased-array) acquisitions need the polar
        # refocus pipeline so the sector geometry is preserved; every other geometry
        # uses the linear one.
        if not args.refocus:
            continue
        cfg = PARAMETERS[path.stem]
        if cfg.get("refocus"):
            refocus_key = "refocus_sector" if cfg["pipeline"] == "sector" else "refocus"
            reconstruct(path, pipeline_key=refocus_key, suffix="_zea_refocus_bmode.png")


if __name__ == "__main__":
    main()
