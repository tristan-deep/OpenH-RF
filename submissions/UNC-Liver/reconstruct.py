"""Reconstruct a B-mode image from a fullwave-abdominal-wall zea file using a zea.Pipeline.

The transmit sequence is full synthetic aperture on a curvilinear array, so the
reconstruction grid is polar: rays emanate from the centre of curvature, which sits
at ``z = -distance_to_apex`` in the file's coordinate frame (the array apex is at
``z = 0``).

Usage:
    python reconstruct.py sample.hdf5 --out bmode.png
    python reconstruct.py sample.hdf5 --save-yaml pipeline.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import matplotlib
import numpy as np
import zea
from zea.display import scan_convert_2d
from zea.ops import (
    Beamform,
    Cast,
    Demodulate,
    EnvelopeDetect,
    LogCompress,
    Normalize,
    Pipeline,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reconstruction grid, matching the polar grid the dataset's reference
# beamformed_data was formed on (640 radial x 128 lateral).
GRID_PARAMETERS = {
    "grid_type": "polar",
    "polar_limits": (-0.3, 0.3),
    "grid_size_z": 640,
    "grid_size_x": 128,
    "f_number": 2.0,
    "dynamic_range": (-60, 0),
}


def grid_parameters(
    probe_radius: float, r_min: float = 0.005, n_r: int = 640, dr: float = 1.0405405405405406e-04
) -> dict:
    """Grid parameters for zea's polar_pixel_grid in the file's coordinate frame.

    ``polar_pixel_grid`` uses ``rlims = (zlims[0], zlims[1] + distance_to_apex)``,
    so ``zlims[0]`` is a radius from the centre of curvature while ``zlims[1]`` is a
    depth from ``z = 0``. ``r_min`` is the shallowest depth below the array surface.
    """
    r_max = r_min + n_r * dr
    return {
        **GRID_PARAMETERS,
        "distance_to_apex": probe_radius,
        "zlims": (probe_radius + r_min, r_max),
    }


def build_pipeline() -> zea.Pipeline:
    """RF channel data -> log-compressed B-mode."""
    return Pipeline(
        [
            Cast(dtype="float32"),
            Demodulate(),
            Beamform(beamformer="delay_and_sum"),
            EnvelopeDetect(),
            Normalize(),
            LogCompress(),
        ],
    )


def reconstruct(
    zea_path: Path, pipeline: zea.Pipeline | None = None, frame: int = 0
) -> tuple[np.ndarray, zea.Parameters]:
    """Return (bmode_db, parameters) for one frame of a zea file."""
    zea.init_device()
    pipeline = pipeline or build_pipeline()

    with zea.File(str(zea_path)) as f:
        geom = np.asarray(f.probe.probe_geometry)
        params = f.load_parameters(**grid_parameters(_apex_radius(geom)))
        raw = f.data.raw_data[:]

    inputs = pipeline.prepare_parameters(params)
    outputs = pipeline(**{pipeline.key: raw}, **inputs, return_numpy=True)
    return outputs[pipeline.output_key][frame], params


def _apex_radius(geom: np.ndarray) -> float:
    """Recover the array radius from element positions (apex at z = 0).

    Elements lie on an arc of radius R centred at (0, 0, -R), so
    ``x^2 + (z + R)^2 = R^2``  =>  ``x^2 + z^2 + 2 R z = 0``. Solved as a
    least-squares fit over all elements rather than per-element, which is
    ill-conditioned for the elements nearest the apex (z -> 0).
    """
    x, z = geom[:, 0].astype(np.float64), geom[:, 2].astype(np.float64)
    return float(-np.sum(z * (x**2 + z**2)) / (2.0 * np.sum(z**2)))


def main() -> None:
    """CLI entry point: reconstruct a B-mode and optionally save the pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zea_file", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("bmode.png"))
    parser.add_argument("--save-yaml", type=Path, default=None)
    args = parser.parse_args()

    pipeline = build_pipeline()
    bmode, _ = reconstruct(args.zea_file, pipeline=pipeline, frame=args.frame)

    if args.save_yaml is not None:
        config = pipeline.to_config()
        with zea.File(str(args.zea_file)) as f:
            geom = np.asarray(f.probe.probe_geometry)
        config["parameters"] = grid_parameters(_apex_radius(geom))
        config.to_yaml(str(args.save_yaml))
        print(f"Saved pipeline to {args.save_yaml}")

    with zea.File(str(args.zea_file)) as f:
        geom = np.asarray(f.probe.probe_geometry)
    grid = grid_parameters(_apex_radius(geom))
    radius = grid["distance_to_apex"]

    # Scan-convert the polar image into a physical sector. rho is measured from the
    # centre of curvature, so the displayed depth axis is rho - radius.
    # rho is measured from the centre of curvature; work in mm so the returned
    # Cartesian limits come back in mm too.
    rho_range = ((grid["zlims"][0]) * 1e3, (grid["zlims"][1] + radius) * 1e3)
    sector, sc = scan_convert_2d(
        bmode,
        rho_range=rho_range,
        theta_range=grid["polar_limits"],
        fill_value=-60.0,
        distance_to_apex=0.0,
    )
    sector = keras.ops.convert_to_numpy(sector)
    xlims = [float(v) for v in keras.ops.convert_to_numpy(sc["x_lim"])]
    zlims = [float(v) for v in keras.ops.convert_to_numpy(sc["z_lim"])]
    # Shift the depth axis so 0 mm is the array apex rather than the centre of curvature.
    extent = (xlims[0], xlims[1], zlims[1] - radius * 1e3, zlims[0] - radius * 1e3)

    fig, ax = plt.subplots(figsize=(6.5, 7))
    im = ax.imshow(sector, cmap="gray", vmin=-60, vmax=0, extent=extent)
    ax.set_title(f"Fullwave abdominal wall B-mode (DAS), frame {args.frame}")
    ax.set_xlabel("Lateral position [mm]")
    ax.set_ylabel("Depth [mm]")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved reconstruction to {args.out}")


if __name__ == "__main__":
    main()
