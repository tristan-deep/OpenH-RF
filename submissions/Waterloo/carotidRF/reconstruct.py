# SPDX-License-Identifier: Apache-2.0
"""Reference reconstruction for the UW-CarotidRF dataset.

Defines a delay-and-sum beamforming pipeline in code (DAS -> envelope ->
normalize -> log-compress) and, together with a display dynamic range, saves it
to pipeline.yaml. The B-mode is reconstructed directly from the raw channel data
and shown next to the stored (LITMUS) B-mode as a sanity check on the recorded
acquisition parameters.

The dynamic range lives in pipeline.yaml (`parameters.dynamic_range`); tweak it
there and it is picked up for display. If the vector-flow fields are present, a
third panel overlays the vector velocity field on the stored B-mode using
`draw_velocity_field` — a single self-contained (numpy + matplotlib) helper
reproduced below from the LITMUS core Python package
(`litmus.core_py.visualization`), so this script has no dependency on the full
LITMUS GPU stack.

Usage:
    python reconstruct.py --input hdf5/Acq0.hdf5 --frame 100

@ LITMUS Research Group, University of Waterloo, 2026.
"""

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from zea.ops import (
    BandPassFilter,
    Beamform,
    Cast,
    Demodulate,
    EnvelopeDetect,
    LogCompress,
    Normalize,
)

import zea
from zea import Config, File, Pipeline

HERE = Path(__file__).parent
DEFAULT_INPUT = HERE / "hdf5" / "Acq0.hdf5"
DEFAULT_PIPELINE = HERE / "pipeline.yaml"
DEFAULT_OUTPUT = HERE / "reconstruct_output.png"

DYNAMIC_RANGE = [-50, 0]  # dB; written to pipeline.yaml, tweak it there


def draw_velocity_field(ax, vx, vz, power, threshold, extent, density=0.02, cmap="jet", vmax=None):
    """Overlay power-Doppler-masked velocity vectors on `ax` as a quiver plot.

    Reproduced from the LITMUS core Python package
    (`litmus.core_py.visualization.vector_field`), LITMUS Research Group,
    University of Waterloo, 2026. `extent` is [left, right, bottom, top] in mm.
    """
    nz, nx = vx.shape
    x = np.linspace(extent[0], extent[1], nx)
    z = np.linspace(extent[3], extent[2], nz)
    X, Z = np.meshgrid(x, z)

    step = max(1, int(round(np.sqrt(1.0 / density))))
    U = vx[::step, ::step]
    V = -vz[::step, ::step]  # negate so +axial velocity points up on screen
    mag = np.sqrt(U**2 + V**2)

    invalid = (power[::step, ::step] < threshold) | np.isnan(mag)
    U = np.where(invalid, np.nan, U)
    V = np.where(invalid, np.nan, V)
    mag = np.where(invalid, np.nan, mag)

    scale = vmax if (vmax and vmax > 0) else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q = ax.quiver(
            X[::step, ::step],
            Z[::step, ::step],
            U,
            V,
            mag,
            cmap=cmap,
            angles="xy",
            scale_units="xy",
            scale=scale,
            pivot="middle",
            width=0.01,
        )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    return q


def build_config() -> Config:
    """Define the delay-and-sum B-mode pipeline in code, plus the display range."""
    config = Pipeline(
        operations=[
            Cast(dtype="float32"),
            BandPassFilter(passband=(3e6, 7e6)),
            Demodulate(),
            Beamform(beamformer="delay_and_sum", num_patches=1000),
            EnvelopeDetect(),
            Normalize(),
            LogCompress(),
        ],
    ).to_config()
    config["parameters"] = {"dynamic_range": DYNAMIC_RANGE}
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument(
        "--power-threshold",
        type=float,
        default=58.0,
        help="Power Doppler threshold for masking velocity vectors (dB)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found. Run convert.py first.")

    zea.init_device()

    # pipeline.yaml is the source of truth (pipeline + dynamic_range); (re)create
    # it only if missing so manual tweaks to the dynamic range are preserved.
    if args.pipeline.exists():
        config = Config.from_path(str(args.pipeline))
    else:
        config = build_config()
        config.to_yaml(str(args.pipeline))
    pipeline = Pipeline.from_config(config)

    with File(str(args.input)) as f:
        frame = min(max(0, args.frame), f.data.image.values.shape[0] - 1)

        raw = f.data.raw_data[frame : frame + 1]  # (1, n_tx, n_ax, n_el, 1)
        stored = f.data.image.values[frame]  # stored LITMUS B-mode (dB)

        # Reconstruct on the same grid as the stored B-mode so the panels line up.
        coords = f.data.image.coordinates[:]  # (z, x, 3), last axis [x, y, z] in metres
        parameters = f.load_parameters(
            **config.get("parameters", {}),
            grid_size_z=coords.shape[0],
            grid_size_x=coords.shape[1],
            xlims=[float(coords[..., 0].min()), float(coords[..., 0].max())],
            zlims=[float(coords[..., 2].min()), float(coords[..., 2].max())],
        )

        has_velocity = all(
            k in f.data for k in ("vector_velocity_x", "vector_velocity_z", "power_doppler")
        )
        if has_velocity:
            vx = f.data.vector_velocity_x.values[frame]
            vz = f.data.vector_velocity_z.values[frame]
            power = f.data.power_doppler.values[frame]

    inputs = pipeline.prepare_parameters(parameters)
    recon = pipeline(data=raw, **inputs, return_numpy=True)["data"][0]
    extent = [v * 1e3 for v in parameters.extent_imshow]  # metres -> mm
    vmin, vmax_db = parameters.dynamic_range

    zea.visualize.set_mpl_style()
    n_panels = 3 if has_velocity else 2
    # Size each panel to the image aspect ratio so the axes hug the B-mode.
    panel_h = 5.0
    img_aspect = (extent[1] - extent[0]) / (extent[2] - extent[3])
    fig, axes = plt.subplots(
        1, n_panels, figsize=(panel_h * img_aspect * n_panels, panel_h), constrained_layout=True
    )
    imshow_kw = dict(cmap="gray", vmin=vmin, vmax=vmax_db, extent=extent)

    axes[0].imshow(stored, **imshow_kw)
    axes[0].set_title("B-mode (litmus)")
    axes[1].imshow(recon, **imshow_kw)
    axes[1].set_title("B-mode (zea)")

    if has_velocity:
        mag = np.sqrt(vx**2 + vz**2)
        valid = (power >= args.power_threshold) & ~np.isnan(mag)
        v_scale = float(np.percentile(mag[valid], 99)) if np.any(valid) else 1.0
        axes[2].imshow(stored, **imshow_kw)
        q = draw_velocity_field(axes[2], vx, vz, power, args.power_threshold, extent, vmax=v_scale)
        q.set_clim(0, v_scale)
        axes[2].set_title("Vector flow overlay (litmus)")

    for ax in axes:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(f"UW-CarotidRF — frame {frame}")
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
