# SPDX-License-Identifier: Apache-2.0
"""Reference reconstruction for the UW-MuscleRF dataset.

Defines a delay-and-sum beamforming pipeline in code (DAS -> envelope ->
normalize -> log-compress), saves it to pipeline.yaml, and reconstructs a B-mode
directly from the raw channel data. The reconstruction is shown next to the
stored B-mode as a sanity check on the recorded acquisition parameters.

Usage:
    python reconstruct.py --input hdf5/Acq_p64_Calf_Left_calf_Inside_Pressure.hdf5 --frame 9

@ LITMUS Research Group, University of Waterloo, 2026.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from zea import File, Pipeline

HERE = Path(__file__).parent
DEFAULT_INPUT = HERE / "data" / "Acq_p64_Calf_Left_calf_Inside_Pressure.hdf5"
DEFAULT_PIPELINE = HERE / "pipeline.yaml"
DEFAULT_OUTPUT = HERE / "reconstruct_output.png"


def build_pipeline() -> Pipeline:
    """Define the delay-and-sum B-mode pipeline in code."""
    return Pipeline(
        operations=[
            Cast(dtype="float32"),
            BandPassFilter(passband=(3e6, 7e6)),
            Demodulate(),
            Beamform(beamformer="delay_and_sum", num_patches=1000),
            EnvelopeDetect(),
            Normalize(),
            LogCompress(),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--frame", type=int, default=9)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found. Run convert.py first.")

    zea.init_device()

    pipeline = build_pipeline()
    pipeline.to_yaml(str(args.pipeline))

    with File(str(args.input)) as f:
        frame = min(max(0, args.frame), f.data.image.values.shape[0] - 1)

        raw = f.data.raw_data[frame : frame + 1]  # (1, n_tx, n_ax, n_el, 1)
        stored = f.data.image.values[frame]  # stored B-mode (dB)

        # Reconstruct on the same grid as the stored B-mode so the panels line up.
        coords = f.data.image.coordinates[:]  # (z, x, 3), last axis [x, y, z] in metres
        parameters = f.load_parameters(
            grid_size_z=coords.shape[0],
            grid_size_x=coords.shape[1],
            xlims=[float(coords[..., 0].min()), float(coords[..., 0].max())],
            zlims=[float(coords[..., 2].min()), float(coords[..., 2].max())],
        )

    inputs = pipeline.prepare_parameters(parameters)
    recon = pipeline(data=raw, **inputs, return_numpy=True)["data"][0]
    extent = [v * 1e3 for v in parameters.extent_imshow]  # metres -> mm

    zea.visualize.set_mpl_style()
    # Size each panel to the image aspect ratio so the axes hug the B-mode.
    panel_h = 5.5
    img_aspect = (extent[1] - extent[0]) / (extent[2] - extent[3])
    fig, axes = plt.subplots(
        1, 2, figsize=(panel_h * img_aspect * 2, panel_h), constrained_layout=True
    )
    imshow_kw = dict(cmap="gray", vmin=-60, vmax=0, extent=extent)

    axes[0].imshow(stored, **imshow_kw)
    axes[0].set_title("B-mode (litmus)")
    axes[1].imshow(recon, **imshow_kw)
    axes[1].set_title("B-mode (zea)")

    for ax in axes:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(f"UW-MuscleRF — frame {frame}")
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
