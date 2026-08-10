"""Beamform a robotic ultrasound OpenH-RF HDF5 sample and compare B-modes.

Renders the zea reconstruction next to the Verasonics VSX B-mode stored in the
file as `data/image`, which is the reference the reconstruction should match.

Each steering angle is acquired three times with the 64-element transmit and
receive aperture walked across the array, so a reconstruction has to compound
all 21 acquisitions to cover the full probe.

The tail of the pipeline reproduces the Verasonics display mapping, so the grey
levels are directly comparable to the stored B-mode.

The reconstruction parameters (f-number, lens correction, grid limits) are
recorded in pipeline.yaml under the ``parameters:`` key alongside the operation
chain, so the whole recipe is reproducible from that one file.

Usage:
    python reconstruct.py
    python reconstruct.py --input my_file.hdf5 --frame 100
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zea
from keras import ops
from zea.internal.core import DataTypes
from zea.internal.registry import ops_registry
from zea.ops import (
    BandPassFilter,
    Cast,
    DelayAndSum,
    Demodulate,
    EnvelopeDetect,
    Normalize,
    PatchedGrid,
    ReshapeGrid,
    TOFCorrection,
)
from zea.ops.base import Operation


HERE = Path(__file__).parent
DEFAULT_INPUT = HERE / "data/thyroid_phantom/longitudinal_scan_1.hdf5"


PARAMETERS = {
    "f_number": 1.155,
    "apply_lens_correction": True,
}

def coords_to_imshow_mm(coords):
    """openh-rf per-pixel coordinates (z, x, 3), last axis [x, y, z] in metres
    -> mpl imshow extent [left, right, bottom, top] in mm."""
    x = coords[..., 0]
    z = coords[..., 2]
    return [x.min() * 1e3, x.max() * 1e3, z.max() * 1e3, z.min() * 1e3]

@ops_registry("power_compress")
class PowerCompress(Operation):
    """Power (gamma) compression: raises the normalised envelope to ``exponent``.
    """

    def __init__(self, exponent=0.5, **kwargs):
        super().__init__(
            input_data_type=DataTypes.ENVELOPE_DATA,
            output_data_type=DataTypes.IMAGE,
            **kwargs,
        )
        self.exponent = exponent

    def call(self, **kwargs):
        data = kwargs[self.key]
        return {self.output_key: ops.power(data, self.exponent)}


def build_pipeline(passband):
    return zea.Pipeline(
        operations=[
            Cast(dtype="float32"),
            BandPassFilter(passband=passband),
            Demodulate(),
            zea.Pipeline(
                operations=[
                    PatchedGrid(
                        operations=[TOFCorrection(), DelayAndSum()],
                        num_patches=200,
                    ),
                    ReshapeGrid(),
                ],
                validate=True,
            ),
            EnvelopeDetect(),
            Normalize(),
            PowerCompress(exponent=0.5),
        ],
    )


def reconstruct_frame(f, frame, config_path):
    """Beamform one frame onto the axial extent of the stored B-mode.

    Compounds all transmits: each steering angle is acquired three times with
    the aperture walked across the array, so every acquisition is needed to
    cover the full probe.
    """
    n_tx = f.scan.polar_angles.shape[0]
    selected = list(range(n_tx))
    raw = np.asarray(f.data.raw_data[frame : frame + 1, selected]).copy()

    sound_speed = float(np.asarray(f.scan.sound_speed))
    initial_times = np.asarray(f.scan.initial_times, dtype=np.float64)
    start_depth = float(initial_times[0]) * sound_speed / 2

    display_coords = f.data.image.coordinates[:]
    end_depth = float(display_coords[..., 2].max())
    display_axial_spacing = display_coords[1, 0, 2] - display_coords[0, 0, 2]
    img_coords = display_coords[round(start_depth / display_axial_spacing) :].copy()
    img_coords[..., 2] = np.linspace(start_depth, end_depth, img_coords.shape[0])[:, None]

    parameters = {
        **PARAMETERS,
        "xlims": [float(img_coords[..., 0].min()), float(img_coords[..., 0].max())],
        "zlims": [float(img_coords[..., 2].min()), float(img_coords[..., 2].max())],
        "selected_transmits": selected,
    }
    params = f.load_parameters(**parameters)

    center_frequency = float(np.asarray(f.scan.center_frequency).reshape(-1)[0])
    bandwidth_fraction = float(np.asarray(f.probe.probe_bandwidth_percent).reshape(-1)[0]) / 100.0
    passband = (
        center_frequency * (1 - bandwidth_fraction / 2),
        center_frequency * (1 + bandwidth_fraction / 2),
    )

    pipeline = build_pipeline(passband)
    inputs = pipeline.prepare_parameters(params)
    generated = pipeline(
        data=raw,
        **inputs,
        return_numpy=True,
    )["data"][0]

    # to_config() only serialises the operation chain, so the reconstruction
    # parameters are attached explicitly before writing.
    config = pipeline.to_config()
    config["parameters"] = parameters
    config.to_yaml(str(config_path))

    # The grid starts at the imaging start depth, so pad the near field back on
    # to line the result up with the stored Verasonics B-mode.
    axial_spacing = (end_depth - start_depth) / (generated.shape[0] - 1)
    generated = np.pad(generated, ((round(start_depth / axial_spacing), 0), (0, 0)))
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--frame", type=int, default=0, help="Zero-based frame index to reconstruct.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found. Run convert.py first.")

    output = args.output or args.input.with_name(f"{args.input.stem}_reconstructed.png")
    config_path = args.input.with_name("pipeline.yaml")

    zea.init_device()

    with zea.File(str(args.input)) as f:
        display_coords = f.data.image.coordinates[:]
        verasonics = np.asarray(f.data.image.values[args.frame])
        generated = reconstruct_frame(f, args.frame, config_path)
        print(f"Reconstructed: {generated.shape}")

    extent = coords_to_imshow_mm(display_coords)
    fig, axes = plt.subplots(1, 2, figsize=(11, 8))
    axes[0].imshow(
        generated,
        aspect="auto",
        cmap="gray",
        extent=extent
    )
    axes[0].set_title("ZEA reconstruction")
    axes[0].set_xlabel("Lateral [mm]")
    axes[0].set_ylabel("Depth [mm]")

    axes[1].imshow(
        verasonics,
        aspect="auto",
        cmap="gray",
        extent=extent,
    )
    axes[1].set_title("Verasonics VSX B-mode")
    axes[1].set_xlabel("Lateral [mm]")

    fig.suptitle(f"Frame {args.frame} — {args.input.name}")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
