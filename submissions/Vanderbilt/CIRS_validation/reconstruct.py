"""Reconstruct: beamform the Verasonics plane-wave phantom using DAS.

Defines a delay-and-sum beamforming pipeline in code, saves it (together with some
beamforming parameters) to pipeline.yaml, then loads that YAML back and runs it on
the HDF5 file created by convert.py. The resulting B-mode image is saved as a PNG file.

Usage:
    python examples/templates/verasonics/reconstruct.py
    python examples/templates/verasonics/reconstruct.py --input my_file.hdf5
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from zea.ops import (
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

DEFAULT_INPUT = HERE / "CIRS_sample.hdf5"
DEFAULT_OUTPUT = HERE / "CIRS_bmode_final.png"
CONFIG = HERE / "pipeline.yaml"

PARAMETERS = {
    "grid_size_x": 1084,
    "grid_size_z": 636,
    "dynamic_range": [-80, 0],
    "zlims": [0, 0.18],
    "xlims": [-0.08, 0.06],
    "apply_lens_correction": False,
    "f_number": 0,
}


def build_pipeline() -> Pipeline:
    """Define the delay-and-sum beamforming pipeline in code."""
    return Pipeline(
        operations=[
            Cast(dtype="float32"),
            Demodulate(),
            Beamform(
                beamformer="delay_and_sum",
                num_patches=400,
                enable_pfield=True,
            ),
            EnvelopeDetect(),
            Normalize(),
            LogCompress(),
        ],
    )


def write_config(pipeline: Pipeline, path: Path) -> None:
    """Serialize the pipeline and acquisition parameters to a YAML config file."""
    config = pipeline.to_config()
    config["parameters"] = PARAMETERS
    config.to_yaml(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device ID (e.g. 'cuda:0', 'auto:1', or 'cpu')",
    )
    args = parser.parse_args()

    zea.init_device(device=args.device, verbose=False)

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found. Run convert.py first.")

    # Define the beamforming pipeline in code, save it (with the acquisition
    # parameters) to pipeline.yaml, then load that YAML back in.
    write_config(build_pipeline(), CONFIG)
    config = Config.from_path(str(CONFIG))

    # Load file: read acquisition parameters (with config overrides) and raw RF data
    with File(str(args.input)) as f:
        parameters = f.load_parameters(**config.parameters)

        # Only grab and beamform the first frame
        raw = f.data.raw_data[0:1, ...]  # (n_frames, n_tx, n_ax, n_el, 1) — RF

    # Build and run the beamforming pipeline loaded from pipeline.yaml
    pipeline = Pipeline.from_config(config)
    inputs = pipeline.prepare_parameters(parameters)

    outputs = pipeline(data=raw, **inputs, return_numpy=True)

    # Convert the output tensor to a NumPy array and save as PNG
    recon = outputs["data"]  # (n_frames, grid_z, grid_x)
    image = zea.display.to_8bit(recon[0], dynamic_range=parameters.dynamic_range)

    zea.visualize.set_mpl_style()
    plt.imshow(
        image,
        extent=parameters.extent_imshow,
        cmap="gray",
    )
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.savefig(str(args.output), bbox_inches="tight", dpi=100)

    print(f"Reconstructed  : {recon.shape}")
    print(f"Saved          : {args.output}")


if __name__ == "__main__":
    main()
