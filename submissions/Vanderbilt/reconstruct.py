"""Reconstruct: beamform the CIRS phantom, fundamental cardiac data, and harmonic cardiac data using DAS.

Defines a delay-and-sum beamforming pipeline in code, saves it (together with some
beamforming parameters) to pipeline.yaml, then loads that YAML back and runs it on
the HDF5 file. The resulting B-mode image is saved as a PNG file in the same input directory.
All data can be reconstructed with the same pipeline.

Each dataset contains 32 frames which can be specified with the n_frames argument. To beamform just the first frame,
set number_of_frames to 1 (default). To beamform the whole cineloop, set it to 32.

Usage:
    python path/to/reconstruct.py --input my_file.hdf5 --n_frames number_of_frames
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from zea.ops import (
    Beamform,
    Cast,
    Demodulate,
    EnvelopeDetect,
    LogCompress,
    Normalize,
    ScanConvert,
)

import zea
from zea import Config, File, Pipeline

HERE = Path(__file__).parent
CONFIG = HERE / "pipeline.yaml"

# P4-2v is a phased array (sector scan), so beamform on a polar grid and
# scan convert to Cartesian for display, rather than beamforming directly
# on a Cartesian grid. polar_limits is pinned to the actual transmit angle
# range (+/-45 deg)
PARAMETERS = {
    "grid_type": "polar",
    "polar_limits": [-np.pi / 4, np.pi / 4],
    "grid_size_x": 1084,
    "grid_size_z": 636,
    "dynamic_range": [-60, 0],
    "zlims": [0, 0.18],
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
            ScanConvert(),
        ],
    )


def write_config(pipeline: Pipeline, path: Path) -> None:
    """Serialize the pipeline and acquisition parameters to a YAML config file."""
    config = pipeline.to_config()
    config["parameters"] = PARAMETERS
    config.to_yaml(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--n_frames", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device ID (e.g. 'cuda:0', 'auto:1', or 'cpu')",
    )
    args = parser.parse_args()
    
    input_folder, input_filename = os.path.split(args.input)
    output_path = os.path.join(input_folder, input_filename.replace('hdf5', 'png'))

    zea.init_device(device=args.device, verbose=False)

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found. Run convert.py first.")

    # Define the beamforming pipeline in code, save it (with the acquisition
    # parameters) to pipeline.yaml, then load that YAML back in.
    write_config(build_pipeline(), CONFIG)
    config = Config.from_path(str(CONFIG))

    frames = list(range(args.n_frames))
    # Load file: read acquisition parameters (with config overrides) and raw RF data
    with File(str(args.input)) as f:
        parameters = f.load_parameters(**config.parameters)

        # Only grab and beamform the first frame
        raw = f.data.raw_data[frames, ...]  # (n_frames, n_tx, n_ax, n_el, 1) — RF

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
    plt.savefig(str(output_path), bbox_inches="tight", dpi=100)

    print(f"Reconstructed  : {recon.shape}")
    print(f"Saved          : {output_path}")


if __name__ == "__main__":
    main()
