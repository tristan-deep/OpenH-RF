"""Reconstruct an OpenPros speed-of-sound map with a pretrained network.

This example reads the limited-view ultrasound waveform data written to the ZEA
HDF5 format by ``convert.py``, converts it back to the tensor layout used by the
OpenPros models, and applies the same signed-log and min-max preprocessing used
by the official OpenPros implementation. The normalized data are passed through
the pretrained ``zea`` InversionNet model, whose normalized prediction is mapped
back to the physical speed-of-sound range (1300--3600 m/s). Finally, the script
plots the predicted and ground-truth SOS maps side by side and saves the figure.

The pipeline can be defined directly in Python, serialized to ``pipeline.yaml``,
or restored from that YAML file with ``--load_config``.

Usage:
    python reconstruct.py
"""

import argparse
import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "jax"

import keras
import matplotlib.pyplot as plt
from custom_ops import LogTransform, MyRearrange
from network_ops import InversionNetInference
from zea.ops import Normalize

import zea
from zea import Config, File, Pipeline

HERE = Path(__file__).parent
INPUT = HERE / "openpros_sample.hdf5"
CONFIG = HERE / "pipeline.yaml"
OUTPUT = HERE / "pred_sos.png"


def plot_comparison(sos, pred, path):
    _, ax = plt.subplots(1, 2, figsize=(7, 6))
    im = ax[0].imshow(sos[0, :, :, 0], cmap="gray", vmin=1300, vmax=1700)
    ax[0].set_title("Ground Truth SOS Map")
    ax[1].imshow(keras.ops.convert_to_numpy(pred)[0, :, :, 0], cmap="gray", vmin=1300, vmax=1700)
    ax[1].set_title("Predicted SOS Map")
    for axis in ax:
        axis.set_xlabel("X (mm)")
        axis.set_xticks(range(0, 161, 40), labels=range(0, 61, 15))
    ax[0].set_ylabel("Z (mm)")
    ax[0].set_yticks(range(0, 401, 80), labels=range(0, 151, 30))
    ax[1].set_yticks(range(0, 401, 80), [])
    plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.2, pad=0.04, label="SOS (m/s)")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write_config",
        action="store_true",
        help="Write the pipeline and parameters to a YAML config file.",
    )
    parser.add_argument(
        "--load_config",
        action="store_true",
        help="Load the pipeline and parameters from a YAML config file.",
    )
    parser.add_argument(
        "--use_zea_vis_style",
        action="store_true",
        help="Use ZEA's visualization style.",
    )
    args = parser.parse_args()

    zea.init_device(verbose=False)

    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found. Run convert.py first.")

    if args.load_config:
        config = Config.from_path(str(CONFIG))
        pipeline = Pipeline.from_config(config)
    else:
        # Match the official OpenPros inference preprocessing and postprocessing:
        # restore the original four acquisition blocks (SS, SR, RR, RS), apply
        # the sign-preserving log1p transform with k=1e5, and min-max normalize
        # its configured data range to [-1, 1]. After network inference, undo
        # the label normalization by mapping the prediction from [-1, 1] to the
        # physical SOS range of 1300--3600 m/s.
        pipeline = Pipeline(
            operations=[
                MyRearrange(),  # rearrange data to the layout expected by the network
                LogTransform(data_min=-0.25, data_max=0.45, k=1e5),
                Normalize(output_range=(-1, 1)),
                InversionNetInference(preset="inversionnet-openpros"),
                Normalize(input_range=(-1, 1), output_range=(1300, 3600)),
            ]
        )

    if args.write_config:
        config = pipeline.to_config()
        print(config)
        config.to_yaml(str(CONFIG))
        print(f"Wrote pipeline and parameters to {CONFIG}")

    with File(INPUT) as f:
        raw = f.data.raw_data[:]
        sos = f.data.sos_map.values[:]  # gt

    print(f"raw_data shape: {raw.shape}")
    print(f"ground truth shape: {sos.shape}")
    outputs = pipeline(data=raw)["data"]
    print(f"reconstructed shape: {outputs.shape}")
    if args.use_zea_vis_style:
        zea.visualize.set_mpl_style()
    plot_comparison(sos, outputs, OUTPUT)
    print(f"Saved comparison plot: {OUTPUT}")


if __name__ == "__main__":
    main()
