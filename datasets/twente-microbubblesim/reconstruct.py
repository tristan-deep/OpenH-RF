# SPDX-License-Identifier: Apache-2.0
"""Example reconstruction script for the twente-microbubblesim dataset of OpenH-RF.

Dataset link: https://huggingface.co/datasets/nvidia/OpenH-RF/tree/main/twente-microbubblesim

B-mode reconstruction of simulated plane-wave microbubble channel data.

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

# backend library so CUDA devices are not visible during device discovery.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import zea
from utils import (
    bubble_coordinates_cm,
    image_extent_cm,
    plot_bmode,
    run_bmode,
)

HERE = Path(__file__).resolve().parent

PULSE_NAMES = (
    "DPT",
    "L1.7",
    "L2.5",
    "L3.4",
    "LDC",
    "LUC",
    "REF",
    "S1.7",
    "S2.5",
    "S3.4",
    "SDC",
    "SUC",
)
PULSE_TO_TRACK = {pulse: index for index, pulse in enumerate(PULSE_NAMES)}


# --- Inputs -----------------------------------------------------------------
# Defaults stream straight from the published corpus. Swap any of these for a
# local path to run against your own copy.
PULSE = "REF"  # Pulse label to reconstruct, for example REF, DPT, or L1.7
PATH = "hf://nvidia/OpenH-RF/twente-microbubblesim/data/Monodispers/RFDATA00002.hdf5"
# Pipeline YAML to load; otherwise the selected track's saved pipeline
CONFIG_PATH = f"hf://nvidia/OpenH-RF/twente-microbubblesim/pipeline/pipeline_track_{PULSE_TO_TRACK[PULSE]}_{PULSE}.yaml"
# (must match the track in CONFIG_PATH; track 0 is DPT)
OUT = None  # Output PNG filename + Path; otherwise uses population and pulse names
DYNAMIC_RANGE = (-30.0, 0.0)  # Display dynamic range in dB
SHOW_BUBBLES = True  # Draw the red bubble ground-truth overlay


def population_name(path: Path) -> str:
    """Return the population name encoded in a dataset path."""

    parts = [part.lower() for part in path.parts]
    if any("sonovue" in part for part in parts):
        return "SonoVue"
    if any("monodispers" in part for part in parts):
        return "Monodispers"
    return "OpenH-RF"


def default_config_path(pulse: str) -> Path:
    """Return the saved pipeline for the selected pulse track."""

    track_index = PULSE_TO_TRACK[pulse]
    return HERE / "pipeline" / f"pipeline_track_{track_index}_{pulse}.yaml"


def default_output_path(path: Path, pulse: str) -> Path:
    population = population_name(path)
    return HERE / "output" / f"{population}_{pulse}.png"


def main() -> None:
    zea.init_device("cpu")
    zea.visualize.set_mpl_style()

    config_path = str(CONFIG_PATH) if CONFIG_PATH is not None else default_config_path(PULSE)
    input_path = str(PATH)
    if not input_path.lower().endswith(".hdf5"):
        raise ValueError("PATH must point directly to a .hdf5 acquisition file")
    track_index = PULSE_TO_TRACK[PULSE]
    output_path = Path(OUT) if OUT else default_output_path(Path(input_path), pulse=PULSE)

    file, parameters, image, custom = run_bmode(
        input_path,
        config_path,
        track_index=track_index,
        dynamic_range=tuple(DYNAMIC_RANGE),
        xlims_cm=(-1.4, 1.4),
    )
    try:
        extent_cm = image_extent_cm(parameters, custom, image.shape, xlims_cm=(-1.4, 1.4))
        bubble_x_cm, bubble_z_cm = bubble_coordinates_cm(custom)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plot_bmode(
            image,
            extent_cm,
            output_path,
            bubble_x_cm=bubble_x_cm,
            bubble_z_cm=bubble_z_cm,
            show_bubbles=SHOW_BUBBLES,
            title=(f"{population_name(Path(input_path))} — {PULSE} — {Path(input_path).stem}"),
        )
    finally:
        file.close()
    print("Successfully saved image to", output_path)


if __name__ == "__main__":
    main()
