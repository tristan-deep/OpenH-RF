#!/usr/bin/env python3
"""Reconstruct B-mode and inspect the embedded QUS targets in a zea HDF5 file.

Defines a delay-and-sum beamforming pipeline in code, saves it (together with
the reconstruction parameters) to pipeline.yaml, then loads that YAML back and
runs it on the requested acquisition. The resulting B-mode image is saved as a
PNG, and the embedded QUS targets (BSC, Nakagami) for the same frame are
printed alongside it.

Usage:
    python reconstruct.py --input data/ac1_15m_SK.hdf5 --frame 0
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from zea.ops import Beamform, Cast, Demodulate, EnvelopeDetect, LogCompress, Normalize

import zea
from zea import Config, File, Pipeline

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "pipeline.yaml"
DEFAULT_INPUT = HERE / "data" / "ac1_15m_SK.hdf5"
DEFAULT_OUTPUT = HERE / "ac1_15m_SK_pipeline.png"

# Common shallow reference grid used for every acquisition.
PARAMETERS = {
    "grid_size_x": 300,
    "grid_size_z": 597,
    "xlims": [-0.021965, 0.021965],
    "zlims": [0.0029568189236411113, 0.02498497892364108],
    "dynamic_range": [-60, 0],
}


def build_pipeline() -> Pipeline:
    """Define the delay-and-sum B-mode pipeline in code."""
    return Pipeline(
        operations=[
            Cast(dtype="float32"),
            Demodulate(),
            Beamform(beamformer="delay_and_sum", num_patches=100),
            EnvelopeDetect(),
            Normalize(output_range=[0.0, 1.0]),
            LogCompress(),
        ],
    )


def write_config(pipeline: Pipeline, path: Path) -> None:
    """Serialize the pipeline and reconstruction parameters to pipeline.yaml."""
    config = pipeline.to_config()
    config["parameters"] = PARAMETERS
    config.to_yaml(str(path))


def reconstruct_one(
    zea_file: Path, config: Config, frame: int, device: str | None
) -> tuple[np.ndarray, object]:
    zea.init_device(device=device, verbose=False)
    pipeline = Pipeline.from_config(config)

    with File(str(zea_file)) as f:
        frame_count = f.data.raw_data.shape[0]
        if not 0 <= frame < frame_count:
            raise ValueError(f"--frame must be in [0, {frame_count - 1}]")
        parameters = f.load_parameters(**config.parameters)
        raw = f.data.raw_data[frame : frame + 1]

    inputs = pipeline.prepare_parameters(parameters)
    outputs = pipeline(data=raw, **inputs, return_numpy=True)
    bmode = np.asarray(outputs["data"][0])
    return bmode, parameters


def save_png(image_db: np.ndarray, parameters, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extent_mm = [float(v) * 1e3 for v in parameters.extent_imshow]
    fov_ratio = abs(extent_mm[2] - extent_mm[3]) / abs(extent_mm[1] - extent_mm[0])
    fig, ax = plt.subplots(figsize=(6.2, 2.0 + 4.8 * fov_ratio), constrained_layout=True)
    im = ax.imshow(
        image_db,
        cmap="gray",
        vmin=float(parameters.dynamic_range[0]),
        vmax=float(parameters.dynamic_range[1]),
        extent=extent_mm,
        aspect="equal",
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Lateral position (mm)")
    ax.set_ylabel("Axial depth (mm)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Envelope (dB)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def summarize_qus(zea_file: Path, frame: int) -> None:
    with File(str(zea_file)) as file:
        data = file.data
        raw_frame = data.raw_data[frame, 0, :, :, 0]
        bsc = data.theoretical_bsc.values[frame]
        frequency_labels = data.theoretical_bsc.labels[...]
        frequencies_hz = np.asarray(
            [float(label.removeprefix("frequency_hz=")) for label in frequency_labels]
        )
        m_frame = data.nakagami_m_per_frame.values[frame]
        m_pooled = data.nakagami_m_pooled.values[frame]
        omega_frame = data.nakagami_omega_per_frame.values[frame]
        omega_pooled = data.nakagami_omega_pooled.values[frame]
        coordinates_shape = data.nakagami_m_pooled.coordinates.shape
        bsc_unit = data.theoretical_bsc.unit[()]
        m_unit = data.nakagami_m_pooled.unit[()]
        omega_unit = data.nakagami_omega_pooled.unit[()]

    print("QUS targets")
    print(f"  RF frame       : shape={raw_frame.shape}, dtype={raw_frame.dtype}")
    print(f"  QUS grid       : shape={m_pooled.shape}, coordinates={coordinates_shape} [x,y,z] m")
    print(
        f"  Theoretical BSC: shape={bsc.shape}, unit={bsc_unit}, "
        f"band={frequencies_hz[0] * 1e-6:.6f}-{frequencies_hz[-1] * 1e-6:.6f} MHz"
    )

    for name, direct, pooled, unit in (
        ("Nakagami m", m_frame, m_pooled, m_unit),
        ("Nakagami omega", omega_frame, omega_pooled, omega_unit),
    ):
        error = direct - pooled
        print(
            f"  {name:<15}: unit={unit}, frame-to-pooled "
            f"MAE={np.mean(np.abs(error)):.6g}, "
            f"RMSE={np.sqrt(np.mean(error**2)):.6g}, "
            f"correlation={correlation(direct, pooled):.6f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="Input OpenH-RF/zea HDF5 file."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to reconstruct.")
    parser.add_argument(
        "--device", type=str, default=None, help="Optional zea device, e.g. cpu, cuda:0, auto:0."
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    # Define the beamforming pipeline in code, save it (with the reconstruction
    # parameters) to pipeline.yaml, then load that YAML back in.
    write_config(build_pipeline(), CONFIG)
    config = Config.from_path(str(CONFIG))

    image_db, parameters = reconstruct_one(args.input, config, args.frame, args.device)
    save_png(image_db, parameters, args.output, f"{args.input.stem}, frame {args.frame}")

    print(f"Input          : {args.input}")
    print(f"Pipeline       : {CONFIG}")
    print(f"Output         : {args.output}")
    print(
        f"Reconstruction : shape={image_db.shape}, min={float(np.nanmin(image_db)):.2f}, max={float(np.nanmax(image_db)):.2f}"
    )
    summarize_qus(args.input, args.frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
