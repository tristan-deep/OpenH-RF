"""Reconstruct: beamform one rotation frame of the eSAF rotational 3D US dataset.

Loads a zea .hdf5 acquisition (raw RF channel data, one normal plane-wave
transmit per rotation frame), reads the acquisition parameters, and runs the
delay-and-sum beamforming pipeline configured in pipeline.yaml. The resulting
B-mode image is saved as a PNG, together with a plot of the per-frame probe
rotation angle (this dataset's special data: the linear array rotates 180
degrees about its axial axis, 1 degree per frame) so downstream users know how
to interpret the frame axis.

Usage:
    python reconstruct.py
    python reconstruct.py --input data/baseline_R45_H8__point_z080_r4.hdf5
    python reconstruct.py --frame-index 45
"""

import os

# Default to the jax backend so the script runs without exporting KERAS_BACKEND.
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import zea
from zea import Config, File, Pipeline

HERE = Path(__file__).parent
DEFAULT_INPUT = HERE / "data" / "coarse_pitch_0p3__point_z080_r4.hdf5"
DEFAULT_OUTPUT = HERE / "outputs" / "reconstruct_example.png"
CONFIG = HERE / "pipeline.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="rotation frame to beamform (default: frame closest to +-90 deg "
        "rotation magnitude, where an off-axis target lies in the imaging plane; "
        "works for signed encoder angles too, e.g. measured scans that run 0 -> -180)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found.")

    zea.init_device()

    # Load the whole processing chain + parameters from pipeline.yaml
    config = Config.from_path(str(CONFIG))

    # Load file: acquisition parameters (with config overrides), one raw RF frame,
    # and this dataset's special data — the probe rotation angle per frame.
    # Files are single-track (channel RF + scan); the paired eSAF label volume
    # lives in the custom group (custom/saf_bmode/{values,coordinates}, via f.custom).
    with File(str(args.input)) as f:
        parameters = f.load_parameters(**config.parameters)
        # Rotation angle per frame lives in metadata/probe_pose (euler_xyz [rad],
        # nominal 1 Hz sampling — one pose per rotation frame): the array rotates
        # about its axial (z) axis, so the angle is the z Euler component.
        rotation_angles_deg = np.degrees(np.asarray(f.metadata.probe_pose.rotation)[:, 2]).ravel()
        n_frames = f.data.raw_data.shape[0]
        frame = args.frame_index
        if frame is None:  # default: frame where the probe has rotated ~90 deg
            # Select on rotation *magnitude* so signed encoder angles work too
            # (measured scans run 0 -> ~-180 deg; a plain |angle - 90| would pick
            # frame 0 for those). |abs(angle) - 90| picks the ~90 deg frame either way.
            frame = int(np.argmin(np.abs(np.abs(rotation_angles_deg) - 90.0)))
        raw = f.data.raw_data[frame : frame + 1]  # (1, n_tx, n_ax, n_el, 1) — RF

    print(f"raw_data frame : {raw.shape} ({n_frames} frames in file)")
    print(
        f"selected frame : {frame} @ {rotation_angles_deg[frame]:.1f} deg "
        f"(|rotation| {abs(rotation_angles_deg[frame]):.1f} deg)"
    )

    # Build and run the beamforming pipeline defined in pipeline.yaml
    pipeline = Pipeline.from_config(config)
    inputs = pipeline.prepare_parameters(parameters)
    outputs = pipeline(data=raw, **inputs)

    recon = np.array(outputs["data"])  # (1, grid_z, grid_x), log-compressed dB
    image = zea.display.to_8bit(recon[0], dynamic_range=parameters.dynamic_range, pillow=False)

    # B-mode + rotation-angle plot (how to interpret the frame axis)
    zea.visualize.set_mpl_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.3]})

    mm = plt.FuncFormatter(lambda v, _: f"{v * 1e3:.0f}")
    axes[0].imshow(image, extent=parameters.extent_imshow, cmap="gray", aspect="equal")
    axes[0].xaxis.set_major_formatter(mm)
    axes[0].yaxis.set_major_formatter(mm)
    axes[0].set(
        title=f"B-mode frame {frame} @ {rotation_angles_deg[frame]:.0f} deg",
        xlabel="Lateral [mm]",
        ylabel="Depth [mm]",
    )

    axes[1].plot(np.arange(n_frames), rotation_angles_deg, color="tab:blue")
    axes[1].axvline(frame, color="tab:red", linewidth=1.2, label="beamformed frame")
    axes[1].set(
        title="Probe rotation angle per frame\n(array rotates about its axial axis)",
        xlabel="Frame index",
        ylabel="Rotation angle [deg]",
    )
    axes[1].grid(True, color="0.85", linewidth=0.8)
    axes[1].legend(loc="upper left", fontsize=8, frameon=False)

    fig.suptitle(f"OpenH-RF eSAF rotational 3D US — {args.input.name}", y=0.98)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(args.output), bbox_inches="tight", dpi=150)

    print(f"Reconstructed  : {recon.shape}")
    print(f"Saved          : {args.output}")


if __name__ == "__main__":
    main()
