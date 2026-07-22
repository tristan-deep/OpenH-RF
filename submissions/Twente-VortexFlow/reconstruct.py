"""Reconstruct B-mode images from the OpenH-RF flow-phantom dataset.

Loads the pipeline definition from pipeline_<track>.yaml (one per track),
runs it on the specified zea HDF5 file, and saves a PNG side-by-side of all
tracks.

Usage:
    python reconstruct.py
    python reconstruct.py --input AcqData_PVoltage80_TVoltage3.4.hdf5
    python reconstruct.py --input AcqData_PVoltage120_TVoltage7.1.hdf5 --frame 20
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

import zea
from zea.ops import Beamform, Cast, Demodulate, EnvelopeDetect, LogCompress, Normalize

HERE = Path(__file__).parent

# Mapping from track label to the pipeline YAML shipped with this submission.
PIPELINE_YAMLS = {
    "short imaging pulse": HERE / "pipeline_short_imaging_pulse.yaml",
    "chirp":               HERE / "pipeline_chirp.yaml",
}

# Frame index used for the reference reconstruction.
DEFAULT_FRAME = 10
DEFAULT_XLIMS = (-0.08, 0.08)
DEFAULT_ZLIMS = (0.05, 0.10)
DEFAULT_GRID_SIZE_Z = 480


def build_pipeline() -> zea.Pipeline:
    """Build the default DAS B-mode pipeline.

    Chain: Cast → Demodulate → Beamform (DAS) → EnvelopeDetect → Normalize → LogCompress

    - Cast: converts raw_data to float32 (required before any float ops).
    - Demodulate: required for RF data (n_ch=1); no-op for IQ (n_ch=2).
    - Beamform: delay-and-sum with 100 patches for memory efficiency.
    - EnvelopeDetect / Normalize / LogCompress: standard B-mode display chain.
    """
    return zea.Pipeline(
        operations=[
            Cast(dtype="float32"),
            Demodulate(),
            Beamform(beamformer="delay_and_sum", num_patches=100),
            EnvelopeDetect(),
            Normalize(),
            LogCompress(),
        ],
        validate=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=HERE / "AcqData_PVoltage80_TVoltage3.4.hdf5",
        help="Path to a submission .hdf5 file (default: AcqData_PVoltage80_TVoltage3.4.hdf5)",
    )
    parser.add_argument(
        "--frame", type=int, default=DEFAULT_FRAME,
        help="Frame index to reconstruct (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "reference_bmode.png",
        help="Output PNG path (default: %(default)s)",
    )
    parser.add_argument(
        "--output-2x1", type=Path, default=HERE / "reference_mapping.png",
        help="Output PNG path for 2x1 paired view (default: %(default)s)",
    )
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else HERE / args.input
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            f"Pass --input with a file inside {HERE}."
        )

    zea.init_device()

    with zea.File(str(input_path)) as f:
        tracks = list(f.tracks)
        n_tracks = len(tracks)
        fig, axes = plt.subplots(n_tracks, 1, figsize=(6 * n_tracks, 6))
        if n_tracks == 1:
            axes = [axes]
        track_panels = []

        for ax, track in zip(axes, tracks):
            label = track.label

            # Load or build the pipeline for this track.
            yaml_path = PIPELINE_YAMLS.get(label)
            if yaml_path and yaml_path.exists():
                # Load-and-run: use the saved YAML shipped with the submission.
                pipeline = zea.Pipeline.from_path(str(yaml_path))
            else:
                # Fallback: build the default pipeline in code.
                pipeline = build_pipeline()

            # Reconstruct one frame.
            params = track.load_parameters()
            # Apply fixed image limits before parameter prep so beamforming uses this FOV.
            params.xlims = DEFAULT_XLIMS
            params.zlims = DEFAULT_ZLIMS
            # Enforce square reconstruction sampling (same physical pixel size in x and z).
            params.grid_size_z = DEFAULT_GRID_SIZE_Z
            x_span = params.xlims[1] - params.xlims[0]
            z_span = params.zlims[1] - params.zlims[0]
            params.grid_size_x = max(1, int(round(params.grid_size_z * x_span / z_span)))
            raw = track.data.raw_data[args.frame : args.frame + 1, ...]
            inputs = pipeline.prepare_parameters(params)
            outputs = pipeline(data=raw, **inputs)

            import keras
            recon = keras.ops.convert_to_numpy(outputs["data"])[0]
            extent_mm = [v * 1e3 for v in params.extent_imshow]

            image_frame = None
            image_group = getattr(track.data, "image", None)
            if image_group is not None and hasattr(image_group, "values"):
                image_values = image_group.values[args.frame : args.frame + 1, ...]
                image_np = keras.ops.convert_to_numpy(image_values)
                if image_np.shape[0] > 0:
                    image_frame = np.squeeze(image_np[0])
                    # Rotate to match ultrasound orientation.
                    image_frame = np.flipud(np.rot90(image_frame, 3))
            track_panels.append((label, recon, extent_mm, image_frame))

            ax.imshow(recon, cmap="gray", vmin=-50, vmax=0, extent=extent_mm, aspect="equal")
            ax.set_title(f"Track: {label}\nFile: {input_path.name}, Frame {args.frame}")
            ax.set_xlabel("Lateral [mm]")
            ax.set_ylabel("Depth [mm]")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved {args.output}")

    # Build requested 2x1 view: top = short imaging pulse ultrasound, bottom = matching camera image.
    if track_panels:
        selected_panel = None
        for label, recon, extent_mm, image_frame in track_panels:
            if label.strip().lower() == "short imaging pulse":
                selected_panel = (label, recon, extent_mm, image_frame)
                break
        if selected_panel is None:
            selected_panel = track_panels[0]

        label, recon, extent_mm, image_frame = selected_panel
        fig2, axes2 = plt.subplots(2, 1, figsize=(7, 10))

        axes2[0].imshow(recon, cmap="gray", vmin=-50, vmax=0, extent=extent_mm, aspect="equal")
        axes2[0].set_title(f"Ultrasound ({label}) - frame {args.frame}")
        axes2[0].set_xlabel("Lateral [mm]")
        axes2[0].set_ylabel("Depth [mm]")

        if image_frame is not None:
            axes2[1].imshow(image_frame, cmap="gray", aspect="equal")
            axes2[1].set_title(f"Camera image - frame {args.frame}")
            axes2[1].set_xlabel("X [px]")
            axes2[1].set_ylabel("Y [px]")
        else:
            axes2[1].text(0.5, 0.5, "No camera image available", ha="center", va="center")
            axes2[1].set_title("Camera image unavailable")
            axes2[1].set_axis_off()

        fig2.tight_layout()
        fig2.savefig(args.output_2x1, dpi=150)
        plt.close(fig2)
        print(f"Saved {args.output_2x1}")


if __name__ == "__main__":
    main()
