"""Reconstruct: beamform L11-5v thyroid patient scans (full session) using DAS.

Adapted from ``thyroid_test/subjects_wo_hd5/reconstruct.py``. Three modes:

1. Single-scan mode (``--input``): beamform one frame from one HDF5 file and
   save it as a PNG. With no arguments at all, picks the first HDF5 file
   found alongside this script.

       KERAS_BACKEND=jax python reconstruct.py
       KERAS_BACKEND=jax python reconstruct.py --input subjects/1_1.hdf5
       KERAS_BACKEND=jax python reconstruct.py --input subjects/1_1.hdf5 --frame 10

2. Random grid mode (default, or explicit ``--data-dir`` with no
   ``--systematic``): pick a few random scans from a directory of converted
   HDF5 files, a few random frames from each, and save a single PNG grid
   (rows = scans, columns = frames).

       KERAS_BACKEND=jax python reconstruct.py --data-dir subjects --output subjects/grid.png

3. Systematic grid mode (``--systematic``): fixed, evenly-spaced frame
   numbers (1-indexed, matching the raw rawdata_{frame}of4 numbering) for a
   specific list of patients -- e.g. to check where a scan's settled imaging
   state begins after a diagnostic ``convert.py --keep-all-frames`` run.
   Rows = patients, columns = the sampled frame numbers.

       KERAS_BACKEND=jax python reconstruct.py --systematic --patients 1_1,10_1 \\
           --frame-start 1 --frame-step 5 --frame-count 15 \\
           --data-dir /tmp/test --output /tmp/test/search_grid.png

Requires ``KERAS_BACKEND=jax`` in the environment -- zea defaults to
tensorflow, which isn't installed here. With it set, jax auto-selects the
GPU with no extra flag needed.
"""

import os

os.environ["MPLBACKEND"] = "Agg"  # use non-interactive backend for matplotlib

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import zea
from zea import Config, File, Pipeline

HERE = Path(__file__).parent
CONFIG = HERE / "pipeline.yaml"


def reconstruct_single(args, config):
    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found. Run convert.py first.")
    output_path = args.output or args.input.with_suffix(".png")

    with File(str(args.input)) as f:
        n_frames_available = f.data.raw_data.shape[0]
        frame = args.frame if args.frame is not None else 0
        # apply_lens_correction (pipeline.yaml) + the probe's lens_thickness/
        # lens_sound_speed (set by convert.py) apply the lens correction
        # automatically -- no manual initial_times shift needed here.
        parameters = f.load_parameters(**config.parameters)
        raw = f.data.raw_data[frame : frame + 1]  # (1, n_tx, n_ax, n_el, 1) -- RF

    print(f"frame index      : {frame} / {n_frames_available - 1}")
    print(f"raw_data shape   : {raw.shape}")
    print(f"grid             : {parameters.grid.shape}  (z, x, 3)")
    if parameters.lens_thickness is not None:
        print(
            f"lens correction  : thickness={parameters.lens_thickness * 1e3:.3f}mm, "
            f"c_lens={parameters.lens_sound_speed:.0f}m/s"
        )

    pipeline = Pipeline.from_config(config)
    inputs = pipeline.prepare_parameters(parameters)
    outputs = pipeline(data=raw, **inputs)

    recon = np.array(outputs["data"])  # (1, grid_z, grid_x)
    image = zea.display.to_8bit(recon[0], dynamic_range=parameters.dynamic_range)
    # NOTE: parameters.extent_imshow is in meters, not mm -- scale explicitly.
    extent_mm = [v * 1e3 for v in parameters.extent_imshow]

    zea.visualize.set_mpl_style()
    plt.imshow(image, extent=extent_mm, cmap="gray")
    plt.xlabel("X (mm)")
    plt.ylabel("Z (mm)")
    plt.savefig(str(output_path), bbox_inches="tight", dpi=100)

    print(f"Reconstructed  : {recon.shape}")
    print(f"Saved          : {output_path}")


def reconstruct_grid(args, config):
    hdf5_files = sorted(args.data_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {args.data_dir}. Run convert.py first.")

    rng = random.Random(args.seed)
    n_scans = min(args.n_scans, len(hdf5_files))
    chosen_files = rng.sample(hdf5_files, n_scans)

    pipeline = Pipeline.from_config(config)
    rows = []  # (scan_name, [(frame_idx, image, extent_mm), ...]) per scan

    for path in chosen_files:
        with File(str(path)) as f:
            parameters = f.load_parameters(**config.parameters)
            n_frames_available = f.data.raw_data.shape[0]
            if args.min_frame < n_frames_available:
                candidate_indices = range(args.min_frame, n_frames_available)
            else:
                candidate_indices = range(n_frames_available)
            n_frames = min(args.n_frames, len(candidate_indices))
            frame_indices = sorted(rng.sample(candidate_indices, n_frames))
            raw = f.data.raw_data[frame_indices]

        inputs = pipeline.prepare_parameters(parameters)
        outputs = pipeline(data=raw, **inputs)
        recon = np.array(outputs["data"])  # (n_frames, grid_z, grid_x)
        extent_mm = [v * 1e3 for v in parameters.extent_imshow]
        images = [zea.display.to_8bit(r, dynamic_range=parameters.dynamic_range) for r in recon]
        rows.append((path.stem, list(zip(frame_indices, images, [extent_mm] * len(images)))))
        print(f"raw_data shape   : {raw.shape}  ({path.stem})")

    _save_grid(rows, args.output or args.data_dir / "random_grid.png", frame_label_offset=0)


def reconstruct_systematic_grid(args, config):
    """Fixed, evenly-spaced 1-indexed frame numbers per patient, one row per
    patient. Frame *numbers* here match the raw rawdata_{frame}of4 numbering
    (1-indexed); array index = frame_number - 1."""
    patients = [p.strip() for p in args.patients.split(",") if p.strip()]
    if not patients:
        raise ValueError("--systematic requires --patients (comma-separated patient IDs).")

    frame_numbers = list(
        range(
            args.frame_start, args.frame_start + args.frame_step * args.frame_count, args.frame_step
        )
    )

    pipeline = Pipeline.from_config(config)
    rows = []

    for patient in patients:
        path = args.data_dir / f"{patient}.hdf5"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run convert.py for patient {patient!r} first."
            )

        with File(str(path)) as f:
            parameters = f.load_parameters(**config.parameters)
            n_frames_available = f.data.raw_data.shape[0]
            valid = [n for n in frame_numbers if 1 <= n <= n_frames_available]
            if len(valid) < len(frame_numbers):
                print(
                    f"{patient}: only {n_frames_available} frames available, "
                    f"dropping frame numbers > {n_frames_available}"
                )
            array_indices = [n - 1 for n in valid]
            raw = f.data.raw_data[array_indices]

        inputs = pipeline.prepare_parameters(parameters)
        outputs = pipeline(data=raw, **inputs)
        recon = np.array(outputs["data"])
        extent_mm = [v * 1e3 for v in parameters.extent_imshow]
        images = [zea.display.to_8bit(r, dynamic_range=parameters.dynamic_range) for r in recon]
        rows.append((patient, list(zip(valid, images, [extent_mm] * len(images)))))
        print(f"raw_data shape   : {raw.shape}  ({patient}, frame numbers {valid})")

    _save_grid(rows, args.output or args.data_dir / "search_grid.png", frame_label_offset=0)


def _save_grid(rows, output_path, frame_label_offset):
    zea.visualize.set_mpl_style()

    n_scans = len(rows)
    n_cols = max(len(row_cells) for _, row_cells in rows)
    fig, axes = plt.subplots(n_scans, n_cols, figsize=(3 * n_cols, 3.4 * n_scans), squeeze=False)
    for row, (scan_name, row_cells) in enumerate(rows):
        for col in range(n_cols):
            ax = axes[row][col]
            if col >= len(row_cells):
                ax.axis("off")
                continue
            frame_idx, image, extent_mm = row_cells[col]
            ax.imshow(image, extent=extent_mm, cmap="gray")
            ax.set_title(f"{scan_name}\nframe {frame_idx + frame_label_offset}", fontsize=9)
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Z (mm)")

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=100)
    print(f"Saved          : {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="Single HDF5 file to reconstruct.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=HERE / "subjects",
        help="Directory of HDF5 files to sample from (grid modes, used when --input is not given).",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame index (single-scan mode). Defaults to the first frame in the file.",
    )
    parser.add_argument(
        "--n-scans", type=int, default=3, help="Number of random scans (grid mode)."
    )
    parser.add_argument(
        "--n-frames", type=int, default=4, help="Random frames per scan (grid mode)."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (grid mode).")
    parser.add_argument(
        "--min-frame",
        type=int,
        default=0,
        help="Lowest frame index to sample from per scan (random grid mode).",
    )
    parser.add_argument(
        "--systematic",
        action="store_true",
        help="Use systematic grid mode: fixed frame numbers per patient (see --patients, "
        "--frame-start/--frame-step/--frame-count) instead of random sampling.",
    )
    parser.add_argument(
        "--patients",
        default=None,
        help="Comma-separated patient IDs, e.g. '1_1,10_1' (systematic mode).",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=1,
        help="First frame number, 1-indexed (systematic mode).",
    )
    parser.add_argument(
        "--frame-step", type=int, default=5, help="Step between frame numbers (systematic mode)."
    )
    parser.add_argument(
        "--frame-count", type=int, default=15, help="Number of frames to sample (systematic mode)."
    )
    args = parser.parse_args()

    if args.input is None and not args.systematic and not args.data_dir.exists():
        found = sorted(HERE.glob("*.hdf5"))
        if found:
            args.input = found[0]

    config = Config.from_path(str(CONFIG))

    if args.input is not None:
        reconstruct_single(args, config)
    elif args.systematic:
        reconstruct_systematic_grid(args, config)
    else:
        reconstruct_grid(args, config)


if __name__ == "__main__":
    main()
