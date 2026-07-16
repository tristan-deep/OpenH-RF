# SPDX-License-Identifier: CC-BY-4.0
"""Reconstruct: DAS reflectivity images of the 2D and 3D ring-array USCT phantoms.

Defines a round-trip time-of-flight Delay-And-Sum pipeline in code, saves it
(together with the reconstruction parameters) to pipeline.yaml, then loads that
YAML back and runs it on the HDF5 file. Both sub-datasets — the 2D full-ring
(256 transmits) and the 3D ring (64 transmits) — reconstruct with the same
pipeline; only the imaging grid differs, and that is read from the file.

USCT does not fit zea's standard B-mode pipeline: the transmits are individual
point sources firing in turn, not a wavefront steered from the receive aperture,
so `zea.ops.Beamform`'s steered-wavefront time-of-flight model does not apply.
`zea.ops.USCTReflectivityDAS` is the dedicated operation for this geometry — for
every pixel it coherently sums the analytic channel signal over all
transmit/receive pairs at the round-trip delay, rejecting the direct
through-transmission arrival (which dwarfs the backscatter) and apodizing to keep
only backscatter geometries.

The imaging grid defaults to the footprint and resolution of the ground-truth
sound-speed map stored in the file, so the reconstruction is directly comparable,
pixel for pixel, with the ground truth. This is the sanity check the OpenH-RF
guide asks for: if the recorded geometry and timing are right, the bright skin
boundary traces the ground-truth contour.

By default the delays assume the file's constant `sound_speed`. With --sos_map,
the ground-truth sound-speed map is fed to the same operation, which replaces the
constant-c delays with a straight-ray integral of the local slowness — the
best-case delay model, since it uses the true medium.

Usage:
    python reconstruct.py --input data/2d/phantom_xxx.hdf5
    python reconstruct.py --input data/2d/phantom_xxx.hdf5 --sos_map
    python reconstruct.py --input data/3d/phantom_xxx.hdf5 --fov 0.13 --num_pixels 512
"""

import os

# zea picks a Keras backend at import; default to torch (this project's env) but
# honour an explicit KERAS_BACKEND (e.g. export KERAS_BACKEND=jax) if set.
os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from zea.ops import (
    Cast,
    LogCompress,
    Normalize,
    PatchedGrid,
    ReshapeGrid,
    USCTReflectivityDAS,
)

import zea
from zea import Config, File, Pipeline

HERE = Path(__file__).parent
CONFIG = HERE / "pipeline.yaml"

# The ring lies in the XZ imaging plane, so the reconstruction grid is Cartesian
# and centred on the ring. `ylims` is pinned to zero: the ring images a single
# plane, so the grid must stay 2D (left unset, zea would infer an elevation
# extent from the probe and build a volume).
PARAMETERS = {
    "grid_type": "cartesian",
    "ylims": [0.0, 0.0],
    "dynamic_range": [-40, 0],
}


def build_pipeline() -> Pipeline:
    """Define the USCT delay-and-sum reflectivity pipeline in code."""
    return Pipeline(
        operations=[
            Cast(dtype="float32"),
            # Every pixel is independent, so reconstruct the grid in patches to
            # bound peak memory, then reshape the flat result to an image. The
            # 5-cycle 1.5 MHz toneburst is ~3.3 us long; a 2.5 us guard (~0.75 of
            # a pulse) past the direct arrival rejects through-transmission while
            # keeping backscatter from near the ring.
            PatchedGrid(
                operations=[
                    USCTReflectivityDAS(tx_chunk=4, transmission_guard_s=2.5e-6),
                ],
                num_patches=64,
            ),
            ReshapeGrid(),
            Normalize(),
            LogCompress(),
        ],
        # Python-level loops over transmit chunks: not jittable.
        jit_options="pipeline",
        # One acquisition at a time: `data` is (n_tx, ...) with no leading frame
        # axis. The pipeline propagates this to every operation it contains.
        with_batch_dim=False,
    )


def write_config(pipeline: Pipeline, path: Path) -> None:
    """Serialize the pipeline and reconstruction parameters to a YAML config file."""
    config = pipeline.to_config()
    config["parameters"] = PARAMETERS
    config.to_yaml(str(path))


def check_ring_in_imaging_plane(file: File):
    """The ring must lie in the XZ imaging plane (y = elevation), as zea expects.

    A ring stored in the XY plane still reconstructs — every element projects onto
    a line — but yields a meaningless image, and lets zea infer an elevation extent
    from the ring, turning the grid into a volume that exhausts GPU memory. Both
    are far easier to understand as an error here.
    """
    probe_geometry = file.probe.probe_geometry[:]
    elevation = np.abs(probe_geometry[:, 1]).max()
    in_plane = np.abs(probe_geometry[:, [0, 2]]).max()
    if elevation > 0.01 * in_plane:
        raise ValueError(
            f"{file.path}: probe_geometry spans {2 * elevation * 1e3:.1f} mm in y "
            f"(elevation) against {2 * in_plane * 1e3:.1f} mm in-plane, so the ring is "
            "not in the XZ imaging plane. Run fix_uploaded_files.py (see FEEDBACK.md)."
        )


def ground_truth(file: File):
    """Ground-truth maps and their in-plane (x, z) axes, read from the zea file."""
    coords = file.data.sos_map.coordinates[:]  # (n_z, n_x, 3)
    return {
        "sos": file.data.sos_map.values[0],
        "attenuation": file.data.attenuation_map.values[0],
        "x": coords[0, :, 0],
        "z": coords[:, 0, 2],
    }


def ring_radius(file: File):
    """Radius of the transducer ring [m], from the in-plane element positions."""
    probe_geometry = file.probe.probe_geometry[:]
    return float(np.linalg.norm(probe_geometry[:, [0, 2]], axis=-1).mean())


def grid_limits(gt, radius, fov=None, num_pixels=None):
    """Imaging grid: the ground-truth footprint, clipped to the ring interior.

    The 3D ground-truth maps are wider than the ring (232 mm across a 222 mm ring),
    so imaging their full footprint would put the transducer ring itself inside the
    grid, where it reconstructs as a bright ring that dominates the normalization
    and buries the phantom. A square of half-width `h` has corners at `h*sqrt(2)`,
    so keeping `h <= 0.9 * radius / sqrt(2)` leaves a 10% margin to the elements.
    The 2D maps are already well inside their ring, so this leaves them untouched.
    """
    half = min(np.abs(gt["x"]).max(), np.abs(gt["z"]).max(), 0.636 * radius)
    if fov is not None:
        half = fov / 2
    if num_pixels is None:
        # Keep the ground-truth pixel pitch, so the panels stay comparable.
        num_pixels = int(round(2 * half / (gt["x"][1] - gt["x"][0]))) + 1
    return {
        "xlims": [-half, half],
        "zlims": [-half, half],
        "grid_size_x": num_pixels,
        "grid_size_z": num_pixels,
    }


def crop_to_grid(gt, grid):
    """Crop the ground-truth maps to the imaging grid, for a like-for-like figure."""
    keep_x = (gt["x"] >= grid["xlims"][0]) & (gt["x"] <= grid["xlims"][1])
    keep_z = (gt["z"] >= grid["zlims"][0]) & (gt["z"] <= grid["zlims"][1])
    return {
        "sos": gt["sos"][np.ix_(keep_z, keep_x)],
        "attenuation": gt["attenuation"][np.ix_(keep_z, keep_x)],
        "x": gt["x"][keep_x],
        "z": gt["z"][keep_z],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--fov",
        type=float,
        default=None,
        help="square field of view [m] (default: the ground-truth map footprint, "
        "clipped to the ring interior)",
    )
    parser.add_argument(
        "--num_pixels",
        type=int,
        default=None,
        help="output image is num_pixels x num_pixels (default: ground-truth resolution)",
    )
    parser.add_argument(
        "--sos_map",
        action="store_true",
        help="use the ground-truth sound-speed map for straight-ray corrected "
        "delays (default: the file's constant sound_speed)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device ID (e.g. 'cuda:0', 'auto:1', or 'cpu')",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found. Run convert_2d_to_zea.py first.")
    suffix = "_sos.png" if args.sos_map else ".png"
    output_path = args.input.with_name(args.input.stem + suffix)

    zea.init_device(device=args.device, verbose=True)

    # Define the pipeline in code, save it (with the reconstruction parameters)
    # to pipeline.yaml, then load that YAML back in.
    write_config(build_pipeline(), CONFIG)
    config = Config.from_path(str(CONFIG))

    # Load file: acquisition parameters (with config overrides) and raw RF data.
    with File(str(args.input)) as f:
        check_ring_in_imaging_plane(f)
        gt_full = ground_truth(f)
        grid = grid_limits(gt_full, ring_radius(f), args.fov, args.num_pixels)
        gt = crop_to_grid(gt_full, grid)
        parameters = f.load_parameters(**config.parameters, **grid)
        raw = f.data.raw_data[0]  # (n_tx, n_ax, n_el, 1) — RF, one frame

    # SoS-corrected delays: pass the ground-truth map (uncropped, so rays that
    # leave the imaging grid still see the phantom) as call-time data. This is
    # a runtime input, not a pipeline parameter, so pipeline.yaml is unchanged.
    sos_inputs = {}
    if args.sos_map:
        sos_inputs = {
            "sos_map": gt_full["sos"].astype(np.float32),
            "sos_grid_x": gt_full["x"].astype(np.float32),
            "sos_grid_z": gt_full["z"].astype(np.float32),
        }

    # Build and run the pipeline loaded from pipeline.yaml.
    pipeline = Pipeline.from_config(config)
    inputs = pipeline.prepare_parameters(parameters, **sos_inputs)

    zea.log.info("Running pipeline on RF data...")
    outputs = pipeline(data=raw, **inputs, return_numpy=True)

    recon = outputs["data"]  # (grid_z, grid_x) — log-compressed reflectivity

    # The ground-truth maps share the reconstruction's frame, so the reflective
    # skin boundary should trace the ground-truth contour panel for panel.
    gt_extent = [gt["x"].min(), gt["x"].max(), gt["z"].max(), gt["z"].min()]

    zea.visualize.set_mpl_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    recon_title = "DAS reflectivity [dB]" + (" (SoS-corrected)" if args.sos_map else "")
    panels = [
        (recon, recon_title, "gray", parameters.extent_imshow),
        (gt["sos"], "Ground-truth sound speed [m/s]", "viridis", gt_extent),
        (gt["attenuation"], "Ground-truth attenuation [dB/m/Hz]", "magma", gt_extent),
    ]
    for ax, (image, title, cmap, extent) in zip(axes, panels):
        handle = ax.imshow(image, cmap=cmap, extent=extent)
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        # Tie the colorbar axes to the image axes so it matches the panel height.
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        fig.colorbar(handle, cax=cax)
    fig.suptitle(args.input.name)
    fig.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight", dpi=100)

    print(f"Reconstructed  : {recon.shape}")
    print(f"Saved          : {output_path}")


if __name__ == "__main__":
    main()
