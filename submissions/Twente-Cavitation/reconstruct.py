"""Reconstruct: passive acoustic map (PAM) of one cavitation acquisition, saved as PNG.

This is a *passive* acquisition: the L11-4v never transmits (the file records
`tx_apodizations` as all zeros); a separate 2.25 MHz single-element transducer
insonifies the tube with 444 us pulses and the array only receives. Running
zea's standard pulse-echo pipeline on this file fails twice over:

1. All-zero `tx_apodizations` make the transmit-delay model return inf -> NaN.
2. Pulse-echo delays sample each pixel at time (z + r)/c after "transmit" —
   but the cavitation signal only reaches the array ~50 us into the frame, so
   pixels near the tube are sampled before any signal has arrived.

Both are fixed with two parameter overrides — no custom operations needed:

- `tx_apodizations` -> ones      (restores a valid delay computation)
- `initial_times`   -> [-t_c]    (shifts every pixel's sampling instant to
                                  t_c, a moment *during* the insonification)

With those overrides the standard chain (Cast -> Demodulate -> Beamform ->
EnvelopeDetect) aligns purely on receive curvature, i.e. one-way passive
beamforming. Averaging the envelope energy over several sampling instants t_c
and over frames approximates time-exposure-acoustics PAM (Gyongy & Coussios).
The map focuses at the tube (~x=0, z=21 mm) with the axial tails expected for
a narrowband continuous source.

Usage:
    python reconstruct.py
    python reconstruct.py --input my_file.hdf5 --device cpu
    python reconstruct.py --input my_file.hdf5 --output my_map.png --frames 20 --device cuda:0
"""

import os

# Select backend before importing zea/keras.
os.environ["KERAS_BACKEND"] = "jax"
os.environ["MPLBACKEND"] = "Agg"  # non-interactive backend for matplotlib

import argparse
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
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
DEFAULT_INPUT = HERE / "cavitation_bubbles_100kPa_01mL.hdf5"
CONFIG = HERE / "pipeline.yaml"

# Grid at half-wavelength sampling over the full aperture.
PARAMETERS = {
    "grid_size_x": 387,
    "grid_size_z": 577,
    "xlims": [-0.019, 0.019],
    "zlims": [0.002, 0.060],
    "apply_lens_correction": False,
}

# Sampling instants t_c within the 444 us insonification window. Each run of
# the pipeline images the received field at one instant; averaging their
# envelope energies is the "time exposure" integration of classic PAM.
SAMPLING_INSTANTS = np.linspace(100e-6, 400e-6, 6)

# Contributor-suggested minimum variance (PR #490), else delay-and-sum.
BEAMFORMER = "minimum_variance"
BEAMFORMER_KWARGS = {"subarray_size": 32, "diagonal_loading": 1e-2}


def build_pipeline() -> Pipeline:
    """Standard zea chain up to envelope detection (energy averaging happens
    across sampling instants/frames, so normalization comes afterwards)."""
    return Pipeline(
        operations=[
            Cast(dtype="float32"),
            Demodulate(),  # RF (n_ch=1) -> IQ before beamforming
            Beamform(beamformer=BEAMFORMER, num_patches=8, **BEAMFORMER_KWARGS),
            EnvelopeDetect(),
        ],
        validate=True,
        with_batch_dim=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: input file name with a .png extension)",
    )
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to average")
    parser.add_argument(
        "--device",
        type=str,
        default="auto:1",
        help="Device to use (e.g. 'cpu', 'cuda:0', or 'auto:1')",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_suffix(".png")

    zea.init_device(device=args.device, verbose=False)

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found.")

    # Build the pipeline in code and save it (with parameters) to pipeline.yaml,
    # then load that YAML back in so the shipped YAML is exactly what runs.
    pipeline = build_pipeline()
    config = pipeline.to_config()
    config["parameters"] = PARAMETERS
    config.to_yaml(str(CONFIG))
    config = Config.from_path(str(CONFIG))
    pipeline = Pipeline.from_config(config)

    with File(str(args.input)) as f:
        parameters = f.load_parameters(**config.parameters)
        raw = f.data.raw_data[: args.frames]  # (n_frames, n_tx, n_ax, n_el, 1)

    # Passive-acquisition overrides (see module docstring).
    parameters["tx_apodizations"] = np.ones_like(np.asarray(parameters["tx_apodizations"]))

    # Accumulate envelope energy over sampling instants (and frames).
    energy = None
    for t_c in SAMPLING_INSTANTS:
        parameters["initial_times"] = np.array([-t_c], dtype=np.float32)
        inputs = pipeline.prepare_parameters(parameters)
        env = keras.ops.convert_to_numpy(pipeline(data=raw, **inputs)["data"])
        frame_energy = (env**2).mean(axis=0)  # average over frames
        energy = frame_energy if energy is None else energy + frame_energy
    amplitude = np.sqrt(energy / len(SAMPLING_INSTANTS))

    # Normalize + log-compress the averaged map with the same zea operations.
    post = Pipeline(operations=[Normalize(), LogCompress()], with_batch_dim=False)
    pam_db = keras.ops.convert_to_numpy(post(data=amplitude)["data"])

    zea.visualize.set_mpl_style()
    extent_mm = [v * 1e3 for v in parameters.extent_imshow]
    plt.figure(figsize=(6, 8))
    plt.imshow(np.clip(pam_db, -25, 0), extent=extent_mm, cmap="inferno")
    plt.colorbar(label="dB")
    plt.xlabel("X (mm)")
    plt.ylabel("Z (mm)")
    plt.title(
        f"Passive acoustic map, {BEAMFORMER}\n"
        f"{args.frames} frames x {len(SAMPLING_INSTANTS)} instants — "
        "microbubble cavitation in a 200 um tube"
    )
    plt.savefig(str(args.output), bbox_inches="tight", dpi=100)
    plt.close()

    iz, ix = np.unravel_index(np.argmax(amplitude), amplitude.shape)
    zg = np.linspace(*PARAMETERS["zlims"], amplitude.shape[0])
    xg = np.linspace(*PARAMETERS["xlims"], amplitude.shape[1])
    print(
        f"Map            : {amplitude.shape}, peak at x={xg[ix] * 1e3:.2f} mm, z={zg[iz] * 1e3:.2f} mm"
    )
    print(f"Saved          : {args.output}")


if __name__ == "__main__":
    main()
