# SPDX-License-Identifier: CC-BY-4.0
"""Reconstruct: transmission-mode attenuation FBP for the US4US ring-array USCT data.

This is a full ring of point-source transmit elements with a receive sub-aperture
on the far side of the ring (not surrounding the transmitter), so it's a
transmission (amplitude-attenuation) acquisition, not reflection B-mode.
`zea.ops.Beamform`/`USCTReflectivityDAS` don't apply, so this defines four custom
operations (https://zea.readthedocs.io/en/latest/pipeline.html#custom-operations):

    TransmissionAttenuation -> ApertureRearrange -> ResortSinogram
        -> FilteredBackprojection -> (built-in) GaussianBlur

`ApertureRearrange`'s `aperture_offset` (receive channel `k` of transmit `t` is
ring element `(t + offset + k) mod 1024`, centered on the antipode) isn't
documented anywhere in the file; confirmed via `validate_geometry` below, which
predicts the direct-arrival sample from real element positions and compares it
to the measured envelope peak.

Usage:
    python reconstruct.py --input yezitronix-b-rg-1.2.hdf5
    python reconstruct.py --input s8_r.hdf5 --frame 0
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from keras import ops
from mpl_toolkits.axes_grid1 import make_axes_locatable
from zea.internal.registry import ops_registry
from zea.ops import Cast, GaussianBlur, Operation

import zea
from zea import Config, File, Pipeline

HERE = Path(__file__).parent
CONFIG = HERE / "pipeline.yaml"

N_EL = 1024  # ring elements
N_RX = 512  # receive sub-aperture width
APERTURE_OFFSET = (N_EL - N_RX) // 2  # 256


@ops_registry("transmission_attenuation")
class TransmissionAttenuation(Operation):
    """Windowed peak RF amplitude per (tx, rx) pair, converted to attenuation via
    ``-20 * log10(amp / reference)``. ``reference=1.0`` is arbitrary units (no
    water-reference calibration)."""

    def __init__(self, reference: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.reference = reference

    def call(self, **kwargs):
        raw = kwargs[self.key]  # (n_tx, n_ax, n_rx, n_ch)
        raw = ops.squeeze(raw, axis=-1)  # (n_tx, n_ax, n_rx)
        n_ax = raw.shape[1]
        # `keras.ops.bartlett` requires its length argument to be a concrete
        # (non-traced) Python int even under jit, which a shape-derived value
        # here isn't - so the triangular window is built directly instead.
        i = ops.arange(n_ax, dtype="float32")
        half = (n_ax - 1) / 2.0
        window = (1.0 - ops.abs((i - half) / half))[None, :, None]
        amp = ops.max(ops.abs(raw * window), axis=1)  # (n_tx, n_rx)
        att = -20 * ops.log10(amp / self.reference)
        return {self.output_key: att}


@ops_registry("aperture_rearrange")
class ApertureRearrange(Operation):
    """Cyclically re-indexes each transmit's receive columns to a consistent
    relative frame, so `ResortSinogram` can treat rows as consecutive projection
    angles. `aperture_offset` is not documented in the file — see module
    docstring."""

    def __init__(self, aperture_offset: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.aperture_offset = APERTURE_OFFSET if aperture_offset is None else aperture_offset

    def call(self, **kwargs):
        att = kwargs[self.key]  # (n_tx, n_rx)
        n_tx, n_rx = att.shape
        rel_start = n_tx // 2 - self.aperture_offset
        rel_end = n_tx // 2 + self.aperture_offset
        tx = ops.arange(n_tx)
        rx_end = ops.mod(tx + rel_end, n_tx)
        rx_start = ops.mod(tx + rel_start, n_tx)
        shift = ops.where(rx_end < rx_start, rx_end + 1, 0)  # (n_tx,)
        col = ops.mod(ops.arange(n_rx)[None, :] + shift[:, None], n_rx)
        rearranged = ops.take_along_axis(att, col, axis=1)
        return {self.output_key: rearranged}


@ops_registry("resort_sinogram")
class ResortSinogram(Operation):
    """Doubles angular sampling (n_tx -> 2*n_tx) and halves detector count
    (n_rx -> n_rx // 2) by interleaving receive channels across transmits, into a
    proper parallel-projection sinogram."""

    def call(self, **kwargs):
        att = kwargs[self.key]  # (n_tx, n_rx)
        n_tx, n_rx = att.shape
        rx = ops.arange(n_rx // 2)
        tx = ops.arange(n_tx)
        RX, TX = ops.meshgrid(rx, tx, indexing="xy")
        src_tx = ops.mod(TX + n_rx // 4 - RX, n_tx)  # (n_tx, n_rx // 2)

        # output[2t, r] = att[src_tx[t, r], 2r]; output[2t+1, r] = att[src_tx[t, r], 2r+1].
        # Gathering the even/odd columns separately along axis 0 with the same
        # src_tx index, then interleaving via stack+reshape, is equivalent to that
        # scatter but keras.ops.take_along_axis-only (no fancy 2D indexing needed).
        even = ops.take_along_axis(att[:, 0::2], src_tx, axis=0)
        odd = ops.take_along_axis(att[:, 1::2], src_tx, axis=0)
        sinogram = ops.reshape(ops.stack([even, odd], axis=1), (n_tx * 2, n_rx // 2))
        return {self.output_key: sinogram}


@ops_registry("filtered_backprojection")
class FilteredBackprojection(Operation):
    """Ram-Lak (ramp) filter + backprojection onto a square grid sized to the
    detector count, vectorized over all angles in one gather-and-sum."""

    @staticmethod
    def _ramp_filter_half(n_detectors: int):
        """Ram-Lak filter, rfft-domain half. `sinogram` is real, so `rfft`/`irfft`
        (real in, real out) are used below instead of a full complex `fft`/`ifft`
        - there's no plain complex `ifft` in keras.ops anyway."""
        half = n_detectors // 2
        n = ops.concatenate([ops.arange(1.0, half + 1, 2.0), ops.arange(half - 1.0, 0.0, -2.0)])
        odd = -1.0 / (np.pi * n) ** 2
        even = ops.concatenate([ops.convert_to_tensor([0.25]), ops.zeros((half - 1,))])
        f = ops.reshape(ops.stack([even, odd], axis=1), (n_detectors,))
        real, _ = ops.rfft(f)
        return 2 * real

    def call(self, **kwargs):
        sinogram = kwargs[self.key]  # (n_angles, n_detectors)
        n_angles, n_detectors = sinogram.shape

        ramp_half = self._ramp_filter_half(n_detectors)
        real, imag = ops.rfft(sinogram)
        filtered = ops.irfft((real * ramp_half, imag * ramp_half), fft_length=n_detectors)

        # Backprojection geometry (which detector index each pixel/angle maps to)
        # depends only on (n_angles, n_detectors).
        half = n_detectors // 2
        angles = ops.arange(n_angles, dtype="float32") * (2 * np.pi / n_angles)
        x = ops.arange(n_detectors) - half
        X, Y = ops.meshgrid(x, x)  # (n_detectors, n_detectors)
        t = X[None] * ops.cos(angles)[:, None, None] + Y[None] * ops.sin(angles)[:, None, None]
        idx_raw = ops.cast(ops.round(t + half), "int32")  # (n_angles, D, D)
        # Skip out-of-range detector indices per angle rather than clamping them
        # to the edge, which would smear a repeated edge value into the image.
        valid = ops.logical_and(idx_raw >= 0, idx_raw < n_detectors)
        idx_safe = ops.clip(idx_raw, 0, n_detectors - 1)

        gathered = ops.take_along_axis(
            ops.repeat(filtered[:, None, :], n_detectors, axis=1), idx_safe, axis=2
        )  # (n_angles, D, D)
        image = ops.sum(ops.where(valid, gathered, 0.0), axis=0)
        return {self.output_key: image}


def build_pipeline(reference: float = 1.0, smooth_sigma: float = 2.0) -> Pipeline:
    return Pipeline(
        operations=[
            Cast(dtype="float32"),
            TransmissionAttenuation(reference=reference),
            ApertureRearrange(aperture_offset=APERTURE_OFFSET),
            ResortSinogram(),
            FilteredBackprojection(),
            GaussianBlur(sigma=smooth_sigma, axes=(-2, -1)),
        ],
        with_batch_dim=False,
    )


def write_config(pipeline: Pipeline, path: Path) -> None:
    config = pipeline.to_config()
    config.to_yaml(str(path))


def parse_dead_channels(description: str) -> np.ndarray:
    """Dead-element indices, parsed from the file's root `description` text."""
    match = re.search(r"turned off[^:]*:\s*([\d,\s]+?)\s*\(numbering", description)
    if not match:
        return np.array([], dtype=np.int64)
    return np.array([int(x) for x in match.group(1).split(",") if x.strip()], dtype=np.int64)


def ring_center_and_radius(probe_geometry: np.ndarray) -> tuple[np.ndarray, float]:
    """Ring center [x, z] and radius [m], measured (this ring is centered at
    z ~ 0.13 m, not the origin)."""
    xz = probe_geometry[:, [0, 2]]
    center = xz.mean(axis=0)
    radius = float(np.linalg.norm(xz - center, axis=-1).mean())
    return center, radius


def rx_element_indices(aperture_offset: int, n_tx: int = N_EL, n_rx: int = N_RX) -> np.ndarray:
    """Physical ring-element index for every (transmit, raw receive channel)
    pair: `(n_tx, n_rx)`."""
    k = np.arange(n_rx)
    return (np.arange(n_tx)[:, None] + aperture_offset + k[None, :]) % n_tx


def validate_geometry(rf, tx_position, rx_positions, sound_speed, initial_time, fs):
    """Predicted vs. measured direct-arrival sample, per receive channel, for one
    transmit. Agreement confirms the ring geometry, `initial_times`, and the
    aperture offset at once. `rf` is `(n_ax, n_rx)` real RF for one transmit."""
    distance = np.linalg.norm(rx_positions - tx_position, axis=-1)
    predicted = (distance / sound_speed - initial_time) * fs
    envelope = np.abs(rf)
    measured = envelope.argmax(axis=0)
    strong = envelope.max(axis=0) > 0.2 * envelope.max()
    residual = float(np.median(np.abs(measured[strong] - predicted[strong])))
    return predicted, measured, residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="a *.hdf5 file in this folder")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--aperture_offset", type=int, default=APERTURE_OFFSET)
    parser.add_argument(
        "--smooth_sigma", type=float, default=2.0, help="post-FBP Gaussian smoothing"
    )
    args = parser.parse_args()

    input_path = args.input if args.input.exists() else HERE / args.input
    if not input_path.exists():
        raise FileNotFoundError(f"{args.input} not found (looked in cwd and {HERE}).")
    output_path = input_path.with_name(f"{input_path.stem}_reconstruct.png")

    # Build, save to pipeline.yaml, then reload before running
    write_config(build_pipeline(smooth_sigma=args.smooth_sigma), CONFIG)
    config = Config.from_path(str(CONFIG))
    pipeline = Pipeline.from_config(config)
    pipeline.operations[2].aperture_offset = args.aperture_offset

    with File(str(input_path)) as f:
        probe_geometry = f.probe.probe_geometry[:]
        center, radius = ring_center_and_radius(probe_geometry)
        sound_speed = float(f.scan.sound_speed)
        fs = float(f.scan.sampling_frequency)
        initial_times = f.scan.initial_times[:]
        dead_elements = parse_dead_channels(str(f.attrs.get("description", "")))
        zea.log.info(f"Dead elements parsed from description: {dead_elements.tolist()}")

        raw = np.asarray(f.data.raw_data[args.frame])  # (n_tx, n_ax, n_rx, n_ch)

        stored_image = stored_sinogram = None
        if "image" in [e.name for e in f.custom]:
            stored_image = np.asarray(f.custom.image.data[args.frame])
        if "sinogram" in [e.name for e in f.custom]:
            stored_sinogram = np.asarray(f.custom.sinogram.data[args.frame])

    rx_idx = rx_element_indices(args.aperture_offset)  # (n_tx, n_rx)
    ring_xz = probe_geometry[:, [0, 2]].astype(np.float32)

    predicted, measured, residual = validate_geometry(
        raw[0, :, :, 0].astype(np.float32),
        ring_xz[0],
        ring_xz[rx_idx[0]],
        sound_speed,
        float(initial_times[0]),
        fs,
    )
    zea.log.info(f"Direct-arrival check: median |measured - predicted| = {residual:.2f} samples")

    outputs = pipeline(data=raw, return_numpy=True)
    recon = outputs["data"]  # (n_detectors, n_detectors)

    zea.visualize.set_mpl_style()
    n_panels = 2 + (stored_sinogram is not None) + (stored_image is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5))

    rf_db = 20 * np.log10(np.abs(raw[0, :, :, 0]) / (np.abs(raw[0, :, :, 0]).max() + 1e-12) + 1e-10)
    axes[0].imshow(rf_db, aspect="auto", cmap="gray", vmin=-60, vmax=0)
    axes[0].plot(np.arange(N_RX), predicted, "r--", lw=1.0, label="predicted direct arrival")
    axes[0].plot(np.arange(N_RX), measured, "c:", lw=0.7, alpha=0.6, label="measured (argmax)")
    axes[0].set_ylim(raw.shape[1], 0)
    axes[0].legend(loc="lower left", fontsize=7)
    axes[0].set_title(f"RF, transmit 0 [dB]\ngeometry check: residual {residual:.2f} samples")
    axes[0].set_xlabel("Receive channel")
    axes[0].set_ylabel("Axial sample")

    extent = [center[0] - radius, center[0] + radius, center[1] + radius, center[1] - radius]
    vmin, vmax = np.percentile(recon, [1, 99])
    handle = axes[1].imshow(recon, cmap="magma", extent=extent, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Attenuation FBP (zea.Pipeline)\n({input_path.name}, frame {args.frame})")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Z (m)")
    cax = make_axes_locatable(axes[1]).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(handle, cax=cax)

    panel_idx = 2
    if stored_sinogram is not None:
        axes[panel_idx].imshow(stored_sinogram, cmap="viridis", aspect="auto")
        axes[panel_idx].set_title("Stored sinogram (custom/sinogram)")
        axes[panel_idx].set_xlabel("Receive channel")
        axes[panel_idx].set_ylabel("Transmit")
        panel_idx += 1
    if stored_image is not None:
        axes[panel_idx].imshow(stored_image, cmap="magma")
        axes[panel_idx].set_title("Stored FBP (custom/image)")
        panel_idx += 1

    fig.suptitle(f"{input_path.name} — frame {args.frame}")
    fig.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight", dpi=100)

    print(f"Reconstructed  : {recon.shape}")
    print(f"Saved          : {output_path}")


if __name__ == "__main__":
    main()
