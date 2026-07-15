# SPDX-License-Identifier: CC-BY-4.0
"""Reference reconstruction for the 2D ring-array USCT zea sub-dataset.

Loads one converted ``.hdf5`` (zea) acquisition and reconstructs a Delay-And-Sum
(DAS) reflectivity image directly from the raw RF channel data, using the ring
geometry, sampling rate, and time-zero **read back from the zea file**. This is
the sanity check requested by the OpenH-RF guide: if the recorded geometry /
timing are correct, the DAS image is sharp and spatially aligned with the
ground-truth SOS / attenuation maps stored in the same file.

The beamformer is a single-acquisition port of ``BatchDualChannelDAS`` from the
project's training code (``ddpm_das_waveform.py``): a round-trip time-of-flight
DAS that produces two channels — a full-aperture reflectivity image and a
90-degree partial-aperture image. It is wrapped as a **custom registered
``zea.ops.Operation``** (``ring_das_reflectivity``) so the whole reconstruction
is expressed as a ``zea.Pipeline`` and saved to ``pipeline.yaml``.

Usage:
    python reconstruct.py                       # first file in ./data
    python reconstruct.py data/phantom_xxx.hdf5 # a specific acquisition
    python reconstruct.py --save-pipeline       # (re)write pipeline.yaml and exit

"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")  # zea picks a Keras backend at import

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import zea
from zea import File
from zea.ops import Operation
from zea.internal.registry import ops_registry

HERE = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# DAS reflectivity beamformer (single acquisition; ported from BatchDualChannelDAS)
# ---------------------------------------------------------------------------

@torch.no_grad()
def das_reflectivity(waveform, sensors_xy, dt, t0, *, c0, num_pixels, fov_size,
                     aperture_deg=90, tx_chunk=32, device="cpu"):
    """Round-trip TOF Delay-And-Sum over the 2D full-ring array.

    (The 3D set uses a different beamformer — see reconstruct_3d.py.)

    Args:
        waveform   : (Tx, Time, Rx) float tensor of raw RF channel data.
        sensors_xy : (n_el, 2) element positions [m] (from the zea file).
        dt, t0     : sampling interval [s] and time origin [s] (from the file).
        c0         : assumed background sound speed [m/s].
        num_pixels : output image is num_pixels x num_pixels.
        fov_size   : field of view [m] (square, centred on the ring).

    Returns:
        (2, H, H) float32 tensor: [full-aperture, partial-aperture] reflectivity.
    """
    waveform = waveform.to(device)
    Tx, Time, Rx = waveform.shape
    H = num_pixels
    NP = H * H

    # Pixel grid (centred on the ring, matching the GT map convention).
    grid_1d = torch.linspace(-fov_size / 2, fov_size / 2, H, device=device)
    py, px = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
    pixels = torch.stack([px.flatten(), py.flatten()], dim=1)        # (NP, 2)

    sensors = torch.as_tensor(sensors_xy, dtype=torch.float32, device=device)
    dist = torch.cdist(sensors, pixels)                              # (n_el, NP)
    idx_base = dist / (c0 * dt)                                      # one-way index
    t0_idx = t0 / dt

    # Partial-aperture receive mask (rx within +/- aperture_deg/2 of tx).
    n_valid = int(round(aperture_deg / 360.0 * Tx))
    half = n_valid // 2
    rx_mask = torch.zeros(Tx, Tx, dtype=torch.bool, device=device)
    for tx in range(Tx):
        for off in range(-half, n_valid - half):
            rx_mask[tx, (tx + off) % Tx] = True

    wave = waveform.permute(0, 2, 1).contiguous().float()           # (Tx, Rx, Time)
    das_full = torch.zeros(NP, device=device)
    das_part = torch.zeros(NP, device=device)

    for tx0 in range(0, Tx, tx_chunk):
        tx1 = min(tx0 + tx_chunk, Tx)
        # Round-trip index: tof(tx->pixel) + tof(pixel->rx) - t0.
        tidx = (idx_base[tx0:tx1].unsqueeze(1)
                + idx_base.unsqueeze(0) - t0_idx).long()             # (c, Rx, NP)
        valid = (tidx >= 0) & (tidx < Time)
        tidx.clamp_(0, Time - 1)

        amps = torch.gather(wave[tx0:tx1], dim=2, index=tidx)        # (c, Rx, NP)
        amps *= valid

        das_full += amps.sum(0).sum(0)
        amps *= rx_mask[tx0:tx1].unsqueeze(-1)
        das_part += amps.sum(0).sum(0)

    # Orientation: element k of the channel data sits at probe_geometry[k]
    # (theta_0 = -pi), so the natural pixel-grid view already aligns row/col with
    # the ground-truth maps — no extra flip is needed here. (Equivalent to a
    # flipud+fliplr of the theta_0 = 0 convention used in the training code.)
    full_img = das_full.view(H, H)
    part_img = das_part.view(H, H)
    return torch.stack([full_img, part_img], 0)


# ---------------------------------------------------------------------------
# Custom zea operation wrapping the DAS (so the pipeline serialises to yaml)
# ---------------------------------------------------------------------------

@ops_registry("ring_das_reflectivity")
class RingDAS(Operation):
    """Round-trip TOF DAS reflectivity for a ring array, as a zea Operation.

    Beamforming parameters (``aperture_deg``, ``tx_chunk``) are serialised to
    ``pipeline.yaml``; the data-dependent geometry/timing (element positions,
    dt, t0, c0, grid) are passed at call time from the acquisition itself.
    """

    def __init__(self, aperture_deg=90, tx_chunk=32, **kwargs):
        # Defaults suited to a torch round-trip-DAS; serialised values override.
        kwargs.setdefault("jit_compile", False)
        kwargs.setdefault("jittable", False)
        kwargs.setdefault("with_batch_dim", False)
        super().__init__(**kwargs)
        self.aperture_deg = aperture_deg
        self.tx_chunk = tx_chunk

    def call(self, **kwargs):
        wf = kwargs[self.key]                                        # (Tx, Time, Rx)
        das = das_reflectivity(
            wf, kwargs["sensors_xy"], kwargs["dt"], kwargs["t0"],
            c0=kwargs["c0"], num_pixels=kwargs["num_pixels"],
            fov_size=kwargs["fov_size"], aperture_deg=self.aperture_deg,
            tx_chunk=self.tx_chunk, device=str(wf.device),
        )
        return {self.output_key: das}


def build_pipeline(aperture_deg=90):
    return zea.Pipeline(operations=[
        RingDAS(aperture_deg=aperture_deg, key="raw_data", output_key="reflectivity"),
    ])


# ---------------------------------------------------------------------------
# Load one zea acquisition
# ---------------------------------------------------------------------------

def load_acquisition(path):
    """Read raw data + geometry + GT from a converted zea file (via the zea API)."""
    with File(str(path), "r") as f:
        raw = np.asarray(f.data.raw_data[:])          # (1, Tx, n_ax, n_el, 1)
        s = f.scan
        fs = float(s.sampling_frequency)
        t0 = float(np.asarray(s.initial_times)[0])
        c0 = float(s.sound_speed)
        sensors_xy = np.asarray(f.probe.probe_geometry)[:, :2]   # (n_el, 2) [m]

        sos = np.asarray(f.data.sos_map.values)[0]                # (H, W) m/s
        atten = np.asarray(f.data.attenuation_map.values)[0]      # (H, W) dB/m/Hz
        coords = np.asarray(f.data.sos_map.coordinates)           # (H, W, 3)
        dx = float(coords[0, 1, 0] - coords[0, 0, 0])
        tissue = ""
        for ce in f.custom:
            if ce.name == "tissue":
                v = np.asarray(ce.data).item()
                tissue = v.decode() if isinstance(v, bytes) else str(v)

    waveform = raw[0, :, :, :, 0]                     # (Tx, Time, Rx)
    return dict(waveform=waveform, sensors_xy=sensors_xy, fs=fs, t0=t0, c0=c0,
                sos=sos, atten=atten, dx=dx, tissue=tissue)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", type=Path, default=None,
                    help="zea .hdf5 acquisition (default: first under ./data)")
    ap.add_argument("--out", type=Path, default=HERE / "example_output.png")
    ap.add_argument("--pipeline", type=Path, default=HERE / "pipeline.yaml")
    ap.add_argument("--aperture-deg", type=int, default=90)
    ap.add_argument("--save-pipeline", action="store_true",
                    help="(re)write pipeline.yaml from the zea.Pipeline and exit")
    args = ap.parse_args()

    if args.save_pipeline:
        build_pipeline(args.aperture_deg).to_yaml(str(args.pipeline))
        print(f"Wrote {args.pipeline}")
        return

    # Load the pipeline from pipeline.yaml (the registered op is resolved by name).
    if args.pipeline.exists():
        pipeline = zea.Pipeline.from_config(zea.Config.from_path(str(args.pipeline)))
    else:
        pipeline = build_pipeline(args.aperture_deg)

    path = args.file or next((HERE / "data").rglob("phantom_*.hdf5"))
    print(f"Reconstructing: {path}  (device={DEVICE})")

    acq = load_acquisition(path)
    H = acq["sos"].shape[0]
    fov = H * acq["dx"]                               # square FOV [m]
    print(f"  Tx/Rx={acq['waveform'].shape[0]}  T={acq['waveform'].shape[1]}  "
          f"fs={acq['fs']/1e6:.3f} MHz  t0={acq['t0']*1e6:.3f} us  c0={acq['c0']:.0f} m/s")
    print(f"  image {H}x{H}  fov={fov*1e3:.1f} mm")

    # Run the zea pipeline. Data-dependent geometry/timing are passed as kwargs.
    outputs = pipeline(
        raw_data=torch.from_numpy(acq["waveform"].astype(np.float32)).to(DEVICE),
        sensors_xy=acq["sensors_xy"], dt=1.0 / acq["fs"], t0=acq["t0"],
        c0=acq["c0"], num_pixels=H, fov_size=fov,
    )
    das = outputs["reflectivity"].cpu().numpy()

    # ---- figure: DAS reflectivity (full + partial) next to the GT maps ----
    half_mm = fov * 1e3 / 2
    ext = [-half_mm, half_mm, -half_mm, half_mm]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = [
        (das[0], "DAS reflectivity (full aperture)", "gray"),
        (das[1], f"DAS reflectivity ({args.aperture_deg} deg partial)", "gray"),
        (acq["sos"], f"GT sound speed [m/s] ({acq['tissue']})", "viridis"),
        (acq["atten"], "GT attenuation [dB/cm/MHz]", "magma"),
    ]
    for ax, (img, title, cmap) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, extent=ext, origin="lower")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{path.name} — reference DAS reconstruction from zea channel data",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
