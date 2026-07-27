# SPDX-License-Identifier: Apache-2.0
"""Convert an SLA pre-beamforming .mat file into the OpenH-RF (zea) format.

Source: SLA_preBF_collection/{SLA_preBF,f_SLA_preBF}/<cine>_SLA_preBF.mat
    samples  (MATLAB 680 x 64 x 140 x F) complex int16
             = depth-samples x elements x Rx-lines x frames
             (h5py reads it reversed: (F, 140, 64, 680))
    labels   (MATLAB 652 x 1 x 140 x F) complex single
             = beamformed image (delay-and-sum target), depth x 1 x lines x frames

In-vivo cardiac phased-array sector scan: 140 transmit beams steered over a
~75 deg sector, one image line per transmit. `samples` is the required pre-BF
channel IQ; `labels` is the paired delay-and-sum reconstruction, stored as
beamformed_data.

KNOWN ISSUE (documented by the contributors): the consolidated .mat files do not
carry the full acquisition header, so the exact axial sample rate and the
transmit focus/steering geometry are NOT stored. The scan geometry below is a
best estimate (init_params.m + the dataset's reference beamformer); the depth
scale is therefore approximate. The paired `labels` are the authoritative
reconstruction target.

Usage:
    python convert.py <cine>_SLA_preBF.mat --output out.hdf5
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import h5py
import numpy as np
from zea import File
from zea.beamform.pixelgrid import polar_pixel_grid

HERE = Path(__file__).parent
CREDIT = (
    "Technion – Israel Institute of Technology. "
    "Contributors: S. Vedula, O. Senouf, D. Zadok, A. Bronstein."
)
PROBE_NAME = "GE 3Sc-RS"          # GE 3Sc-RS phased-array probe
US_MACHINE = "GE Vivid S70"       # ultrasound machine
ELEMENT_HEIGHT = 13e-3            # m (elevation aperture; init_params Height / proposal "13 mm")

# Best-estimate acquisition geometry (init_params.m + reference beamformer).
C0 = 1540.0          # m/s          (init_params c0)
PITCH = 300e-6       # m            (init_params Pitch)
F0 = 2.5e6           # Hz           (cardiac fundamental, per contributor / proposal)
FS_ESTIMATE = 6.0e6  # Hz           (best-estimate axial IQ rate; NOT stored)
SECTOR_DEG = 75.0    # deg          (SLA2MLA sector; +/-37.5 deg about boresight)


def convert(
    path: Path, output_path: Path, anatomy: str = "cardiac", subject_id: str | None = None
) -> Path:
    with h5py.File(str(path), "r") as h:
        s = h["samples"]                          # (F, 140, 64, 680) compound int16
        n_frames, n_tx, n_el, n_ax = s.shape

        raw = np.empty((n_frames, n_tx, n_ax, n_el, 2), dtype=np.float32)
        for f in range(n_frames):
            fr = s[f]                             # (n_tx, n_el, n_ax) compound
            raw[f, ..., 0] = np.transpose(fr["real"], (0, 2, 1)).astype(np.float32)
            raw[f, ..., 1] = np.transpose(fr["imag"], (0, 2, 1)).astype(np.float32)

        lab = h["labels"]                         # (F, 140, 1, 652) compound single
        n_lab_ax = lab.shape[3]
        bf = np.empty((n_frames, n_lab_ax, n_tx, 2), dtype=np.float32)  # (F, z, x, ch)
        for f in range(n_frames):
            lf = lab[f][:, 0, :]                  # (n_tx, n_lab_ax) compound
            bf[f, ..., 0] = lf["real"].T.astype(np.float32)   # (n_lab_ax, n_tx)
            bf[f, ..., 1] = lf["imag"].T.astype(np.float32)

    # Phased-array geometry: 64 elements centred on boresight.
    x = (np.arange(n_el) - (n_el - 1) / 2.0) * PITCH
    probe_geometry = np.stack(
        [x, np.zeros_like(x), np.zeros_like(x)], axis=-1
    ).astype(np.float32)

    # Per-line steering angles: 140 lines spanning +/- SECTOR/2 about boresight.
    polar_angles = np.deg2rad(
        np.arange(n_tx) * (SECTOR_DEG / n_tx) - SECTOR_DEG / 2.0
    ).astype(np.float32)

    # Best-estimate polar coordinates for the paired target grid (z x line).
    # Depth scale is APPROXIMATE (fs not stored).
    z_max = n_lab_ax * C0 / (2.0 * FS_ESTIMATE)
    bf_coords = polar_pixel_grid(
        polar_limits=(float(polar_angles.min()), float(polar_angles.max())),
        zlims=(0.0, z_max),
        num_radial_pixels=n_lab_ax,
        num_polar_pixels=n_tx,
    ).astype(np.float32)

    data = {
        "raw_data": raw,
        "beamformed_data": {
            "values": bf,                         # (F, z, x, n_ch=2) complex IQ
            "coordinates": bf_coords,             # (z, x, 3) — best-estimate, approximate
            "labels": ["I", "Q"],
            "description": (
                "Paired delay-and-sum reconstruction supplied with the dataset "
                "(complex IQ). Depth scale approximate: axial rate not stored."
            ),
        },
    }

    scan = {
        "sampling_frequency": float(FS_ESTIMATE),
        "center_frequency": float(F0),
        "demodulation_frequency": float(F0),
        "sound_speed": float(C0),
        "initial_times": np.zeros(n_tx, dtype=np.float32),
        "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
        "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
        "focus_distances": np.full(n_tx, np.inf, dtype=np.float32),
        "transmit_origins": np.zeros((n_tx, 3), dtype=np.float32),
        "polar_angles": polar_angles,
        "azimuth_angles": np.zeros(n_tx, dtype=np.float32),
    }

    probe = {
        "name": PROBE_NAME,
        "type": "phased",
        "probe_geometry": probe_geometry,
        "element_width": np.float32(0.9 * PITCH),
        "element_height": np.float32(ELEMENT_HEIGHT),
        "probe_center_frequency": np.float32(F0),
    }

    sid = subject_id or path.stem.replace("_SLA_preBF", "").replace("CARD", "").lower()
    metadata = {
        "subject": {"id": sid, "type": "human"},
        "credit": CREDIT,
        "annotations": {"anatomy": anatomy, "label": "in vivo", "view": "apical four-chamber (A4C)"},
    }

    File.create(
        str(output_path),
        data=data,
        scan=scan,
        probe=probe,
        metadata=metadata,
        us_machine=US_MACHINE,
        description=(
            "In-vivo cardiac phased-array sector scan: 140 transmit beams steered "
            "over +/-37.5 deg, one image line per transmit; pre-beamformed channel "
            "IQ (raw_data) paired with the delay-and-sum target (beamformed_data). "
            "Axial rate/focus not stored; geometry is best-estimate and depth scale "
            "approximate."
        ),
        overwrite=True,
    )
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_path", type=Path)
    p.add_argument("--output", type=Path, default=HERE / "sla_sample.hdf5")
    p.add_argument("--anatomy", default="cardiac")
    p.add_argument("--subject-id", default=None, help="clean id, e.g. a1 / f2")
    args = p.parse_args()

    out = convert(args.input_path, args.output, args.anatomy, args.subject_id)
    print(f"Saved: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    with File(str(out)) as f:
        print("data:", list(f.data))
        print("raw_data:", f.data.raw_data.shape, f.data.raw_data.dtype)


if __name__ == "__main__":
    main()
