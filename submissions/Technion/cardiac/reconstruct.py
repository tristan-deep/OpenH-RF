# SPDX-License-Identifier: Apache-2.0
"""Reference B-mode reconstruction for the SLA (cardiac) zea files.

The repo's own reconstruction of these acquisitions is the conventional
delay-and-sum image produced by ``code/dataset_creation/createDS.m`` ->
``SLA2MLA.m`` -> ``code/CREANUIS/do_dynamic_focalization_CREANUIS_new.m``. That
output is shipped verbatim in each file as ``beamformed_data`` (the paired
target), so the exact, authoritative reconstruction is simply the stored
``beamformed_data`` — this script envelope-detects, log-compresses, and
scan-converts it for display, using the per-pixel polar ``coordinates`` stored
alongside it.

(Reconstructing an image from ``raw_data`` by classical DAS is the learning task
this paired dataset is built for; because the CREANUIS acquisition header — exact
axial rate / transmit focus — is not stored, a classical raw->image beamforming
is only approximate in depth scale, which is why the delivered reference is the
paired target rather than a re-beamform. See README "Known Issues".)

Usage:
    python reconstruct.py data/a1.hdf5 --frame 15 --out bmode_a1.png
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent


def env_db(iq, dr=50.0):
    e = np.abs(iq[..., 0] + 1j * iq[..., 1])
    e = e / (e.max() + 1e-12)
    return np.clip(20 * np.log10(e + 1e-12), -dr, 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("zea_file", type=Path)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", type=Path, default=HERE / "bmode.png")
    p.add_argument("--dr", type=float, default=50.0, help="dynamic range (dB)")
    args = p.parse_args()

    with h5py.File(str(args.zea_file), "r") as h:
        g = h["tracks/track_0"]
        bf = g["data/beamformed_data/values"][args.frame]  # (z, x, 2) complex IQ
        coords = g["data/beamformed_data/coordinates"][()]  # (z, x, 3) m, best-estimate
    bmode = env_db(bf, args.dr)
    X = coords[..., 0] * 1e3  # lateral mm
    Z = coords[..., 2] * 1e3  # depth mm (approximate scale)

    fig, ax = plt.subplots(figsize=(5.0, 6))
    pcm = ax.pcolormesh(X, Z, bmode, cmap="gray", shading="auto", vmin=-args.dr, vmax=0)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("lateral [mm]")
    ax.set_ylabel("depth [mm]  (scale approximate)")
    ax.set_title(f"{args.zea_file.stem} — frame {args.frame}\npaired DAS target (repo reconstruction)")
    fig.colorbar(pcm, ax=ax, label="dB", fraction=0.046)
    fig.tight_layout()
    fig.savefig(str(args.out), dpi=130, bbox_inches="tight")
    print(f"Saved: {args.out}  (target {bf.shape[:2]})")


if __name__ == "__main__":
    main()
