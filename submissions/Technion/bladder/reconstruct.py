# SPDX-License-Identifier: Apache-2.0
"""Reference B-mode reconstruction for the SLT (bladder / phantom) zea files.

Faithful port of the dataset's own production beamformer,
``code/processing/IQBF.m`` (Technion SLT/MLT toolbox): per-line receive
delay-and-sum on the pre-beamformed channel IQ, with

  * the exact two-way delay law ``t_delay = (r + |p - x_e|) / c``,
    ``r = t*c/2``, ``t = (0..Ns-1)/fs``  (sltBFTRYIQ.m / IQBF.m),
  * per-channel IQ carrier phase rotation ``exp(+i*2*pi*fDem*(t_delay - t))``,
  * the IQBF.m dynamic expanding receive aperture (f-number = 1): at depth ``r``
    only elements within ``±r/2`` of the beam origin contribute, clipped to the
    physical array (rectangular / boxcar apodization),

then envelope detection, log compression, and sector scan conversion.

Reading straight from the converted zea file (``raw_data`` + ``scan`` + ``probe``)
also confirms the conversion end-to-end: the repo's own beamformer run on the
converted file reproduces the repo's B-mode. A `zea.Pipeline` equivalent is saved
in ``pipeline.yaml``.

Usage:
    python reconstruct.py data/s2.hdf5 --frame 54 --out bmode_s2.png
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent


def load(zea_path: Path, frame: int):
    with h5py.File(str(zea_path), "r") as h:
        g = h["tracks/track_0"]
        sc = g["scan"]
        raw = g["data/raw_data"][frame]  # (n_tx, n_ax, n_el, 2)
        fs = float(sc["sampling_frequency"][()])
        fdem = float(sc["demodulation_frequency"][()])
        c = float(sc["sound_speed"][()])
        theta = sc["polar_angles"][()].astype(np.float64)  # (n_tx,)
        elem = h["probe/probe_geometry"][()][:, 0].astype(np.float64)  # (n_el,)
    iq = raw[..., 0] + 1j * raw[..., 1]  # (n_tx, n_ax, n_el)
    return iq, fs, fdem, c, theta, elem


def beamform(iq, fs, fdem, c, theta, elem):
    """Exact IQBF.m: dynamic-aperture receive DAS per transmit line."""
    n_tx, ns, nch = iq.shape
    t = np.arange(ns) / fs
    r = t * c / 2.0
    w0 = 2 * np.pi * fdem
    ap_start = -np.minimum(r / 2.0, 0.0 - elem[0])   # (ns,) expanding aperture
    ap_end = np.minimum(r / 2.0, elem[-1] - 0.0)
    img = np.zeros((ns, n_tx), dtype=complex)
    for ii in range(n_tx):
        x_rx = r * np.sin(theta[ii])
        z_rx = r * np.cos(theta[ii])
        line = np.zeros(ns, dtype=complex)
        for n in range(nch):
            t_del = (r + np.sqrt((x_rx - elem[n]) ** 2 + z_rx ** 2)) / c
            v = np.interp(t_del, t, iq[ii, :, n], left=0.0, right=0.0)
            phi = w0 * (t_del - t)
            mask = (elem[n] >= ap_start) & (elem[n] <= ap_end)
            line += v * (np.cos(phi) + 1j * np.sin(phi)) * mask
        img[:, ii] = line
    return img, r


def env_db(x, dr=60.0):
    e = np.abs(x)
    e = e / (e.max() + 1e-12)
    return np.clip(20 * np.log10(e + 1e-12), -dr, 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("zea_file", type=Path)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", type=Path, default=HERE / "bmode.png")
    p.add_argument("--dr", type=float, default=60.0, help="dynamic range (dB)")
    args = p.parse_args()

    iq, fs, fdem, c, theta, elem = load(args.zea_file, args.frame)
    img, r = beamform(iq, fs, fdem, c, theta, elem)
    bmode = env_db(img, args.dr)

    x = (r[:, None] * np.sin(theta[None, :])) * 1e3  # mm
    z = (r[:, None] * np.cos(theta[None, :])) * 1e3
    fig, ax = plt.subplots(figsize=(5.5, 6))
    pcm = ax.pcolormesh(x, z, bmode, cmap="gray", shading="auto", vmin=-args.dr, vmax=0)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("lateral [mm]")
    ax.set_ylabel("depth [mm]")
    ax.set_title(f"{args.zea_file.stem} — frame {args.frame}  (IQBF.m reference beamformer)")
    fig.colorbar(pcm, ax=ax, label="dB", fraction=0.046)
    fig.tight_layout()
    fig.savefig(str(args.out), dpi=130, bbox_inches="tight")
    print(f"Saved: {args.out}  (polar {img.shape}, depth 0-{r[-1]*1e3:.0f} mm)")


if __name__ == "__main__":
    main()
