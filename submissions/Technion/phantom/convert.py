# SPDX-License-Identifier: Apache-2.0
"""Convert an SLT pre-beamforming .mat file into the OpenH-RF (zea) format.

Source: SLT_preBF_collection/SLT_preBF/<subject>_SLT_preBF.mat
    IQelementsSLT_filt  (MATLAB 696 x 64 x 180 x F) complex double
                        = samples x elements x transmit-lines x frames
                        (h5py reads it reversed: (F, 180, 64, 696))
    thetaTX             (180,) transmit steering angles, radians
    specs               acquisition parameters

Single-line-transmit phased-array sector scan: 180 steered transmits over a
+/-45.13 deg sector, per-element IQ recorded before receive beamforming.

Usage:
    python convert.py <subject>_SLT_preBF.mat --output out.hdf5
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import h5py
import numpy as np
from scipy.io import loadmat
from zea import File

HERE = Path(__file__).parent
# Element positions used by the dataset's own beamformer (64 el, 0.3 mm pitch).
ELEM_POS_MAT = (
    HERE / ".." / ".." / "data" / "US_data" / "SLT_preBF_collection"
    / "code" / "processing" / "3Sc_elem_pos.mat"
)


def _read_probe_geometry() -> np.ndarray:
    """Element (x, y, z) positions in metres, shape (n_el, 3)."""
    pos = loadmat(str(ELEM_POS_MAT))["elements_positions"].astype(np.float32)
    assert pos.shape == (64, 3), pos.shape
    return pos


def convert(
    path: Path, output_path: Path, anatomy: str = "bladder", subject_id: str | None = None
) -> Path:
    with h5py.File(str(path), "r") as h:
        ds = h["IQelementsSLT_filt"]              # (F, 180, 64, 696) compound
        n_frames, n_tx, n_el, n_ax = ds.shape

        # (F, n_tx, n_el, n_ax) -> (F, n_tx, n_ax, n_el, n_ch=2 [I, Q])
        raw = np.empty((n_frames, n_tx, n_ax, n_el, 2), dtype=np.float32)
        for f in range(n_frames):                 # frame-by-frame to bound memory
            fr = ds[f]                            # (n_tx, n_el, n_ax) compound
            i = np.transpose(fr["real"], (0, 2, 1)).astype(np.float32)  # (tx, ax, el)
            q = np.transpose(fr["imag"], (0, 2, 1)).astype(np.float32)
            raw[f, ..., 0] = i
            raw[f, ..., 1] = q

        theta = h["thetaTX"][:].ravel().astype(np.float32)             # (n_tx,)
        specs = {k: float(h["specs"][k][()].ravel()[0]) for k in h["specs"]}

    fs = specs["IQSampleRate"]                    # two-way IQ sample rate, Hz
    fdem = specs["DemodulationFrequency"]         # IQ demod (received) centre, Hz
    c = specs["SpeedOfSound"]                     # m/s
    probe_geometry = _read_probe_geometry()

    data = {"raw_data": raw}

    scan = {
        "sampling_frequency": float(fs),
        "center_frequency": float(fdem),          # received/harmonic centre (== demod)
        "demodulation_frequency": float(fdem),
        "sound_speed": float(c),
        "initial_times": np.zeros(n_tx, dtype=np.float32),   # StartDepth == 0
        "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
        "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
        "focus_distances": np.full(n_tx, np.inf, dtype=np.float32),
        "transmit_origins": np.zeros((n_tx, 3), dtype=np.float32),    # common apex
        "polar_angles": theta,                    # steering angle per transmit
        "azimuth_angles": np.zeros(n_tx, dtype=np.float32),
    }

    pitch = float(np.median(np.diff(probe_geometry[:, 0])))
    probe = {
        "name": "64-element phased array (0.3 mm pitch)",
        "type": "phased",
        "probe_geometry": probe_geometry,
        "element_width": np.float32(0.9 * pitch),
        "probe_center_frequency": np.float32(fdem),
    }

    sid = subject_id or path.stem.replace("_SLT_preBF", "").lower()
    stype = "phantom" if anatomy == "phantom" else "human"
    label = "phantom" if anatomy == "phantom" else "in vivo"
    metadata = {
        "subject": {"id": sid, "type": stype},
        "annotations": {"anatomy": anatomy, "label": label},
    }

    File.create(
        str(output_path),
        data=data,
        scan=scan,
        probe=probe,
        metadata=metadata,
        description=(
            "SLT single-line-transmit phased-array sector scan, per-element IQ "
            "before receive beamforming (Technion cardiac/bladder study, 2018)."
        ),
        overwrite=True,
    )
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_path", type=Path)
    p.add_argument("--output", type=Path, default=HERE / "slt_sample.hdf5")
    p.add_argument("--anatomy", default="bladder", help="bladder (in-vivo) or phantom")
    p.add_argument("--subject-id", default=None, help="clean id, e.g. a1 / ph")
    args = p.parse_args()

    out = convert(args.input_path, args.output, args.anatomy, args.subject_id)
    print(f"Saved: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    with File(str(out)) as f:
        print("data:", list(f.data))
        print("raw_data:", f.data.raw_data.shape, f.data.raw_data.dtype)
        print("scan:", f.scan)


if __name__ == "__main__":
    main()
