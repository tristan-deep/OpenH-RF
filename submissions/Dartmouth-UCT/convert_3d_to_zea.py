"""Convert the 3D ring-array USCT k-Wave waveforms into the zea HDF5 format.

One ``.hdf5`` file is written per acquisition (one phantom z-plane = 64 focused
single-element transmit events). Everything is saved through ``zea.File.create``:
the raw RF channel data and full acquisition description (PURE-calibrated ring
geometry, sampling, time-zero), the ground-truth maps as zea ``sos_map`` /
``attenuation_map`` fields (with per-pixel coordinates), and acquisition metadata
as zea ``CustomElement``s.

Input layout (``-v7`` MATLAB files, loaded with scipy):
  dense_3D/datasets_res0.29_split{N}/{id}/
    kWave_phantom_{id}_zoff{±k}.mat  -> full_dataset (T=2161, Rx=256, Tx=64),
                                        time (T,), z_off, phantom_z_idx
    gt_slice_{id}_zoff{±k}.mat       -> gt_sos_slice (Nx,Ny), gt_atten_slice,
                                        phantom_z_idx, z_off, dx, iz_ring

Output:
  <out_dir>/phantom_{id}_zoff{±k}.hdf5   (zea File)

Usage:
  python convert_3d_to_zea.py --check                 # validate pairing/shapes (no zea)
  python convert_3d_to_zea.py                          # full conversion (needs zea)
  python convert_3d_to_zea.py --limit 3
"""

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")  # zea picks a Keras backend at import

import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent           # .../OpenH-RF
SIM_DIR = HERE.parent / "Simulations"            # .../Simulations
DATA_3D = SIM_DIR / "dense_3D"
CALIB_MAT = SIM_DIR / "CalibrationPos_PURE_v2.mat"

# Geometry / physics (from kwave_simulation3D_phantom_v3.m and the OpenH-RF README)
N_ELEMENTS = 256
TX_ELEMS = np.arange(0, 256, 4)           # 64 transmit elements (stride 4), 0-based
N_TX = len(TX_ELEMS)
CENTER_FREQ_HZ = 1.5e6                     # 5-cycle toneburst centre frequency
N_CYCLES = 5                               # toneburst cycle count
C_WATER = 1500.0                           # nominal background sound speed [m/s]
ALPHA_POWER = 1.01                         # k-Wave alpha_power
# Emission-centroid offset: the saved 3D time vector starts at the toneburst
# onset (t=0), so we shift t=0 to the pulse centroid to match the 2D convention
# and make time-of-flight reconstruction correct. = (len-1)/2 * dt.
_FS = 12e6
EMISSION_OFFSET_S = (round(N_CYCLES / CENTER_FREQ_HZ * _FS) - 1) / 2 / _FS  # 1.625e-6 s
# GT attenuation maps are in dB/cm/MHz; zea AttenuationMap base unit is dB/m/Hz.
DBCMMHZ_TO_DBMHZ = 1e-4
ELEMENT_WIDTH_M = 0.558e-3                  # element_diameter
ELEMENT_HEIGHT_M = 19e-3                    # element_height (elevation aperture)
ELEMENT_FOCUS_M = 75e-3                     # geometric elevation focus (R_focus)
CREDIT = ("Thayer School of Engineering, Dartmouth College; "
          "University of Rochester Medical Center")

WAVE_RE = re.compile(r"^kWave_phantom_(.+)_zoff([+-]\d+)\.mat$")


# ---------------------------------------------------------------------------
# Ring geometry (PURE calibration; same centring as the simulation)
# ---------------------------------------------------------------------------

def ring_geometry():
    """Return PURE ring element positions and per-transmit descriptors.

    Matches kwave_simulation3D_phantom_v3.m: x = pos_ele[:,0]-mean,
    y = pos_ele[:,2]-mean, z = 0 (ring in the x-y plane).
    """
    pos = sio.loadmat(str(CALIB_MAT))["pos_ele"]          # (256, 3)
    x = pos[:, 0] - pos[:, 0].mean()
    y = pos[:, 2] - pos[:, 2].mean()
    z = np.zeros_like(x)
    probe_geometry = np.column_stack([x, y, z]).astype(np.float32)

    # tx_apodizations: transmit event k fires element TX_ELEMS[k] (stride-4).
    tx_apod = np.zeros((N_TX, N_ELEMENTS), dtype=np.float32)
    tx_apod[np.arange(N_TX), TX_ELEMS] = 1.0

    return {
        "probe_geometry": probe_geometry,
        "tx_apodizations": tx_apod,
        "transmit_origins": probe_geometry[TX_ELEMS].copy(),    # (64, 3)
        "azimuth_angles": np.zeros(N_TX, dtype=np.float32),
        "polar_angles": np.zeros(N_TX, dtype=np.float32),
        "focus_distances": np.zeros(N_TX, dtype=np.float32),    # in-plane: point/diverging
    }


# ---------------------------------------------------------------------------
# Discovery & loading
# ---------------------------------------------------------------------------

def discover():
    """Return acquisition items across all splits, paired with their GT files."""
    items = []
    for split in sorted(DATA_3D.glob("datasets_*")):
        for ph in sorted(d for d in split.iterdir() if d.is_dir()):
            for wf in sorted(ph.glob("kWave_phantom_*_zoff*.mat")):
                m = WAVE_RE.match(wf.name)
                if not m:
                    continue
                pid, zoff = m.group(1), m.group(2)
                gt = ph / f"gt_slice_{pid}_zoff{zoff}.mat"
                items.append({"pid": pid, "zoff": zoff, "split": split.name,
                              "wave": wf, "gt": gt})
    return items


def load_acquisition(item, dtype):
    """Load + assemble one acquisition into zea-ready arrays.

    raw_data : (1, n_tx=64, n_ax=2161, n_el=256, 1)
    """
    wf = sio.loadmat(str(item["wave"]))
    full = wf["full_dataset"]                       # (T, Rx=256, Tx=64)
    time_vec = wf["time"].ravel().astype(np.float32)
    # (T, Rx, Tx) -> (Tx, T, Rx) = (n_tx, n_ax, n_el)
    raw = np.transpose(full, (2, 0, 1)).astype(dtype)
    raw_data = raw[np.newaxis, ..., np.newaxis]     # (1, n_tx, n_ax, n_el, 1)
    fs = np.float32(1.0 / (time_vec[1] - time_vec[0]))

    gt = sio.loadmat(str(item["gt"]))
    sos = gt["gt_sos_slice"].astype(np.float32)
    atten = gt["gt_atten_slice"].astype(np.float32)
    dx = float(np.ravel(gt["dx"])[0])
    extra = {
        "phantom_z_idx": int(np.ravel(gt["phantom_z_idx"])[0]),
        "iz_ring": int(np.ravel(gt["iz_ring"])[0]),
        "z_off": int(np.ravel(wf["z_off"])[0]),
    }
    return raw_data, time_vec, sos, atten, dx, fs, extra


# ---------------------------------------------------------------------------
# zea writing
# ---------------------------------------------------------------------------

def build_scan(geo, time_vec, fs, c0):
    return {
        "sampling_frequency": fs,
        "center_frequency": np.float32(CENTER_FREQ_HZ),
        "demodulation_frequency": np.float32(0.0),
        "sound_speed": np.float32(c0),
        # Shift t = 0 to the emission centroid (the saved time vector starts at
        # the toneburst onset), matching the 2D convention for correct TOF.
        "initial_times": np.full(N_TX, time_vec[0] - EMISSION_OFFSET_S,
                                 dtype=np.float32),
        "t0_delays": np.zeros((N_TX, N_ELEMENTS), dtype=np.float32),
        "tx_apodizations": geo["tx_apodizations"],
        "transmit_origins": geo["transmit_origins"],
        "azimuth_angles": geo["azimuth_angles"],
        "polar_angles": geo["polar_angles"],
        "focus_distances": geo["focus_distances"],
    }


def map_coordinates(shape, dx):
    """Per-pixel `[x, y, z]` positions in metres, centred on the ring (z=0)."""
    H, W = shape
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    x = (jj - (W - 1) / 2.0) * dx
    y = (ii - (H - 1) / 2.0) * dx
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def ground_truth_maps(sos, atten, dx):
    """Build zea `sos_map` + `attenuation_map` data dicts (per-pixel coordinates)."""
    coords = map_coordinates(sos.shape, dx)
    return {
        "sos_map": {
            "values": sos[np.newaxis].astype(np.float32),          # (1, H, W) m/s
            "coordinates": coords,
            "unit": "m/s",
        },
        "attenuation_map": {
            "values": (atten * DBCMMHZ_TO_DBMHZ)[np.newaxis].astype(np.float32),
            "coordinates": coords,
            "unit": "dB/m/Hz",
            "gamma": np.float32(ALPHA_POWER),
        },
    }


def convert(item, geo, out_dir, dtype, File, CustomElement):
    raw_data, time_vec, sos, atten, dx, fs, extra = load_acquisition(item, dtype)
    c0 = float(sos[0, 0])                            # background sound speed [m/s]

    scan = build_scan(geo, time_vec, fs, c0)
    probe = {
        "name": "ring_3D_PURE_256",
        "type": "custom",
        "probe_geometry": geo["probe_geometry"],
        "element_width": np.float32(ELEMENT_WIDTH_M),
        "element_height": np.float32(ELEMENT_HEIGHT_M),
        "probe_center_frequency": np.float32(CENTER_FREQ_HZ),
    }
    data = {"raw_data": raw_data, **ground_truth_maps(sos, atten, dx)}
    metadata = {"subject": {"id": f"phantom_{item['pid']}", "type": "phantom"},
                "credit": CREDIT}
    custom = [
        CustomElement(name="z_off", data=np.array(extra["z_off"], dtype=np.int32),
                      description="Z-offset of the ring plane (voxels)", unit="-"),
        CustomElement(name="phantom_z_idx",
                      data=np.array(extra["phantom_z_idx"], dtype=np.int32),
                      description="Phantom z-slice index seen by the ring plane", unit="-"),
        CustomElement(name="element_focus", data=np.array(ELEMENT_FOCUS_M, dtype=np.float32),
                      description="Geometric elevation focus of the focused elements",
                      unit="m"),
    ]

    out_path = out_dir / f"phantom_{item['pid']}_zoff{item['zoff']}.hdf5"
    File.create(
        str(out_path),
        data=data, scan=scan, probe=probe, metadata=metadata, custom=custom,
        description=("Simulated 3D ring-array USCT RF acquisition (k-Wave, "
                     "digital breast phantom)"),
        overwrite=True,
    )
    return out_path, raw_data.shape


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=HERE / "data" / "3d")
    ap.add_argument("--dtype", choices=["float32"], default="float32",
                    help="raw_data storage dtype (zea allows float32 or int16).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check", action="store_true",
                    help="validate pairing + shapes only; do not import/write zea")
    args = ap.parse_args()

    geo = ring_geometry()
    items = discover()
    if args.limit:
        items = items[:args.limit]

    # Guard against output-name collisions across splits.
    seen = {}
    for it in items:
        name = f"phantom_{it['pid']}_zoff{it['zoff']}"
        if name in seen and seen[name] != it["split"]:
            print(f"  ⚠ name collision across splits: {name} "
                  f"({seen[name]} vs {it['split']})")
        seen[name] = it["split"]

    File = CustomElement = None
    if not args.check:
        from zea import File as _File
        from zea.data.file import CustomElement as _CE
        File, CustomElement = _File, _CE
        args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 3D: {len(items)} acquisitions ===")
    done = missing = errors = 0
    for i, item in enumerate(items, 1):
        if not item["gt"].exists():
            print(f"  [{i}/{len(items)}] {item['wave'].name}  ⚠ missing GT -> SKIP")
            missing += 1
            continue
        try:
            if args.check:
                raw, tv, sos, atten, dx, fs, extra = load_acquisition(item, np.float32)
                print(f"  [{i}/{len(items)}] {item['pid']}_zoff{item['zoff']}  "
                      f"raw={raw.shape} T={tv.shape[0]} fs={fs/1e6:.3f}MHz t0={tv[0]:.2e} "
                      f"sos={sos.shape}[{sos.min():.0f},{sos.max():.0f}] dx={dx*1e3:.3f}mm")
            else:
                out_path, shape = convert(item, geo, args.out_dir,
                                          np.dtype(args.dtype), File, CustomElement)
                print(f"  [{i}/{len(items)}] {out_path.name}  raw={shape}")
            done += 1
        except Exception as e:
            print(f"  [{i}/{len(items)}] {item['pid']}_zoff{item['zoff']}  ✗ {e}")
            errors += 1

    mode = "validated" if args.check else "written"
    print(f"\nDone. {done} {mode}, missing={missing}, errors={errors}.")


if __name__ == "__main__":
    main()
