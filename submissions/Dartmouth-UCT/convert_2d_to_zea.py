"""Convert the 2D ring-array USCT k-Wave waveforms into the zea HDF5 format.

One ``.hdf5`` file is written per acquisition (one phantom slice = all 256
single-element transmit events). Everything is saved through ``zea.File.create``:
the raw RF channel data and full acquisition description (ring geometry, sampling,
time-zero), the ground-truth maps as zea ``sos_map`` / ``attenuation_map`` fields
(with per-pixel coordinates), and phantom metadata as zea ``CustomElement``s.

Inputs (per dataset, see DATASETS below):
  - waveforms_fp16/{base}_waveform.npy   fp16  (Tx=256, T=867, Rx=256)
  - time_vectors/{base}_time.npy         fp32  (T=867,)   (t=0 at pulse centroid)
  - <gt_dir>/phantom_{id}_z{z}.npy       fp64  (2, 230, 230) -> [SOS, attenuation]
    where base == "kWave_phantom_{id}_z{z}".

Output:
  - <out_dir>/<dataset>/phantom_{id}_z{z}.hdf5   (zea File)

Usage:
  # validate pairing + shapes only (no zea needed; safe to run in the `ut` env):
  python convert_2d_to_zea.py --check

  # full conversion (needs an env with `zea` installed):
  python convert_2d_to_zea.py --dataset both
  python convert_2d_to_zea.py --dataset dense --limit 5
"""

import argparse
import os
import re
from pathlib import Path

# zea bootstraps a Keras backend at import time; default to an installed one so
# the script runs without exporting KERAS_BACKEND first. An explicit value wins.
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

# ---------------------------------------------------------------------------
# Paths & acquisition constants (2D set, per process_single_phantom.m / README)
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent           # .../OpenH-RF
SIM_DIR = HERE.parent / "Simulations"            # .../Simulations
PHANTOM_ROOT = Path("/home/yujia/Desktop/usct-breast-phantom")

DATASETS = {
    "dense": {
        "waveform_dir": SIM_DIR / "dense" / "waveforms_fp16",
        "time_dir": SIM_DIR / "dense" / "time_vectors",
        "gt_dir": PHANTOM_ROOT / "dataset_SOS_02_22_npy",
        "tissue": "dense breast",
    },
    "fatty": {
        "waveform_dir": SIM_DIR / "fatty" / "waveforms_fp16",
        "time_dir": SIM_DIR / "fatty" / "time_vectors",
        "gt_dir": PHANTOM_ROOT / "dataset_SOS_fatty_npy",
        "tissue": "fatty breast",
    },
}

# Geometry / physics (from process_single_phantom.m and the OpenH-RF README)
N_ELEMENTS = 256          # ring elements (= n_tx = n_rx)
RING_RADIUS_M = 60e-3     # ring radius [m]
CENTER_FREQ_HZ = 1.0e6    # Gaussian pulse centre frequency
C_WATER = 1500.0          # background/reference sound speed [m/s]
GT_DX_M = 0.30e-3         # ground-truth pixel size [m] (centered crop)
ALPHA_POWER = 1.01        # k-Wave alpha_power -> zea AttenuationMap.gamma
# The GT attenuation maps are in dB/cm/MHz; zea's AttenuationMap base unit is
# dB/m/Hz, with 1 dB/cm/MHz = 1e-4 dB/m/Hz.
DBCMMHZ_TO_DBMHZ = 1e-4
FRAC_BW = 0.75            # Gaussian pulse fractional bandwidth (-> 75% probe BW)
CREDIT = ("Thayer School of Engineering, Dartmouth College; "
          "University of Rochester Medical Center")

WAVE_RE = re.compile(r"^(kWave_phantom_(.+)_z(-?\d+))_waveform\.npy$")


# ---------------------------------------------------------------------------
# Ring geometry (identical ordering to the saved channel data)
# ---------------------------------------------------------------------------

def ring_geometry():
    """Return ring element positions and per-transmit descriptors.

    Channel ordering matches process_single_phantom.m: element k corresponds to
    theta_k = -pi + k * 2*pi/N, placed at radius RING_RADIUS_M in the x-y plane.

    Returns dict of arrays:
        probe_geometry   (N, 3) float32  [m]  (x, y, z=0)
        tx_apodizations  (N, N) float32       identity (single-element transmits)
        transmit_origins (N, 3) float32  [m]  transmitting element position
        azimuth_angles   (N,)   float32       0 (not steered)
        polar_angles     (N,)   float32       0
        focus_distances  (N,)   float32       0 (unfocused point source)
    """
    theta = -np.pi + np.arange(N_ELEMENTS) * (2.0 * np.pi / N_ELEMENTS)
    x = RING_RADIUS_M * np.cos(theta)
    y = RING_RADIUS_M * np.sin(theta)
    z = np.zeros_like(x)
    probe_geometry = np.column_stack([x, y, z]).astype(np.float32)

    geo = {
        "probe_geometry": probe_geometry,
        "tx_apodizations": np.eye(N_ELEMENTS, dtype=np.float32),
        "transmit_origins": probe_geometry.copy(),
        "azimuth_angles": np.zeros(N_ELEMENTS, dtype=np.float32),
        "polar_angles": np.zeros(N_ELEMENTS, dtype=np.float32),
        "focus_distances": np.zeros(N_ELEMENTS, dtype=np.float32),
    }
    # Nominal element width = ring pitch (elements are idealized point sources).
    geo["element_width"] = np.float32(2.0 * np.pi * RING_RADIUS_M / N_ELEMENTS)
    return geo


# ---------------------------------------------------------------------------
# Pairing & loading
# ---------------------------------------------------------------------------

def discover(cfg):
    """Return a list of acquisition items found in this dataset's waveform dir."""
    items = []
    for wf in sorted(Path(cfg["waveform_dir"]).glob("kWave_phantom_*_waveform.npy")):
        m = WAVE_RE.match(wf.name)
        if not m:
            continue
        base, pid, z = m.group(1), m.group(2), m.group(3)
        time_path = Path(cfg["time_dir"]) / f"{base}_time.npy"
        gt_path = Path(cfg["gt_dir"]) / f"phantom_{pid}_z{z}.npy"
        items.append({
            "base": base, "pid": pid, "z": z,
            "wave": wf, "time": time_path, "gt": gt_path,
        })
    return items


def load_acquisition(item, dtype):
    """Load + assemble one acquisition into zea-ready arrays.

    Returns (raw_data, time_vec, sos, atten, sampling_frequency).
        raw_data : (n_frames, n_tx, n_ax, n_el, n_ch) = (1, 256, 867, 256, 1)
        time_vec : (T,) float32 [s]
        sos      : (Nz, Nx) float32 [m/s]
        atten    : (Nz, Nx) float32 [dB/cm/MHz]
    """
    # waveform is already (Tx, T, Rx) = zea (n_tx, n_ax, n_el); add frame & channel.
    wf = np.load(item["wave"]).astype(dtype)         # (256, 867, 256) -> (n_tx, n_ax, n_el)
    raw_data = wf[np.newaxis, ..., np.newaxis]        # (1, n_tx, n_ax, n_el, n_ch)

    time_vec = np.load(item["time"]).astype(np.float32).ravel()  # (T,)
    fs = np.float32(1.0 / (time_vec[1] - time_vec[0]))

    gt = np.load(item["gt"])                          # (2, Nz, Nx) fp64
    sos = gt[0].astype(np.float32)
    atten = gt[1].astype(np.float32)

    return raw_data, time_vec, sos, atten, fs


# ---------------------------------------------------------------------------
# zea writing
# ---------------------------------------------------------------------------

def build_scan(geo, time_vec, fs):
    """Assemble the zea `scan` dict (acquisition / beamforming parameters)."""
    n_tx = N_ELEMENTS
    return {
        "sampling_frequency": fs,
        "center_frequency": np.float32(CENTER_FREQ_HZ),
        "demodulation_frequency": np.float32(0.0),   # raw RF, no demodulation
        "sound_speed": np.float32(C_WATER),          # reference/background SOS
        # t = 0 is the emission centroid; first sample sits at time_vec[0].
        "initial_times": np.full(n_tx, time_vec[0], dtype=np.float32),
        "t0_delays": np.zeros((n_tx, N_ELEMENTS), dtype=np.float32),
        "tx_apodizations": geo["tx_apodizations"],
        "transmit_origins": geo["transmit_origins"],
        "azimuth_angles": geo["azimuth_angles"],
        "polar_angles": geo["polar_angles"],
        "focus_distances": geo["focus_distances"],
        # NB: probe_geometry lives in the `probe` dict, not `scan` (ScanSpec).
    }


def map_coordinates(shape, dx):
    """Per-pixel `[x, y, z]` positions in metres, centred on the ring (z=0).

    Returns an array of shape `(*shape, 3)` for a map of `shape = (H, W)`.
    """
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


def convert(item, cfg, geo, out_dir, dtype, File, CustomElement):
    raw_data, time_vec, sos, atten, fs = load_acquisition(item, dtype)

    scan = build_scan(geo, time_vec, fs)
    probe = {
        "name": "ring_2D_256",
        "type": "custom",
        "probe_geometry": geo["probe_geometry"],
        "element_width": geo["element_width"],
        "probe_center_frequency": np.float32(CENTER_FREQ_HZ),
        "probe_bandwidth_percent": np.float32(FRAC_BW * 100.0),
    }
    data = {"raw_data": raw_data, **ground_truth_maps(sos, atten, GT_DX_M)}
    metadata = {"subject": {"id": f"phantom_{item['pid']}", "type": "phantom"},
                "credit": CREDIT}
    custom = [
        CustomElement(name="tissue", data=np.array(cfg["tissue"]),
                      description="Breast tissue class of the source phantom", unit="-"),
        CustomElement(name="z_slice", data=np.array(int(item["z"]), dtype=np.int32),
                      description="Phantom z-slice index (from the source filename)",
                      unit="-"),
    ]

    out_path = out_dir / f"phantom_{item['pid']}_z{item['z']}.hdf5"
    File.create(
        str(out_path),
        data=data,
        scan=scan,
        probe=probe,
        metadata=metadata,
        custom=custom,
        description=(
            "Simulated 2D ring-array USCT RF acquisition (k-Wave, "
            f"{cfg['tissue']} digital breast phantom)"
        ),
        overwrite=True,
    )
    return out_path, raw_data.shape


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["dense", "fatty", "both"], default="both")
    ap.add_argument("--out-dir", type=Path, default=HERE / "data" / "2d",
                    help="output root (default: OpenH-RF/data/2d)")
    ap.add_argument("--dtype", choices=["float32"], default="float32",
                    help="raw_data storage dtype. zea's raw_data spec only allows "
                         "float32 or int16; float16 is silently upcast to float32, so "
                         "float32 is the lossless native option for this fp16 source.")
    ap.add_argument("--limit", type=int, default=None, help="convert at most N per dataset")
    ap.add_argument("--check", action="store_true",
                    help="validate pairing + array shapes only; do not import/write zea")
    args = ap.parse_args()

    names = ["dense", "fatty"] if args.dataset == "both" else [args.dataset]
    geo = ring_geometry()

    File = CustomElement = None
    if not args.check:
        from zea import File as _File  # noqa: imported lazily so --check needs no zea
        from zea.data.file import CustomElement as _CE
        File, CustomElement = _File, _CE

    grand_total = 0
    for name in names:
        cfg = DATASETS[name]
        items = discover(cfg)
        if args.limit:
            items = items[:args.limit]
        out_dir = args.out_dir  # dense + fatty mixed into one flat folder
        if not args.check:
            out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {name}: {len(items)} acquisitions ===")
        missing, done, errors = 0, 0, 0
        for i, item in enumerate(items, 1):
            if not item["time"].exists() or not item["gt"].exists():
                miss = "time" if not item["time"].exists() else "gt"
                print(f"  [{i}/{len(items)}] {item['base']}  ⚠ missing {miss} -> SKIP")
                missing += 1
                continue
            try:
                if args.check:
                    raw, tv, sos, atten, fs = load_acquisition(item, np.float32)
                    print(f"  [{i}/{len(items)}] {item['base']}  raw={raw.shape} "
                          f"T={tv.shape[0]} fs={fs/1e6:.3f}MHz "
                          f"sos={sos.shape}[{sos.min():.0f},{sos.max():.0f}] "
                          f"atten[{atten.min():.4f},{atten.max():.4f}]")
                else:
                    out_path, shape = convert(item, cfg, geo, out_dir,
                                              np.dtype(args.dtype), File, CustomElement)
                    print(f"  [{i}/{len(items)}] {out_path.name}  raw={shape}")
                done += 1
            except Exception as e:
                print(f"  [{i}/{len(items)}] {item['base']}  ✗ {e}")
                errors += 1
        print(f"  -> ok={done} missing={missing} errors={errors}")
        grand_total += done

    mode = "validated" if args.check else "written"
    print(f"\nDone. {grand_total} acquisitions {mode}.")


if __name__ == "__main__":
    main()
