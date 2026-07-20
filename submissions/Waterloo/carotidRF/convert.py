# SPDX-License-Identifier: Apache-2.0
"""Convert raw LITMUS RF frames into the openh-rf (zea) HDF5 format.

Reference / provenance script. It documents exactly how the shipped `hdf5/*.hdf5`
acquisitions were produced from the raw LITMUS acquisition frames. It is **not**
runnable from this folder alone: it requires the LITMUS core Python package
(`litmus.core_py`, GPU beamforming + Doppler), the acquisition spreadsheet, and
the per-frame `rf_frame*.mat` files, none of which are shipped here.

@ LITMUS Research Group, University of Waterloo, 2026.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Litmus imports (GPU beamforming + vector Doppler)
from litmus.core_py import (
    CPUcolorFlowProcessorLargeBatch_wParam,
    MultiAngleDopplerLargeBatch,
    Param,
    PixelMap,
    PWImage_formation_GPU_wParam,
    RFfilter5MHz,
    VectorFlow,
    bflowHighPass_general,
    bfType,
    vectorDoppler,
    vectorDopplerELSVD_PWI,
)
from scipy.io import loadmat
from zea.beamform.pixelgrid import cartesian_pixel_grid

# Zea imports
import zea
from zea import File

CREDIT = (
    "Hassan Nahas, Jason Y. -H. Hsu, Theresa Gu, Adrian J. Y. Chee, "
    "Alfred C. H. Yu. @ LITMUS @ University of Waterloo, Canada. 2026"
)

# Write files with the current zea file format. Bump this to the latest release
# before re-converting; the check below blocks conversion on an older zea so the
# output never carries a stale `zea_version`.
MIN_ZEA_VERSION = "0.1.3"

SCRIPT_DIR = Path(__file__).parent.resolve()


def require_latest_zea():
    """Block conversion unless zea is at least MIN_ZEA_VERSION."""
    from packaging.version import parse as parse_version

    installed = getattr(zea, "__version__", "0")
    if parse_version(installed) < parse_version(MIN_ZEA_VERSION):
        raise SystemExit(
            f"zea {installed} is too old: this converter requires zea >= {MIN_ZEA_VERSION} "
            f"so the output files carry the latest zea file format. "
            f"Upgrade with `pip install -U zea` and re-run."
        )


def resolve_path(p):
    """Resolve a path relative to the CWD or, failing that, this script's dir."""
    p_path = Path(p)
    if p_path.is_absolute() or p_path.exists():
        return str(p_path)
    script_rel = SCRIPT_DIR / p_path
    return str(script_rel if script_rel.exists() else p_path)


def main():
    # Block early if zea is too old, before any (slow) loading/beamforming.
    require_latest_zea()

    # Probe geometry: (128, 2) lateral/axial positions -> (128, 3) with y = 0.
    arrayMat = loadmat(resolve_path("l14_5_array.mat"))
    probe_geometry = arrayMat["probe_geom"]
    probe_geometry = np.concatenate(
        (probe_geometry, np.zeros((probe_geometry.shape[0], 1))), axis=1
    ).astype(np.float32)

    path_to_metadata = resolve_path("Database 1_wsl_local.xlsx")
    metadata_df = pd.read_excel(path_to_metadata, sheet_name="Acq")

    parser = argparse.ArgumentParser(
        description="Convert RF data and compute B-mode/Doppler fields."
    )
    parser.add_argument(
        "acqIDs",
        type=int,
        nargs="*",
        default=[6],
        help="Acquisition row index (or indices) in the Excel database. Use -1 for all.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of RF frames to load/process (default: all available)",
    )
    parser.add_argument(
        "--novectorflow",
        action="store_true",
        help="Only compute B-mode; skip all vector-velocity computation.",
    )
    args = parser.parse_args()
    acq_list = args.acqIDs
    max_frames = args.max_frames
    novectorflow = args.novectorflow

    if -1 in acq_list:
        acq_list = list(range(len(metadata_df)))

    for acqID in acq_list:
        if acqID < 0 or acqID >= len(metadata_df):
            print(
                f"Warning: acquisition row {acqID} out of bounds (0..{len(metadata_df) - 1}). Skipping."
            )
            continue

        print(f"\n{'=' * 50}")
        print(f"Starting pipeline for acquisition row {acqID} (row {acqID + 1} in Excel)...")
        print(f"{'=' * 50}")

        acq_metadata = metadata_df.iloc[acqID]
        needsDealiasing = bool(acq_metadata["Needs Dealiasing"])

        # Locate the raw frames + LITMUS parameter file for this acquisition.
        acq_id = int(acq_metadata["AcqID"]) if "AcqID" in acq_metadata else acqID
        samplePath = resolve_path(os.path.join("litmus_raw", f"Acq{acq_id}"))
        filepath = samplePath
        path_to_litmus_param = resolve_path(acq_metadata["Param"])

        if not os.path.exists(samplePath):
            print(f"Warning: RF frame directory '{samplePath}' does not exist. Skipping.")
            continue

        print(f"RF filepath: {samplePath}")
        print(f"Param path:  {path_to_litmus_param}")

        # Load acquisition parameters via the LITMUS loader.
        p = Param()
        p.acq = p.acq.load(path_to_litmus_param)
        numOfFirings = len(p.acq.txEventsPerBatch)

        # Count RF frames on disk and clamp to a multiple of the firings-per-frame.
        try:
            available_frames = len(
                [
                    f
                    for f in os.listdir(samplePath)
                    if f.startswith("rf_frame") and f.endswith(".mat")
                ]
            )
        except OSError:
            available_frames = 0

        target_frames = max_frames if max_frames is not None else (available_frames or 3000)
        if available_frames > 0:
            target_frames = min(target_frames, available_frames)

        numOfRFframes = (target_frames // numOfFirings) * numOfFirings or numOfFirings
        print(f"Processing {numOfRFframes} RF frames...")

        # Load the first frame to learn the (n_ax, n_el) shape.
        try:
            rf_data_first = loadmat(f"{samplePath}/rf_frame1.mat")["rf_data"]
        except Exception as e:
            print(f"Error loading first RF frame for row {acqID}: {e}. Skipping.")
            continue
        n_ax, n_el = rf_data_first.shape

        # Load all RF frames.
        rf_data = np.zeros((numOfRFframes, n_ax, n_el), dtype=np.float32)
        print(f"Loading {numOfRFframes} RF frames from disk...")
        try:
            for i in range(numOfRFframes):
                rf_data[i] = loadmat(f"{samplePath}/rf_frame{i + 1}.mat")["rf_data"].astype(
                    np.float32
                )
        except Exception as e:
            print(f"Error loading RF frames for row {acqID}: {e}. Skipping.")
            continue

        # (n_frames, n_tx, n_ax, n_el, 1)
        rawData = np.reshape(rf_data, (numOfRFframes // numOfFirings, numOfFirings, n_ax, n_el, 1))
        n_frames, n_tx, n_ax, n_el, _ = rawData.shape

        # --- Beamforming (GPU DAS, dual receive-angle compounding) ---
        p.bf.type = bfType.DAS
        p.bf.startFrame = 1
        p.bf.endFrame = numOfRFframes
        p.bf.skipFrame = 1
        p.bf.numFiringPerFrame = numOfFirings
        p.bf.deg_rx = np.zeros(p.bf.numFiringPerFrame, dtype=np.float32)
        p.bf.pixelMap = PixelMap([-19.0e-3, 0.0], [19.0e-3, 30.0e-3], 1e-4, 1e-4)
        p.bf.prefilter = RFfilter5MHz(p.acq.fs)
        p.bf.apertSetting = "full"
        p.bf.apertSize = 64
        p.bf.apodization = np.hanning(p.bf.apertSize).astype(np.float32)
        p.bf.Fnum = 1.5
        p.bf.gain = 1.0
        p.bf.lensCorrection = np.zeros(p.bf.numFiringPerFrame, dtype=np.float32)

        first_tx = p.acq.txEventsPerBatch[0]
        t_delay = float(first_tx.pulse.numCycles) / (2.0 * float(first_tx.pulse.f0))
        p.bf.delay = t_delay * np.ones(p.bf.numFiringPerFrame, dtype=np.float32)

        print("Beamforming (dual receive-angle compounding)...")
        p.bf.deg_rx = +15 * np.ones(p.bf.numFiringPerFrame, dtype=np.float32)
        HRI_PW_Prx = PWImage_formation_GPU_wParam(filepath, p, progress="on")
        p.bf.deg_rx = -15 * np.ones(p.bf.numFiringPerFrame, dtype=np.float32)
        HRI_PW_Nrx = PWImage_formation_GPU_wParam(filepath, p, progress="on")
        HRI_PW = 0.5 * (HRI_PW_Nrx + HRI_PW_Prx)

        Bmode = 20.0 * np.log10(np.abs(HRI_PW))
        Bmode[~np.isfinite(Bmode)] = 0.0
        processed_Bmode = np.transpose(Bmode, (2, 0, 1))

        upper_left = p.bf.pixelMap.UpperLeft
        bottom_right = p.bf.pixelMap.BottomRight
        processed_bmode_coords = cartesian_pixel_grid(
            xlims=(upper_left[0], bottom_right[0]),
            zlims=(upper_left[1], bottom_right[1]),
            grid_size_x=int(p.bf.pixelMap.numW),
            grid_size_z=int(p.bf.pixelMap.numH),
        ).astype(np.float32)

        # --- Vector flow (multi-angle Doppler -> least-squares vector Doppler) ---
        if not novectorflow:
            p.flow = VectorFlow()
            p.flow.clutterfilter = bflowHighPass_general(0.1, 0.15, 100)
            p.flow.ensemble = 64
            p.flow.slide = 1
            p.flow.ensembleS = 1
            p.flow.slideS = 1

            if p.bf.numFiringPerFrame == 2:
                p.flow.RxSteering = [-10.0, 10.0]
                p.flow.TxRxCombinations = np.array([[1, 1, 2, 2], [1, 2, 1, 2]], dtype=np.int32)
            else:
                p.flow.RxSteering = [-10.0, 0.0, 10.0]
                p.flow.TxRxCombinations = np.array([[1, 1, 1], [1, 2, 3]], dtype=np.int32)

            print("Computing Doppler frequencies...")
            vel_combine_multi, _, _ = MultiAngleDopplerLargeBatch(
                filepath, p, PWImage_formation_GPU_wParam
            )

            HRI_PW_filled = np.nan_to_num(HRI_PW, nan=0.0, posinf=0.0, neginf=0.0)
            _, pow_vals, powS, _ = CPUcolorFlowProcessorLargeBatch_wParam(HRI_PW_filled, p)

            print("Computing vector velocities...")
            velocitiesMap = vectorDoppler(p, vel_combine_multi, pow_vals[0] > 0)

            # Pad the (missing) final ensemble slice with NaN so the velocity maps
            # line up frame-for-frame with the B-mode stack.
            nan_complex = np.full(
                (velocitiesMap.shape[0], velocitiesMap.shape[1], 1),
                np.nan + 1j * np.nan,
                dtype=np.complex64,
            )
            nan_real = np.full(
                (pow_vals[0].shape[0], pow_vals[0].shape[1], 1), np.nan, dtype=np.float32
            )

            processed_vector = np.transpose(
                np.concatenate([velocitiesMap, nan_complex], axis=2), (2, 0, 1)
            )
            processed_power = np.transpose(
                np.concatenate([pow_vals[0], nan_real], axis=2), (2, 0, 1)
            )

            if needsDealiasing:
                print("Computing dealiased vector velocities...")
                if p.bf.numFiringPerFrame == 2:
                    p.flow.RxSteering = [10.0, 3.0, -6.0, -10.0, 6.0, -3.0, -10.0]
                    p.flow.TxRxCombinations = np.array(
                        [[1, 1, 1, 1, 2, 2, 2], [1, 2, 3, 4, 5, 6, 7]], dtype=np.int32
                    )
                else:
                    p.flow.RxSteering = [10.0, 6.0, 3.0, 0.0, -3.0, -6.0, -10.0]
                    p.flow.TxRxCombinations = np.array([[1, 1, 1], [1, 2, 3]], dtype=np.int32)
                p.flow.ELSVDshift = 1

                vel_combine_multi_deal, _, _ = MultiAngleDopplerLargeBatch(
                    filepath, p, PWImage_formation_GPU_wParam
                )
                velocitiesMap_dealiased, _, _ = vectorDopplerELSVD_PWI(
                    p, vel_combine_multi_deal, powS, pow_vals[0] > 0
                )
                processed_vector_deal = np.transpose(
                    np.concatenate([velocitiesMap_dealiased, nan_complex], axis=2), (2, 0, 1)
                )

        # --- Assemble zea metadata + data package ---
        print("Formatting metadata for zea HDF5 creation...")
        txAngles = []
        delayProfiles = []
        for event in p.acq.txEventsPerBatch:
            angle = event.txAngle
            txAngles.append(
                float(angle[0]) if isinstance(angle, (list, np.ndarray)) else float(angle)
            )
            dp = event.delayProfile
            delayProfiles.append(
                np.array(dp if isinstance(dp, (list, np.ndarray)) else [dp], dtype=np.float32)
            )
        delayProfiles = np.array(delayProfiles, dtype=np.float32).reshape(n_tx, n_el)

        first_event = p.acq.txEventsPerBatch[0]
        f0_val = float(first_event.pulse.f0)
        pri_val = float(first_event.pri)

        scan = {
            "sampling_frequency": np.float32(p.acq.fs),
            "center_frequency": np.float32(f0_val),
            "demodulation_frequency": np.float32(f0_val),
            "initial_times": np.zeros(n_tx, dtype=np.float32) + t_delay,
            "t0_delays": delayProfiles,
            "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
            "focus_distances": np.full(n_tx, np.inf),
            "transmit_origins": np.zeros((n_tx, 3)),
            "polar_angles": np.deg2rad(np.array(txAngles, dtype=np.float32)),
            "sound_speed": np.float32(p.acq.sos),
            "time_to_next_transmit": pri_val * np.ones((n_frames, n_tx), dtype=np.float32),
        }

        probe = {"name": "L14-5", "probe_geometry": probe_geometry, "type": "linear"}

        age_val = acq_metadata["Age"]
        subject = {
            "id": str(acq_metadata["Participant ID"]),
            "type": "human",
            "age": np.uint8(int(age_val) if not pd.isna(age_val) else 0),
            "sex": acq_metadata["Sex"],
        }
        annotations = {
            "anatomy": acq_metadata["Artery"],
            "view": acq_metadata["View"],
            "label": acq_metadata["Condition"],
        }
        metadata = {"credit": CREDIT, "subject": subject, "annotations": annotations}

        # Each map sets the dedicated `unit` field (machine-readable); `description`
        # is the human-readable name, without the unit baked in.
        data_package = {
            "raw_data": rawData,
            "image": {
                "values": processed_Bmode - np.max(processed_Bmode),
                "coordinates": processed_bmode_coords,
                "unit": "dB",
                "description": "B-mode",
            },
        }
        if not novectorflow:
            data_package["vector_velocity_x"] = {
                "values": processed_vector.real,
                "coordinates": processed_bmode_coords,
                "unit": "m/s",
                "description": "Vector velocity, lateral (x) component",
            }
            data_package["vector_velocity_z"] = {
                "values": processed_vector.imag,
                "coordinates": processed_bmode_coords,
                "unit": "m/s",
                "description": "Vector velocity, axial (z) component",
            }
            data_package["power_doppler"] = {
                "values": processed_power,
                "coordinates": processed_bmode_coords,
                "unit": "dB",
                "description": "Power Doppler",
            }
            if needsDealiasing:
                data_package["vector_velocity_x_deal"] = {
                    "values": processed_vector_deal.real,
                    "coordinates": processed_bmode_coords,
                    "unit": "m/s",
                    "description": "Dealiased vector velocity, lateral (x) component",
                }
                data_package["vector_velocity_z_deal"] = {
                    "values": processed_vector_deal.imag,
                    "coordinates": processed_bmode_coords,
                    "unit": "m/s",
                    "description": "Dealiased vector velocity, axial (z) component",
                }

        description = (
            f"UW-LITMUS-CarotidRF High frame rate in vivo carotid artery acquisition made "
            f"using {p.acq.scanner}. See annotations for details on anatomy, view, and condition."
        )

        hdf5_dir = resolve_path("hdf5")
        os.makedirs(hdf5_dir, exist_ok=True)
        hdf5_file_path = os.path.join(hdf5_dir, f"Acq{acqID}.hdf5")

        print(f"Writing zea file to: {hdf5_file_path}...")
        File.create(
            hdf5_file_path,
            data=data_package,
            scan=scan,
            description=description,
            probe=probe,
            metadata=metadata,
            overwrite=True,
        )
        print(f"Successfully created HDF5 zea file: {hdf5_file_path}")


if __name__ == "__main__":
    main()
