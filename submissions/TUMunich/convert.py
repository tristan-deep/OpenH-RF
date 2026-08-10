"""Convert recorded Verasonics sweeps for one patient into a single OpenH-RF file.

The input is a patient folder holding one subfolder per sweep, each with
``rf_data/``, ``beamformed_data/`` and ``robot/``. Those sweep folders are found
at any depth, so both ``Patient/session/`` and the newer ``Patient/session/
angle_p005_0deg/`` (grouped by probe tilt) layouts work. Every sweep is read,
their frames concatenated into a single track, and the robot tracking attached
as a raw pose stream. A per-frame ``sweep_index`` recording which sweep each frame
came from is saved as a zea custom element, readable as ``f.custom.sweep_index``.

Uncompressed .vrs files are read natively in Python from the flat metadata.hdf5
written by the panel; MATLAB is not used during conversion.
"""

import argparse
import os
import struct
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import h5py
import matplotlib.image as mpimg
import numpy as np

import zea
from zea.beamform.pixelgrid import cartesian_pixel_grid
from zea.data.file import CustomElement, File


HERE = Path(__file__).parent
DEFAULT_OUTPUT = HERE / "robotic_ultrasound_sample.hdf5"
VRS_PATTERN = "L11_5gHPlaneWaveAnglesRF*.vrs"
VRS_DTYPE = np.dtype("<i2")


def read_vrs(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        handle.read(4)
        has_timestamp = handle.read(1)[0]
        if has_timestamp:
            handle.read(6)
        for _ in range(3):
            handle.seek(struct.unpack("<Q", handle.read(8))[0], os.SEEK_CUR)
        dimensions = tuple(value for value in struct.unpack("<4Q", handle.read(32)) if value > 1)
        number_of_points, = struct.unpack("<Q", handle.read(8))
        handle.read(1)
        data = np.fromfile(handle, dtype=VRS_DTYPE, count=number_of_points)
    return data.reshape(dimensions, order="F")


def load_metadata(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Record the session with the updated operator panel."
        )
    with h5py.File(path, "r") as h5:
        metadata = {
            "tx_delays_seconds": np.asarray(h5["tx_delays_seconds"]).T.squeeze(),
            "tx_apodizations": np.asarray(h5["tx_apodizations"]).T.squeeze(),
            "polar_angles_rad": np.asarray(h5["polar_angles_rad"]).T.squeeze(),
            "azimuth_angles_rad": np.asarray(h5["azimuth_angles_rad"]).T.squeeze(),
            "initial_times_seconds": np.asarray(h5["initial_times_seconds"]).T.squeeze(),
            "focus_distances_m": np.asarray(h5["focus_distances_m"]).T.squeeze(),
            "transmit_origins_m": np.asarray(h5["transmit_origins_m"]).T.squeeze(),
            "receive_start_samples": np.asarray(h5["receive_start_samples"]).T.squeeze(),
            "receive_end_samples": np.asarray(h5["receive_end_samples"]).T.squeeze(),
            "receive_channel_by_element": np.asarray(
                h5["receive_channel_by_element"]
            ).T.squeeze(),
            "time_to_next_transmit_seconds": np.asarray(
                h5["time_to_next_transmit_seconds"]
            ).T.squeeze(),
            "probe_geometry_m": np.asarray(h5["probe_geometry_m"]).T.squeeze(),
            "waveform_one_way": np.asarray(h5["waveform_one_way"]).T.squeeze(),
            "waveform_two_way": np.asarray(h5["waveform_two_way"]).T.squeeze(),
            "tgc_gain_curve": np.asarray(h5["tgc_gain_curve"]).T.squeeze(),
            "sound_speed_m_per_s": float(h5["sound_speed_m_per_s"][0, 0]),
            "sampling_frequency_hz": float(h5["sampling_frequency_hz"][0, 0]),
            "demodulation_frequency_hz": float(
                h5["demodulation_frequency_hz"][0, 0]
            ),
            "center_frequency_hz": float(h5["center_frequency_hz"][0, 0]),
            "probe_center_frequency_hz": float(
                h5["probe_center_frequency_hz"][0, 0]
            ),
            "probe_bandwidth_percent": float(h5["probe_bandwidth_percent"][0, 0]),
            "probe_element_width_m": float(h5["probe_element_width_m"][0, 0]),
            # Verasonics Trans.lensCorrection, i.e. the one-way lens path length.
            # The panel records it under a "_wavelengths" name, but this probe is
            # defined with Trans.units = 'mm' (ElementPos and elementWidth are
            # likewise written in mm by scripts/save_verasonics_metadata.m), so
            # the recorded value is in millimetres. It is the physical lens
            # thickness, used directly as probe.lens_thickness.
            "probe_lens_thickness_m": float(
                h5["probe_lens_correction_wavelengths"][0, 0]
            )
            * 1e-3,
            "bmode_start_depth_mm": float(h5["bmode_start_depth_mm"][0, 0]),
            "bmode_end_depth_mm": float(h5["bmode_end_depth_mm"][0, 0]),
            "bmode_extent_m": np.asarray(h5["bmode_extent_m"]).T.squeeze(),
        }
    return metadata


def load_paired_files(
    session_dir: Path, max_frames: int | None
) -> tuple[list[Path], list[Path], np.ndarray]:
    rf_folder = session_dir / "rf_data"
    bmode_folder = session_dir / "beamformed_data"

    frame_slice = slice(None, max_frames)
    vrs_files = sorted(rf_folder.glob(VRS_PATTERN))[frame_slice]
    bmode_files = sorted(bmode_folder.glob("BMode_*.png"))[frame_slice]
    timestamps = np.loadtxt(
        bmode_folder / "bmode_timestamps.txt",
        skiprows=1,
        usecols=2,
        ndmin=1,
    )[frame_slice]
    # A recording stopped mid-frame leaves a trailing B-mode with no .vrs (or the
    # reverse), so pair by frame index and drop whatever hangs off the end.
    paired = min(len(vrs_files), len(bmode_files), len(timestamps))
    unpaired = max(len(vrs_files), len(bmode_files), len(timestamps)) - paired
    if unpaired:
        print(
            f"{session_dir.name}: dropping {unpaired} unpaired trailing frame(s) "
            f"(RF {len(vrs_files)}, B-mode {len(bmode_files)}, "
            f"timestamps {len(timestamps)})."
        )
    return (
        vrs_files[:paired],
        bmode_files[:paired],
        timestamps[:paired],
    )


def expand_vrs_frame(frame: np.ndarray, metadata: dict) -> np.ndarray:
    """Expand multiplexed 64-channel VRS data to the 128 probe elements.

    The VRS frame stores all receive events consecutively as (rows, 64 hardware
    channels). This separates the receive events, extracts their valid sample
    windows, and maps each hardware channel to its physical probe element.
    Inactive probe elements remain zero.

    Returns data shaped as (transmits, axial samples, probe elements).
    """
    if frame.ndim != 2:
        raise ValueError(f"Expected a two-dimensional VRS frame, got {frame.shape}.")
    starts = metadata["receive_start_samples"].astype(int) - 1
    ends = metadata["receive_end_samples"].astype(int)
    channel_map = metadata["receive_channel_by_element"].astype(int)
    n_tx, n_elements = channel_map.shape
    n_ax = int(ends[0] - starts[0])
    output = np.zeros((n_tx, n_ax, n_elements), dtype=frame.dtype)
    for tx in range(n_tx):
        block = frame[starts[tx] : ends[tx], :]
        if block.shape[0] != n_ax:
            raise ValueError(f"Receive window {tx} has inconsistent axial size {block.shape[0]}.")
        active_elements = np.flatnonzero(channel_map[tx] > 0)
        source_channels = channel_map[tx, active_elements] - 1
        output[tx][:, active_elements] = block[:, source_channels]
    return output


def frame_time_tags_seconds(frame: np.ndarray, start_samples: np.ndarray) -> np.ndarray:
    """Per-transmit hardware time tags of one .vrs frame, in seconds since midnight.
    """
    lsb = frame[start_samples - 1, 0].astype(np.uint16).astype(np.uint64)
    msb = frame[start_samples, 0].astype(np.uint16).astype(np.uint64)
    return ((msb << np.uint64(16)) | lsb).astype(np.float64) * 25e-6


def load_bmode_images(files: list[Path]) -> np.ndarray:
    frames = []
    for path in files:
        image = mpimg.imread(path)
        if image.ndim == 3:
            image = image[..., 0]
        if np.issubdtype(image.dtype, np.floating):
            image = np.rint(np.clip(image, 0, 1) * 255)
        frames.append(image.astype(np.uint8))
    return np.stack(frames)


def bmode_coordinates(metadata: dict, image_shape: tuple[int, int]) -> np.ndarray:
    x_min, x_max, z_max, z_min = metadata["bmode_extent_m"]
    return cartesian_pixel_grid(
        xlims=(x_min, x_max),
        zlims=(z_min, z_max),
        grid_size_x=image_shape[1],
        grid_size_z=image_shape[0],
    ).astype(np.float32)


def load_pose_stream(robot_dir: Path) -> dict:
    """Parse robot/end_effector.txt into a raw probe-pose stream.

    Each line is ``timestamp tx ty tz qx qy qz qw`` with translation already in
    metres and an xyzw quaternion, sampled continuously (~50 Hz).
    """
    table = np.loadtxt(robot_dir / "end_effector.txt", ndmin=2)
    return {
        "timestamps": table[:, 0],
        "translation": table[:, 1:4].astype(np.float32),
        "rotation": table[:, 4:8].astype(np.float32),
    }


def open_raw_data_scratch(output_path: Path, shape: tuple[int, ...], dtype) -> np.memmap:
    """Allocate the raw_data buffer on disk instead of in RAM.

    A full patient is tens of gigabytes of channel data, more than fits in
    memory, so the frames are written into a scratch file next to the output and
    h5py streams from that mapping when the file is created. The scratch file is
    unlinked as soon as it is mapped: the mapping stays valid, but the space is
    handed back by the OS when this process exits, including after a crash.
    """
    scratch_path = output_path.with_suffix(".raw_data.tmp")
    n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    free_bytes = os.statvfs(output_path.parent).f_bavail * os.statvfs(output_path.parent).f_frsize
    if n_bytes > free_bytes:
        raise OSError(
            f"raw_data needs {n_bytes / 1e9:.1f} GB of scratch space in "
            f"{output_path.parent}, which has {free_bytes / 1e9:.1f} GB free."
        )
    print(f"Buffering {n_bytes / 1e9:.1f} GB of raw_data via {scratch_path}")
    buffer = np.memmap(scratch_path, dtype=dtype, mode="w+", shape=shape)
    scratch_path.unlink()
    return buffer


def sweep_start_time(sweep: Path) -> float:
    """Unix time of a sweep's first robot pose sample."""
    with (sweep / "robot" / "end_effector.txt").open() as handle:
        return float(next(handle).split()[0])


def convert_patient(
    patient_dir: Path,
    output_path: Path,
    max_frames: int | None,
) -> None:
    """Convert every sweep under a patient folder into one OpenH-RF file."""
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {patient_dir}")

    # A sweep is any folder holding rf_data/. Older sessions put those directly
    # under the patient folder, newer ones group them one level deeper by probe
    # tilt (session/angle_p005_0deg/), so search at any depth. Order by
    # acquisition time, not by name: the tilt-angle names do not sort
    # chronologically, and both the pose stream and the RF time tags have to
    # keep increasing across the concatenated sweeps.
    sweeps = sorted(
        (p.parent for p in patient_dir.rglob("rf_data") if p.is_dir()),
        key=sweep_start_time,
    )
    if not sweeps:
        raise FileNotFoundError(
            f"No sweeps under {patient_dir}: expected folders holding rf_data/, "
            "beamformed_data/ and robot/."
        )
    reference = load_metadata(sweeps[0] / "rf_data" / "metadata.hdf5")

    # First pass: count frames so the big arrays can be preallocated instead of
    # stacking every sweep in memory at once.
    per_sweep_files = []
    for sweep in sweeps:
        per_sweep_files.append((sweep, *load_paired_files(sweep, max_frames)))

    total_frames = sum(len(vrs) for _, vrs, _, _ in per_sweep_files)
    probe_frame = expand_vrs_frame(read_vrs(per_sweep_files[0][1][0]), reference)
    n_tx, n_ax, n_el = probe_frame.shape

    start_samples = reference["receive_start_samples"].astype(int)
    raw_data = open_raw_data_scratch(
        output_path, (total_frames, n_tx, n_ax, n_el, 1), probe_frame.dtype
    )
    time_tags = np.zeros((total_frames, n_tx), dtype=np.float64)
    sweep_index = np.zeros(total_frames, dtype=np.int32)
    image_list = []
    pose_streams = []

    # Second pass: fill the preallocated arrays one sweep at a time.
    cursor = 0
    for sweep_id, (sweep, vrs_files, bmode_files, timestamps) in enumerate(per_sweep_files):
        for offset, path in enumerate(vrs_files):
            frame = read_vrs(path)
            raw_data[cursor + offset, ..., 0] = expand_vrs_frame(frame, reference)
            time_tags[cursor + offset] = frame_time_tags_seconds(frame, start_samples)
        n = len(vrs_files)
        sweep_index[cursor : cursor + n] = sweep_id
        image_list.append(load_bmode_images(bmode_files))
        pose_streams.append(load_pose_stream(sweep / "robot"))
        cursor += n
        print(f"Loaded sweep {sweep_id} ({sweep.relative_to(patient_dir)}): {n} frames")

    images = np.concatenate(image_list, axis=0)
    coordinates = bmode_coordinates(reference, images.shape[1:])
    if images.shape[1:] != coordinates.shape[:2]:
        raise ValueError(
            f"B-mode image shape {images.shape[1:]} does not match recorded metadata "
            f"{coordinates.shape[:2]}."
        )

    # The RF time tags are seconds since local midnight (Verasonics clock). Put
    # the robot pose on the same axis by subtracting that day's local midnight
    # from its Unix timestamps
    pose_timestamps = np.concatenate([p["timestamps"] for p in pose_streams])
    midnight = datetime.fromtimestamp(pose_timestamps[0]).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    pose_since_midnight = pose_timestamps - midnight.timestamp()
    probe_pose = {
        "translation": np.concatenate([p["translation"] for p in pose_streams]),
        "rotation": np.concatenate([p["rotation"] for p in pose_streams]),
        "rotation_representation": "quaternion_xyzw",
        "start_time_offset": np.float32(pose_since_midnight[0] - time_tags[0, 0]),
        "timestamps": (pose_since_midnight - pose_since_midnight[0]).astype(np.float32),
    }

    time_to_next_transmit = np.diff(time_tags.reshape(-1)).astype(np.float32)
    scan = {
        "time_to_next_transmit": time_to_next_transmit,
        "t0_delays": reference["tx_delays_seconds"],
        "tx_apodizations": reference["tx_apodizations"].astype(np.float32),
        "sampling_frequency": reference["sampling_frequency_hz"],
        "polar_angles": reference["polar_angles_rad"],
        "azimuth_angles": reference["azimuth_angles_rad"],
        "center_frequency": np.full(n_tx, reference["center_frequency_hz"]),
        "demodulation_frequency": reference["demodulation_frequency_hz"],
        "sound_speed": reference["sound_speed_m_per_s"],
        "initial_times": reference["initial_times_seconds"].astype(np.float32),
        "focus_distances": reference["focus_distances_m"],
        "transmit_origins": reference["transmit_origins_m"],
        "waveforms_one_way": reference["waveform_one_way"],
        "waveforms_two_way": reference["waveform_two_way"],
        "tgc_gain_curve": reference["tgc_gain_curve"],
    }
    probe = {
        "name": "L11-5gH",
        "type": "linear",
        "probe_center_frequency": reference["probe_center_frequency_hz"],
        "probe_bandwidth_percent": np.asarray(reference["probe_bandwidth_percent"]),
        "probe_geometry": reference["probe_geometry_m"],
        "element_width": reference["probe_element_width_m"],
        "lens_thickness": np.float32(reference["probe_lens_thickness_m"]),
        "lens_sound_speed": 1000.0,
    }
    custom = [
        CustomElement(
            name="sweep_index",
            data=sweep_index,
            description=(
                "Zero-based index of the tracked sweep each frame belongs to. The "
                f"{len(per_sweep_files)} sweeps over the same phantom are concatenated "
                "into a single track."
            ),
            unit="unitless",
        ),
    ]
    data = {
        "raw_data": raw_data,
        "image": {
            "values": images,
            "coordinates": coordinates,
            "description": "Saved log-compressed B-mode frames from the robotic ultrasound panel."
        },
    }
    File.create(
        path=str(output_path),
        data=data,
        scan=scan,
        probe=probe,
        metadata={
            "subject": {"id": "cirs_074_thyroid_phantom", "type": "phantom"},
            # "subject": {"id": "cirs_054gs_phantom", "type": "phantom"},
            # "subject": {"id": "blue_phantom_bpa304hp_arm", "type": "phantom"},
            "credit": "CAMP, Technical University of Munich (TUM).",
            "probe_pose": probe_pose,
        },
        description=(
            "Robotically tracked freehand ultrasound of a CIRS Model 074 thyroid ultrasound "
            "training phantom, a slightly enlarged thyroid gland in an anthropomorphic neck of "
            "Zerdine hydrogel, with the trachea, common carotid artery and internal jugular vein "
            "as internal landmarks and one cyst plus one isoechoic stiff lesion per lobe. "
            "Verasonics Vantage NXT with an L11-5gH linear probe (128 elements, 0.30 mm pitch, "
            "7.6 MHz) held by a KUKA LBR robot. Each frame is a 7-angle plane-wave acquisition "
            "(-18 to +18 deg), 21 walking 64-element mux acquisitions per frame, raw "
            "pre-beamformed channel data with the measured end-effector pose attached per frame. "
            "Seven (longitudinal) or 5 (transverse) tracked sweeps over the same are "
            "are concatenated, with respective out-of-plane angles [(-15), -10, -5, 0, 5, 10, (15)] degrees ."
        ),
        # description=(
        #     "Robotically tracked freehand ultrasound of a CIRS Model 054GS general-purpose "
        #     "phantom. Verasonics Vantage NXT with an L11-5gH linear probe (128 elements, "
        #     "0.30 mm pitch, 7.6 MHz) held by a KUKA LBR robot. Each frame is a 7-angle "
        #     "plane-wave acquisition (-18 to +18 deg), 21 walking 64-element mux acquisitions "
        #     "per frame, raw pre-beamformed channel data with the measured end-effector pose "
        #     "attached per frame. A in-plane lateral sweep is performed covering approx. 7 cm"
        #     # "Seven tracked sweeps over the same phantom "
        #     # "are concatenated, with respective out-of-plane angles [-15, -10, -5, 0, 5, 10, 15] degrees ."
        # ),
        # description=(
        #     "Robotically tracked freehand ultrasound of a Blue Phantom Gen II PICC, PIV and "
        #     "Arterial Line Vascular Access training model (BPA304-HP), an upper-extremity arm "
        #     "phantom with a nine-vessel system (cephalic, basilic, medial cubital, radial and "
        #     "ulnar veins plus brachial, radial and ulnar arteries) embedded in self-healing "
        #     "tissue-mimicking material. Verasonics Vantage NXT with an L11-5gH linear probe "
        #     "(128 elements, 0.30 mm pitch, 7.6 MHz) held by a KUKA LBR robot. Each frame is a "
        #     "7-angle plane-wave acquisition (-18 to +18 deg), 21 walking 64-element mux "
        #     "acquisitions per frame, raw pre-beamformed channel data with the measured "
        #     "end-effector pose attached per frame. Seven tracked sweeps over the same phantom "
        #     "are concatenated, with respective out-of-plane angles [-15, -10, -5, 0, 5, 10, 15] degrees ."
        # ),
        custom=custom,
        overwrite=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Patient folder containing sweep subfolders.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-frames", type=int, help="Convert only the first N paired frames per sweep."
    )
    args = parser.parse_args()
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    zea.init_device()
    convert_patient(args.input, args.output, args.max_frames)
    print(f"Done. {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
