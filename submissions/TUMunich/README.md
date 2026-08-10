---
pretty_name: "OpenH-RF — Robotic Tracked Ultrasound (Verasonics L11-5gH, arm, CIRS 054GS and CIRS 074 thyroid phantoms)"
license: cc-by-4.0
task_categories:
  - image-to-3d
tags:
  - ultrasound
  - rf
  - openh-rf
  - 3d
  - robotic
  - freehand
language:
  - en
size_categories:
  - 10K<n<100K
---

# Robotic Tracked Ultrasound — Verasonics L11-5gH, vascular arm, CIRS 054GS and CIRS 074 thyroid phantoms

## Dataset Description

Robotically tracked freehand ultrasound of three tissue-mimicking phantoms, acquired with a Verasonics
Vantage NXT and an L11-5gH linear probe held by a KUKA LBR robot:

- a **Blue Phantom Gen II PICC, PIV and Arterial Line Vascular Access training model (BPA304-HP)** —
  an upper-extremity arm phantom with a nine-vessel system (cephalic, basilic, medial cubital, radial
  and ulnar veins plus brachial, radial and ulnar arteries) in self-healing tissue-mimicking material;
- a **CIRS Model 054GS** General Purpose Ultrasound Phantom;
- a **CIRS Model 074** Thyroid Ultrasound Training Phantom — a slightly enlarged thyroid gland in an
  anthropomorphic neck of Zerdine® hydrogel, with the trachea, common carotid artery and internal
  jugular vein as internal landmarks and one cyst plus one isoechoic stiff lesion per lobe.

Each frame is a multi-angle plane-wave acquisition of raw pre-beamformed channel data, and every
frame carries the measured robot end-effector pose, so the sweeps can be compounded into 3D.
**Multiple tracked sweeps were recorded from each phantom**: the sweeps belonging to one scan area are
concatenated into a single track within that file (multi-angle acquisition, except the synth aperture dataset).
The phantom was not moved between the scans: they are overlapping and tracking information is calibrated with respect to each other.
For the multi-angle data, individual sweeps are always recorded between two defined poses with varying out-of-plane rotation (around the x-axis).
The first sweep refers to -15 degree, with every following sweep raising that angle by 5 degree until it reaches 15 degree (in total 7 scans).
For the two transverse scans in the CIRS phantom, we only recorded -15 to +10 degree, due to the limited scan area of the robot.
For the two transverse scans in the thyroid phantom, we only recorded -10 to +10 degree, due to the limited scan area of the robot.
`custom/sweep_index` identifies the originating sweep per frame. This is **phantom** data (no human
or animal subjects). The intended contribution is a raw-channel-data benchmark for tracked freehand
reconstruction and beamforming.

## Dataset Contributor(s)

CAMP (Computer Aided Medical Procedures), Technical University of Munich (TUM).
Primary contact: Felix Dülmer <felix.duelmer@tum.de>.

## Dataset Creation Date

07/16/2026

## License / Terms of Use

CC BY 4.0. The data is phantom-derived and carries no patient consent or IP encumbrances, so it is
cleared for CC BY 4.0.

## Intended Usage

Primary task: **robotic tracked ultrasound acquisition** — compounding the per-frame robot poses with
the tracked sweeps for freehand 3D reconstruction. The raw channel data also supports advanced
beamforming research.

## Dataset Characterization

- **Data Collection Method:** phantom
- **Labeling Method:** N/A (no annotations; a co-registered robot pose stream is provided per frame)
- **Acquisition system:** Verasonics Vantage NXT; L11-5gH linear array, 128 elements, 0.30 mm pitch,
  7.6 MHz center frequency, 76.8% fractional bandwidth; 30.3 MHz receive sampling; 1540 m/s assumed
  sound speed. Transmit: 7 plane-wave angles (−18° … +18° in 6° steps), each acquired with three
  walking 64-element mux sub-apertures (21 acquisitions per frame). Imaging depth: 50 mm for the arm phantom 
  and 60 mm for the CIRS phantom.

## Dataset Format

*zea* HDF5. The recorded Verasonics `.vrs` files are read natively in Python (no MATLAB): the
multiplexed 64-channel receive apertures are expanded to the full 128-element probe dimension, and
the per-transmit hardware time tags embedded in the channel data are decoded to build
`time_to_next_transmit`. The transmit-time-gain-compensation (TGC) applied on the Verasonics hardware
is left in the raw data; no additional filtering, decimation, or demodulation is applied before
packaging.

## Dataset Quantification

- **Phantoms:** 3 (vascular access arm phantom, CIRS 054GS, CIRS 074 thyroid)
- **Files:** 11 zea HDF5 files across 3 sub-dataset folders
- **Sweeps:** multiple tracked sweeps per phantom, concatenated per file
- **Frames / acquisitions:** 17,852 frames in total; ~200–425 frames per sweep
  (i.e. for one file with 7 sweeps, ~2100 frames)
- **Total size on disk:** 56 GiB
- **Train / val / test split:** N/A

| Folder | Files | Frames | Sweeps per file |
|---|---|---|---|
| `arm_phantom/` | 2 | 5,018 | 7 |
| `cirs_phantom/` | 5 | 6,204 | 7 (longitudinal), 6 (transverse), 1 (synth. aperture) |
| `thyroid_phantom/` | 4 | 6,630 | 7 (longitudinal), 5 (transverse) |

The shapes below are for the CIRS 054GS file. The thyroid phantom files have identical per-frame
shapes and imaging geometry (60 mm depth, 2816 axial samples). **The arm phantom files were acquired
at 50 mm depth and so have 2560 axial samples** — their `raw_data` is `(#frames, 21, 2560, 128, 1)`.
Otherwise the layout is the same across all three, differing only in the leading frame count and the
number of sweeps.

| Field | Shape | dtype | Units | Description |
|---|---|---|---|---|
| `tracks/track_0/data/raw_data` | `(#frames, 21, 2816, 128, 1)` | int16 | ADC counts | Raw RF channel data (frames, transmits, axial samples, elements, 1) |
| `tracks/track_0/data/image` | `(#frames, 593, 379)` | uint8 | 8-bit intensity | Verasonics VSX B-mode reference per frame |
| `tracks/track_0/scan/*` | — | float32 | SI | Acquisition geometry/timing: `t0_delays`, `tx_apodizations`, `polar_angles`, `initial_times`, `time_to_next_transmit`, `sound_speed`, `center_frequency`, `sampling_frequency`, `tgc_gain_curve`, waveforms |
| `probe/*` | — | float32 | SI | L11-5gH `probe_geometry` (128, 3), `element_width`, `probe_bandwidth_percent`, `lens_thickness`, `lens_sound_speed` |
| `metadata/probe_pose/translation` | `(N, 3)` | float32 | m | Robot end-effector translation stream (~50 Hz) |
| `metadata/probe_pose/rotation` | `(N, 4)` | float32 | quaternion xyzw | Robot end-effector orientation stream |
| `custom/sweep_index` | `(#frames,)` | int32 | — | Which of the  sweeps each frame belongs to  |

The probe pose is an independently sampled stream; `probe_pose.start_time_offset` and
`time_to_next_transmit` place it on the same clock as the frames so the pose can be interpolated at
each frame's acquisition time.

`sweep_index` is the one quantity that has no home in the zea schema, so it is stored as a zea
custom element (written via `File.create(custom=[...])`, read back as `f.custom.sweep_index`).

### Acoustic lens

`probe/lens_thickness` is the Verasonics `Trans.lensCorrection` one-way lens path (0.6 mm; this probe
is defined with `Trans.units = 'mm'`). `probe/lens_sound_speed` is **not** reported by Verasonics and
is set to a nominal 1000 m/s for the silicone lens — override it with
`convert.py --lens-sound-speed`. Storing the lens physically means the reconstruction only sets
`apply_lens_correction: true` and zea applies its refraction model itself, rather than the
reconstruction patching `initial_times` by hand.

## Subject Metadata

- **Subjects:** 3 tissue-mimicking phantoms (no human/animal subjects).
- **Phantom models:**
  - Blue Phantom Gen II PICC, PIV and Arterial Line Vascular Access training model (BPA304-HP) —
    upper-extremity arm phantom, nine-vessel system, self-healing tissue-mimicking material.
  - CIRS Model 054GS (General Purpose Ultrasound Phantom).
  - CIRS Model 074 (Thyroid Ultrasound Training Phantom) — thyroid gland in an anthropomorphic neck,
    Zerdine® hydrogel, trachea/carotid/jugular landmarks, one cyst and one isoechoic stiff lesion per
    lobe.
- **Sweeps:** multiple tracked freehand sweeps per phantom.
- **Probe model:** Verasonics L11-5gH.

## Data Validation

`reconstruct.py` runs a `zea.Pipeline` that beamforms a frame from the raw channel data and compares
it to the stored Verasonics B-mode. The chain is: cast → band-pass (transducer band) → demodulate →
delay-and-sum compounding over all 21 transmits → envelope detect → normalize → power compression.

Both the operation chain and the reconstruction parameters are written to `pipeline.yaml`: the
`parameters:` key holds `f_number` (the Verasonics senscutoff aperture, 1.155), the grid limits, and
`apply_lens_correction: true`. Everything else — probe geometry, sound speed, the lens — is read from
the file by `File.load_parameters()`, so the whole recipe is reproducible from the HDF5 plus that one
YAML. A reference reconstruction is in `reconstructed.png`.

## Known Issues

- N/A

## Ethical Considerations

Phantom data only — no human or animal subjects, no PHI, and no IRB/consent required. No
de-identification is applicable.
