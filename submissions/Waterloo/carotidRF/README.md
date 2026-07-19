---
pretty_name: UW-CarotidRF
license: cc-by-4.0
task_categories:
  - other
tags:
  - ultrasound
  - rf
  - openh-rf
  - medical-imaging
  - in-vivo
  - carotid-artery
  - beamforming
  - vector-doppler
  - vector-flow-imaging
language:
  - en
size_categories:
  - n<4.5M
---

# UW-Carotid RF

Dataset consisting of raw RF data and vector velocity measurements of carotid
arteries acquired in in vivo carotid artery studies conducted by LITMUS @
University of Waterloo. The dataset consists of longitudinal and cross-sectional
images of the common and internal carotid arteries respectively.

This directory follows the OpenH-RF Sub-Dataset Submission Guide. The
**submission files** are:

| File | Role |
|------|------|
| `hdf5/*.hdf5` | Acquisition(s) in [`zea` file format](https://zea.readthedocs.io/en/v0.1.0a3/data-acquisition.html) (one file per acquisition) |
| [`reconstruct.py`](reconstruct.py) | Reference reconstruction → `.png`, driven by a `zea.Pipeline`, with an optional vector-flow overlay |
| [`pipeline.yaml`](pipeline.yaml) | The saved reconstruction pipeline |
| [`convert.py`](convert.py) | Provenance: how the raw LITMUS frames were converted to the `zea` HDF5 format (reference only; not runnable from this folder) |
| [`README.md`](README.md) | This data card |
| [`LICENCE.txt`](LICENCE.txt) | CC BY 4.0 |


# Data Card — CarotidRF

## Dataset Description

This is a dataset consisting of raw RF frames (plane wave) and vector flow
profiles of the carotid arteries (Common Carotid Artery and Internal Carotid
Artery) in humans, acquired using a programmable research scanner configured for
high frame rate vector flow imaging. The data was collected as part of studies
conducted by the LITMUS research group at the University of Waterloo, focusing on
carotid artery hemodynamics during baseline and physiological maneuvers (such as
the Valsalva Maneuver, head-down tilt, and supine postures).

## Dataset Contributor(s)

Hassan Nahas, Jason Y. -H. Hsu, Theresa Gu, Adrian J. Y. Chee, Alfred C. H. Yu

Correspondence emails:
hassan.nahas@uwaterloo.ca
jason.hsu@uwaterloo.ca
theresa.gu@uwaterloo.ca
alfred.yu@uwaterloo.ca

## Dataset Creation Date

07/16/2026

## License / Terms of Use

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en).

All human studies were approved by the University of Waterloo’s Human Research
Ethics Board (ORE #46278). All included data was acquired from participants who
provided both written and verbal consent prior to participating in the study
regarding public data sharing.

## Intended Usage

Developing, benchmarking, and evaluating methods for ultrasound image
reconstruction, motion estimation, clutter filtering, multi-angle Doppler
processing, and vector flow imaging (VFI) in carotid artery imaging.

## Dataset Characterization

- **Data Collection Method:** In vivo imaging of human carotid arteries (Common Carotid Artery and Internal Carotid Artery).
- **Labeling Method:** Categorized by target artery (Anatomy), view direction (Longitudinal or Cross-sectional), and physiological condition/maneuver (Baseline, Valsalva Maneuver, Valsalva Maneuver – Supine, Valsalva Maneuver – Head Down Tilt, Head Down Tilt).
- **Acquisition system:**
  Raw RF data was acquired from programmable research scanners (US4R/US4R-Lite, US4US, Warsaw, Poland) equipped with an L14-5 linear array transducer.

## Dataset Format

Submitted in the [`zea` file format](https://zea.readthedocs.io/en/v0.1.0a3/data-acquisition.html) (one HDF5 file per acquisition).

Per-sample contents of the converted HDF5:

| Group / field | Shape | Dtype | Units | Description |
|---|---|---|---|---|
| `metadata/credit` | `[]` | str | -- | Dataset attribution (LITMUS @ University of Waterloo) |
| `metadata/subject/id` | `[]` | str | -- | Participant ID (needed for subject-wise splits) |
| `metadata/subject/type` | `[]` | str | -- | Subject type (`human`) |
| `metadata/subject/age` | `[]` | uint8 | years | Subject age (`0` if not available) |
| `metadata/subject/sex` | `[]` | str | -- | Subject sex (`M` or `F`) |
| `metadata/annotations/anatomy` | `[]` | str | -- | Target artery (e.g., `Common Carotid Artery`, `Internal Carotid Artery`) |
| `metadata/annotations/view` | `[]` | str | -- | View orientation (`Longitudinal` or `Cross-sectional`) |
| `metadata/annotations/label` | `[]` | str | -- | Physiological condition (e.g., `Baseline`, `Valsalva Maneuver`) |
| `probe/name` | `[]` | str | -- | Probe identifier (`L14-5`) |
| `probe/type` | `[]` | str | -- | Probe structure type (`linear`) |
| `probe/probe_geometry` | `[128, 3]` | float32 | m | Transducer element Cartesian positions (x, y, z) |
| `data/raw_data` | `[n_frames, n_tx, n_ax, n_el, 1]` | float32 | -- | Raw RF channel data |
| `data/image` | `[n_frames, z, x]` (+ `coordinates` `[z, x, 3]`) | float32 | dB | Stored log-compressed B-mode intensity (relative to peak) |
| `data/vector_velocity_x` | `[n_frames, z, x]` (+ `coordinates` `[z, x, 3]`) | float32 | m/s | Lateral component of vector velocity ($v_x$) |
| `data/vector_velocity_z` | `[n_frames, z, x]` (+ `coordinates` `[z, x, 3]`) | float32 | m/s | Axial component of vector velocity ($v_z$) |
| `data/power_doppler` | `[n_frames, z, x]` (+ `coordinates` `[z, x, 3]`) | float32 | dB | Power Doppler intensity |
| `scan/*` | -- | -- | -- | Probe geometry, sampling/center/demodulation frequency, t0 delays, sound speed, transmit angles, focus distances, transmit origins, apodizations, PRI... |

All `coordinates` arrays are per-pixel Cartesian positions in meters, last axis
`[x, y, z]` (y = 0 for 2-D maps).

> **Note on dealiasing:** acquisitions that require it also carry
> `data/vector_velocity_x_deal` and `data/vector_velocity_z_deal` (dealiased
> $v_x$/$v_z$). The two example acquisitions shipped here do not require
> dealiasing and therefore omit those fields.

## Shipped Example Acquisitions

Two example acquisitions are included under `hdf5/` as a representative subset of
the full dataset:

| File | Subject | Anatomy | View | Condition | Frames |
|---|---|---|---|---|---|
| `hdf5/Acq0.hdf5` | 1 | Common Carotid Artery | Longitudinal | Baseline | 500 |
| `hdf5/Acq1.hdf5` | 1 | Common Carotid Artery | Longitudinal | Baseline | 500 |

Each frame comprises 2 steered plane-wave transmits (`n_tx = 2`), 2048 axial
samples, and 128 receive channels. The frame count in these examples is truncated
for demonstration; full acquisitions contain the frame counts described below.

## Dataset Quantification

Data was collected from 8 participants, spanning carotid arteries (Common Carotid
Artery and Internal Carotid Artery) in both longitudinal and cross-sectional
views. In total, the dataset consists of 93 acquisitions, containing
36,000/60,000 frames of raw RF data per acquisition.

## Subject Metadata

| Metric | Value |
| :--- | :--- |
| **Total Number of Subjects** | 8 |
| **Total Number of Files (Acquisitions)** | 93 |
| **Sex Composition** | M: 6 (75.0%), F: 2 (25.0%) |
| **Total RF Frames** | 4,476,000 |

## Known Issues

- Acquisitions made with US4R-lite include x2 receive multiplexing, so the full frame is constructed from two transmissions (receiving the first 64 channels first, then the second 64 channels).
- Participants may overlap with other UWaterloo submissions.
- Due to hardware, some acquisitions may have elevated Doppler noise on the right side of the image.
- Some acquisitions may have not hit the target view optimally, leading to poor flow detection.
- Due to the reduced PRF with the US4R-lite, some internal carotid artery acquisitions may contain sporadic/minor aliasing.

## Beamforming and Processing

1. **Pre-Filtering:**
   Channel RF data is pre-filtered using a 5 MHz bandpass filter before beamforming.
2. **GPU-Accelerated Beamforming (DAS):**
   Beamforming is carried out via a GPU-accelerated Delay-and-Sum (DAS) module.
   - **Aperture & Apodization:** 64-element Hanning window apodization and an F-number of 1.5.
   - **Dual Angle-Compounding:** Beamforming for B-mode and power Doppler is performed twice with opposite receive angles ($+15^{\circ}$ and $-15^{\circ}$). The final high-resolution beamformed image (HRI) is the average of these two acquisitions:
     $$HRI = \frac{HRI_{+15^{\circ}} + HRI_{-15^{\circ}}}{2}$$
   - **Reconstruction Grid:** Cartesian coordinates mapped by a `PixelMap` representing a lateral range of $[-19, 19]\text{ mm}$ and axial depth of $[0, 30]\text{ mm}$ at $0.1\text{ mm}$ spatial resolution.
3. **Clutter Filtering:**
   Clutter filtering is performed on the beamformed ensemble using a high-pass wall filter (normalized cut-off frequencies of 0.1 and 0.15, filter length of 100).
4. **Multi-Angle Doppler Frequency Estimation:**
   Angle-specific Doppler frequencies are computed using an ensemble size of 64 frames with a step size of 1.
   - For acquisitions with 2 tx angles, we used the following Tx-Rx angles:
     Tx: [-10, -10, 10, 10]; Rx: [-10, 10, -10, 10]
   - For acquisitions with 1 tx angle:
     Tx: [-10, -10, -10]; Rx: [-10, 0, 10]
5. **Vector Doppler Velocity Estimation:**
   Lateral ($v_x$) and axial ($v_z$) velocity components are computed from the multi-angle Doppler frequency estimates using least-squares estimation.

The full LITMUS processing pipeline (GPU DAS beamforming + multi-angle vector
Doppler) is documented in [`convert.py`](convert.py). That script is included for
provenance and reproducibility; it depends on the LITMUS core Python package and
the raw acquisition frames, so it is not runnable from this folder alone.

Papers relevant to our pipeline:

Y. -H. Hsu (2025). High-frame-rate ultrasound characterization of carotid pulse waves to assess cerebrovascular resistance [Doctoral dissertation, University of Waterloo]. UWSpace. https://hdl.handle.net/10012/21738

H. Nahas, B. Y. S. Yiu, A. J. Y. Chee, T. Ishii and A. C. H. Yu, "Bedside Ultrasound Vector Doppler Imaging System With GPU Processing and Deep Learning," in IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, vol. 72, no. 8, pp. 1079-1094, Aug. 2025, doi: 10.1109/TUFFC.2025.3582773

B. Y. S. Yiu and A. C. H. Yu, "Least-Squares Multi-Angle Doppler Estimators for Plane-Wave Vector Flow Imaging," in IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, vol. 63, no. 11, pp. 1733-1744, Nov. 2016, doi: 10.1109/TUFFC.2016.2582514

## Data Validation

[`reconstruct.py`](reconstruct.py) builds a `zea.Pipeline` of DAS beamforming →
envelope detection → normalization → log-compression **in code** and
reconstructs a B-mode directly from `raw_data`, showing the raw-to-image flow
without any config file. It also saves the pipeline to
[`pipeline.yaml`](pipeline.yaml) as a shareable recipe. Comparing the
reconstruction against the stored (LITMUS) B-mode is a sanity check that the
acquisition parameters and probe geometry are recorded correctly, and serves as
a reproducible reference reconstruction.

When the vector-flow fields (`vector_velocity_x/z` + `power_doppler`) are present,
a third panel overlays the vector velocity field on the stored B-mode. The
overlay uses `draw_velocity_field`, a single self-contained (numpy + matplotlib)
helper reproduced inside `reconstruct.py` from the LITMUS core Python package
(`litmus.core_py.visualization`), so the script has no dependency on the full
LITMUS GPU stack.

The result is written to `reconstruct_output.png`:

![reference reconstruction](reconstruct_output.png)

### Example Usage of reconstruct.py

```bash
# Reconstruct the default file (hdf5/Acq0.hdf5) at frame 100
python reconstruct.py

# Reconstruct a specific file and frame, and adjust the power-Doppler mask
python reconstruct.py --input hdf5/Acq1.hdf5 --frame 250 --power-threshold 55.0
```

## Ethical Considerations

All human studies were approved by the University of Waterloo’s Human Research
Ethics Board (ORE #46278). All included data was acquired from participants who
provided both written and verbal consent prior to participating in the study
regarding public data sharing.

## Citation

```bibtex
@phdthesis{Hsu2025Carotid,
  title  = {High-frame-rate ultrasound characterization of carotid pulse waves to assess cerebrovascular resistance},
  author = {Hsu, Jason Y.-H.},
  school = {University of Waterloo},
  year   = {2025},
  note   = {UWSpace, https://hdl.handle.net/10012/21738},
}
```
