---
pretty_name: "USTB - Phantom (Verasonics L7-4 / P4)"
license: cc-by-4.0
task_categories:
  - image-to-image
language:
  - en
tags:
  - ultrasound
  - rf
  - openh-rf
  - phantom
  - cirs
size_categories:
  - n<1K
---

# USTB - Phantom (Verasonics L7-4 / P4)

Part of the **UltraSound ToolBox (USTB) Channel Capture Collection** contributed to the
[OpenH-RF](https://github.com/open-h/OpenH-RF) initiative. All acquisitions are stored in the
*zea* HDF5 file format (zea_version 0.1.0) and contain raw pre-beamformed
channel data (`/data/raw_data`).

## Dataset Description

Tissue-mimicking and table-top phantom channel-capture data acquired on a Verasonics Vantage 256 with L7-4 linear and P4 phased-array probes. The collection spans coherent plane-wave compounding (CPWC), focused imaging (FI), synthetic transmit aperture (STA) and diverging-wave (DW) sequences for resolution, contrast, dynamic-range and point-spread-function evaluation, including CIRS tissue-mimicking phantom targets.

## Dataset Contributors

University of Oslo (UiO), Department of Informatics. Primary contact: Ole Marius Hoel Rindal (omrindal@ifi.uio.no). Team: Ole Marius Hoel Rindal, Yucel Karabiyik, Sven Peter Nasholm, Andreas Austeng.

## Dataset Creation Date

06/23/2026 (packaging date; original acquisitions/simulations were produced between 2016 and 2023).

## License / Terms of Use

Released under **Creative Commons Attribution 4.0 International (CC BY 4.0)** — see the `LICENCE`
file at the submission root (this license is also declared in the YAML frontmatter above). The
contributed data is cleared for this license. The UltraSound ToolBox (USTB) Channel Capture Collection, University of Oslo. Contributed to OpenH-RF. Zenodo record 20261898.

## Intended Usage

Generalized reconstruction, image-quality assessment (resolution, contrast, dynamic range), and beamforming research (RFP task 6.1). Several acquisitions provide multiple transmit schemes on the same target for cross-method comparison. (OpenH-RF RFP task 6.1 Generalized Reconstruction).

## Dataset Characterization

- **Data Collection Method:** phantom
- **Labeling Method:** Derived (known phantom geometry, e.g. CIRS Model 040GSE where applicable).
- **Acquisition system:** probe(s) L7-4, P4-1, P4-2;
  element positions stored in `/probe/probe_geometry` (meters); center frequency, sampling
  frequency and sound speed stored per acquisition in `/scan` (see per-sample feature table).

## Dataset Format

All acquisitions are stored in the **zea** HDF5 file format. Each `.hdf5` file is a single
acquisition with raw channel data `/data/raw_data` of shape
`(n_frames, n_tx, n_ax, n_el, n_ch)` and a fully populated `/scan` group describing the transmit
sequence (delays, focus distances, steering angles, apodization, timing). Data type: RF (n_ch=1).
No demodulation or decimation was applied during packaging beyond conversion from the USTB
Ultrasound File Format (UFF) to zea; RF data is demodulated inside the reconstruction pipeline.

## Dataset Quantification

- **Number of acquisitions:** 15
- **Total channel-capture frames:** 27
- **Train / validation / test split:** not predefined (research dataset).
- **Total size on disk:** 1448 MB

Per-acquisition summary:

| Acquisition | frames | transmits | samples | elements | n_ch | fs (MHz) | fc (MHz) | size (MB) |
|---|---|---|---|---|---|---|---|---|
| `experimental_dynamic_range_phantom` | 1 | 128 | 3840 | 128 | 1 | 40.8 | 5.00 | 252 |
| `experimental_STAI_dynamic_range` | 1 | 128 | 1920 | 128 | 1 | 20.4 | 5.00 | 125 |
| `FI_P4_cysts_center` | 1 | 128 | 2048 | 64 | 1 | 10.9 | 2.72 | 38 |
| `FI_P4_point_scatterers` | 6 | 128 | 1792 | 64 | 1 | 10.9 | 2.72 | 195 |
| `L7_CPWC_193328` | 3 | 15 | 1920 | 128 | 1 | 20.8 | 5.20 | 30 |
| `L7_CPWC_TheGB` | 1 | 11 | 1920 | 128 | 1 | 20.8 | 5.00 | 7 |
| `L7_DW_TheGB` | 1 | 25 | 1920 | 128 | 1 | 20.8 | 5.21 | 14 |
| `L7_FI_IUS2018` | 1 | 128 | 1920 | 128 | 1 | 20.8 | 5.21 | 62 |
| `L7_FI_TheGB` | 1 | 128 | 1920 | 128 | 1 | 20.8 | 5.21 | 61 |
| `L7_FI_Verasonics` | 1 | 128 | 1920 | 128 | 1 | 20.8 | 5.21 | 125 |
| `L7_FI_Verasonics_CIRS` | 1 | 128 | 1920 | 128 | 1 | 20.8 | 5.21 | 125 |
| `L7_FI_Verasonics_CIRS_points` | 1 | 128 | 1920 | 128 | 1 | 20.8 | 5.21 | 125 |
| `L7_STA_TheGB` | 1 | 128 | 1920 | 128 | 1 | 20.8 | 5.00 | 60 |
| `P4_FI_121444_45mm_focus` | 6 | 128 | 1280 | 64 | 1 | 11.9 | 2.98 | 151 |
| `STAI_UFF_CIRS_phantom` | 1 | 128 | 2688 | 128 | 1 | 20.8 | 5.00 | 78 |

Per-sample feature table:

| Field | Shape | Dtype | Units | Description |
|---|---|---|---|---|
| `data/raw_data` | `(n_frames, n_tx, n_ax, n_el, n_ch)` | float32 | a.u. | Raw pre-beamformed RF channel data |
| `scan/sampling_frequency` | `scalar` | float32 | Hz | A/D sampling frequency |
| `scan/center_frequency` | `scalar` | float32 | Hz | Transmit pulse center frequency |
| `scan/demodulation_frequency` | `scalar` | float32 | Hz | Demodulation (carrier) frequency |
| `scan/sound_speed` | `scalar` | float32 | m/s | Assumed medium speed of sound |
| `scan/initial_times` | `(n_tx,)` | float32 | s | A/D start time per transmit |
| `scan/t0_delays` | `(n_tx, n_el)` | float32 | s | Per-element transmit fire times |
| `scan/tx_apodizations` | `(n_tx, n_el)` | float32 | - | Per-element transmit apodization |
| `scan/focus_distances` | `(n_tx,)` | float32 | m | Focus distance per transmit (0 = plane wave) |
| `scan/polar_angles` | `(n_tx,)` | float32 | rad | Transmit steering (polar) angle |
| `scan/transmit_origins` | `(n_tx, 3)` | float32 | m | Transmit beam origin (x, y, z) |
| `probe/probe_geometry` | `(n_el, 3)` | float32 | m | Element positions (x, y, z) |

## Subject Metadata

No human or animal subjects. Targets: CIRS tissue-mimicking phantoms and lab phantoms (wire/point targets, hypo/hyperechoic inclusions, dynamic-range targets). Probes: L7-4 (128 el.), P4 phased array.

## Data Validation

A Delay-And-Sum `zea.Pipeline` is provided in the **`pipeline.yaml` at the submission root** and
run by the single **`reconstruct.py` at the submission root**:
`cast -> demodulate -> delay-and-sum beamform -> envelope detect -> normalize -> log compress`
(RF is demodulated in-pipeline; IQ uses a baseband pipeline). The script is geometry-driven and
recurses into every sub-dataset folder; running `python reconstruct.py` from the root reconstructs
every `.hdf5` in the collection (or pass a folder-qualified path for a single acquisition) and writes
`<name>_zea_bmode.png` next to each file as a portable check that the recorded geometry and timing
are correct.

The reference B-mode images committed alongside the data (`<name>_bmode.png`) are produced with the
UltraSound ToolBox (USTB) MATLAB Delay-And-Sum beamformer — the exact per-dataset reconstruction
used in the public USTB dataset catalog (https://unioslo.github.io/USTB/datasets.html), with
scanline transmit apodization for focused/sector acquisitions and correct sector-scan geometry.
These are the recommended reference reconstructions for visual verification.

## Known Issues

Mixed transmit schemes across files (CPWC / FI / STA / DW). Single-frame acquisitions for most phantoms. The reference reconstruction uses a single B-mode pipeline; per-scheme tuning may improve image quality.

## Ethical Considerations

Phantom / table-top acquisitions; no human or animal subjects are involved. No ethical considerations beyond standard laboratory practice.
