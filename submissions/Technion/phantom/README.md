---
pretty_name: "OpenH-RF — Technion/ISTA Phantom Pre-Beamformed Channel Data"
license: cc-by-4.0
task_categories:
  - image-to-image
tags:
  - ultrasound
  - iq
  - openh-rf
  - beamforming
  - phantom
  - 3d
language:
  - en
size_categories:
  - n<1K
---

# OpenH-RF — Tissue-mimicking phantom pre-beamformed RF channel data

## Dataset Description

Pre-beamformed ultrasound **channel data** from a tissue-mimicking phantom,
acquired on the same 64-element phased array and single-line-transmit sector
scheme as the in-vivo collection, for **calibration and verification**. Contains
resolvable point targets and an anechoic cyst — a clean reference for validating
beamforming and reconstruction. 12 frames, one acquisition.

## Dataset Contributor(s)

Sanketh Vedula (Princeton University; Broad Institute; Technion),
Ortal Senouf (EPFL; Technion), Dean Zadok (Carnegie Mellon University; Technion),
Alex M. Bronstein (ISTA; Technion — PI). Primary contact: svedula@ist.ac.at.

## Dataset Creation Date

Source data 2018; converted to the OpenH-RF (zea) format 07/16/2026.

## License / Terms of Use

CC BY 4.0 (proposal §8).

## Intended Usage

Calibration and end-to-end verification of the beamforming/reconstruction
pipeline (point-target resolution, cyst contrast). Phantom tier (×1).

## Dataset Characterization

- **Data Collection Method:** phantom — GAMMEX 403GS LE tissue-mimicking phantom,
  acquired on the same probe as the in-vivo collection for calibration.
- **Labeling Method:** N/A (calibration target; known phantom geometry).
- **Acquisition system:** 64-element phased array, 0.30 mm pitch, single-line
  transmit, 180 lines over ±45.13° (≈90.25° FOV), IQ demodulated at 3.44 MHz.

## Dataset Format

zea file format, a single HDF5 file `data/ph.hdf5`. Source complex samples
repackaged to `float32` I/Q (`n_ch = 2`), values verbatim. Carries
`metadata/subject/{id=ph, type=phantom}` and
`metadata/annotations/{anatomy=phantom, label=phantom}`.

## Dataset Quantification

- **Frames / acquisitions:** 12 frames · 1 acquisition.
- **Total size on disk:** ~0.8 GB.

| Field | Shape | dtype | Units | Description |
|---|---|---|---|---|
| `data/raw_data` | `(12, 180, 696, 64, 2)` | float32 | a.u. | pre-BF channel IQ: frames × tx-lines × axial × elements × {I, Q} |
| `scan/sampling_frequency` | scalar | float32 | Hz | 3.333 MHz |
| `scan/center_frequency`, `demodulation_frequency` | scalar | float32 | Hz | 3.44 MHz |
| `scan/sound_speed` | scalar | float32 | m/s | 1540 |
| `scan/polar_angles` | `(180,)` | float32 | rad | ±45.13° steered lines |
| `probe/probe_geometry` | `(64, 3)` | float32 | m | element positions, 0.30 mm pitch |

## Subject Metadata

N/A — inanimate phantom (GAMMEX 403GS LE). `annotations.anatomy = phantom`.

## Data Validation

`reconstruct.py` is a faithful port of the dataset's own production beamformer
`code/processing/IQBF.m` (dynamic-aperture receive delay-and-sum with per-channel
IQ phase rotation), reading straight from the converted zea file. Reference
output: `bmode_ph.png` — resolvable point targets and a well-defined anechoic cyst
at ~65 mm, matching the source collection's reference render (confirms the
conversion end-to-end). `pipeline.yaml` provides the equivalent `zea.Pipeline`.

## Known Issues

- Same scan scheme and probe as the in-vivo bladder collection (GE
  tissue-harmonic); acquired as its calibration reference. GAMMEX 403GS LE.

## Ethical Considerations

None — inanimate phantom, no human or animal subjects.
