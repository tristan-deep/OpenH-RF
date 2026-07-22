---
pretty_name: "OpenH-RF — Technion/ISTA Cardiac Pre-Beamformed Channel Data"
license: cc-by-4.0
task_categories:
  - image-to-image
tags:
  - ultrasound
  - iq
  - openh-rf
  - beamforming
  - cardiac
  - 3d
language:
  - en
size_categories:
  - n<1K
---

# OpenH-RF — Cardiac pre-beamformed RF channel data (paired with DAS targets)

## Dataset Description

Real, **in-vivo human** pre-beamformed ultrasound **channel data** for cardiac
imaging: per-element I/Q recorded before receive beamforming on a 64-element
phased array (single-line acquisition, sector scan). Each frame is **paired with
its conventional delay-and-sum reconstruction** (stored as `beamformed_data`),
making this a ready-made input→target set for learned reconstruction /
beamforming. 777 frames across 25 cine loops from six subjects (a–f).

## Dataset Contributor(s)

Sanketh Vedula (Princeton University; Broad Institute; Technion),
Ortal Senouf (EPFL; Technion), Dean Zadok (Carnegie Mellon University; Technion),
Alex M. Bronstein (ISTA; Technion — PI). Primary contact: svedula@ist.ac.at.

## Dataset Creation Date

Acquired 2018; converted to the OpenH-RF (zea) format 07/16/2026.

## License / Terms of Use

CC BY 4.0. The data is the contributors' own research acquisition, cleared for
CC BY 4.0 with no third-party IP encumbrances.

## Intended Usage

Primary: **generalized reconstruction** (§6.1) — learning to map raw per-element
channel data to a focused image (learned receive/transmit beamforming,
super-resolution, clutter suppression), trained and evaluated against the paired
delay-and-sum target. Secondary: motion estimation across the cardiac cine loops
(§6.4) and anatomy/cohort interpretation (§6.5).

## Dataset Characterization

- **Data Collection Method:** in-vivo human (research platform) — GE experimental
  breadboard system with raw per-element channel access.
- **Labeling Method:** derived ground truth — the paired `beamformed_data` is the
  conventional delay-and-sum reconstruction of each frame (reference beamformer
  released with the dataset).
- **Acquisition system:** 64-element phased array, 0.30 mm pitch, sector scan,
  140 acquisition lines over a ~75° sector (±37.5°); 1.75-cycle 2.5 MHz transmit
  on the 28 central elements, elevation aperture 13 mm, elevation focus 100 mm,
  transmit depth focus 71 mm.

## Dataset Format

zea file format, one HDF5 file per cine loop (`data/<subject><clip>.hdf5`, e.g.
`a1.hdf5` = subject a, clip 1; `f2.hdf5` = patient-set subject f). The source
complex `int16` samples were repackaged to `float32` I/Q with I and Q on the
final channel axis (`n_ch = 2`); values are otherwise verbatim. Each file carries
`metadata/subject/{id,type=human}` and `metadata/annotations/{anatomy=cardiac,
label=in vivo}`.

## Dataset Quantification

- **Frames / cines / subjects:** 777 frames · 25 cine loops · 6 subjects (a–f).
- **Train / val / test split:** N/A (contributor to define; the `f2` patient set
  is a natural held-out cine).
- **Total size on disk:** ~19 GB.

| Field | Shape | dtype | Units | Description |
|---|---|---|---|---|
| `data/raw_data` | `(n_frames, 140, 680, 64, 2)` | float32 | a.u. | pre-BF channel IQ: frames × tx-lines × axial × elements × {I, Q} |
| `data/beamformed_data.values` | `(n_frames, 652, 140, 2)` | float32 | a.u. | paired delay-and-sum target (complex IQ): frames × depth × line × {I, Q} |
| `data/beamformed_data.coordinates` | `(652, 140, 3)` | float32 | m | per-pixel polar coordinates (⚠️ depth scale approximate) |
| `scan/sampling_frequency` | scalar | float32 | Hz | 6.0 MHz — **best estimate**, axial rate not stored (see Known Issues) |
| `scan/center_frequency`, `demodulation_frequency` | scalar | float32 | Hz | 2.5 MHz (cardiac fundamental) |
| `scan/sound_speed` | scalar | float32 | m/s | 1540 |
| `scan/polar_angles` | `(140,)` | float32 | rad | ±37.5° steered lines |
| `probe/probe_geometry` | `(64, 3)` | float32 | m | element positions, 0.30 mm pitch |

## Subject Metadata

Six subjects (a–e main set, f patient set), 777 frames across 25 cine loops.
In-vivo human; no PHI stored (only `subject.id` a1…f2, `subject.type = human`,
`anatomy = cardiac`). Age/sex distribution: to be supplied by the contributor.

## Data Validation

The dataset's own reconstruction is the conventional delay-and-sum image produced
by the repo pipeline (`createDS.m` → `SLA2MLA.m` →
`do_dynamic_focalization_CREANUIS_new.m`), shipped verbatim in each file as
`beamformed_data`. `reconstruct.py` therefore renders that **paired DAS target**
(the authoritative, repo-exact reconstruction), envelope-detected, log-compressed,
and scan-converted using the stored per-pixel coordinates. Reference output:
`bmode_a1.png`. `pipeline.yaml` provides the equivalent standard `zea.Pipeline`.
A classical raw→image beamform is the learning task this paired set is built for;
its depth scale is approximate because the CREANUIS axial rate is not stored (see
Known Issues).

## Known Issues

- **Axial sample rate not stored.** The consolidated source `.mat` files do not
  carry the acquisition header, so `sampling_frequency` (6.0 MHz) is a best
  estimate and the reconstructed depth scale is approximate. The paired
  `beamformed_data` target is exact in value but shares this approximate depth
  axis.
- **Sector-angle convention.** Lines are stored as ±37.5° centred about
  boresight; this is physically correct for a phased array and is what the
  reference beamformer reproduces.

## Ethical Considerations

**Privacy safeguards (HIPAA and GDPR).** Pre-beamformed RF channel data contains
no facial or otherwise identifying imagery. All records are de-identified to the
HIPAA Safe Harbor standard, with direct identifiers removed and any dates
generalized to bands. As an EU institution we additionally comply with GDPR,
holding any pseudonymized subject identifiers separately on access-controlled
storage and never sharing them. The released data are de-identified and contain
only the channel signals and acquisition metadata.

**Ethics.** The data were collected under ethical best practices on healthy
volunteers.
