---
pretty_name: "OpenH-RF - Resolve Stroke Saddle-Array Reference B-modes (Transcranial CEUS + Phantom)"
license: cc-by-4.0
task_categories:
  - other
tags:
  - ultrasound
  - iq
  - openh-rf
  - matrix-probe
  - diverging-wave
  - b-mode
  - transcranial
  - clinical
  - phantom
language:
  - en
size_categories:
  - n<1K
---

# OpenH-RF - Resolve Stroke Saddle-Array Reference B-modes

## Dataset Description

Single-frame anatomical B-mode acquisitions from the "saddle" imaging sequence of
Resolve Stroke's SYLVER ultrasound device, using a 32×32 matrix probe. Each raw
acquisition contains, alongside the multi-thousand-frame contrast-enhanced
ultrasound (CEUS) sequence, one
wide-angle diverging-wave frame (the *saddle* sequence: `x_ang` −24°…+24° in nine
steps, no elevation steering, cylindrical elevation focus) received on the full
1024-element aperture through four consecutive 256-element receive events per
transmit. Beamformed, this single frame gives a sector B-mode of the
imaging plane: the structural reference view acquired at the same probe placement
as the contrast (CEUS) recording.

This directory contains one such reference B-mode per dataset: 20 clinical
transcranial acquisitions (SCULPT study) and 1 static matrix-probe imaging phantom
(21 files total). It is the structural companion to the OpenH-RF Resolve Stroke
clinical CEUS clip submission, and the two use the same anonymized subject codes.

## Dataset Contributor(s)

Aitana Waelbroeck\*, Carl Ferlay\*, Arthur Chavignon\*, Maxence Reberol\*, Vincent Hingot\*

\* Resolve Stroke (29 Rue du Faubourg Saint-Jacques, 75014 Paris)

Contact email: maxence.reberol@resolvestroke.com

## Dataset Creation Date

07/09/2026

## License / Terms of Use

CC BY 4.0 (see `LICENCE`). Data is released under Creative Commons Attribution
4.0 International, which permits commercial use with attribution. (The Python
scripts in this directory carry their own `SPDX-License-Identifier: Apache-2.0`
header; the dataset itself is CC BY 4.0.)

## Intended Usage

Matrix-probe diverging-wave beamforming research, anatomical B-mode
reconstruction, and structural reference for the companion transcranial CEUS
flow/perfusion datasets (OpenH-RF request-for-proposals task group 6.2, Blood Flow).

## Dataset Characterization

- Data collection method: 10 human subjects, 20 acquisitions (2 per subject:
  different side and/or session), transcranial through the temporal acoustic
  window, plus 1 static imaging phantom (wire/point targets).
- Labeling method: none (a single unlabeled anatomical frame per file).
- Acquisition system: SYLVER (Resolve Stroke's ultrasound device). SN2672 32×32
  matrix probe, 0.50 mm pitch, 0.30 mm kerf; transmit center frequency ≈ 2.031 MHz,
  sound speed 1540 m/s. The *saddle* sequence transmits 9 diverging waves steered
  `x_ang` −24°…+24° (6° steps), `y_ang = 0`, virtual source at −50 mm, with an
  elevation (saddle) focus at 120 mm. The system receives 256 channels at a time,
  so each transmit is fired four times in a row, once per 256-element receive
  sub-aperture; the four receptions are stacked into 1024 virtual elements
  (element index = `aperture·256 + element`, matching PyCompute's `apElemPos`
  ordering), giving the full probe on receive. Channel data is digital down-converted (DDC) baseband IQ, so `sampling_frequency`
  (≈ 2.031 MHz) is the post-decimation IQ rate and equals `demodulation_frequency`.

## Dataset Format

One zea HDF5 file per dataset under `data/`, each holding a single frame of DDC IQ
channel data (`data/raw_data`, last axis `[I, Q]`), in the zea HDF5 format, root `zea_version` 0.1.6, validated `compliant: true` against `validate_zea_spec.py`.

Files are named `<sp_id>[-<side>][-<n>].hdf5` (anonymized subject code, imaging
side, and a sequential index when a subject/side has more than one acquisition);
the phantom is `PMP01.hdf5`.

The hardware time-gain compensation is baked into `raw_data`; `scan/tgc_gain_curve`
is the applied (non-linear) gain per axial sample. `reconstruct.py` beamforms the
stored IQ directly (it does **not** undo the TGC, so deeper structure stays bright);
divide `raw_data` by `scan/tgc_gain_curve` first to recover true channel amplitudes.

## Dataset Quantification

**Current OpenH-RF release:** 21 HDF5 files; 175.70 MB (175,702,016 bytes) stored; root `zea_version` **0.1.6**. Sizes include all HDF5 contents and use decimal units (MB = 10^6 bytes, GB = 10^9 bytes, TB = 10^12 bytes), not decoded-array memory or original-source download sizes.

- Files: 21 (20 clinical + 1 phantom)
- Frames per file: 1
- Transmits per frame: 9 (diverging waves)
- **Stored HDF5 size:** 175.70 MB (175,702,016 bytes).

| Field | Shape | dtype | Units | Description |
|---|---|---|---|---|
| `data/raw_data` | `(1, 9, 384, 1024, 2)` | int16 | n/a | IQ channel data: frames × tx × axial × elements × {I, Q} |
| `probe/probe_geometry` | `(1024, 3)` | float32 | m | Virtual element positions (full 1024-element probe) |
| `scan/t0_delays` | `(9, 1024)` | float32 | s | Per-transmit, per-element transmit delays |
| `scan/tx_apodizations` | `(9, 1024)` | float32 | n/a | Per-element transmit weights (all 1) |
| `scan/focus_distances` | `(9,)` | float32 | m | Virtual-source distance (diverging wave, −0.05) |
| `scan/polar_angles` | `(9,)` | float32 | rad | Transmit steering (−24°…+24° x) |
| `scan/azimuth_angles` | `(9,)` | float32 | rad | Transmit steering (0) |
| `scan/transmit_origins` | `(9, 3)` | float32 | m | Transmit origin |
| `scan/initial_times` | `(9,)` | float32 | s | ADC start time |
| `scan/tgc_gain_curve` | `(384,)` | float32 | n/a | Hardware TGC applied to `raw_data`; divide by it to undo |
| `scan/sampling_frequency` | scalar | float32 | Hz | 2031250 (post-DDC IQ rate) |
| `scan/center_frequency` | scalar | float32 | Hz | 2031250 |
| `scan/demodulation_frequency` | scalar | float32 | Hz | 2031250 |
| `scan/sound_speed` | scalar | float32 | m/s | 1540 |

## Subject Metadata

- Type: 10 human subjects (20 acquisitions) + 1 phantom
- Subject IDs: `SP01`–`SP10` (anonymized), `PMP01` (phantom). Each subject
  contributes 2 files (different side and/or session).
- Anatomy: Brain (transcranial), via the temporal acoustic window.

## Data Validation

`reconstruct.py` runs a standard `zea.Pipeline` (cast → DAS beamform → envelope →
normalize → log-compress), configured in `pipeline.yaml`, to reconstruct a B-mode
from the IQ channel data. The probe is a 2D matrix array insonified by
diverging-wave transmits, so it beamforms on a polar (sector) grid (a fan spanning
the divergence angle in the x-z plane at y = 0, apex at the virtual source) and
scan-converts the result.

The montage below shows the reconstruction of all 21 files, one panel per dataset.
The phantom (PMP01) shows a regular column of point targets, which checks the depth
scaling and geometry.

![Saddle-array B-modes for all 21 datasets](../assets/saddle_bmode_montage.png)

Set up the OpenH-RF environment once (clone <https://github.com/open-h/OpenH-RF>
and run `uv sync` in it), then reconstruct any file:

```
uv run --project /path/to/OpenH-RF python reconstruct.py
```

`reconstruct.py` streams `PMP01.hdf5` from the Hub by default; set `INPUT` at the top
of the script to another of the 21 files (or a local path). The B-mode PNG is written
next to the script as `<file>_bmode.png`.

## Ethical Considerations

Human-subject data. Channel data was acquired during the SCULPT clinical study
(National registration number (ID RCB): 2025-A00023-46; NCT07324421), a prospective
monocentric trial conducted at CHU Gui de Chauliac (Montpellier, France) under approval
from the French ethics committee (Comité de Protection des Personnes, CPP), comparing
cerebral perfusion from Resolve Stroke's SYLVER ultrasound system with routine perfusion
CT in ICU/CCU patients, using SonoVue® as the contrast agent.

The dataset contains no direct personal identifiers: subjects are referenced only by
an anonymized study code, and no name, date of birth, or operator identifiers are
stored in the released files. The only hardware field is the probe model (`SN2672`),
which is identical across all files and identifies the study device, not any subject.

## Known Issues

- `raw_data` has the hardware TGC baked in; `scan/tgc_gain_curve` is that (non-linear)
  applied curve. `reconstruct.py` beamforms the stored IQ as-is (leaving the TGC in,
  which keeps deep structure bright); a user wanting true channel amplitudes must
  divide `raw_data` by the curve.
- `sampling_frequency ≈ center_frequency` because the data is DDC baseband IQ (see
  Dataset Characterization), not an RF acquisition.
