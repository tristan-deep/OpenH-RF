---
pretty_name: "OpenH-RF — Technion Bladder Pre-Beamformed Channel Data"
license: cc-by-4.0
task_categories:
  - image-to-image
tags:
  - ultrasound
  - iq
  - openh-rf
  - beamforming
  - bladder
  - 3d
language:
  - en
size_categories:
  - 1K<n<10K
---

# OpenH-RF — Bladder pre-beamformed RF channel data

## Dataset Description

Real, **in-vivo human** pre-beamformed ultrasound **channel data** for bladder
imaging: per-element I/Q recorded before receive beamforming on a 64-element
phased array, single-line transmit, 180 steered lines over a ~90° sector. 1,508
frames across 14 sweeps from seven subjects. Acquired on a GE research system in
tissue-harmonic mode; the harmonic echo is demodulated to I/Q at 3.44 MHz and
band-pass filtered. No paired image is supplied — the B-mode is reproduced from
the channel data by the released beamformer.

## Dataset Contributor(s)

Sanketh Vedula, Ortal Senouf, Dean Zadok, Alex M. Bronstein (PI) —
Technion – Israel Institute of Technology. Primary contact: sanketh@campus.technion.ac.il.

## Dataset Creation Date

Source data 2018; converted to the OpenH-RF (zea) format 07/16/2026.

## License / Terms of Use

CC BY 4.0. The contributors confirm intent to release under CC BY 4.0 with no
third-party IP encumbrances (proposal §8).

## Intended Usage

Primary: **generalized reconstruction** (§6.1) — learned receive beamforming and
image reconstruction from raw channel data. The quasi-static bladder is also
suited to multi-line-transmission (MLT) emulation and high-frame-rate research,
and to anatomy/cohort interpretation (§6.5).

## Dataset Characterization

- **Data Collection Method:** in-vivo human (research platform) — GE research
  ultrasound system with raw per-element channel access, tissue-harmonic mode.
- **Labeling Method:** N/A — no per-frame image label; the `zea.Pipeline` in
  `pipeline.yaml` reconstructs a B-mode from the channel data for validation.
- **Acquisition system:** 64-element phased array, 0.30 mm pitch, single-line
  transmit, 180 lines over ±45.13° (≈90.25° FOV). Per proposal: 2.56-cycle
  1.6 MHz transmit, no transmit apodization, tissue-harmonic mode, harmonic echo
  demodulated to I/Q at 3.44 MHz and filtered, ~18 fps; transversal plane with
  slow longitudinal probe sweep to decorrelate frames.

## Dataset Format

zea file format, one HDF5 file per sweep (`data/<subject>.hdf5`, e.g. `a1.hdf5`,
`ak.hdf5`, `s2.hdf5`). The source complex `double` samples were repackaged to
`float32` I/Q with I and Q on the final channel axis (`n_ch = 2`); values are
otherwise verbatim (band-pass filtered baseband IQ, as archived). Each file
carries `metadata/subject/{id,type=human}`, `metadata/credit`, and
`metadata/annotations/{anatomy=bladder, label=in vivo, view=transversal}`.

## Dataset Quantification

- **Frames / sweeps / subjects:** 1,508 frames · 14 sweeps · 7 subjects.
- **Train / val / test split:** N/A (contributor to define).
- **Total size on disk:** ~90 GB.

| Field | Shape | dtype | Units | Description |
|---|---|---|---|---|
| `data/raw_data` | `(n_frames, 180, 696, 64, 2)` | float32 | a.u. | pre-BF channel IQ: frames × tx-lines × axial × elements × {I, Q} |
| `scan/sampling_frequency` | scalar | float32 | Hz | 3.333 MHz (IQ sample rate, from `specs`) |
| `scan/center_frequency`, `demodulation_frequency` | scalar | float32 | Hz | 3.44 MHz (tissue-harmonic demod, from `specs`) |
| `scan/sound_speed` | scalar | float32 | m/s | 1540 |
| `scan/polar_angles` | `(180,)` | float32 | rad | ±45.13° steered lines (`thetaTX`) |
| `probe/probe_geometry` | `(64, 3)` | float32 | m | element positions, 0.30 mm pitch |

## Subject Metadata

**Seven in-vivo human volunteers**, 14 sweeps, 1,508 frames. (The proposal's
"six" was an undercount; verified from the acquisitions to be seven distinct
volunteers.) No phantom is included in this collection — the calibration phantom
is a separate submission (`../phantom/`). No PHI stored: only anonymized
`subject.id`, `subject.type = human`, and `annotations.anatomy = bladder`.
Age and sex were not recorded for these acquisitions.

| Subject | Sweeps (files) | Frames |
|---|---|---|
| A | `a1`, `a2` | 215 |
| AK | `ak` | 107 |
| H | `h1`, `h2` | 216 |
| O | `o1` | 108 |
| OK | `ok1`, `ok2` | 216 |
| P | `p1a`, `p1b`, `p2a`, `p2b` | 430 |
| S | `s1`, `s2` | 216 |

## Data Validation

`reconstruct.py` reconstructs a B-mode from `raw_data` using the `zea.Pipeline`
defined in `pipeline.yaml`: delay-and-sum beamforming on a polar scanline grid
(one image line per transmit, receive dynamic focusing at f-number 1) → envelope
detection → normalization → log compression → sector scan conversion. Run it on
any file to reproduce a reference frame:

```
python reconstruct.py data/s2.hdf5 --frame 54 --out bmode_s2.png
```

Reference output: `bmode_s2.png`. The pipeline matches the acquisition's own
receive-beamforming geometry (`code/processing/`), so the reconstruction
reproduces the expected sector B-mode.

## Known Issues

- **No paired image target** (unlike the cardiac set); the B-mode is derived from
  the channel data, not supplied.
- **Transmit fundamental (1.6 MHz) not stored** — only the 3.44 MHz demodulation
  frequency is in the files, so `center_frequency` equals the demodulation
  frequency.

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
