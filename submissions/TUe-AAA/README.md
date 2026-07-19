---
pretty_name: "OpenH-RF — TU/e PULS/e — Abdominal Aortic Aneurysm (C5-2v) channel data"
license: cc-by-4.0
task_categories:
  - image-to-image
tags:
  - ultrasound
  - rf
  - openh-rf
  - beamforming
  - motion-estimation
  - abdominal-aortic-aneurysm
  - curved-array
language:
  - en
---

# Abdominal Aortic Aneurysm (AAA) curved-array channel data

## Dataset Description

This dataset contains ultrasound channel data acquired in vivo from patients with abdominal aortic aneurysms (AAA). The dataset consists of ultrafast acquisitions obtained using a curved array ultrasound transducer operating in diverging wave transmission mode with a Verasonics system. The data capture raw radio frequency (RF) channel signals at high frame rates, enabling access to the full spatiotemporal information.

## Dataset Contributor(s)

PULS/e group
Department of Biomedical Engineering
Eindhoven University of Technology
contact: Hans-Martin Schwab (h.schwab@tue.nl)

## Dataset Creation Date

July 2026

## License / Terms of Use

CC BY 4.0.

## Intended Usage

Advanced beamforming
Motion estimation

## Dataset Characterization

- **Data collection method:** clinical
- **Acquisition system:** Verasonics Vantage, C5-2v curved array, 128 elements, center frequency 3.6 MHz
- **Transmit sequence:** 15 steered diverging waves (polar angles −12° … +12°)

## Dataset Format

zea file format. Subject metadata is stored under `metadata/subject` (`age`, `sex`, `bmi`); attribution under `metadata/credit`.

## Dataset Quantification

- **Samples / frames:** 500 acquisitions
- **Total size on disk:** 5.7 GB

| Field | Shape | dtype | Units | Description |
|---|---|---|---|---|
| `raw_data` | `(N_frames, N_tx, N_ax, N_el, 1)` | int16 | — | Raw RF channel data |

## Subject Metadata

Patients are dominantly male, aged 63–90, and scanned in the Netherlands.

## Data Validation

A `zea.Pipeline` (cast → axial window → demodulate → DAS beamforming → envelope detection → normalization → log compression) is defined in `pipeline.yaml`. Run `reconstruct.py` to reproduce the reference B-mode image (`AAApatient01_bmode.png`).

## Known Issues

N/A

## Ethical Considerations

The experiment protocol was approved by the local ethics review board of the Catharina Hospital Eindhoven on 22 December 2023 (ID-number: nWMO-2023.115) and written informed consent was obtained from each patient prior to scanning.
