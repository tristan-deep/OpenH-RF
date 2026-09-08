---
pretty_name: "OpenH-RF — OpenPros Limited-View Prostate USCT"
license: cc-by-4.0
task_categories:
  - other
tags:
  - ultrasound
  - rf
  - openh-rf
  - zea
  - usct
  - speed-of-sound
  - prostate
  - simulation
language:
  - en
size_categories:
  - 100K<n<1M
---

# OpenPros - Limited-View Prostate Ultrasound Computed Tomography

## Dataset Description

[OpenPros](https://open-pros.github.io/) is a large-scale benchmark for limited-view prostate ultrasound computed tomography (USCT). Each example pairs simulated full-waveform RF data from transabdominal and transrectal acquisition paths with an anatomically realistic two-dimensional speed-of-sound (SOS) map. The source OpenPros phantoms are derived from expert-annotated clinical MRI/CT anatomy and experimental measurements of ex vivo prostate specimens; the RF measurements in this package are simulated rather than acquired in vivo. The intended research task is quantitative SOS reconstruction from limited-angle ultrasound data.

## Dataset Contributor(s)

OpenPros was created by Hanchen Wang, Yixuan Wu, Yinan Feng, Peng Jin, Luoyuan Zhang, Shihang Feng, James Wiskin, Baris Turkbey, Peter A. Pinto, Bradford J. Wood, Songting Luo, Yinpeng Chen, Emad Boctor, and Youzuo Lin. The affiliations include the University of North Carolina at Chapel Hill, Johns Hopkins University, the National Institutes of Health, the Pennsylvania State University, QT Imaging, Iowa State University, and Google DeepMind. 
- **Corresponding author:** Youzuo Lin (`yzlin@unc.edu`)
- **Source repository:** <https://github.com/hanchenwang/OpenPros>
- **Dataset website:** <https://open-pros.github.io/>

## Dataset Creation Date

05/18/2025 (initial public release of OpenPros).

## License / Terms of Use

[Creative Commons Attribution 4.0 International license (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

## Intended Usage

- Training and evaluating limited-angle prostate USCT reconstruction methods.
- Reconstructing quantitative two-dimensional SOS maps from full-waveform data.
- Benchmarking learned, physics-guided, and operator-learning inverse methods.
- Studying robustness to noise and generalization across prostate specimens or patient-level anatomies.
- Reproducing inference with the pretrained OpenPros InversionNet baseline.

## Dataset Characterization

- **Data collection method:** Synthetic FDTD wave-propagation simulations through anatomically realistic digital prostate phantoms.
- **Ground-truth method:** SOS maps derived from expert-annotated MRI/CT anatomy, measured ex vivo prostate SOS values, and literature tissue properties.
- **Anatomy:** Prostate and surrounding male pelvic anatomy, including tissue heterogeneity and, depending on the slice, bony structures and lesions.
- **Acquisition geometry:** Limited-view configuration with 10 sources and 161 receivers on the abdominal/body-surface side and another 10 sources and 161 receivers on the transrectal side.
- **Excitation:** 1 MHz peak-frequency Ricker wavelet.
- **Sampling:** 10 MHz (`dt = 1e-7 s`), 1,000 samples, or 100 microseconds per waveform.
- **Image grid:** 401 axial by 161 lateral samples at 0.375 mm spacing, covering approximately 150 mm by 60 mm.

## Dataset Format

The package uses the `zea` HDF5 format. Run [`convert.py`](convert.py) to create  `openpros_sample.hdf5` from the original OpenPros NumPy arrays. The converter does not demodulate, decimate, filter, or normalize the RF values. It adds a singleton channel dimension and changes the original four 10-transmit blocks into a physical `20 transmits × 322 receivers` representation:

| HDF5 region | Source side | Receiver side | Original transmit channels |
|---|---|---|---|
| `raw_data[:, 0:10, :, 0:161, :]` | Body surface | Body surface | `0:10` |
| `raw_data[:, 0:10, :, 161:322, :]` | Body surface | Rectum | `10:20` |
| `raw_data[:, 10:20, :, 161:322, :]` | Rectum | Rectum | `20:30` |
| `raw_data[:, 10:20, :, 0:161, :]` | Rectum | Body surface | `30:40` |

The file also stores source positions, probe geometry, scan parameters, subject identifiers, SOS values, and Cartesian SOS coordinates. Placeholder scan fields needed for format compatibility are described under **Known Issues**.

## Dataset Quantification

The full OpenPros release contains 280,000 paired samples (approximately 6.8 TB) with an official split of 224,000 training, 28,000 validation, and 28,000 test samples. It is derived from four patient-level clinical anatomies and 62 ex vivo prostate specimens. Each source NumPy file used here contains 1,140 examples. 

By default, `flag_single_sample = True` in `convert.py`, so the packaged `openpros_sample.hdf5` contains only the first example and is approximately 20.5 MiB. Set the flag to `False` to convert all 1,140 examples in the selected source pair.

| Field | Shape in default sample | dtype | Units | Description |
|---|---|---|---|---|
| `raw_data` | `(1, 20, 1000, 322, 1)` | float32 | — | Simulated full-waveform pressure/RF data |
| `sos_map.values` | `(1, 401, 161, 1)` | float32 | m/s | Ground-truth speed-of-sound map |
| `sos_map.coordinates` | `(401, 161, 3)` | float32 | m | Cartesian `(x, y, z)` grid coordinates |

The leading dimension becomes `1140` when the complete selected source pair is converted.

## Subject Metadata

The converted file identifies its content as a simulation and stores the composite subject ID (e.g., `1_2021-03-16`). This corresponds to OpenPros patient-level anatomy `3_01` and prostate-level anatomy `2021-03-16`. The full dataset contains four patient-level anatomy IDs (`3_01` through `3_04`) and 62 date-based prostate IDs. No direct identifiers or individual-level demographic attributes are included. 

## Data Validation

[`reconstruct.py`](reconstruct.py) defines a `zea.Pipeline` that reproduces the OpenPros InversionNet preprocessing and postprocessing:

1. Restore the original acquisition-block order: body/body, body/rectum, rectum/rectum, and rectum/body.
2. Apply the sign-preserving logarithmic transform `sign(x) * log1p(abs(1e5 * x))`.
3. Min-max normalize the configured transformed input range to `[-1, 1]`.
4. Run the pretrained InversionNet model.
5. Denormalize its output from `[-1, 1]` to the physical SOS range `1300–3600 m/s`.

Run the end-to-end example with:

```bash
python convert.py
python reconstruct.py
```

The reconstruction script checks the input shapes and writes `pred_sos.png`, a side-by-side comparison of the predicted and ground-truth SOS maps. Use `--write_config` to serialize the pipeline to `pipeline.yaml`, `--load_config` to restore it, and `--use_zea_vis_style` to apply the ZEA plotting style.

## Known Issues

- `focus_distances`, `t0_delays`, `tx_apodizations`, and related transmit fields are compatibility placeholders because the simulation uses external point sources rather than a conventional focused array transmission.


## Ethical Considerations

The packaged RF data and SOS labels are simulated and contain no directly identifying patient information.
