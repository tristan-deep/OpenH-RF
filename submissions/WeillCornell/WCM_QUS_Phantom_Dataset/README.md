---
license: cc-by-4.0
pretty_name: OpenH-RF QUS Phantom Dataset
task_categories:
  - image-to-image
  - feature-extraction
tags:
  - ultrasound
  - rf
  - openh-rf
  - quantitative-ultrasound
  - phantom
  - raw-channel-data
language:
  - en
size_categories:
  - n<1K
---

# OpenH-RF QUS Phantom Dataset

## Dataset Description

Pre-beamformed RF channel data from three homogeneous tissue-mimicking
phantoms. Every zea HDF5 acquisition is self-contained: it includes raw RF,
model-derived theoretical backscatter coefficient (BSC) targets, direct
per-frame Nakagami estimates, and 20-frame pooled Nakagami references.

## Dataset Contributor(s)

- Shangke Liu, Cornell University and Weill Cornell Medicine
- Tipu Sultan
- Jonathan Mamou

Contact: Shangke Liu, shl4035@med.cornell.edu

## Dataset Creation Date

07/16/2026 (submission package). Original acquisition timestamps were not
retained.

## License / Terms of Use

CC BY 4.0. See `LICENCE`. Commercial and non-commercial reuse is permitted
with attribution. The contributors confirm that these phantom data are cleared
for release under CC BY 4.0.

## Intended Usage

1. Reconstruct B-mode images from OpenH-RF/zea channel data.
2. Use one RF frame to predict the broadcast theoretical material-model BSC
   target. This is one of three phantom curves, not a local experimental BSC
   measurement.
3. Use one RF frame to predict Nakagami `m` or `omega`, using the direct
   per-frame fit as a baseline and the 20-frame pooled fit as the statistical
   reference. The pooled reference includes the evaluated input frame.

All QUS maps are public reference targets, not hidden competition labels.

## Dataset Characterization

- **Collection:** homogeneous phantoms; no human or animal data
- **System:** Verasonics Vantage 256 with GE9LD linear array
- **Probe:** 192 elements, 0.23 mm pitch, 5.2083 MHz center frequency
- **Acquisition:** one normal-incidence plane wave per frame
- **Sampling:** 20.8333 MHz RF; sound-speed metadata 1540 m/s
- **Design:** 3 phantoms x 2 operators x 20 probe placements x 20 frames
- **Labels:** Faran model-derived BSC and derived Nakagami maps

## Dataset Format

```text
data/<scan_id>.hdf5                         self-contained zea acquisitions
raw_data/phantom_<id>/<scan_id>.mat         source Verasonics workspaces
scripts/example_qus_task.py                 minimal input/target example
figures/reference_bmode/                    submitted zea reconstruction
figures/bmode/                              per-acquisition QC B-mode PNGs
reconstruct.py                              OpenH-RF reconstruction entry point
pipeline.yaml                               submitted zea pipeline
```

Each HDF5 file follows zea 0.1.1 and has one track. The raw RF is int16 and was
not demodulated, decimated, or resampled. Conversion only maps the axes and adds
the singleton transmit and channel dimensions required by zea. The maps are
derived products.

| Field under `tracks/track_0/data/` | Shape | dtype | Unit | Meaning |
|---|---|---|---|---|
| `raw_data` | `(20,1,n_ax,192,1)` | `int16` | unitless ADC counts | Raw RF input |
| `theoretical_bsc/values` | `(20,z,x,157)` | `float32` | `m^-1 sr^-1` | Broadcast model BSC target |
| `nakagami_m_per_frame/values` | `(20,z,x)` | `float32` | `1` | Direct single-frame shape estimate |
| `nakagami_omega_per_frame/values` | `(20,z,x)` | `float32` | `a.u.^2` | Direct single-frame spread estimate |
| `nakagami_m_pooled/values` | `(20,z,x)` | `float32` | `1` | 20-frame pooled shape reference |
| `nakagami_omega_pooled/values` | `(20,z,x)` | `float32` | `a.u.^2` | 20-frame pooled spread reference |

All map groups are under `tracks/track_0/data/` and contain float32 `values`,
float32 `coordinates` in `[x,y,z]` order and metres, a description, unit, min,
and max. The BSC map additionally contains 157 string `labels` in the form
`frequency_hz=<value>`. The 157 channels span
3.3162434896-6.4900716146 MHz.

Map sizes are `(z,x)=(26,35)` for `15m` and `18m`, and `(53,35)` for `60m`.
The windows are approximately 2.96 mm axial by 4.37 mm lateral with 75%
overlap. Nakagami maps use normal-incidence delay-and-sum beamforming, Hann
receive apodization, and intensity-domain maximum-likelihood fitting, with
`omega = E[A^2]`.

The theoretical BSC is a phantom-material property. Because each phantom is
homogeneous, its BSC curve is spatially invariant and shared by all 20 frames.
The pooled Nakagami maps are also repeated along the HDF5 frame axis so that
every single-frame input is paired with the same 20-frame reference. That
pooled estimate includes the selected input frame and is a consistency target,
not independent ground truth. The per-frame maps contain distinct estimates
for each frame.

## Quick Start

Use the OpenH-RF zea environment to reconstruct one frame:

```bash
KERAS_BACKEND=jax python reconstruct.py \
  --input data/ac1_15m_SK.hdf5 \
  --pipeline pipeline.yaml \
  --output /tmp/ac1_15m_SK_pipeline.png
```

Load and inspect one frame and all associated QUS references:

```bash
python scripts/example_qus_task.py data/ac1_15m_SK.hdf5 --frame 0
```

The example requires only NumPy and h5py. It reports the input and target
shapes plus direct per-frame-to-pooled Nakagami errors.

For Nakagami predictions, report MAE/RMSE and map correlation against the
pooled reference, with the supplied per-frame estimate as the baseline. For
BSC predictions, report the frequency range and unit and evaluate linear BSC
or explicitly state the logarithmic reference used for dB conversion.

## Dataset Quantification

- 3 phantoms: `15m`, `18m`, `60m`
- 2 operator IDs: `SK`, `TP`
- 120 acquisitions and 2400 frames
- 120 source MAT files and 120 self-contained zea HDF5 files
- Approximately 3.2 GB total
- No predefined train/validation/test split

Keep all frames from one acquisition in the same split.

## Subject Metadata

Phantom-only data with zero human or animal subjects. HDF5 metadata contains
only phantom, acquisition-view, and scan labels; no human anatomical,
demographic, clinical, pathology, or PHI fields are included.

## Data Validation

All 120 HDF5 files pass zea 0.1.1 `File.validate()` and
`File.validate_spec()`. The raw RF arrays are element-wise identical to the
corresponding source MAT acquisitions after documented axis mapping.

![Reference zea reconstruction of ac1_15m_SK frame 0](figures/reference_bmode/ac1_15m_SK_pipeline.png)

The submitted reference uses frame 0 and the common 597 x 300 zea grid at
approximately 3-25 mm depth. `figures/bmode/` contains separate visual QC
images using the 20-frame mean envelope and each workspace's full PData field
of view. These QC images are not expected to be pixel-identical to the
submitted single-frame zea reconstruction.

## Known Issues

- Acquisition timestamps, PRF, explicit TGC curves, and emitted waveforms are
  unavailable; do not use the frames for calibrated temporal or flow analysis.
- Repeated frames are strongly correlated.
- The BSC targets are model-derived references, not independent experimental
  calibration measurements, and there are three unique phantom curves.
- The 20-frame pooled Nakagami maps are lower-variance statistical references,
  not independent physical ground truth.
- Nakagami `omega` depends on gain, attenuation, beam sensitivity, and RF
  amplitude scale; it is not system independent.
- This release contains Verasonics data only.

## Ethical Considerations

Phantom-only measurements with no human participants, animal subjects,
clinical metadata, PHI, consent, or IRB requirements.
