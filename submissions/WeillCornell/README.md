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

Pre-beamformed RF channel data from three homogeneous tissue-mimicking
phantoms. Each zea HDF5 acquisition is self-contained and includes raw RF,
model-derived theoretical backscatter coefficient (BSC), direct per-frame
Nakagami maps, and 20-frame pooled Nakagami references.

## Contributors

- Shangke Liu, Cornell University and Weill Cornell Medicine
- Tipu Sultan
- Jonathan Mamou

Contact: Shangke Liu, shl4035@med.cornell.edu

## Dataset Summary

- Verasonics Vantage 256 and GE9LD linear array
- 192 elements, 0.23 mm pitch, 5.2083 MHz center frequency
- One normal-incidence plane wave per frame
- 20.8333 MHz RF sampling; 1540 m/s sound-speed metadata
- 3 phantoms x 2 operators x 20 probe placements x 20 frames
- 120 acquisitions and 2400 frames
- Phantom-only data; no human or animal subjects, clinical metadata, or PHI
- No predefined train/validation/test split

Keep all frames from one acquisition in the same split.

## Files

```text
data/<scan_id>.hdf5                         self-contained zea acquisitions
raw_data/phantom_<id>/<scan_id>.mat         original Verasonics workspaces
figures/bmode/                              per-acquisition B-mode QC
figures/reference_bmode/                    reference zea reconstruction
reconstruct.py                              B-mode and QUS example
pipeline.yaml                               zea reconstruction pipeline
LICENSE                                     CC BY 4.0 license
```

Scan IDs use `ac<number>_<phantom>_<operator>`, for example
`ac1_15m_SK`. Phantom IDs are `15m`, `18m`, and `60m`; operator IDs are
`SK` and `TP`; acquisition numbers run from 1 to 20.

## HDF5 Contents

Each file follows zea 0.1.3 and contains one track. Raw RF is stored as `int16`
ADC counts without demodulation, decimation, or resampling.

| Field under `tracks/track_0/data/` | Shape | dtype | Unit | Meaning |
|---|---|---|---|---|
| `raw_data` | `(20,1,n_ax,192,1)` | `int16` | ADC counts | Raw RF input |
| `theoretical_bsc/values` | `(20,z,x,157)` | `float32` | `m^-1 sr^-1` | Model-derived BSC target |
| `nakagami_m_per_frame/values` | `(20,z,x)` | `float32` | `1` | Direct single-frame shape estimate |
| `nakagami_omega_per_frame/values` | `(20,z,x)` | `float32` | `a.u.^2` | Direct single-frame spread estimate |
| `nakagami_m_pooled/values` | `(20,z,x)` | `float32` | `1` | 20-frame pooled shape reference |
| `nakagami_omega_pooled/values` | `(20,z,x)` | `float32` | `a.u.^2` | 20-frame pooled spread reference |

The five QUS products are zea Map fields with `values`, Cartesian
`coordinates` in `[x,y,z]` order and metres, `description`, `unit`, `min`, and
`max`. The BSC field also contains 157 `labels` of the form
`frequency_hz=<value>`, spanning 3.3162435-6.4900716 MHz.

Map grids are `(z,x)=(26,35)` for `15m` and `18m`, and `(53,35)` for `60m`.
The analysis windows are approximately 2.96 mm axial by 4.37 mm lateral with
75% overlap. Nakagami maps use normal-incidence delay-and-sum beamforming,
Hann receive apodization, and intensity-domain maximum-likelihood fitting,
with `omega = E[A^2]`.

The theoretical BSC is a homogeneous phantom material property: each phantom
has one curve, broadcast over space and all 20 frames. The pooled Nakagami maps
are also repeated along the frame axis so every single-frame input has the same
20-frame reference. The pooled estimate includes the selected input frame.

Metadata uses `metadata.subject.type="phantom"` and a phantom ID. Anatomical
annotations and anatomical-view annotations are intentionally omitted.
Plane-wave acquisition is recorded in the file description, and the linear
array geometry is recorded in the probe fields.

## Suggested Tasks

1. Reconstruct B-mode from one frame of raw RF.
2. Predict the phantom's theoretical BSC curve from one RF frame. This is a
   material-model target, not a local experimentally estimated BSC curve.
3. Predict pooled Nakagami `m` or `omega` from one RF frame. Use the supplied
   direct per-frame map as the single-frame baseline.

For Nakagami predictions, report MAE/RMSE and map correlation against the
pooled reference. For BSC predictions, report the frequency range and unit and
state whether evaluation uses linear BSC or a specified dB conversion.

All QUS maps are public reference targets, not hidden competition labels.

## Quick Start

Use an OpenH-RF environment with `zea==0.1.3` and a supported Keras backend:

```bash
KERAS_BACKEND=jax python reconstruct.py \
  --input data/ac1_15m_SK.hdf5 \
  --pipeline pipeline.yaml \
  --output /tmp/ac1_15m_SK_pipeline.png \
  --frame 0
```

The script saves the B-mode PNG, then prints the selected frame's RF shape,
BSC band and unit, QUS map shapes, and direct per-frame-to-pooled Nakagami
errors.

## Validation

All 120 HDF5 files pass zea 0.1.3 `File.validate()` and
`File.validate_spec()`. The RF arrays are element-wise identical to the source
MAT acquisitions after the documented axis mapping. All embedded maps were
checked for shape, dtype, finite values, coordinates, labels, min/max, and the
documented frame-broadcast behavior. Files use zea 0.1.3's default
Blosc/Zstd+bitshuffle compression.

![Reference zea reconstruction of ac1_15m_SK frame 0](figures/reference_bmode/ac1_15m_SK_pipeline.png)

The reference uses frame 0 and the common 597 x 300 zea grid at approximately
3-25 mm depth. `figures/bmode/` contains visual QC images using the 20-frame
mean envelope and each workspace's full field of view, so they are not expected
to be pixel-identical to the single-frame reference reconstruction.

## Limitations

- Acquisition timestamps, PRF, explicit TGC curves, and emitted waveforms are
  unavailable; do not use these frames for calibrated temporal or flow analysis.
- Repeated frames are strongly correlated.
- BSC targets are model-derived references, not independent experimental BSC
  measurements; there are three unique phantom curves.
- Pooled Nakagami maps are lower-variance statistical references, not
  independent physical ground truth.
- Nakagami `omega` depends on gain, attenuation, beam sensitivity, and RF
  amplitude scale; it is not system-independent.
- This release contains Verasonics data only.

## License

CC BY 4.0. See `LICENSE`. Commercial and non-commercial reuse is permitted
with attribution. The contributors confirm that these phantom data are cleared
for release under CC BY 4.0.

Dataset package prepared on 07/20/2026.
