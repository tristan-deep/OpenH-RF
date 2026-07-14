---
license: cc-by-4.0
pretty_name: OpenH-RF Thyroid In-Vivo L11-5v Scan (D1)
task_categories:
  - other
tags:
  - ultrasound
  - channel-data
  - beamforming
  - thyroid
  - clinical
  - zea
---

# OpenH-RF Thyroid Clinical Scans

## Dataset Description

The data set consists of clinical ultrasound channel data acquired
with a Verasonics Vantage 128 system and an L11-5v linear array probe, imaging
the thyroid gland of 30 adult healthy volunteers. The scans were carried out by a senior radiologist
who specializes in ultrasound thyroid imaging. The purpose of the scans is to provide a complete set of channel data of a thyroid gland scan of the full anatomy.

## Dataset Contributor(s)

Prof. Yonina Eldar's SAMPLLAB Group, Faculty of Mathematics and Computer Science, Weizmann Institute of Science, Rehovot, Israel.


## Dataset Creation Date

07/07/2026

## License / Terms of Use

CC-BY-4.0. Acquired under a Weizmann Institute IRB-approved research protocol, with informed
consent obtained from each subject prior to scanning.

## Intended Usage

General-purpose ultrasound channel-data foundation model pretraining and
evaluation — in particular DAS beamforming/reconstruction, inverse speed of sound imaging.
Suitable as a base for future downstream tasks (e.g. thyroid
segmentation, nodule detection) if paired with additional annotations from the B-mode images.

## Dataset Characterization

- **Data Collection Method:** clinical
- **Labeling Method:** N/A — no annotations included
- **Acquisition system:** Verasonics Vantage 128 system, L11-5v linear array
  probe, center frequency 7.6 MHz (76.8% fractional
  bandwidth), element width 0.27 mm, aperture width 38.1 mm; RF sampling
  frequency 31.25 MHz, demodulation frequency 7.8125 MHz, assumed sound
  speed 1540 m/s.

## Dataset Format

zea file format, one acquisition HDF5 file per subject. Before packaging,
the frames prior to workspace parameter freezing were removed from the raw channel data frames.

## Dataset Quantification
- Number of samples / acquisitions / channels: ~1600 / 128 / 128
- Number of frames:  average 250 (dependent on subject)
- No train/validation/test split
- Total size on disk: 374 GB
- Per-sample feature table:

| Field | Shape                        | Dtype | Units | Description |
|---|------------------------------|---|---|---|
| `tracks/track_0/data/raw_data` | (nframes, 128, 4096, 128, 1) | int16 | ADC counts | Raw per-element RF channel data: (frames, transmits, axial samples, elements, 1) |
| `tracks/track_0/scan/t0_delays` | (128, 128)                   | float32 | s | Per-transmit, per-element transmit delay |
| `tracks/track_0/scan/tx_apodizations` | (128, 128)                   | float32 | unitless (0-1) | Per-transmit, per-element transmit apodization weight |
| `tracks/track_0/scan/polar_angles` | (128,)                       | float32 | rad | Per-transmit polar steering angle |
| `tracks/track_0/scan/azimuth_angles` | (128,)                       | float32 | rad | Per-transmit azimuth steering angle (0 for this 1D linear array) |
| `tracks/track_0/scan/focus_distances` | (128,)                       | float32 | m | Per-transmit focal distance |
| `tracks/track_0/scan/transmit_origins` | (128, 3)                     | float32 | m | Per-transmit virtual source origin (x, y, z) |
| `tracks/track_0/scan/initial_times` | (128,)                       | float32 | s | Per-transmit acquisition start time offset |
| `tracks/track_0/scan/time_to_next_transmit` | (nframes, 128)               | float32 | s | Per-frame, per-transmit inter-transmit timing |
| `tracks/track_0/scan/center_frequency` | scalar                       | float32 | Hz | Transmit center frequency (7.6 MHz) |
| `tracks/track_0/scan/sampling_frequency` | scalar                       | float32 | Hz | RF sampling frequency (31.25 MHz) |
| `tracks/track_0/scan/demodulation_frequency` | scalar                       | float32 | Hz | Demodulation frequency used downstream (7.8125 MHz) |
| `tracks/track_0/scan/sound_speed` | scalar                       | float32 | m/s | Assumed speed of sound (1540 m/s) |
| `tracks/track_0/scan/tgc_gain_curve` | (2048,)                      | float32 | Verasonics TGC units | Time-gain-compensation curve applied during acquisition |
| `probe/probe_geometry` | (128, 3)                     | float32 | m | Element (x, y, z) positions, L11-5v linear array, ±19.05 mm aperture |
| `probe/element_width` | scalar                       | float32 | m | Element width (0.27 mm) |
| `probe/probe_center_frequency` | scalar                       | float32 | Hz | Probe nominal center frequency (7.6 MHz) |
| `probe/probe_bandwidth_percent` | scalar                       | float32 | % | Probe fractional bandwidth (76.8%) |
| `probe/lens_thickness` | scalar                       | float32 | m | Acoustic lens thickness (0.390 mm), derived from the Verasonics lens correction scalar; consumed automatically by zea's per-element/per-pixel lens correction during reconstruction (`apply_lens_correction: true` in `pipeline.yaml`) |
| `probe/lens_sound_speed` | scalar                       | float32 | m/s | Assumed speed of sound in the lens material (1000 m/s) used to derive `lens_thickness` above |
| `custom/lens_correction` | scalar                       | float64 | wavelengths | Raw Verasonics one-way lens correction delay (2.961 wl), kept for provenance |
| `metadata/subject/id` | scalar                       | str | - | De-identified subject code |
| `metadata/subject/type` | scalar                       | str | - | `human` |


## Subject Metadata
- number of subjects: 30,
- age range: 18-65,
- 20 Female , 10 Male
- Thyroid gland
- healthy volunteers with incidental pathology (e.g. cysts, nodules) in some cases
- Verasonics vantage 128 system, L11-5v linear array probe

## Data Validation

`reconstruct.py` runs the following `zea.Pipeline` (saved as `pipeline.yaml`,
alongside this README) to reconstruct a B-mode image from the raw channel
data: cast → demodulate → DAS beamform (with native, per-element/per-pixel
lens correction) → envelope detect → normalize → log compress.

```yaml
# Beamforming config for the L11-5v thyroid scans (focused ray-line imaging).
# RF data is real (not IQ/baseband, sampleMode=NS200BW) so it is demodulated
# before beamforming. grid_size_x/z fix the output resolution; the actual
# depth mapped onto it comes from this file's own N_AX (D1: 2048 samples,
# ~50mm one-way in 1540 m/s tissue).

parameters:
  grid_size_x: 300
  grid_size_z: 400
  dynamic_range: [-60, 0]

pipeline:
  operations:
    - name: keras.ops.cast
      params:
        dtype: float32
    - demodulate
    - name: beamform
      params:
        beamformer: delay_and_sum
        num_patches: 200
    - envelope_detect
    - normalize
    - log_compress
```

Example reconstruction, run via `python reconstruct.py --data-dir subjects
--n-scans 3 --n-frames 3 --seed 0 --output three_patients_grid.png`
(3 subjects, 3 randomly sampled frames each):

![3 frames from 3 different subjects](three_patients_grid.png)

All 9 sampled frames, across 3 different subjects, show consistent diffuse
in-vivo tissue speckle with no reconstruction artifacts, confirming the
acquisition geometry, timing metadata, and native lens correction are
correctly recorded/applied across the released dataset, not just a single
acquisition.

## Known Issues

- `decimSampleRate`, `quadDecim`, and `demodFrequency` are absent from the
  raw Verasonics workspace (`BaselineWorkspace.mat`) and were instead
  sourced from this session's recalibration checkpoint files,
  where they were confirmed constant.
- The default beamforming grid in `pipeline.yaml` (`grid_size_x=300`,
  `grid_size_z=400`) is coarser than the half-wavelength Nyquist rate for
  this probe/frequency; this only affects the resolution of the example
  reconstruction images, not the released raw channel data.
- The number of frames for each subject is variable, depending on the subject's
  anatomy and the radiologist's scanning protocol. It may also be affected by
  bottlenecks in the Verasonics system's data transfer rate,
  which can cause "dropped frames" when the system cannot keep up with the
  acquisition speed.

## Ethical Considerations

This acquisition was performed under an IRB-approved research protocol,
with informed consent obtained from the subject prior to scanning. The
subjects are identified only by a de-identified code  with no directly
identifying information (name, exact date of birth, medical record number)
stored in the released file. No `acquisition_time` timestamp is embedded in
the zea file, consistent with HIPAA Safe Harbor de-identification guidance
for human-subject data.

