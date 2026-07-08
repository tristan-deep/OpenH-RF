---
pretty_name: "OpenH-RF — UltraSound ToolBox (USTB) Channel Capture Collection"
license: cc-by-4.0
task_categories:
  - image-to-image
language:
  - en
tags:
  - ultrasound
  - rf
  - openh-rf
  - beamforming
  - channel-data
size_categories:
  - n<1K
---

# UltraSound ToolBox (USTB) Channel Capture Collection — OpenH-RF Submission

Contributed by the **University of Oslo (UiO), Department of Informatics** (the USTB team)
to the [OpenH-RF](https://github.com/open-h/OpenH-RF) initiative. All data is pre-beamformed
(raw channel-capture) ultrasound stored in the **zea** HDF5 format (`zea_version 0.1.0`,
≥ required `0.1.0a3`) and released under **CC BY 4.0**.

## Contents

41 acquisitions packaged into six application sub-datasets. Each sub-dataset folder contains its
zea `.hdf5` files, a Hugging Face–style `README.md` data card, and one reference B-mode PNG per
acquisition. The **`reconstruct.py`, `pipeline.yaml`, and the CC BY 4.0 `LICENCE` live once at the
submission root** — the reconstruction reconstructs every sub-dataset (the pipeline is identical
across folders), and the single LICENCE covers the whole collection (each data card also declares
`license: cc-by-4.0` in its YAML frontmatter).

| Folder | Sub-dataset | Tier | RFP task | Acq. |
|---|---|---|---|---|
| `A_cardiac/` | In-vivo cardiac (Verasonics P4-2) | in-vivo human (research) | 6.1 Generalized Reconstruction | 3 |
| `B_carotid/` | In-vivo carotid (Verasonics L7-4) | in-vivo human (research) | 6.1 Generalized Reconstruction | 3 |
| `C_verasonics_phantom/` | Phantom (Verasonics L7-4 / P4) | phantom | 6.1 Generalized Reconstruction | 15 |
| `D_alpinion_phantom/` | Phantom (Alpinion L3-8) | phantom | 6.1 Generalized Reconstruction | 4 |
| `E_simulation/` | Simulation (Field II) | simulation | 6.1 Generalized Reconstruction | 12 |
| `F_motion/` | Motion estimation (SWE / ARFI, L7-4) | phantom | 6.4 Motion Estimation | 4 |

Total: **41 acquisitions**, ~10 GB on disk.

## How to reconstruct

Every acquisition is reconstructable from the file alone — all acquisition parameters live in the
zea `/scan` and `/probe` groups. A single `reconstruct.py` at the submission root reconstructs
every acquisition in every sub-dataset folder:

```bash
export KERAS_BACKEND=jax
python reconstruct.py                          # every .hdf5 in all sub-folders
python reconstruct.py A_cardiac/<file>.hdf5    # a single acquisition
```

All reconstruction choices live in **`parameters.yaml`** — one entry per acquisition giving its
`pipeline` (which `zea.Pipeline` to use), display window (`zlims`/`xlims` in mm) and `dynamic_range`.
Every acquisition uses the same workflow (load parameters → run the pipeline → plot); there is no
per-file logic in the script. Each `pipeline` value maps to a `zea.Pipeline` YAML at the root:

| `pipeline` | Used for | `zea.Pipeline` |
|---|---|---|
| `scanline` | focused linear (FI) scans | `pipeline_scanline.yaml`: `beamform` on a `grid_type: scanline` grid with `enable_receive_apodization: true` — one focused transmit per image line. The artifact-free reconstruction for a walking-focus linear scan (compounding focused beams onto a shared grid leaves a band at the focal depth). |
| `scanline_sector` | steered focused scans (fixed origin, varying angle) | `pipeline_scanline_sector.yaml`: same scanline setup along steered rays, displayed on fan geometry from `parameters.grid`. Preserves speckle that compounding would smooth away. |
| `sector` | phased-array focused sector scans | `pipeline_sector.yaml`: polar grid + scan conversion; pressure-field-weighted DAS with peaked weighting (≈ scanline) and `focal_region_length`. |
| `iq` | baseband IQ (`n_ch == 2`, e.g. PICMUS) | `pipeline_iq.yaml` (no demodulation step). |
| `compound` | non-focused linear (plane-wave / diverging / STA / SWE-ARFI) | `pipeline.yaml`: coherent compounding, no pfield (these insonify the whole field of view). |

The pipelines share the same shape as zea's regular B-mode pipeline (`cast → apply_window →
demodulate → beamform → envelope → normalize → log_compress`). Scanline imaging
uses the same `beamform` op with a scanline grid plus receive apodization rather than a dedicated
scanline-specific operation. Two zea features used in this submission are scanline receive
apodization (`enable_receive_apodization`) and `focal_region_length` (focal-region delay blending,
Rindal et al., IUS 2018).

An acquisition may also set `refocus: true` in `parameters.yaml` (enabled for a few many-angle
CPWC acquisitions). For those, `reconstruct.py` additionally runs `pipeline_refocus.yaml` —
REFoCUS transmit-encoding recovery (Bottenus, 2018) that inverts the transmit-encoding matrix to
recover the multistatic (full-matrix-capture) dataset before pfield DAS — and writes a
`<name>_zea_refocus_bmode.png` alongside the standard reconstruction, to demonstrate the option.

## Reference images

Each acquisition ships with:
- `<name>_zea_bmode.png` — produced by the root `reconstruct.py` (zea); the reproducible reference.
- `<name>_zea_bmode_old.png` — the initial v2 zea reconstruction (kept for comparison).
- `<name>_bmode.png` — the canonical USTB MATLAB Delay-And-Sum reconstruction from the public
  [USTB dataset catalog](https://unioslo.github.io/USTB/datasets.html), for cross-validation.
- `<name>_zea_refocus_bmode.png` — REFoCUS variant, only for acquisitions with `refocus: true`.

## Licensing & attribution

Released under **CC BY 4.0**. Please cite the UltraSound ToolBox (USTB), University of Oslo
(Zenodo record 20261898). `PICMUS_numerical_calib_v2` (group E) was created **in collaboration
with our group** as part of the PICMUS effort and is included on that basis; the other PICMUS
datasets are excluded.

## Contact

Ole Marius Hoel Rindal — omrindal@ifi.uio.no (UiO, USTB).
