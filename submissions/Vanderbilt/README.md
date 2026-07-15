---
pretty_name: "OpenH-RF — Vanderbilt / Multi-Frame Focused Transmit Echocardiography Channel Dataset"
license: cc-by-4.0
task_categories:
  - image-reconstruction            # e.g. image-segmentation, image-classification
tags:
  - ultrasound
  - rf                   # or iq, depending on data_type
  - openh-rf
language:
  - en
size_categories:
  - n<1K                 # 82 ciweloops from 29 patients; each contains 32 frames. 
---


# Dataset Description
This multi-frame focused transmit echocardiography channel dataset contains over 2000 frames of fundamental and harmonic data acquired with the P4-2v probe on Verasonic Vantage 128. This dataset was originally acquired to visualize the left atrial appendage in patients following transesoophageal echocardiography. Some patients have atrial fibrillation which can cause blood clots to form in the appendage.

#Dataset Contributors
Brett Byram (PI), Christopher Khan, Ying-Chun (Preston) Pan, Zoe Marshall

# Dataset Creation Date
2026/07/11

# Liceense/Terms of Use
CC BY 4.0

# Intended Usage
This dataset could be useful for training a domain adaptive network as it captures a wide range of in vivo image quality. It might also be useful for training a model that converts fundamental to harmonic data. It contains view labels provided by the physician at the time of data acquisition, but these views were often slightly modified to accommodate the patient (see Subject Metadata). It could be useful for identifying thrombus in the left atrial appendage as ~ 10% of the dataset came from patients with a thrombus (see Subject Metadata).

# Dataset Characterization
- Data Collection Method: clinical
- Labeling Method: human-annotated 
- Acquisition system: Verasonics Vantage 128, P4-2v (64 elements). Fundamental: 2.72 MHz; Harmonic: 2.08 MHz for transmit, 4.16 MHz for receive. Sampling rate: 10.88 MHz. 

# Dataset Format
All sub-datasets are submitted in the zea file format. No preprocessing. 

# Dataset Quantification
A total of 2624 frames from 82 ciweloops (32 frames each) across 29 patients. Another 2624 frames of harmonic data. 

# Subject Metadata
- imaging_view_name: one of {'apical two chamber', 'apical four chamber', 'parasternal short axis', 'parasternal long axis', 'subxiphoid', 'N/A'}. Because acquisition occurred while patients were recovering from anesthesia, sonographers often deviated from standard view geometry to maximize image quality; as a result, these datasets should not be used to train a view classification network. The 'N/A' label marks acquisitions that didn't approximate any standard view at all — in these cases, abandoning the standard geometry entirely was the only way to obtain reasonable image quality. 

- Notes: unstructured free-text on patient condition and acquisition quality. Common content includes: left atrial appendage occlusion devices (Watchman, Amulet, or Conformal), presence of thrombus, comorbidities such as COPD, and patient motion during acquisition.

# Data Validation
Provided as a separate reconstruct.py script in the root folder. 

# Known Issues
As mentioned in Subject Metadata, these view names should only be used as a reference for interpreting the image and should not be used to train a view classification task. 

# Ethical Considerations
The study was approved by Vanderbilt's IRB.