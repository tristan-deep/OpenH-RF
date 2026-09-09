---
pretty_name: "OpenH-RF — us4us Ring-Array USCT (forearm, breast phantom, water)"
license: cc-by-4.0
task_categories:
  - other
tags:
  - ultrasound
  - rf
  - openh-rf
  - zea
  - usct
  - tomography
  - ring-array
  - attenuation
  - forearm
language:
  - en
size_categories:
  - n<1K
---

## Dataset Description
Raw RF data acquired with the us4us Ltd. us4R system and a Draminski ring probe.

Intended for ultrasound tomographic reconstruction.

Acquisitions were performed on:
- the left and right human forearms of 8 healthy volunteers,
- a Yezitronix B-RG-1.2 breast phantom,
- a single reference slice from water only.

The forearm and phantom acquisitions comprise consecutive slices spaced 9 mm apart in depth (half of the probe elevation).

Each slice was recorded using 1024 subsequent single-element transmissions, and received by a 512-element aperture located opposite the transmitting element.


## Dataset Contributors
PI: Ziemowit Klimonda

Team: Piotr Jarosik, Jakub Rozbicki, Piotr Karwat, Marcin Lewandowski (all @ us4us Ltd. (https://us4us.eu/))


## Dataset Creation Date
09/07/2026


## License
CC BY 4.0


## Intended Usage
Ultrasound computed tomography image reconstruction.


## Dataset Characterization
  * Data Collection Method: phantom, healthy adult human volunteers.
  * Labeling Method: N/A.
  * Acquisition system: 
      - ring probe:
          - probe radius: 130mm,
          - number of elements: 1024, 
          - center frequency: 2MHz, 
          - sampling rate: 8125000.0,
      - us4R research system + host PC,
      - custom positioning system for subject spatial control.
  * Tx/Rx scheme:
      - the scheme consisted of 1024 transmit-receive events,
      - each event consisted of a single probe element transmitting a short pulse (1 period at 2 MHz excitation), while 512 probe elements were used for reception,
      - the 1024 transmit events were performed sequentially, with each probe element transmitting once,
      - for each transmission, the center of the receiving aperture was positioned on the opposite side of the probe relative to the transmitting element,
      - for example, for the 0th transmission, the 0th probe element transmitted, and the receiving aperture consisted of probe elements 256–767.

## Dataset Format
All sub-datasets are provided in the ZEA file format.
No preprocessing was performed on the raw channel data.


## Dataset Quantification
  * Each HDF5 file contains 10 slices of the subject, except `reference_water.hdf5`, which contains a single slice.
  * Each HDF5 file has a size of 4,973,053,792 bytes, except `reference_water.hdf5`, which is 516,269,920 bytes.


## Subject Metadata
The dataset contains data from:
  * breast phantom Yezitronix B-RG-1.2,
  * both forearms of 8 healthy human volunteers (2 females and 6 males, age range [20, 55]),
  * water only (reference data).


## Data Validation
  * The data files contain attenuation sinograms and images reconstructed from raw data using the filtered backprojection algorithm.


## Known Issues
  * The following probe elements should be considered damaged: [367, 409, 457, 764, 775, 776, 368, 390, 470, 739, 792].


## Ethical Considerations
  * All human subjects were healthy adults who voluntarily participated in the measurements.

