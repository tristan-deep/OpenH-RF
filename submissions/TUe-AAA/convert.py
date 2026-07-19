# -*- coding: utf-8 -*-

import os
os.environ["KERAS_BACKEND"] = "jax"

import zea
print("ZEA loaded successfully")


from zea.data.convert.verasonics import VerasonicsFile

VerasonicsFile(r"AAApatient01_vsx.mat").to_zea(r"AAApatient01_zea.hdf5")

""" 
Received errors and warnings during conversion:
zea: WARNING The probe geometry is not ordered as a uniform linear array. Focal distances are not set to infinity for plane waves.
zea: ERROR Could not read Verasonics ImgDataP buffer: too many indices for array: array is 0-dimensional, but 1 were indexed, skipping.
zea: WARNING Probe name 'C5-2v' is not in the list of known probes. Please add it to the _VERASONICS_TO_ZEA_PROBE_NAMES dictionary. Falling back to generic probe.
zea: WARNING Optional MetadataSpec field 'subject' is not set. Description: Subject associated with the study. Defaulted to None.
zea: WARNING Optional MetadataSpec field 'credit' is not set. Description: Credit or attribution for the dataset. Defaulted to None.

"""

" add metadata: 
        
import h5py

with h5py.File("AAApatient01_zea.hdf5", "r+") as f:

    patient = f.require_group("metadata/patient")

    patient["age"] = 67
    patient["sex"] = "M"
    patient["bmi"] = 24.3



" READ parameters:    

with zea.File("AAApatient01_zea.hdf5") as file:
    parameters = file.load_parameters()
    
print(parameters)


"Read file content: 
    
with h5py.File("AAApatient01_zea.hdf5", "r") as f:
    f.visit(print)
