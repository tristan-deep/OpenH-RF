"""Prove the registered-operation pipeline round-trips through YAML."""
import numpy as np, zea
from zea import Config, File, Pipeline
from zea.ops import Normalize
from custom_ops import MyRearrange, LogTransform
from model_ops import InversionNetSOS

zea.init_device(verbose=False)

def build():
    return Pipeline(operations=[
        MyRearrange(),
        LogTransform(data_min=-0.25, data_max=0.45, k=1e5),
        Normalize(output_range=(-1, 1)),
        InversionNetSOS(weights_path="InversionNet_weights_only.pth"),
        Normalize(input_range=(-1, 1), output_range=(1300, 3600)),
    ])

print("--- 1. build + to_yaml ---")
cfg = build().to_config()
cfg.to_yaml("pipeline.yaml")
print("wrote pipeline.yaml OK")

print("--- 2. reload from YAML ---")
loaded = Pipeline.from_config(Config.from_yaml("pipeline.yaml"))
print("Pipeline.from_config OK ->", [type(o).__name__ for o in loaded.operations])

print("--- 3. run both, compare ---")
with File("openpros_sample.hdf5") as f:
    raw = f.data.raw_data[:]
a = build()(data=raw)["data"]
b = loaded(data=raw)["data"]
a_np, b_np = np.asarray(a.cpu()), np.asarray(b.cpu())
print("in-code output :", a_np.shape)
print("from-YAML output:", b_np.shape)
print("max abs diff   :", float(np.abs(a_np - b_np).max()))
print("IDENTICAL" if np.array_equal(a_np, b_np) else "DIFFERS")
