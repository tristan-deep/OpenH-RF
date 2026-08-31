"""Prove the registered-operation pipeline round-trips through YAML.

Three checks, in order of what they establish:

1. ``pipeline.to_config().to_yaml(...)`` succeeds where ``Lambda`` raised
   ``TypeError: Cannot serialize generic 'lambda' operation ...``.
2. ``Pipeline.from_config`` rebuilds it and produces bit-identical output.
3. A **fresh process** loads ``pipeline.yaml`` with neither ``custom_ops`` nor
   ``model_ops`` imported by hand, and runs it. This is the check that catches a
   registration name which is not a real importable module path: registering as
   ``openpros.custom_ops.MyRearrange`` while the file is a top-level
   ``custom_ops.py`` fails here with ``ModuleNotFoundError: No module named
   'openpros'``, even though checks 1 and 2 both pass.

Run from the submission folder (needs openpros_sample.hdf5 and the checkpoint):

    KERAS_BACKEND=torch python verify_pipeline_roundtrip.py
"""

import subprocess
import sys
import textwrap

import numpy as np
import zea
from zea import Config, File, Pipeline
from zea.ops import Normalize

from custom_ops import LogTransform, MyRearrange
from model_ops import InversionNetSOS

zea.init_device(verbose=False)

WEIGHTS = "InversionNet_weights_only.pth"
SAMPLE = "openpros_sample.hdf5"


def build():
    return Pipeline(
        operations=[
            MyRearrange(),
            LogTransform(data_min=-0.25, data_max=0.45, k=1e5),
            Normalize(output_range=(-1, 1)),
            InversionNetSOS(weights_path=WEIGHTS),
            Normalize(input_range=(-1, 1), output_range=(1300, 3600)),
        ]
    )


print("--- 1. build + to_yaml ---")
build().to_config().to_yaml("pipeline.yaml")
print("wrote pipeline.yaml OK")

print("\n--- 2. reload from YAML, compare outputs ---")
loaded = Pipeline.from_config(Config.from_path("pipeline.yaml"))
print("Pipeline.from_config OK ->", [type(o).__name__ for o in loaded.operations])

with File(SAMPLE) as f:
    raw = f.data.raw_data[:]

a = np.asarray(build()(data=raw)["data"].cpu())
b = np.asarray(loaded(data=raw)["data"].cpu())
print("in-code output  :", a.shape)
print("from-YAML output:", b.shape)
print("max abs diff    :", float(np.abs(a - b).max()))
print("IDENTICAL" if np.array_equal(a, b) else "DIFFERS")

print("\n--- 3. cold load in a fresh process (no manual imports) ---")
cold = textwrap.dedent(
    """
    import sys, zea
    from zea import Config, File, Pipeline
    zea.init_device(verbose=False)
    assert "custom_ops" not in sys.modules and "model_ops" not in sys.modules
    p = Pipeline.from_config(Config.from_path("pipeline.yaml"))
    with File("openpros_sample.hdf5") as f:
        raw = f.data.raw_data[:]
    print("cold-loaded and ran ->", tuple(p(data=raw)["data"].shape))
    """
)
r = subprocess.run([sys.executable, "-c", cold], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip()[-400:])
print("COLD LOAD OK" if r.returncode == 0 else "COLD LOAD FAILED")
