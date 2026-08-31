"""Serializable zea operation wrapping the pretrained OpenPros InversionNet.

Replaces ``zea.ops.Lambda(inference, ...)``, which cannot be written to YAML
("Cannot serialize generic 'lambda' operation with an arbitrary callable").

Two rules make this round-trip through ``pipeline.to_yaml`` / ``Pipeline.from_config``:

1. The class is registered under its full module path, so zea re-imports this
   module automatically when loading the YAML.
2. Only YAML-serializable values are constructor arguments. The model is built
   from ``weights_path`` inside ``__init__`` -- never passed in as an object.
"""

import torch
from zea.internal.registry import ops_registry
from zea.ops import Operation

from network import InversionNet


@ops_registry("openpros.model_ops.InversionNetSOS")
class InversionNetSOS(Operation):
    """Run the pretrained InversionNet on rearranged OpenPros channel data."""

    def __init__(self, weights_path: str = "InversionNet_weights_only.pth", **kwargs):
        super().__init__(**kwargs)
        self.weights_path = weights_path
        self.model = InversionNet()
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()

    def call(self, **kwargs):
        # NOTE: a registered Operation keeps the leading frame axis that
        # ``Lambda`` strips, so the input here is (n_frames, 40, n_ax, n_el, 1),
        # not (40, n_ax, n_el, 1). Dropping the trailing channel axis alone
        # already gives conv2d the 4D (batch, C, H, W) it wants, and the result
        # (n_frames, 1, 401, 161) keeps the frame axis the pipeline expects.
        data = kwargs[self.key].squeeze(-1)
        with torch.no_grad():
            output = self.model(data)
        return {self.output_key: output}
