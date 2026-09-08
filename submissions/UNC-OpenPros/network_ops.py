# import os
# os.environ["KERAS_BACKEND"] = "torch"

import torch
from network import InversionNet

from zea.internal.registry import ops_registry
from zea.ops import Operation

@ops_registry('openpros.network_ops.InversionNetInference')
class InversionNetInference(Operation):
    def __init__(self, weights_path:str, **kwargs):
        super().__init__(**kwargs)
        self.weights_path = weights_path
        self.model = InversionNet()
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        self.model.eval()
        

    def call(self, **kwargs):
        data = kwargs[self.key]
        self.model.to(data.device)
        data = data.squeeze(-1)
        with torch.no_grad():
            output = self.model(data).squeeze(1).unsqueeze(-1)
        return {self.output_key: output}
