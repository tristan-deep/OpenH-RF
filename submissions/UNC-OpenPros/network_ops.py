import torch
from network import InversionNet
import zea

device = zea.init_device(verbose=False, hide_others=False)

model = InversionNet()
model.load_state_dict(torch.load('InversionNet_weights_only.pth', map_location='cpu'))
model.to(device)
model.eval()

def inference(data):
    data = data.unsqueeze(0).squeeze(-1).to(device)
    with torch.no_grad():
        output = model(data)
    return output.squeeze(0)
