import torch
from torchvision import models

from torchvision.models.quantization import MobileNet_V2_QuantizedWeights


torch.backends.quantized.engine = 'qnnpack'

print("Building quantized model")
model = models.quantization.mobilenet_v2(weights="DEFAULT", quantize=True)

torch.save(model.state_dict(), "model_quantized.pth")
print("Saved PyTorch Model State to model_quantized.pth")
# model.load_state_dict(torch.load("model_quantized.pth"))

# print(model)
