import torch
from torchvision import models

torch.backends.quantized.engine = 'qnnpack'

print("Building quantized model")
model = models.quantization.mobilenet_v2(quantize=True)

torch.save(model.state_dict(), "model_quantized.pth")
print("Saved PyTorch Model State to model.pth")
model.load_state_dict(torch.load("model_quantized.pth"))

print(model)