import torch
from torchvision import models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

print("Building model")
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
# model.load_state_dict(torch.load("model.pth"))
#
# print(model)
