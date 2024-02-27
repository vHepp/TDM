# %%

# %%
import time
from torchvision import transforms
from PIL import Image
import urllib
import torch
from torchvision import models

# torch.backends.quantized.engine = 'qnnpack'

# %%
# Define model

# model = models.quantization.mobilenet_v2(quantize=True)

model = models.mobilenet_v2()

# model


# %%
print("loading model")

# model.load_state_dict(torch.load("model_quantized.pth"))
model.load_state_dict(torch.load("model.pth"))

# %%
# %%bash
# wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

# %%
# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# %%
# Download an example image from the pytorch website
url, filename = (
    "https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)


with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# %%
# sample execution (requires torchvision)
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
# create a mini-batch as expected by the model
input_batch = input_tensor.unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')


# %%


# get the start time
start = time.time()


# %%
with torch.no_grad():
    output = model(input_batch)


# get the end time
stop = time.time()
# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)

# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

print()
# get the execution time
elapsed_time = stop - start
print('Execution time:', elapsed_time, 'seconds')
