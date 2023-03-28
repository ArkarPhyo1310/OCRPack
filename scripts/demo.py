import math

import torch
from PIL import Image
from torchvision import transforms

from ocrpack.models.text_recog.backbones.vit import VisionTransformer

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
SCALE_SIZE = int(math.floor(IMAGE_SIZE / 0.875))

model = VisionTransformer()
model.load_state_dict(torch.load("./pretrained_weights/vit_base_16_224.pt"))
model.eval()

filename = "./assets/dog.jpg"
img = Image.open(filename).convert('RGB')
transform = transforms.Compose([
    transforms.Resize(SCALE_SIZE, interpolation=transforms.functional.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])
tensor = transform(img).unsqueeze(0)  # transform and add batch dimension

with torch.no_grad():
    out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim=0)
print(probabilities.shape)

# Get imagenet class mappings
with open("./assets/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Print top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
