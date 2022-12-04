import os
import torch
import torch.nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F 
import torchvision.utils as utils
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
import argparse

alexnet = models.resnet50(pretrained=True)

data_transforms = transforms.Compose([
        transforms.Resize((224,224)),             # resize the input to 224x224
        transforms.ToTensor(),              # put the input to tensor format
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the input
        # the normalization is based on images from ImageNet
    ])

test_image_filepath = "test_data/tiger.jpg"
img = Image.open(test_image_filepath)
transformed_img = data_transforms(img)
batch_img = torch.unsqueeze(transformed_img, 0)

alexnet.eval()

output = alexnet(batch_img)

sorted, indices = torch.sort(output, descending=True)
percentage = F.softmax(output, dim=1)[0] * 100.0
results = [percentage[i].item() for i in indices[0][:5]]
print("\nresnet50: print the first 5 classes the testing image belongs to")
for i in range(5):
    print('{:.4f}%'.format(results[i]))

