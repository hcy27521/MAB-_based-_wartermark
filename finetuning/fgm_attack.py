import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from cleverhans.torch.utils import optimize_linear


def preprocess_image(image):
    # Preprocess the image to match the requirements of ResNet18
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    return image


def fast_gradient_method(
    model,
    x,
    eps,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False,
):
    # Function implementation remains the same as in the previous response
    ...


# Load the pretrained ResNet18 model
model = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# Load the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = CIFAR10(root='./data', train=False, download=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

epsilon = 0.1  # Epsilon parameter for the attack
norm = np.inf  # L-infinity norm

# Generate adversarial examples for each image in the dataset
for i, (image, target) in enumerate(dataloader):
    target_tensor = torch.tensor([target])  # Convert the scalar target to a tensor
    adv_example = fast_gradient_method(model, image, epsilon, norm, y=target_tensor)

    # Check if adv_example is not None
    if adv_example is not None:
        # Save the adversarial image to a file
        save_image(adv_example, f"adversarial_image_{i}.png")

    if i >= 10:  # Generate adversarial examples for the first 10 images only (for demonstration)
        break
