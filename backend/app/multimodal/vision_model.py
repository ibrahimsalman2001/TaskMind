# vision_model.py

import torch
from torchvision import models, transforms
from PIL import Image
from typing import List, Tuple

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Preprocessing pipeline for EfficientNet
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    ),
])

# Load pre-trained EfficientNet
model = models.efficientnet_b0(pretrained=True)
model = model.to(device)  # Move model to GPU
model.eval()  # Inference mode

# Get ImageNet labels
with open("D:/FYP/backend/imagenet_classes.txt") as f:
    idx_to_label = [line.strip() for line in f.readlines()]


def classify_frame(image_path: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Classify an image using EfficientNet and return top_k predictions.
    """
    image = Image.open(image_path).convert("RGB")
    tensor = image_transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_idxs = torch.topk(probs, top_k)

    return [(idx_to_label[i], top_probs[j].item()) for j, i in enumerate(top_idxs)]