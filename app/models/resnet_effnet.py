# app/models/resnet_effnet.py

import os
from typing import Dict

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Optional


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")

RESNET_WEIGHTS = os.path.join(MODEL_DIR, "resnet18_finetuned.pt")
EFFNET_WEIGHTS = os.path.join(MODEL_DIR, "efficientnet_b0_finetuned.pt")

IMG_SIZE = 192  # what we used during training


_common_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


class ResNet18Finetuned:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Architecture must match training code
        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 2)

        state = torch.load(RESNET_WEIGHTS, map_location=self.device)
        self.model.load_state_dict(state, strict=True)

        self.model.to(self.device)
        self.model.eval()

        self.transform = _common_transform
        self.class_names = ["Cat", "Dog"]

    def predict(self, image: Image.Image) -> Dict:
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)

        label = self.class_names[idx.item()]
        return {
            "label": label,
            "confidence": float(conf.item()),
        }


class EfficientNetB0Finetuned:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Architecture must match training code
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 2)

        state = torch.load(EFFNET_WEIGHTS, map_location=self.device)
        self.model.load_state_dict(state, strict=True)

        self.model.to(self.device)
        self.model.eval()

        self.transform = _common_transform
        self.class_names = ["Cat", "Dog"]

    def predict(self, image: Image.Image) -> Dict:
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)

        label = self.class_names[idx.item()]
        return {
            "label": label,
            "confidence": float(conf.item()),
        }
