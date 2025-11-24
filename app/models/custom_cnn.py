import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image


class CustomCNNClassifier:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load MobileNetV2 architecture
        self.model = models.mobilenet_v2(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 2)

        # Load your trained weights
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Define transformations (must match train script)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.class_names = ["Cat", "Dog"]

    def predict(self, image: Image.Image):
        img = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)

        label = self.class_names[pred_idx.item()]
        confidence = confidence.item()

        return {
            "label": label,
            "confidence": confidence
        }
