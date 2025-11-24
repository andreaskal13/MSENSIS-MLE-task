# app/inference.py

import os
from typing import Dict

import numpy as np
from PIL import Image
import torch
import cv2

from .models.custom_cnn import CustomCNNClassifier
from .models.pretrained_vit import PretrainedViT
from .models.resnet_effnet import ResNet18Finetuned, EfficientNetB0Finetuned

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
CUSTOM_MODEL_PATH = os.path.join(MODEL_DIR, "custom_cnn.pt")


class InferenceEngine:
    def __init__(self):
        self.custom_model: CustomCNNClassifier | None = None
        self.pretrained_model: PretrainedViT | None = None
        self.resnet_model: ResNet18Finetuned | None = None
        self.efficientnet_model: EfficientNetB0Finetuned | None = None

    # --------- loaders ---------
    def load_custom_model(self) -> CustomCNNClassifier:
        if self.custom_model is None:
            self.custom_model = CustomCNNClassifier(CUSTOM_MODEL_PATH)
        return self.custom_model

    def load_pretrained_model(self) -> PretrainedViT:
        if self.pretrained_model is None:
            self.pretrained_model = PretrainedViT()
        return self.pretrained_model

    def load_resnet_model(self) -> ResNet18Finetuned:
        if self.resnet_model is None:
            self.resnet_model = ResNet18Finetuned()
        return self.resnet_model

    def load_efficientnet_model(self) -> EfficientNetB0Finetuned:
        if self.efficientnet_model is None:
            self.efficientnet_model = EfficientNetB0Finetuned()
        return self.efficientnet_model

    # --------- helpers ---------
    def _load_image(self, file_bytes: bytes) -> Image.Image:
        from io import BytesIO
        return Image.open(BytesIO(file_bytes)).convert("RGB")

    # --------- single-model APIs (existing) ---------
    def predict_custom(self, file_bytes: bytes) -> Dict:
        img = self._load_image(file_bytes)
        model = self.load_custom_model()
        out = model.predict(img)   # already returns dict
        return out

    def predict_pretrained(self, file_bytes: bytes) -> Dict:
        img = self._load_image(file_bytes)
        model = self.load_pretrained_model()
        out = model.predict(img)   # returns dict with label, confidence, raw_label
        return out

    # --------- NEW: multi-model comparison ---------
    def predict_all_models(self, file_bytes: bytes) -> Dict:
        """
        Run prediction on all models (Custom CNN, ResNet18, EfficientNetB0, ViT)
        and return a comparison structure.
        """
        img = self._load_image(file_bytes)

        # Ensure models are loaded
        custom = self.load_custom_model().predict(img)
        resnet = self.load_resnet_model().predict(img)
        effnet = self.load_efficientnet_model().predict(img)
        vit = self.load_pretrained_model().predict(img)

        # Normalize structure; vit already has raw_label
        models_out = [
            {
                "name": "Custom CNN (MobileNetV2)",
                "key": "custom_cnn",
                "label": custom["label"],
                "confidence": float(custom["confidence"]),
                "extra": None,
            },
            {
                "name": "ResNet18 (finetuned)",
                "key": "resnet18",
                "label": resnet["label"],
                "confidence": float(resnet["confidence"]),
                "extra": None,
            },
            {
                "name": "EfficientNet-B0 (finetuned)",
                "key": "efficientnet_b0",
                "label": effnet["label"],
                "confidence": float(effnet["confidence"]),
                "extra": None,
            },
            {
                "name": "ViT (ImageNet, auto-mapped)",
                "key": "vit",
                "label": vit["label"],
                "confidence": float(vit["confidence"]),
                "extra": vit.get("raw_label", None),
            },
        ]

        # Decide "winner": highest confidence (regardless of model)
        best_idx = int(
            max(range(len(models_out)), key=lambda i: models_out[i]["confidence"])
        )
        best_model = models_out[best_idx]

        # Agreement metric: fraction of models that chose the same label as the winner
        same_label_count = sum(
            1 for m in models_out if m["label"] == best_model["label"]
        )
        agreement = same_label_count / len(models_out)

       # Convert list → dict with stable keys
        return {
            "models": {
                "custom": {
                    "label": custom["label"],
                    "confidence": float(custom["confidence"])
                },
                "vit": {
                    "label": vit["label"],
                    "confidence": float(vit["confidence"]),
                    "raw_label": vit.get("raw_label")
                },
                "resnet18": {
                    "label": resnet["label"],
                    "confidence": float(resnet["confidence"])
                },
                "efficientnet_b0": {
                    "label": effnet["label"],
                    "confidence": float(effnet["confidence"])
                }
            },
            "best_model": best_model,
            "agreement": agreement
        }


    # --------- Grad-CAM for custom CNN (unchanged) ---------
    def predict_custom_with_cam(self, file_bytes):
        from torchvision import transforms

        # ---------------------------------------------------------
        # 1) Load image AND DOWNSCALE IT to avoid huge processing
        # ---------------------------------------------------------
        img = self._load_image(file_bytes)

        # 🔥 prevents freezing on 4000×3000 images
        img.thumbnail((256, 256))

        # ---------------------------------------------------------
        # 2) Preprocess for the MobileNetV2 custom CNN
        # ---------------------------------------------------------
        model = self.load_custom_model()
        model.model.eval()

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        x = transform(img).unsqueeze(0).to(model.device)

        # ---------------------------------------------------------
        # 3) Grad-CAM hooks
        # ---------------------------------------------------------
        activations = {}
        gradients = {}

        # MobileNetV2 CAM layer → stable final conv block
        target_layer = model.model.features[18]

        def forward_hook(module, inp, output):
            activations["value"] = output

        def backward_hook(module, grad_in, grad_out):
            gradients["value"] = grad_out[0]

        # Use correct hook API
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

        # ---------------------------------------------------------
        # 4) Forward + backward pass
        # ---------------------------------------------------------
        out = model.model(x)
        pred_class = torch.argmax(out, dim=1)

        score = out[0, pred_class]
        model.model.zero_grad()
        score.backward()

        # ---------------------------------------------------------
        # 5) Retrieve activations + gradients
        # ---------------------------------------------------------
        act = activations["value"][0].cpu().detach().numpy()
        grad = gradients["value"][0].cpu().detach().numpy()

        weights = grad.mean(axis=(1, 2))
        cam = np.zeros(act.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * act[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))

        if cam.max() > 0:
            cam = cam / cam.max()

        # ---------------------------------------------------------
        # 6) Overlay CAM heatmap on resized original
        # ---------------------------------------------------------
        vis_img = img.resize((224, 224))  # prettiest and fast
        heat = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)

        img_np = np.array(vis_img)
        overlay = cv2.addWeighted(img_np, 0.5, heat, 0.5, 0)

        # ---------------------------------------------------------
        # 7) Return JSON-safe results
        # ---------------------------------------------------------
        return {
            "label": "Cat" if pred_class.item() == 0 else "Dog",
            "confidence": float(torch.softmax(out, 1)[0, pred_class].detach()),
            "cam": overlay.tolist()   # small size = no timeout
        }




inference_engine = InferenceEngine()
