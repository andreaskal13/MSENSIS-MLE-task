# import os
# import numpy as np
# from typing import Tuple

# from PIL import Image
# import torch
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# from typing import Optional



# class PretrainedViT:
#     """
#     Wrapper around a pre-trained Vision Transformer.
#     """

#     def __init__(self, model_name: str = "google/vit-base-patch16-224", device: Optional[str] = None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.processor = AutoImageProcessor.from_pretrained(model_name)
#         self.model = AutoModelForImageClassification.from_pretrained(model_name)
#         self.model.to(self.device)
#         self.model.eval()

#     def preprocess(self, image: Image.Image):
#         return self.processor(images=image, return_tensors="pt")

#     def predict(self, image: Image.Image) -> Tuple[str, float]:
#         inputs = self.preprocess(image)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}

#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             logits = outputs.logits
#             probs = torch.softmax(logits, dim=-1)[0]

#         confidence, idx = torch.max(probs, dim=0)
#         label = self.model.config.id2label[idx.item()]
#         return label, confidence.item()


import torch
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification


class PretrainedViT:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.model.to(self.device).eval()

        # All ImageNet labels
        self.id2label = self.model.config.id2label

        # Automatically categorize cat/dog related labels
        self.auto_cat_labels = []
        self.auto_dog_labels = []

        for label in self.id2label.values():
            l = label.lower()

            # Cat-related substrings
            if (
                "cat" in l
                or "kitten" in l
                or "lynx" in l
                or "siamese" in l
                or "persian" in l
                or "tabby" in l
            ):
                self.auto_cat_labels.append(l)

            # Dog-related substrings
            if (
                "dog" in l
                or "puppy" in l
                or "terrier" in l
                or "retriever" in l
                or "shepherd" in l
                or "hound" in l
                or "bulldog" in l
                or "husky" in l
            ):
                self.auto_dog_labels.append(l)

        print(f"Auto-detected {len(self.auto_cat_labels)} cat labels.")
        print(f"Auto-detected {len(self.auto_dog_labels)} dog labels.")

    def predict(self, image: Image.Image):
        # Preprocess
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get highest probability
        confidence, pred_class = torch.max(probs, dim=1)
        confidence = float(confidence)
        raw_label = self.id2label[pred_class.item()].lower()

        # 🔥 Normalization using automatic lists
        if raw_label in self.auto_cat_labels:
            final_label = "Cat"
        elif raw_label in self.auto_dog_labels:
            final_label = "Dog"
        else:
            # fallback decision
            final_label = "Cat" if confidence > 0.5 else "Dog"

        return {
            "label": final_label,
            "confidence": confidence,
            "raw_label": raw_label
        }
