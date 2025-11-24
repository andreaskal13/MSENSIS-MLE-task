import os
import numpy as np
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response  

from .inference import inference_engine



app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="Classifies images of cats and dogs using a custom CNN or a pre-trained ViT.",
    version="1.0.0",
)

# Allow calls from Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"} 



@app.get("/model/info")
def model_info():
    return {                     ### <-- ADDED
        "custom_model": {
            "name": "MobileNetV2 (fine-tuned)",
            "task": "Binary classification (Cat vs Dog)",
        },
        "pretrained_vit": {
            "name": "ViT Base Patch16-224 (ImageNet)",
            "task": "Cat/Dog normalization + breed extraction",
        },
        "labels": ["Cat", "Dog"],
        "version": "1.0.0",
    }




@app.post("/predict/custom")
async def predict_custom(file: UploadFile = File(...)):
    file_bytes = await file.read()
    result = inference_engine.predict_custom(file_bytes)
    return result

@app.post("/predict/pretrained")
async def predict_pretrained(file: UploadFile = File(...)):
    file_bytes = await file.read()
    result = inference_engine.predict_pretrained(file_bytes)
    return result


@app.post("/predict/custom_cam")
async def predict_custom_cam(file: UploadFile = File(...)):
    file_bytes = await file.read()
    result = inference_engine.predict_custom_with_cam(file_bytes)
    return result


@app.post("/predict/all_models")
async def predict_all_models(file: UploadFile = File(...)):
    """
    Run inference on:
      - Custom CNN
      - ResNet18 (finetuned)
      - EfficientNet-B0 (finetuned)
      - ViT (pretrained)
    and return a comparison.
    """
    file_bytes = await file.read()
    result = inference_engine.predict_all_models(file_bytes)
    return result
