# Cats vs Dogs — ML Classification app

Full Stack ML App · FastAPI Backend · Streamlit Frontend · Dockerized

This project implements a complete end-to-end machine learning system for classifying Cats vs Dogs images using four deep learning models:

- Custom CNN (MobileNetV2-based)

- ViT Pretrained

- ResNet18 (finetuned)

- EfficientNet-B0 (finetuned)

### The application includes:

A FastAPI backend serving inference & Grad-CAM explainability

A Streamlit frontend providing interactive comparison UI

Complete training pipelines, plots & metrics

Full Docker support for easy deployment

Optional modern Python environment via uv



## Features
 
### Classifier UI

- Single-model or multi-model prediction

- Side-by-side 4-model comparison

- Grad-CAM explainability (Custom CNN)

### Training Insights

- Accuracy / loss curves

- ROC, PR curves

- Confusion matrices

- Model metadata + checkpoint sizes

- Misclassified sample viewer (disabled in Docker container unless dataset mounted)

### Model Training (alpha version, UI only)

- Run entire training pipelines from UI

- Upload custom dataset ZIP (images/ + labels.csv)

- Versioned training outputs stored under app/models/saved/

- Pretrained models included inside the repository

# Project Structure

```
MSENSIS-MLE-task/
│
├── app/                           # <<< BACKEND SOURCE ROOT (FastAPI)
│   ├── __init__.py
│   ├── main.py                    # FastAPI entrypoint
│   ├── inference.py               # Shared inference helpers
│   │
│   ├── models/                    # All model architectures + saved weights/plots
│   │   ├── __init__.py
│   │   ├── custom_cnn.py
│   │   ├── pretrained_vit.py
│   │   ├── resnet_effnet.py
│   │   │
│   │   └── saved/                
│   │       ├── accuracy_curve.png
│   │       ├── loss_curve.png
│   │       ├── confusion_matrix.png
│   │       ├── compare_acc.png
│   │       ├── compare_loss.png
│   │       ├── compare_cm.png
│   │       ├── custom_cnn_roc.png
│   │       ├── custom_cnn_pr.png
│   │       ├── custom_cnn.pt
│   │       │
│   │       ├── custom_cnn/
│   │       │   └── v1/
│   │       │       ├── accuracy.png
│   │       │       ├── loss.png
│   │       │       ├── confusion.png
│   │       │       ├── roc.png
│   │       │       ├── pr.png
│   │       │       ├── best_model.pt
│   │       │       ├── metrics.json
│   │       │       └── metadata.json
│   │       │
│   │       ├── resnet18_finetuned/
│   │       │   └── v1/ (...)
│   │       │
│   │       └── efficientnet_b0_finetuned/
│   │           └── v1/ (...)
│   │
│   └── data/                     # Backend data root
│       └── images/               # Local datasets (ignored in Docker)
│
│
├── frontend/
│   └── app.py                    # Streamlit UI 
│
│
├── scripts/                      # Training pipeline scripts
│   ├── train_custom_model.py
│   └── train_resnet_efficientnet.py
│
├── config.yaml                   # Global config
│
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
│
├── requirements.txt              # Unified requirements
├── requirements.backend.txt      # For docker build
├── requirements.frontend.txt     # for docker build
│
├── .dockerignore
├── .gitignore
│
└── README.md


```


# Instructions on setting the app

You can setup the application in two ways:

## 1. Run with Docker

Clone repo
```
git clone https://github.com/andreaskal13/MSENSIS-MLE-task
cd MSENSIS-MLE-task
```

then just build and run
```
docker compose up --build
```

UI is accesed by:

```
Frontend: http://localhost:8501
Backend:  http://localhost:8000/docs
```

### Note:
The Docker build mounts only:

app/models/saved/ → pretrained models

data/ → optional user files

The large training dataset is not included inside the repo (see “Dataset” below).

## 2. Run locally with UV
Prequisite: install uv:

```
pip install uv
```

Clone repo
```
git clone https://github.com/andreaskal13/MSENSIS-MLE-task
cd MSENSIS-MLE-task
```


Then run these commands to set up the dependencies:
```
uv venv
uv pip install -r requirements.txt
```

You can run the application by opening a terminal in the directory where you cloned the repo. In a terminal run:

```
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

and in another terminal in the same directory run:

```
uv run streamlit run frontend/app.py
```

# Dataset

In case the user wants to retrain,or experiment, or fine tune further, we attach the dataset that contains images of cats and dogs, and its corresponding labels :

https://crisaeu-my.sharepoint.com/personal/a_kyrizaki_msensis_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fa%5Fkyrizaki%5Fmsensis%5Fcom%2FDocuments%2Fdataset%2Ezip&parent=%2Fpersonal%2Fa%5Fkyrizaki%5Fmsensis%5Fcom%2FDocuments&ga=1

The dataset has the structure:

```
dataset.zip
├── images/
└── labels.csv
```





























