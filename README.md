🐱🐶 Cats vs Dogs — ML Classification Suite

Streamlit Frontend • FastAPI Backend • MobileNetV2 / ViT / ResNet18 / EfficientNet-B0
Dockerized • Production-ready • Clean UV/Python Environment Option

This project delivers a full ML pipeline for classifying cats and dogs, including:
✔ Custom CNN (MobileNetV2 fine-tuned)
✔ Vision Transformer
✔ ResNet18 finetuned
✔ EfficientNet-B0 finetuned
✔ Grad-CAM explainability
✔ Training Insights dashboards
✔ FastAPI inference backend
✔ Streamlit frontend


🚀 Quick Start (Recommended): Run with Docker
1. Clone the project
```
git clone https://github.com/YOUR_USER/cats-dogs-app.git
cd cats-dogs-app
```

2. Build and start
```
docker compose up --build
```
3. Access

Frontend (Streamlit): http://localhost:8501

Backend (FastAPI docs): http://localhost:8000/docs


🐍 Option B: Run Locally Using UV (Fast Python Environment)

Install UV:

```
pip install uv   # or installer from https://astral.sh
```

Create environment (run inside project root):

```
uv venv
uv pip install -r requirements.txt
```


Run backend:

```
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Run frontend:
```
uv run streamlit run frontend/app.py
```

🧱 Project Structure

```
cats-dogs-app/
│  README.md
│  requirements.txt
│  docker-compose.yml
│  Dockerfile.backend
│  Dockerfile.frontend
│  config.yaml
│
├─ app/                 # FastAPI backend
├─ frontend/            # Streamlit UI
├─ scripts/             # Training scripts
├─ data/                # Dataset (ignored in repo)
│   ├─ images/          # 2GB+ not included
│   └─ user_datasets/
└─ app/models/saved/    # Trained models + metrics

```


