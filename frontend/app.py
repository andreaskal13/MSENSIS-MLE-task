import os
from io import BytesIO
import json
import zipfile
import subprocess
import shutil
from datetime import datetime
from typing import Optional  # Python 3.8-safe typing

import numpy as np
import requests
import streamlit as st
from PIL import Image
import yaml

# ============================================================
# HYBRID DOCKER/NON-DOCKER BACKEND AUTO-DETECTION
# ============================================================


def detect_backend_host() -> str:
    """
    Detect backend target automatically:
      • If BACKEND_URL environment variable exists → use it.
      • If inside Docker → use http://catsdogs-backend:8000
      • Otherwise (local run) → use http://127.0.0.1:8000
    """
    env_url = os.getenv("BACKEND_URL")
    if env_url:
        return env_url.rstrip("/")

    running_in_docker = os.path.exists("/.dockerenv") or (
        os.environ.get("DOCKER_CONTAINER", "").lower() == "true"
    )

    if running_in_docker:
        return "http://catsdogs-backend:8000"

    return "http://127.0.0.1:8000"

def running_in_docker():
    return os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER") == "true"



BACKEND_BASE = detect_backend_host()

API_URL_CUSTOM = f"{BACKEND_BASE}/predict/custom"
API_URL_PRETRAINED = f"{BACKEND_BASE}/predict/pretrained"
API_URL_CUSTOM_CAM = f"{BACKEND_BASE}/predict/custom_cam"
API_URL_ALL_MODELS = f"{BACKEND_BASE}/predict/all_models"
API_URL_HEALTH = f"{BACKEND_BASE}/health"

# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Local dev layout:  <project_root>/app/models/saved
LOCAL_MODEL_PLOTS_DIR = os.path.join(ROOT_DIR, "app", "models", "saved")
# Docker layout:     /app/models/saved  (mounted by docker-compose)
DOCKER_MODEL_PLOTS_DIR = "/app/models/saved"

if os.path.exists(DOCKER_MODEL_PLOTS_DIR):
    # Inside Docker container
    MODEL_PLOTS_DIR = DOCKER_MODEL_PLOTS_DIR
else:
    # Local run (uvicorn+streamlit pipeline, no container)
    MODEL_PLOTS_DIR = LOCAL_MODEL_PLOTS_DIR

DATA_DIR = os.path.join(ROOT_DIR, "data", "images")
USER_DATASETS_DIR = os.path.join(ROOT_DIR, "data", "user_datasets")
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")

# ============================================================
# Streamlit Page Config
# ============================================================

st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐱🐶",
    layout="wide",
)

# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1f2937 0, #020617 45%, #000 100%);
        color: #e5e7eb;
    }
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }
    .app-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #9ca3af;
        margin-bottom: 1.4rem;
    }
    .accent {
        color: #22c55e;
    }
    .card {
        padding: 1.2rem 1.4rem;
        border-radius: 0.9rem;
        background: rgba(15,23,42,0.92);
        border: 1px solid rgba(148,163,184,0.35);
        box-shadow: 0 18px 45px rgba(15,23,42,0.9);
        backdrop-filter: blur(10px);
        animation: fadeInUp 0.5s ease-out;
    }
    .card-soft {
        padding: 1.0rem 1.2rem;
        border-radius: 0.8rem;
        background: rgba(15,23,42,0.7);
        border: 1px solid rgba(75,85,99,0.6);
        backdrop-filter: blur(6px);
    }
    .card-title {
        font-weight: 600;
        font-size: 1.02rem;
        margin-bottom: 0.6rem;
    }
    button[data-baseweb="tab"] > div {
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translate3d(0, 18px, 0); }
        to   { opacity: 1; transform: translate3d(0, 0, 0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE (just init, no post-widget mutation)
# ============================================================

if "compare_custom_vit" not in st.session_state:
    st.session_state.compare_custom_vit = False

if "compare_all_models" not in st.session_state:
    st.session_state.compare_all_models = False

# ============================================================
# HELPERS
# ============================================================


def resize_for_preview(image: Image.Image, size=(224, 224)):
    return image.resize(size)


def check_backend_health() -> bool:
    try:
        r = requests.get(API_URL_HEALTH, timeout=3)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def predict_with_api(endpoint: str, file_bytes: bytes):
    try:
        resp = requests.post(
            endpoint,
            files={"file": ("image.jpg", file_bytes, "image/jpeg")},
            timeout=30,
        )
        if resp.status_code != 200:
            return None, f"Server error: {resp.text}"
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Backend/API not reachable."
    except Exception as e:
        return None, str(e)


def safe_show_image(path, caption=""):
    if os.path.exists(path):
        st.image(path, use_container_width=True, caption=caption)
    else:
        st.warning(f"Missing: {os.path.basename(path)}")


def load_metrics(model_name, version="v1"):
    path = os.path.join(MODEL_PLOTS_DIR, model_name, version, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_misclassified(model_name, version="v1"):
    path = os.path.join(MODEL_PLOTS_DIR, model_name, version, "misclassified.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def load_metadata(model_name, version="v1"):
    path = os.path.join(MODEL_PLOTS_DIR, model_name, version, "metadata.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def get_file_size_mb(path):
    if os.path.exists(path):
        return round(os.path.getsize(path) / (1024 * 1024), 2)
    return None


def extract_dataset_zip(uploaded_zip) -> Optional[str]:
    if uploaded_zip is None:
        return None

    os.makedirs(USER_DATASETS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(USER_DATASETS_DIR, f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(BytesIO(uploaded_zip.read()), "r") as zf:
        zf.extractall(out_dir)

    images_dir = os.path.join(out_dir, "images")
    labels_csv = os.path.join(out_dir, "labels.csv")

    if not os.path.exists(images_dir) or not os.path.exists(labels_csv):
        st.error("ZIP must contain `images/` and `labels.csv`.")
        shutil.rmtree(out_dir, ignore_errors=True)
        return None

    st.success(f"Dataset extracted to: {out_dir}")
    return out_dir

# ============================================================
# TRAINING OVERRIDES
# ============================================================


def run_training_with_config_override(
    models_to_train, dataset_dir, epochs, batch_size, max_samples, version_name
):

    if not os.path.exists(CONFIG_PATH):
        st.error(f"Missing config.yaml at {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, "r") as f:
        orig_cfg_text = f.read()
    cfg = yaml.safe_load(orig_cfg_text)

    cfg["data"]["images_dir"] = os.path.join(dataset_dir, "images")
    cfg["data"]["labels_csv"] = os.path.join(dataset_dir, "labels.csv")

    cfg["training"]["epochs"] = int(epochs)
    cfg["training"]["batch_size"] = int(batch_size)
    if max_samples:
        cfg["training"]["max_samples"] = int(max_samples)

    if "custom" in models_to_train:
        cfg["models"]["custom_cnn"]["version"] = version_name
    if "resnet" in models_to_train:
        cfg["models"]["resnet18"]["version"] = version_name
    if "effnet" in models_to_train:
        cfg["models"]["efficientnet_b0"]["version"] = version_name

    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f)

    logs = {}

    try:
        # Custom CNN
        if "custom" in models_to_train:
            script = os.path.join(SCRIPTS_DIR, "train_custom_model.py")
            if os.path.exists(script):
                result = subprocess.run(
                    ["python", script],
                    cwd=ROOT_DIR,
                    capture_output=True,
                    text=True,
                )
                logs["custom_cnn"] = result.stdout + "\n" + result.stderr

        # ResNet + EffNet
        if "resnet" in models_to_train or "effnet" in models_to_train:
            script = os.path.join(SCRIPTS_DIR, "train_resnet_efficientnet.py")
            if os.path.exists(script):
                result = subprocess.run(
                    ["python", script],
                    cwd=ROOT_DIR,
                    capture_output=True,
                    text=True,
                )
                logs["resnet_effnet"] = result.stdout + "\n" + result.stderr

    finally:
        with open(CONFIG_PATH, "w") as f:
            f.write(orig_cfg_text)

    return logs

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    ["Classifier", "Training Insights", "Train Models"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Project:** Cats vs Dogs<br>"
    "**Backend:** FastAPI<br>"
    "**Frontend:** Streamlit",
    unsafe_allow_html=True,
)

# Block classifier-related pages if backend is down
if page in ["Classifier", "Training Insights"]:
    if not check_backend_health():
        st.error(f"❌ Backend API is not running at {BACKEND_BASE}.")
        st.stop()

# ============================================================
# PAGE 1 — CLASSIFIER
# ============================================================

if page == "Classifier":

    st.markdown(
        '<div class="app-title">🐱🐶 Cats vs Dogs Classifier </div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="app-subtitle">'
        'Upload an image, choose a model, inspect predictions & explanations.'
        '</div>',
        unsafe_allow_html=True,
    )

    col_controls, col_info = st.columns([1.1, 1.1])

    # -------- Controls --------
    with col_controls:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload a cat/dog image",
            type=["jpg", "jpeg", "png"],
        )

        # SAFE checkboxes (no on_change, no post-widget mutation)
        compare_custom_vit_value = st.session_state.get("compare_custom_vit", False)
        compare_all_models_value = st.session_state.get("compare_all_models", False)

        compare_custom_vit = st.checkbox(
            "Compare Custom CNN vs ViT",
            key="compare_custom_vit",
            value=compare_custom_vit_value,
            help="Run Custom CNN vs ViT side-by-side.",
        )

        compare_all_models = st.checkbox(
            "Compare ALL 4 Models",
            key="compare_all_models",
            value=compare_all_models_value,
            help="Run all four models on this image.",
        )

        # Derived *effective* flags (mutually exclusive in logic)
        compare_all_active = bool(compare_all_models)
        compare_custom_active = bool(compare_custom_vit and not compare_all_active)

        model_choice = st.selectbox(
            "Primary model",
            ["Custom CNN", "ViT Pretrained", "ResNet18 (finetuned)", "EfficientNet-B0 (finetuned)"],
            disabled=(compare_custom_active or compare_all_active),
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- Info panel --------
    with col_info:
        st.markdown('<div class="card-soft">', unsafe_allow_html=True)
        st.markdown(
            """
            **Models used:**
            - **Custom CNN** → fine-tuned MobileNetV2.
            - **ViT** → Vision Transformer.
            - **ResNet18** → finetuned.
            - **EfficientNet-B0** → finetuned.
            """,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # -------- If image uploaded --------
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()

        try:
            image = Image.open(BytesIO(file_bytes)).convert("RGB")
        except Exception:
            st.error("Invalid image file.")
            st.stop()

        st.markdown("<br>", unsafe_allow_html=True)

        img_col, prev_col = st.columns([1.0, 1.2])

        with img_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Original Image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with prev_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Model-Ready Previews</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.caption("224×224 Models")
                st.image(resize_for_preview(image, (224, 224)))
            with c2:
                st.caption("64×64 (Custom CNN)")
                st.image(resize_for_preview(image, (64, 64)))
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab_pred, tab_explain = st.tabs(["Prediction", "Explainability"])

        # --- PREDICTION TAB ---
        with tab_pred:

            # Helper to robustly read /predict/all_models and normalize to
            # a dict: key -> model_result_dict
            def get_all_models_result(file_bytes):
                result_all, err = predict_with_api(API_URL_ALL_MODELS, file_bytes)
                if err:
                    return None, err
                if not isinstance(result_all, dict):
                    return None, "Unexpected response from backend: expected JSON object."

                models_data = result_all.get("models")
                models_by_key = {}

                # Accept list-of-dicts
                if isinstance(models_data, list):
                    for m in models_data:
                        if not isinstance(m, dict):
                            continue
                        key = m.get("key")
                        if isinstance(key, str):
                            models_by_key[key] = m

                # Or dict-of-dicts
                elif isinstance(models_data, dict):
                    for k, v in models_data.items():
                        if isinstance(v, dict):
                            key = v.get("key") or k
                            if isinstance(key, str):
                                models_by_key[key] = v

                else:
                    return None, (
                        f"Unexpected 'models' type from backend: "
                        f"{type(models_data).__name__}"
                    )

                if not models_by_key:
                    return None, "No valid model entries found in /predict/all_models."

                return models_by_key, None

            if compare_all_active:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">All 4 Models</div>', unsafe_allow_html=True)

                # Call backend once
                result_all, err = predict_with_api(API_URL_ALL_MODELS, file_bytes)
                if err or not isinstance(result_all, dict):
                    st.error(err or "Failed to parse /predict/all_models response.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.stop()

                raw_models = result_all.get("models", None)

                # Normalize to dict
                if isinstance(raw_models, list):
                    models_dict = {
                        entry.get("key"): entry
                        for entry in raw_models
                        if isinstance(entry, dict) and entry.get("key")
                    }
                elif isinstance(raw_models, dict):
                    models_dict = raw_models
                else:
                    st.error("Invalid models structure returned by backend.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.stop()

                # Key mapping: backend keys → display names
                model_display = [
                    ("custom", "Custom CNN"),       # <- key used by backend
                    ("vit", "ViT"),
                    ("resnet18", "ResNet18"),
                    ("efficientnet_b0", "EfficientNet-B0"),
                ]

                cols = st.columns(len(model_display))

                for col, (key, display_name) in zip(cols, model_display):
                    with col:
                        m = models_dict.get(key)

                        if not isinstance(m, dict):
                            st.warning(f"Missing: {display_name}")
                            continue

                        name = m.get("name") or display_name or key
                        label = m.get("label", "N/A")
                        conf = float(m.get("confidence", 0.0) or 0.0)
                        if conf < 0.0:
                            conf = 0.0
                        if conf > 1.0:
                            conf = 1.0

                        st.subheader(name)
                        st.metric("Prediction", label, f"{conf*100:.1f}%")
                        st.progress(conf)

                st.markdown("</div>", unsafe_allow_html=True)

            elif compare_custom_active:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Custom CNN vs ViT</div>', unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    r, e = predict_with_api(API_URL_CUSTOM, file_bytes)
                    if r:
                        st.metric("Custom CNN", r["label"], f"{r['confidence'] * 100:.1f}%")
                        st.progress(r["confidence"])
                    elif e:
                        st.error(e)

                with c2:
                    r, e = predict_with_api(API_URL_PRETRAINED, file_bytes)
                    if r:
                        st.metric("ViT", r["label"], f"{r['confidence'] * 100:.1f}%")
                        st.progress(r["confidence"])
                        if "raw_label" in r:
                            st.caption("ImageNet: " + r["raw_label"])
                    elif e:
                        st.error(e)

                st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Prediction</div>', unsafe_allow_html=True)

                endpoint = None
                model_key = None

                if model_choice == "Custom CNN":
                    endpoint = API_URL_CUSTOM
                elif model_choice == "ViT Pretrained":
                    endpoint = API_URL_PRETRAINED
                elif model_choice == "ResNet18 (finetuned)":
                    endpoint = API_URL_ALL_MODELS
                    model_key = "resnet18"
                elif model_choice == "EfficientNet-B0 (finetuned)":
                    endpoint = API_URL_ALL_MODELS
                    model_key = "efficientnet_b0"

                if model_key is None:
                    result, err = predict_with_api(endpoint, file_bytes)
                else:
                    models_by_key, err = get_all_models_result(file_bytes)
                    result = models_by_key.get(model_key) if (models_by_key and not err) else None
                    if result is None and not err:
                        err = f"Model '{model_key}' missing in /predict/all_models."

                if err or result is None:
                    st.error(err or "Prediction failed.")
                else:
                    conf = float(result["confidence"])
                    if conf < 0.0:
                        conf = 0.0
                    if conf > 1.0:
                        conf = 1.0

                    st.metric(
                        "Prediction",
                        result["label"],
                        f"{conf * 100:.1f}%",
                    )
                    st.progress(conf)
                    if "raw_label" in result:
                        st.caption("ImageNet label: " + result["raw_label"])

                st.markdown("</div>", unsafe_allow_html=True)

        # --- EXPLAINABILITY TAB ---
        with tab_explain:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Grad-CAM (Custom CNN)</div>', unsafe_allow_html=True)

            if st.button("Run Grad-CAM"):
                cam, err = predict_with_api(API_URL_CUSTOM_CAM, file_bytes)
                if err or cam is None:
                    st.error(err or "Grad-CAM failed.")
                else:
                    cam_img = np.array(cam["cam"], dtype=np.uint8)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("Original")
                        st.image(image, use_container_width=True)
                    with c2:
                        st.caption("Grad-CAM")
                        st.image(cam_img, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PAGE 2 — TRAINING INSIGHTS
# ============================================================

elif page == "Training Insights":

    st.markdown(
        '<div class="app-title">Training Insights</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="app-subtitle">'
        'Deep diagnostic metrics from your last training runs.'
        '</div>',
        unsafe_allow_html=True,
    )

    m_custom = load_metrics("custom_cnn")
    m_resnet = load_metrics("resnet18_finetuned")
    m_effnet = load_metrics("efficientnet_b0_finetuned")

    is_docker = running_in_docker()

    tabs = ["Custom CNN", "ResNet18", "EfficientNet-B0", "Comparison"]

    # Only add Misclassified Samples when NOT inside Docker
    if not is_docker:
        tabs.append("Misclassified Samples")

    tabs_obj = st.tabs(tabs)

    # Unpack tabs dynamically
    tab_custom, tab_resnet, tab_effnet, tab_compare = tabs_obj[:4]

    # Optional 5th tab only outside Docker
    tab_mis = tabs_obj[4] if not is_docker else None


    # ----- Custom CNN -----
    with tab_custom:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Custom CNN — MobileNetV2")
        c1, c2, c3 = st.columns(3)
        with c1:
            safe_show_image(os.path.join(MODEL_PLOTS_DIR, "accuracy_curve.png"), "Accuracy")
        with c2:
            safe_show_image(os.path.join(MODEL_PLOTS_DIR, "loss_curve.png"), "Loss")
        with c3:
            safe_show_image(os.path.join(MODEL_PLOTS_DIR, "confusion_matrix.png"), "Confusion")

        c4, c5 = st.columns(2)
        with c4:
            safe_show_image(os.path.join(MODEL_PLOTS_DIR, "custom_cnn_roc.png"), "ROC Curve")
        with c5:
            safe_show_image(os.path.join(MODEL_PLOTS_DIR, "custom_cnn_pr.png"), "PR Curve")

        if m_custom:
            st.json(
                {
                    "ROC AUC": m_custom["roc_auc"],
                    "PR AUC": m_custom["pr_auc"],
                    "Per-class accuracy": m_custom["per_class_accuracy"],
                    "Inference speed (ms/img)": m_custom["inference_ms_per_image"],
                }
            )

        meta = load_metadata("custom_cnn")
        if meta:
            size_mb = get_file_size_mb(meta.get("weights_path", ""))
            st.json(
                {
                    "Model name": meta["model_name"],
                    "Version": meta["version"],
                    "Device": meta["device"],
                    "Image size": meta["img_size"],
                    "Batch size": meta["batch_size"],
                    "Validation split": meta["val_split"],
                    "Max samples": meta["max_samples"],
                    "Weights file": meta["weights_path"],
                    "Checkpoint size (MB)": size_mb,
                }
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ----- ResNet18 -----
    with tab_resnet:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("ResNet18 — Finetuned")
        c1, c2, c3 = st.columns(3)
        with c1:
            safe_show_image(
                os.path.join(MODEL_PLOTS_DIR, "resnet18_finetuned_accuracy.png"),
                "Accuracy",
            )
        with c2:
            safe_show_image(
                os.path.join(MODEL_PLOTS_DIR, "resnet18_finetuned_loss.png"),
                "Loss",
            )
        with c3:
            safe_show_image(
                os.path.join(MODEL_PLOTS_DIR, "resnet18_finetuned_confusion.png"),
                "Confusion",
            )

        c4, c5 = st.columns(2)
        with c4:
            safe_show_image(
                os.path.join(MODEL_PLOTS_DIR, "resnet18_finetuned_roc.png"),
                "ROC Curve",
            )
        with c5:
            safe_show_image(
                os.path.join(MODEL_PLOTS_DIR, "resnet18_finetuned_pr.png"),
                "PR Curve",
            )

        if m_resnet:
            st.json(
                {
                    "ROC AUC": m_resnet["roc_auc"],
                    "PR AUC": m_resnet["pr_auc"],
                    "Per-class accuracy": m_resnet["per_class_accuracy"],
                    "Inference speed (ms/img)": m_resnet["inference_ms_per_image"],
                }
            )

        meta = load_metadata("resnet18_finetuned")
        if meta:
            size_mb = get_file_size_mb(meta.get("weights_path", ""))
            st.json(
                {
                    "Model name": meta["model_name"],
                    "Version": meta["version"],
                    "Device": meta["device"],
                    "Image size": meta["img_size"],
                    "Batch size": meta["batch_size"],
                    "Validation split": meta["val_split"],
                    "Max samples": meta["max_samples"],
                    "Weights file": meta["weights_path"],
                    "Checkpoint size (MB)": size_mb,
                }
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ----- EfficientNet-B0 -----
    with tab_effnet:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("EfficientNet-B0 — Finetuned")

        c1, c2, c3 = st.columns(3)
        with c1:
            safe_show_image(
                os.path.join(MODEL_PLOTS_DIR, "efficientnet_b0_finetuned_accuracy.png"),
                "Accuracy",
            )
        with c2:
            safe_show_image(
                os.path.join(MODEL_PLOTS_DIR, "efficientnet_b0_finetuned_loss.png"),
                "Loss",
            )
        with c3:
            safe_show_image(
                os.path.join(MODEL_PLOTS_DIR, "efficientnet_b0_finetuned_confusion.png"),
                "Confusion",
            )

        c4, c5 = st.columns(2)
        with c4:
            safe_show_image(
                os.path.join(MODEL_PLOTS_DIR, "efficientnet_b0_finetuned_roc.png"),
                "ROC Curve",
            )
        with c5:
            safe_show_image(
                os.path.join(MODEL_PLOTS_DIR, "efficientnet_b0_finetuned_pr.png"),
                "PR Curve",
            )

        if m_effnet:
            st.json(
                {
                    "ROC AUC": m_effnet["roc_auc"],
                    "PR AUC": m_effnet["pr_auc"],
                    "Per-class accuracy": m_effnet["per_class_accuracy"],
                    "Inference speed (ms/img)": m_effnet["inference_ms_per_image"],
                }
            )

        meta = load_metadata("efficientnet_b0_finetuned")
        if meta:
            size_mb = get_file_size_mb(meta.get("weights_path", ""))
            st.json(
                {
                    "Model name": meta["model_name"],
                    "Version": meta["version"],
                    "Device": meta["device"],
                    "Image size": meta["img_size"],
                    "Batch size": meta["batch_size"],
                    "Validation split": meta["val_split"],
                    "Max samples": meta["max_samples"],
                    "Weights file": meta["weights_path"],
                    "Checkpoint size (MB)": size_mb,
                }
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ----- Comparison -----
    with tab_compare:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("ResNet18 vs EfficientNet-B0 — Validation Comparison")

        c1, c2 = st.columns(2)
        with c1:
            safe_show_image(os.path.join(MODEL_PLOTS_DIR, "compare_acc.png"), "Validation Accuracy")
        with c2:
            safe_show_image(os.path.join(MODEL_PLOTS_DIR, "compare_loss.png"), "Validation Loss")

        safe_show_image(os.path.join(MODEL_PLOTS_DIR, "compare_cm.png"), "Confusion Matrices")

        st.markdown("</div>", unsafe_allow_html=True)

        # ----- Misclassified Samples -----
        # Only show this tab when NOT running inside Docker
        if not is_docker and tab_mis:
            with tab_mis:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Misclassified Samples (Validation Set)")

                mis_custom = load_misclassified("custom_cnn")
                mis_res = load_misclassified("resnet18_finetuned")
                mis_eff = load_misclassified("efficientnet_b0_finetuned")

                choice = st.selectbox(
                    "Select model:",
                    ["custom_cnn", "resnet18_finetuned", "efficientnet_b0_finetuned"],
                )

                data = {
                    "custom_cnn": mis_custom,
                    "resnet18_finetuned": mis_res,
                    "efficientnet_b0_finetuned": mis_eff,
                }.get(choice, [])

                if not data:
                    st.info("No misclassified samples found in last training run.")
                else:
                    cols = st.columns(3)
                    for i, item in enumerate(data):
                        img_path = os.path.join(DATA_DIR, item["image_name"])
                        with cols[i % 3]:
                            st.image(
                                img_path,
                                caption=(
                                    f"True: {'Cat' if item['true_label']==0 else 'Dog'} | "
                                    f"Pred: {'Cat' if item['pred_label']==0 else 'Dog'} | "
                                    f"P(Dog): {item['prob_dog']:.2f}"
                                ),
                                use_container_width=True,
                            )

                st.markdown("</div>", unsafe_allow_html=True)

        else:
            # Docker-friendly message (tab hidden OR message shown)
            st.info(
                "⚠️ The 'Misclassified Samples' view is not available in the Docker build.\n\n"
                "The Docker image does not include the full 2GB dataset required for this section.\n"
                "Run the application locally to access misclassified validation images."
            )


# ============================================================
# PAGE 3 — TRAIN MODELS
# ============================================================

else:  # "Train Models"

    st.markdown(
        '<div class="app-title">Train Models (v.0.01 - UI) </div>',
        unsafe_allow_html=True,
    )


    st.warning(
        "⚠️ **Alpha Version (v0.0.1 UI only)** — The interface is functional, "
        "but training pipelines are **not implemented yet**."
    )

    st.markdown(
        '<div class="app-subtitle">'
        'Upload a dataset, configure training, run pipelines.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Expected dataset.zip:
        ```
        dataset.zip
        ├── images/
        └── labels.csv
        ```
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)

    dataset_zip = st.file_uploader("Upload dataset.zip", type=["zip"])

    col_left, col_right = st.columns(2)

    with col_left:
        models_to_train = st.multiselect(
            "Models to train",
            [
                ("Custom CNN (MobileNetV2)", "custom"),
                ("ResNet18 (finetuned)", "resnet"),
                ("EfficientNet-B0 (finetuned)", "effnet"),
            ],
            default=[
                ("Custom CNN (MobileNetV2)", "custom"),
                ("ResNet18 (finetuned)", "resnet"),
                ("EfficientNet-B0 (finetuned)", "effnet"),
            ],
            format_func=lambda x: x[0],
        )
        model_flags = [m[1] for m in models_to_train]

    with col_right:
        default_epochs = 3
        default_batch = 32
        default_max_samples = 2000

        try:
            with open(CONFIG_PATH, "r") as f:
                cfg_def = yaml.safe_load(f)
                default_epochs = cfg_def["training"].get("epochs", default_epochs)
                default_batch = cfg_def["training"].get("batch_size", default_batch)
                default_max_samples = cfg_def["training"].get("max_samples", default_max_samples)
        except Exception:
            pass

        epochs = st.number_input(
            "Epochs", min_value=1, max_value=100, value=int(default_epochs)
        )
        batch_size = st.number_input(
            "Batch size", min_value=8, max_value=256, value=int(default_batch)
        )
        max_samples = st.number_input(
            "Max samples", min_value=100, max_value=50000, value=int(default_max_samples)
        )
        version_name = st.text_input("Model version tag", value="v2")

    start_clicked = st.button("Start Training")

    if start_clicked:
        if not dataset_zip:
            st.error("Upload a dataset.zip first.")
        elif not model_flags:
            st.error("Select at least one model.")
        else:
            dataset_dir = extract_dataset_zip(dataset_zip)
            if dataset_dir:
                with st.spinner("Running training pipelines..."):
                    logs = run_training_with_config_override(
                        models_to_train=model_flags,
                        dataset_dir=dataset_dir,
                        epochs=epochs,
                        batch_size=batch_size,
                        max_samples=max_samples,
                        version_name=version_name,
                    )
                st.success("Training complete.")
                if logs:
                    with st.expander("View training logs"):
                        for name, log in logs.items():
                            st.markdown(f"**{name}**")
                            st.code(log)

    st.markdown("</div>", unsafe_allow_html=True)
