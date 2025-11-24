import os
import random
import json
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
)
import time


# ============================================================
# LOAD CONFIG
# ============================================================
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

SEED = CFG.get("seed", 42)

DATA_CFG = CFG["data"]
TRAIN_CFG = CFG["training"]
MODELS_CFG = CFG["models"]
ARTIFACT_CFG = CFG["artifacts"]

DATA_DIR = DATA_CFG["images_dir"]
LABEL_CSV = DATA_CFG["labels_csv"]
SAVE_DIR = ARTIFACT_CFG["base_dir"]

BATCH_SIZE = TRAIN_CFG["batch_size"]
# custom CNN uses 64x64 by design; allow override via img_size_custom
IMG_SIZE = TRAIN_CFG.get("img_size_custom", 64)
EPOCHS = TRAIN_CFG["epochs"]
MAX_SAMPLES = TRAIN_CFG["max_samples"]
VAL_SPLIT = TRAIN_CFG["val_split"]
LR = TRAIN_CFG["lr"]

SCHED_CFG = TRAIN_CFG.get("lr_scheduler", {})
ES_CFG = TRAIN_CFG.get("early_stopping", {})
ES_PATIENCE = ES_CFG.get("patience", 3)

CUSTOM_CFG = MODELS_CFG.get("custom_cnn", {"name": "custom_cnn", "version": "v1"})


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================
class CatsDogsDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.copy().reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_name"])
        label = 0 if row["label"].lower() == "cat" else 1

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # return filename for misclassified logging
        return img, label, row["image_name"]


# ============================================================
# CSV LOADING + CLEANING
# ============================================================
def load_clean_dataframe():
    df = pd.read_csv(LABEL_CSV)

    # Drop rows with missing
    df = df.dropna(subset=["image_name", "label"])

    # Keep only Cat / Dog
    df = df[df["label"].str.lower().isin(["cat", "dog"])].reset_index(drop=True)

    # SAMPLE subset
    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df = df.sample(MAX_SAMPLES, random_state=SEED).reset_index(drop=True)

    # Remove missing image files
    df["full_path"] = df["image_name"].apply(lambda x: os.path.join(DATA_DIR, x))
    df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

    # Remove corrupted images
    clean_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
        try:
            Image.open(row["full_path"])
            clean_rows.append(row)
        except Exception:
            print(f"⚠️ Skipped corrupted: {row['image_name']}")

    df = pd.DataFrame(clean_rows)
    print(f"Clean dataset size: {len(df)}")

    return df[["image_name", "label"]]


# ============================================================
# TRANSFORMS
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# INFERENCE SPEED BENCHMARK
# ============================================================
def benchmark_inference(model, device, img_size, n_runs=50):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start
    ms_per_image = (elapsed / n_runs) * 1000.0
    return ms_per_image


# ============================================================
# TRAIN FUNCTION (with val, early stopping, scheduler, artifacts)
# ============================================================
def train_model(model, model_name: str, version: str, df, device):
    print(f"\n=== Training {model_name} (version {version}) on device {device} ===")

    # Artifact directories
    artifact_dir = os.path.join(SAVE_DIR, model_name, version)
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)  # flat dir for existing UI

    # Train / Val split
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    split = int(len(df) * (1 - VAL_SPLIT))
    train_idx, val_idx = indices[:split], indices[split:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    ds_train = CatsDogsDataset(train_df, DATA_DIR, train_transform)
    ds_val = CatsDogsDataset(val_df, DATA_DIR, val_transform)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    # Move model to device
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = None
    if SCHED_CFG.get("type", "").lower() == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=SCHED_CFG.get("factor", 0.5),
            patience=SCHED_CFG.get("patience", 1),
        )

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    final_val_labels = []
    final_val_preds = []
    final_val_probs = []
    misclassified = []

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    best_ckpt_path = os.path.join(artifact_dir, "best_model.pt")

    for epoch in range(EPOCHS):
        # ---------------- TRAIN ----------------
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for imgs, labels, _names in tqdm(
            dl_train, desc=f"{model_name} - Epoch {epoch+1}/{EPOCHS} [train]"
        ):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(1)
            running_loss += loss.item() * imgs.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # ---------------- VAL ----------------
        model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        val_running_total = 0

        all_val_labels = []
        all_val_preds = []
        all_val_probs = []
        all_val_names = []

        with torch.no_grad():
            for imgs, labels, names in tqdm(
                dl_val, desc=f"{model_name} - Epoch {epoch+1}/{EPOCHS} [val]"
            ):
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                probs = torch.softmax(outputs, dim=1)[:, 1]  # prob of Dog
                preds = outputs.argmax(1)

                val_running_loss += loss.item() * imgs.size(0)
                val_running_correct += (preds == labels).sum().item()
                val_running_total += labels.size(0)

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())
                all_val_names.extend(list(names))

        val_loss = val_running_loss / val_running_total
        val_acc = val_running_correct / val_running_total

        final_val_labels = all_val_labels
        final_val_preds = all_val_preds
        final_val_probs = all_val_probs

        # Misclassified examples (last epoch)
        misclassified = []
        for name, y_true, y_pred, p in zip(
            all_val_names, all_val_labels, all_val_preds, all_val_probs
        ):
            if y_true != y_pred:
                misclassified.append(
                    {
                        "image_name": name,
                        "true_label": int(y_true),
                        "pred_label": int(y_pred),
                        "prob_dog": float(p),
                    }
                )

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(
            f"{model_name} | Epoch {epoch+1}/{EPOCHS} "
            f"- Train loss: {train_loss:.4f}, acc: {train_acc:.4f} "
            f"- Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        # LR scheduler
        if scheduler is not None:
            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != prev_lr:
                print(f"  → LR reduced: {prev_lr:.6f} → {new_lr:.6f}")

        # Early stopping + checkpoints
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
            # flat path for API / UI
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "custom_cnn.pt"))
            print(f"  → New best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= ES_PATIENCE:
                print(
                    f"Early stopping triggered at epoch {epoch+1} "
                    f"(best epoch = {best_epoch})"
                )
                break

    # ============================================================
    # LOAD BEST MODEL FOR METRICS & BENCHMARK
    # ============================================================
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        model = model.to(device)

    # ============================================================
    # PLOTS (accuracy, loss, confusion, ROC, PR)
    # ============================================================

    # Accuracy plot
    plt.figure(figsize=(5, 4))
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(val_acc_list, label="Val Acc")
    plt.legend()
    plt.title(f"{model_name} - Accuracy per Epoch")
    # flat filenames used by Streamlit for Custom CNN
    acc_flat = os.path.join(SAVE_DIR, "accuracy_curve.png")
    acc_art = os.path.join(artifact_dir, "accuracy.png")
    plt.savefig(acc_flat)
    plt.savefig(acc_art)
    plt.close()

    # Loss plot
    plt.figure(figsize=(5, 4))
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.legend()
    plt.title(f"{model_name} - Loss per Epoch")
    loss_flat = os.path.join(SAVE_DIR, "loss_curve.png")
    loss_art = os.path.join(artifact_dir, "loss.png")
    plt.savefig(loss_flat)
    plt.savefig(loss_art)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(final_val_labels, final_val_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"])
    disp.plot()
    plt.title(f"{model_name} - Confusion Matrix (Val)")
    cm_flat = os.path.join(SAVE_DIR, "confusion_matrix.png")
    cm_art = os.path.join(artifact_dir, "confusion.png")
    plt.savefig(cm_flat)
    plt.savefig(cm_art)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(final_val_labels, final_val_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend()
    roc_flat = os.path.join(SAVE_DIR, "custom_cnn_roc.png")
    roc_art = os.path.join(artifact_dir, "roc.png")
    plt.savefig(roc_flat)
    plt.savefig(roc_art)
    plt.close()

    # Precision–Recall curve
    precision, recall, _ = precision_recall_curve(final_val_labels, final_val_probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"PR (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} - Precision-Recall Curve")
    plt.legend()
    pr_flat = os.path.join(SAVE_DIR, "custom_cnn_pr.png")
    pr_art = os.path.join(artifact_dir, "pr.png")
    plt.savefig(pr_flat)
    plt.savefig(pr_art)
    plt.close()

    # Per-class accuracy
    cm = cm.astype(np.float32)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class_acc_dict = {
        "Cat": float(per_class_acc[0]),
        "Dog": float(per_class_acc[1]),
    }

    # Inference speed on best model
    ms_per_image = benchmark_inference(model, device, IMG_SIZE)

    # Save metrics, metadata, misclassified
    metrics = {
        "model_name": model_name,
        "version": version,
        "epochs_trained": len(train_loss_list),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "train_loss": train_loss_list,
        "val_loss": val_loss_list,
        "train_acc": train_acc_list,
        "val_acc": val_acc_list,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "per_class_accuracy": per_class_acc_dict,
        "inference_ms_per_image": float(ms_per_image),
    }

    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(artifact_dir, "metadata.json"), "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "version": version,
                "artifact_dir": artifact_dir,
                "weights_path": os.path.join(SAVE_DIR, "custom_cnn.pt"),
                "img_size": IMG_SIZE,
                "batch_size": BATCH_SIZE,
                "max_samples": MAX_SAMPLES,
                "val_split": VAL_SPLIT,
                "seed": SEED,
                "device": str(device),
            },
            f,
            indent=2,
        )

    with open(os.path.join(artifact_dir, "misclassified.json"), "w") as f:
        json.dump(misclassified, f, indent=2)

    print(f"Saved metrics and artifacts to: {artifact_dir}")


# ============================================================
# MODEL BUILDER
# ============================================================
def build_custom_cnn():
    # MobileNetV2 pretrained on ImageNet, then fine-tuned
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, 2)  # Cat / Dog
    return m


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = load_clean_dataframe()

    model_name = CUSTOM_CFG.get("name", "custom_cnn")
    version = CUSTOM_CFG.get("version", "v1")

    model = build_custom_cnn()
    train_model(model, model_name, version, df, device)

    print("Custom CNN training DONE.")
