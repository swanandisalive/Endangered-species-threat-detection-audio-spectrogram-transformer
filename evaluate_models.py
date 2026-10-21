"""
evaluate_models.py
===========================================================================
Comprehensive Evaluation Pipeline for EcoAST Project
Evaluates all three models:
  1. AST-FastStride-16x16 (Audio Spectrogram Transformer)
  2. MobileNetV3-Small (timm-based spectrogram image classifier)
  3. EfficientNet-B2 (timm-based spectrogram image classifier)

Metrics Computed:
  - Classification Report (precision, recall, F1 per class)
  - Mean Average Precision (mAP)
  - Validation / Test Loss (CrossEntropyLoss)
  - Confusion Matrix (saved as high-res PNG)
  - Macro & Weighted F1-Score
  - F2-Score (for threat-sensitive evaluation)
  - Per-class Average Precision (AP)
  - PR-AUC curves

Usage:
  python evaluate_models.py --model ast --dataset-dir ./data
  python evaluate_models.py --model mobilenet --dataset-dir ./data
  python evaluate_models.py --model effnet --dataset-dir ./data
  python evaluate_models.py --model all --dataset-dir ./data
  python evaluate_models.py --model ast --dataset-dir ./shared_forest_dataset --use-shared
===========================================================================
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project-specific imports
# ---------------------------------------------------------------------------
from preprocessing_pipeline import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    SAMPLE_RATE,
    build_ast_faststride_16x16_model,
    build_dataset_splits,
    get_dataloaders,
    load_dataset_splits,
    set_seed,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("evaluation_results")
CLASS_NAMES = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]

# Threat classes (higher recall priority)
THREAT_CLASSES = {"Sawing_and_Chainsaw", "Gunshot_and_Explosion", "Engine_and_Vehicle"}

# Color palette for plots
PALETTE = {
    "primary": "#6366f1",      # Indigo
    "secondary": "#8b5cf6",    # Violet
    "accent": "#ec4899",       # Pink
    "success": "#10b981",      # Emerald
    "warning": "#f59e0b",      # Amber
    "danger": "#ef4444",       # Red
    "bg_dark": "#0f172a",      # Slate 900
    "bg_card": "#1e293b",      # Slate 800
    "text": "#e2e8f0",         # Slate 200
    "text_dim": "#94a3b8",     # Slate 400
    "grid": "#334155",         # Slate 700
}

CLASS_COLORS = [
    "#10b981",  # Background_Nature - Emerald
    "#ef4444",  # Sawing_and_Chainsaw - Red
    "#f59e0b",  # Gunshot_and_Explosion - Amber
    "#3b82f6",  # Engine_and_Vehicle - Blue
    "#8b5cf6",  # Wood_Chop_and_Break - Violet
    "#ec4899",  # Human_Vocalization - Pink
    "#14b8a6",  # Dog_Bark - Teal
]


# ===========================================================================
# MODEL BUILDERS
# ===========================================================================
def build_mobilenetv3_model():
    """Builds MobileNetV3-Small with 7-class head using timm."""
    import timm

    model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=NUM_CLASSES)
    return model


def build_efficientnetb2_model():
    """Builds EfficientNet-B2 with 7-class head using timm."""
    import timm

    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=NUM_CLASSES)
    return model


def load_model(model_name: str, device: torch.device):
    """Loads and returns the requested model with weights."""
    weights_dir = Path("trained_model")

    if model_name == "ast":
        print(f"  ├─ Building AST-FastStride-16x16 architecture...")
        model = build_ast_faststride_16x16_model()
        weights_path = weights_dir / "best_ast_faststride_model.pth"
    elif model_name == "mobilenet":
        print(f"  ├─ Building MobileNetV3-Small architecture...")
        model = build_mobilenetv3_model()
        weights_path = weights_dir / "best_mobilenetv3_model.pth"
    elif model_name == "effnet":
        print(f"  ├─ Building EfficientNet-B2 architecture...")
        model = build_efficientnetb2_model()
        weights_path = weights_dir / "best_effnetb2_model.pth"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at: {weights_path}")

    print(f"  ├─ Loading weights from: {weights_path}")
    state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  └─ Model loaded: {total_params:,} parameters")

    return model


# ===========================================================================
# MOBILENETV3 SPECTROGRAM DATASET
# ===========================================================================
class MobileNetSpectrogramDataset(torch.utils.data.Dataset):
    """
    Dataset for MobileNetV3 that converts audio files to 3-channel mel spectrogram images.
    MobileNetV3 expects image-like inputs (3, H, W), so we compute the mel spectrogram
    and replicate it across 3 channels.
    """

    def __init__(self, df, target_sr=SAMPLE_RATE, n_mels=128, target_duration=10.24,
                 img_size=224):
        import torchaudio
        import torchaudio.transforms as T

        self.df = df
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.target_samples = int(target_duration * target_sr)
        self.img_size = img_size
        self._resamplers = {}

        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import torchaudio
        import torchaudio.transforms as T

        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label = row["label"]

        # Load audio
        waveform, sr = torchaudio.load(filepath)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.target_sr:
            if sr not in self._resamplers:
                self._resamplers[sr] = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = self._resamplers[sr](waveform)

        waveform = waveform.squeeze(0)

        # Pad or crop to target length
        if waveform.shape[0] > self.target_samples:
            start = (waveform.shape[0] - self.target_samples) // 2
            waveform = waveform[start: start + self.target_samples]
        elif waveform.shape[0] < self.target_samples:
            pad = self.target_samples - waveform.shape[0]
            waveform = F.pad(waveform, (0, pad), mode="constant", value=0.0)

        # Compute mel spectrogram
        mel = self.mel_transform(waveform.unsqueeze(0))
        mel_db = self.amplitude_to_db(mel)

        # Normalize to [0, 1]
        mel_min = mel_db.min()
        mel_max = mel_db.max()
        if mel_max - mel_min > 0:
            mel_db = (mel_db - mel_min) / (mel_max - mel_min)
        else:
            mel_db = torch.zeros_like(mel_db)

        # Resize to img_size x img_size
        mel_db = F.interpolate(
            mel_db.unsqueeze(0), size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False
        ).squeeze(0)

        # Replicate to 3 channels (RGB)
        img = mel_db.repeat(3, 1, 1)

        # Normalize with ImageNet stats (standard for timm models)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std

        return {
            "input_values": img.float(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ===========================================================================
# INFERENCE ENGINE
# ===========================================================================
def run_inference(model, dataloader, model_name: str, device: torch.device):
    """
    Runs inference on the given dataloader and collects predictions.
    Returns targets, predictions, probabilities, and average loss.
    """
    criterion = nn.CrossEntropyLoss()
    all_targets = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    total_samples = 0

    print(f"\n{'─' * 60}")
    print(f"  Running inference for: {model_name.upper()}")
    print(f"{'─' * 60}")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  Evaluating {model_name}", unit="batch",
                          bar_format="{l_bar}{bar:30}{r_bar}"):
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            if model_name == "ast":
                outputs = model(input_values=inputs)
                logits = outputs.logits
            else:
                # MobileNetV3 / EfficientNet-B2 direct forward
                logits = model(inputs)

            loss = criterion(logits, labels)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total_samples
    return (
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
        avg_loss,
    )


# ===========================================================================
# METRICS COMPUTATION
# ===========================================================================
def compute_all_metrics(targets, preds, probs, loss, model_name: str):
    """Computes all evaluation metrics and returns a structured dict."""
    targets_bin = label_binarize(targets, classes=range(NUM_CLASSES))

    # Per-class AP
    ap_scores = {}
    for i in range(NUM_CLASSES):
        if np.sum(targets_bin[:, i]) > 0:
            ap = average_precision_score(targets_bin[:, i], probs[:, i])
            ap_scores[CLASS_NAMES[i]] = float(ap)

    mAP = float(np.mean(list(ap_scores.values()))) if ap_scores else 0.0

    # ROC-AUC (macro, one-vs-rest)
    try:
        roc_auc_macro = float(roc_auc_score(targets_bin, probs, average="macro", multi_class="ovr"))
    except ValueError:
        roc_auc_macro = None

    # F-scores
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(targets, preds, average="weighted", zero_division=0))
    macro_f2 = float(fbeta_score(targets, preds, beta=2, average="macro", zero_division=0))

    # Per-class F2 for threat classes
    per_class_f2 = {}
    for i in range(NUM_CLASSES):
        class_mask_true = (targets == i).astype(int)
        class_mask_pred = (preds == i).astype(int)
        f2 = fbeta_score(class_mask_true, class_mask_pred, beta=2, zero_division=0)
        per_class_f2[CLASS_NAMES[i]] = float(f2)

    # Accuracy
    accuracy = float(np.mean(targets == preds))

    # Classification report as dict
    cls_report = classification_report(
        targets, preds, target_names=CLASS_NAMES, zero_division=0, output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(targets, preds)

    metrics = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "test_loss": float(loss),
        "accuracy": accuracy,
        "mAP": mAP,
        "per_class_AP": ap_scores,
        "roc_auc_macro": roc_auc_macro,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_f2": macro_f2,
        "per_class_f2": per_class_f2,
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist(),
        "total_samples": int(len(targets)),
    }

    return metrics


# ===========================================================================
# PRETTY-PRINT METRICS
# ===========================================================================
def print_metrics(metrics: dict):
    """Pretty-prints evaluation metrics to console."""
    m = metrics
    model = m["model_name"].upper()

    print(f"\n{'═' * 70}")
    print(f"  EVALUATION RESULTS: {model}")
    print(f"{'═' * 70}")
    print(f"  Test Loss          : {m['test_loss']:.4f}")
    print(f"  Accuracy           : {m['accuracy'] * 100:.2f}%")
    print(f"  mAP (macro)        : {m['mAP']:.4f}")
    if m['roc_auc_macro'] is not None:
        print(f"  ROC-AUC (macro)    : {m['roc_auc_macro']:.4f}")
    print(f"  Macro F1           : {m['macro_f1']:.4f}")
    print(f"  Weighted F1        : {m['weighted_f1']:.4f}")
    print(f"  Macro F2 (threats) : {m['macro_f2']:.4f}")
    print(f"{'─' * 70}")

    print(f"\n  Per-Class Average Precision (AP):")
    for cls_name, ap in m['per_class_AP'].items():
        tag = " ⚠ THREAT" if cls_name in THREAT_CLASSES else ""
        print(f"    {cls_name:<25s} : {ap:.4f}{tag}")
    print(f"    {'─' * 40}")
    print(f"    {'mAP':<25s} : {m['mAP']:.4f}")

    print(f"\n  Per-Class F2-Score:")
    for cls_name, f2 in m['per_class_f2'].items():
        tag = " ⚠ THREAT" if cls_name in THREAT_CLASSES else ""
        print(f"    {cls_name:<25s} : {f2:.4f}{tag}")

    print(f"\n  Classification Report:")
    print(f"  {'─' * 66}")
    cls_rep = classification_report(
        list(range(NUM_CLASSES)), list(range(NUM_CLASSES)),  # dummy, we'll print from dict
        target_names=CLASS_NAMES, zero_division=0
    )
    # Print from stored dict for actual values
    report = m["classification_report"]
    print(f"  {'Class':<25s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    print(f"  {'─' * 66}")
    for cls_name in CLASS_NAMES:
        if cls_name in report:
            r = report[cls_name]
            print(f"  {cls_name:<25s} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {r['support']:>10.0f}")
    print(f"  {'─' * 66}")
    for avg_type in ["macro avg", "weighted avg"]:
        if avg_type in report:
            r = report[avg_type]
            print(f"  {avg_type:<25s} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {r['support']:>10.0f}")

    print(f"\n{'═' * 70}\n")


# ===========================================================================
# VISUALIZATION
# ===========================================================================
def setup_plot_style():
    """Configure matplotlib for dark, premium plots."""
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg_dark"],
        "axes.facecolor": PALETTE["bg_card"],
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["text"],
        "text.color": PALETTE["text"],
        "xtick.color": PALETTE["text_dim"],
        "ytick.color": PALETTE["text_dim"],
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_confusion_matrix(cm, model_name: str, save_path: Path):
    """Plots a styled confusion matrix."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 10))

    # Normalize CM for display
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    # Custom colormap
    cmap = sns.color_palette("blend:#1e293b,#6366f1,#ec4899", as_cmap=True)

    sns.heatmap(
        cm_normalized,
        annot=np.array([[f"{v}\n({p:.0%})" for v, p in zip(row_v, row_p)]
                        for row_v, row_p in zip(cm, cm_normalized)]),
        fmt="",
        cmap=cmap,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        linecolor=PALETTE["grid"],
        cbar_kws={"label": "Proportion", "shrink": 0.8},
        ax=ax,
        annot_kws={"size": 9},
    )

    ax.set_title(f"Confusion Matrix — {model_name.upper()}", pad=20, fontsize=16,
                 fontweight="bold", color=PALETTE["text"])
    ax.set_ylabel("True Label", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=PALETTE["bg_dark"])
    plt.close()
    print(f"  ✓ Confusion matrix saved: {save_path}")


def plot_per_class_metrics(metrics: dict, save_path: Path):
    """Plots per-class precision, recall, F1, and F2 as a grouped bar chart."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(14, 7))
    model_name = metrics["model_name"]

    report = metrics["classification_report"]
    x = np.arange(NUM_CLASSES)
    width = 0.2

    precisions = [report[cn]["precision"] for cn in CLASS_NAMES]
    recalls = [report[cn]["recall"] for cn in CLASS_NAMES]
    f1s = [report[cn]["f1-score"] for cn in CLASS_NAMES]
    f2s = [metrics["per_class_f2"][cn] for cn in CLASS_NAMES]

    bars1 = ax.bar(x - 1.5 * width, precisions, width, label="Precision",
                   color="#6366f1", alpha=0.9, edgecolor="none")
    bars2 = ax.bar(x - 0.5 * width, recalls, width, label="Recall",
                   color="#10b981", alpha=0.9, edgecolor="none")
    bars3 = ax.bar(x + 0.5 * width, f1s, width, label="F1-Score",
                   color="#f59e0b", alpha=0.9, edgecolor="none")
    bars4 = ax.bar(x + 1.5 * width, f2s, width, label="F2-Score",
                   color="#ec4899", alpha=0.9, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-Class Metrics — {model_name.upper()}", fontsize=16,
                 fontweight="bold", pad=15)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", framealpha=0.8, facecolor=PALETTE["bg_card"],
              edgecolor=PALETTE["grid"])
    ax.grid(axis="y", alpha=0.2)

    # Highlight threat classes
    for i, cn in enumerate(CLASS_NAMES):
        if cn in THREAT_CLASSES:
            ax.get_xticklabels()[i].set_color(PALETTE["danger"])
            ax.get_xticklabels()[i].set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=PALETTE["bg_dark"])
    plt.close()
    print(f"  ✓ Per-class metrics saved: {save_path}")


def plot_ap_bar(metrics: dict, save_path: Path):
    """Plots per-class AP as a horizontal bar chart with mAP line."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    model_name = metrics["model_name"]

    ap_scores = metrics["per_class_AP"]
    classes = list(ap_scores.keys())
    values = list(ap_scores.values())

    # Sort by AP
    sorted_pairs = sorted(zip(classes, values), key=lambda x: x[1])
    classes, values = zip(*sorted_pairs)

    colors = [PALETTE["danger"] if c in THREAT_CLASSES else PALETTE["primary"] for c in classes]
    bars = ax.barh(classes, values, color=colors, alpha=0.85, edgecolor="none", height=0.6)

    # mAP line
    mAP = metrics["mAP"]
    ax.axvline(x=mAP, color=PALETTE["accent"], linestyle="--", linewidth=2, alpha=0.8,
               label=f"mAP = {mAP:.4f}")

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                va="center", fontsize=10, color=PALETTE["text"])

    ax.set_xlim(0, 1.1)
    ax.set_title(f"Per-Class Average Precision — {model_name.upper()}", fontsize=16,
                 fontweight="bold", pad=15)
    ax.set_xlabel("Average Precision (AP)")
    ax.legend(loc="lower right", framealpha=0.8, facecolor=PALETTE["bg_card"],
              edgecolor=PALETTE["grid"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=PALETTE["bg_dark"])
    plt.close()
    print(f"  ✓ AP bar chart saved: {save_path}")


def plot_pr_curves(targets, probs, model_name: str, save_path: Path):
    """Plots Precision-Recall curves for each class."""
    setup_plot_style()

    targets_bin = label_binarize(targets, classes=range(NUM_CLASSES))

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(NUM_CLASSES):
        if np.sum(targets_bin[:, i]) > 0:
            precision, recall, _ = precision_recall_curve(targets_bin[:, i], probs[:, i])
            ap = average_precision_score(targets_bin[:, i], probs[:, i])
            linewidth = 2.5 if CLASS_NAMES[i] in THREAT_CLASSES else 1.5
            linestyle = "-" if CLASS_NAMES[i] in THREAT_CLASSES else "--"
            ax.plot(recall, precision, color=CLASS_COLORS[i], linewidth=linewidth,
                    linestyle=linestyle, label=f"{CLASS_NAMES[i]} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(f"Precision-Recall Curves — {model_name.upper()}", fontsize=16,
                 fontweight="bold", pad=15)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.8, facecolor=PALETTE["bg_card"],
              edgecolor=PALETTE["grid"])
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=PALETTE["bg_dark"])
    plt.close()
    print(f"  ✓ PR curves saved: {save_path}")


def plot_comparison_summary(all_metrics: list, save_path: Path):
    """Creates a side-by-side comparison dashboard for 2 or 3 models."""
    if len(all_metrics) < 2:
        return

    setup_plot_style()

    n_models = len(all_metrics)
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    model_names = [m["model_name"].upper() for m in all_metrics]
    colors = [PALETTE["primary"], PALETTE["accent"], PALETTE["success"]][:n_models]

    # Dynamic bar width based on number of models
    width = 0.8 / n_models

    # ── Chart 1 (top-left): Overview metrics ──
    ax = axes[0, 0]
    overview_metrics = ["accuracy", "mAP", "macro_f1", "weighted_f1", "macro_f2"]
    overview_labels = ["Accuracy", "mAP", "Macro F1", "Weighted F1", "Macro F2"]
    x = np.arange(len(overview_metrics))

    for idx, m in enumerate(all_metrics):
        vals = [m[k] for k in overview_metrics]
        offset = (idx - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model_names[idx],
                      color=colors[idx], alpha=0.85, edgecolor="none")
        # Value labels on top
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                    color=PALETTE["text_dim"])

    ax.set_xticks(x)
    ax.set_xticklabels(overview_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Overall Metrics", fontweight="bold", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.legend(framealpha=0.8, facecolor=PALETTE["bg_card"], edgecolor=PALETTE["grid"],
              fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    # ── Chart 2 (top-right): Per-class F1 ──
    ax = axes[0, 1]
    x = np.arange(NUM_CLASSES)
    for idx, m in enumerate(all_metrics):
        f1s = [m["classification_report"][cn]["f1-score"] for cn in CLASS_NAMES]
        offset = (idx - (n_models - 1) / 2) * width
        ax.bar(x + offset, f1s, width, label=model_names[idx],
               color=colors[idx], alpha=0.85, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Class F1 Comparison", fontweight="bold", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.legend(framealpha=0.8, facecolor=PALETTE["bg_card"], edgecolor=PALETTE["grid"],
              fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    # Highlight threat classes
    for i, cn in enumerate(CLASS_NAMES):
        if cn in THREAT_CLASSES:
            ax.get_xticklabels()[i].set_color(PALETTE["danger"])
            ax.get_xticklabels()[i].set_fontweight("bold")

    # ── Chart 3 (bottom-left): Per-class AP ──
    ax = axes[1, 0]
    for idx, m in enumerate(all_metrics):
        aps = [m["per_class_AP"].get(cn, 0) for cn in CLASS_NAMES]
        offset = (idx - (n_models - 1) / 2) * width
        ax.bar(x + offset, aps, width, label=model_names[idx],
               color=colors[idx], alpha=0.85, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Average Precision")
    ax.set_title("Per-Class AP Comparison", fontweight="bold", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.legend(framealpha=0.8, facecolor=PALETTE["bg_card"], edgecolor=PALETTE["grid"],
              fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    for i, cn in enumerate(CLASS_NAMES):
        if cn in THREAT_CLASSES:
            ax.get_xticklabels()[i].set_color(PALETTE["danger"])
            ax.get_xticklabels()[i].set_fontweight("bold")

    # ── Chart 4 (bottom-right): Radar / Spider chart ──
    ax = axes[1, 1]
    ax.set_axis_off()  # Remove default axes, we'll use polar
    radar_ax = fig.add_axes(
        ax.get_position(), polar=True, facecolor=PALETTE["bg_card"]
    )

    radar_metrics = ["accuracy", "mAP", "macro_f1", "weighted_f1", "macro_f2"]
    radar_labels = ["Accuracy", "mAP", "Macro F1", "Weighted F1", "Macro F2"]
    n_vars = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    for idx, m in enumerate(all_metrics):
        vals = [m[k] for k in radar_metrics]
        vals += vals[:1]  # Close the polygon
        radar_ax.plot(angles, vals, "o-", linewidth=2, color=colors[idx],
                      label=model_names[idx], alpha=0.85)
        radar_ax.fill(angles, vals, color=colors[idx], alpha=0.15)

    radar_ax.set_xticks(angles[:-1])
    radar_ax.set_xticklabels(radar_labels, fontsize=9, color=PALETTE["text"])
    radar_ax.set_ylim(0, 1.05)
    radar_ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    radar_ax.tick_params(axis="y", labelsize=7, colors=PALETTE["text_dim"])
    radar_ax.set_title("Radar Overview", fontweight="bold", fontsize=13,
                       color=PALETTE["text"], pad=20)
    radar_ax.grid(color=PALETTE["grid"], alpha=0.3)
    radar_ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.05),
                    framealpha=0.8, facecolor=PALETTE["bg_card"],
                    edgecolor=PALETTE["grid"], fontsize=9)
    radar_ax.set_facecolor(PALETTE["bg_card"])

    fig.suptitle("Model Comparison Dashboard", fontsize=20, fontweight="bold",
                 color=PALETTE["text"], y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=PALETTE["bg_dark"])
    plt.close()
    print(f"  ✓ Comparison dashboard saved: {save_path}")


# ===========================================================================
# MAIN EVALUATION PIPELINE
# ===========================================================================
def evaluate_single_model(
    model_name: str,
    dataset_dir: str,
    use_shared: bool = False,
    batch_size: int = 8,
    num_workers: int = 0,
):
    """Full evaluation pipeline for a single model."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'█' * 70}")
    print(f"  EVALUATING: {model_name.upper()}")
    print(f"  Device: {device} | Batch size: {batch_size}")
    print(f"{'█' * 70}")

    # 1. Load dataset
    print(f"\n[1/4] Loading dataset...")
    if use_shared:
        train_df, val_df, test_df = load_dataset_splits(dataset_dir)
    else:
        _, _, test_df = build_dataset_splits(dataset_dir, max_samples_per_class=700)
        # We also need val_df for proper evaluation
        _, val_df, _ = build_dataset_splits(dataset_dir, max_samples_per_class=700)

    print(f"  └─ Test set: {len(test_df)} samples")

    # 2. Create dataloader
    print(f"\n[2/4] Creating dataloader...")
    if model_name == "ast":
        # Use the existing pipeline's dataloader for AST
        _, _, test_loader, _ = get_dataloaders(
            train_df=test_df,  # dummy
            val_df=test_df,    # dummy
            test_df=test_df,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif model_name == "mobilenet":
        # Custom spectrogram dataset for MobileNetV3 (224x224)
        test_ds = MobileNetSpectrogramDataset(test_df, img_size=224)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )
    elif model_name == "effnet":
        # Custom spectrogram dataset for EfficientNet-B2 (260x260)
        test_ds = MobileNetSpectrogramDataset(test_df, img_size=260)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )

    # 3. Load model & run inference
    print(f"\n[3/4] Loading model...")
    model = load_model(model_name, device)

    start_time = time.time()
    targets, preds, probs, loss = run_inference(model, test_loader, model_name, device)
    elapsed = time.time() - start_time
    print(f"\n  ⏱  Inference completed in {elapsed:.1f}s ({len(targets) / elapsed:.1f} samples/sec)")

    # 4. Compute & save metrics
    print(f"\n[4/4] Computing metrics...")
    metrics = compute_all_metrics(targets, preds, probs, loss, model_name)
    print_metrics(metrics)

    # Create output directory
    model_dir = RESULTS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save plots
    print(f"\n  Generating visualizations...")
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, model_name, model_dir / "confusion_matrix.png")
    plot_per_class_metrics(metrics, model_dir / "per_class_metrics.png")
    plot_ap_bar(metrics, model_dir / "average_precision.png")
    plot_pr_curves(targets, probs, model_name, model_dir / "pr_curves.png")

    # Save metrics JSON
    json_path = model_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics JSON saved: {json_path}")

    return metrics, targets, probs


def main():
    parser = argparse.ArgumentParser(
        description="EcoAST Model Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate AST model with raw FSD50K data
  python evaluate_models.py --model ast --dataset-dir ./data

  # Evaluate MobileNetV3 with raw FSD50K data
  python evaluate_models.py --model mobilenet --dataset-dir ./data

  # Evaluate EfficientNet-B2 with raw FSD50K data
  python evaluate_models.py --model effnet --dataset-dir ./data

  # Evaluate all three models and generate comparison
  python evaluate_models.py --model all --dataset-dir ./data

  # Use pre-split shared dataset
  python evaluate_models.py --model all --dataset-dir ./shared_forest_dataset --use-shared
        """,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["ast", "mobilenet", "effnet", "all"],
        help="Which model(s) to evaluate: ast, mobilenet, effnet, or all",
    )
    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="Path to FSD50K root or shared dataset directory",
    )
    parser.add_argument(
        "--use-shared", action="store_true",
        help="Use pre-split shared dataset (with audio/ and dataset_metadata.csv)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for evaluation (default: 8, lower for CPU)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (default: 0 for Windows/CPU)",
    )

    args = parser.parse_args()

    print(f"\n{'═' * 70}")
    print(f"  🌿 EcoAST — Model Evaluation Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 70}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        models_to_eval = ["ast", "mobilenet", "effnet"]
    else:
        models_to_eval = [args.model]
    all_metrics = []

    for model_name in models_to_eval:
        try:
            metrics, targets, probs = evaluate_single_model(
                model_name=model_name,
                dataset_dir=args.dataset_dir,
                use_shared=args.use_shared,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\n  ✗ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Generate comparison if multiple models evaluated
    if len(all_metrics) >= 2:
        print(f"\n{'█' * 70}")
        print(f"  GENERATING MODEL COMPARISON ({len(all_metrics)} models)")
        print(f"{'█' * 70}")
        plot_comparison_summary(all_metrics, RESULTS_DIR / "model_comparison.png")

        # Print comparison table
        col_width = 14
        header_width = 25 + col_width * len(all_metrics)
        print(f"\n  {'Metric':<25s}", end="")
        for m in all_metrics:
            print(f"  {m['model_name'].upper():>{col_width}s}", end="")
        print()
        print(f"  {'─' * header_width}")
        for key, label in [
            ("accuracy", "Accuracy"),
            ("mAP", "mAP"),
            ("macro_f1", "Macro F1"),
            ("weighted_f1", "Weighted F1"),
            ("macro_f2", "Macro F2"),
            ("test_loss", "Test Loss"),
        ]:
            print(f"  {label:<25s}", end="")
            for m in all_metrics:
                val = m[key]
                print(f"  {val:>{col_width}.4f}", end="")
            print()

        # Identify best model per metric
        print(f"  {'─' * header_width}")
        print(f"  {'BEST →':<25s}", end="")
        for key in ["accuracy", "mAP", "macro_f1", "weighted_f1", "macro_f2"]:
            best_idx = int(np.argmax([m[key] for m in all_metrics]))
            best_name = all_metrics[best_idx]["model_name"].upper()
            # Just print once for the row
            break
        # Actually, print best model for each column
        print()  # newline
        for key, label in [
            ("accuracy", "Accuracy"),
            ("mAP", "mAP"),
            ("macro_f1", "Macro F1"),
            ("weighted_f1", "Weighted F1"),
            ("macro_f2", "Macro F2"),
        ]:
            best_idx = int(np.argmax([m[key] for m in all_metrics]))
            best_name = all_metrics[best_idx]["model_name"].upper()
            print(f"    Best {label}: {best_name} ({all_metrics[best_idx][key]:.4f})")

    print(f"\n{'═' * 70}")
    print(f"  ✓ All results saved to: {RESULTS_DIR.resolve()}")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
