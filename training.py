import os

import numpy as np
import torch
import transformers
from sklearn.metrics import average_precision_score, precision_score
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import ASTConfig, ASTForAudioClassification

transformers.logging.set_verbosity_error()  # Keep console clean during multi-model runs

# Registry of the 3 AST Variants
AST_VARIANTS = {
    "AST-AudioSet-10x10": {
        "checkpoint": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "frequency_stride": 10,
        "time_stride": 10,
    },
    "AST-SpeechCommands-V2": {
        "checkpoint": "MIT/ast-finetuned-speech-commands-v2",
        "frequency_stride": 10,
        "time_stride": 10,
    },
    "AST-FastStride-16x16": {
        "checkpoint": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "frequency_stride": 16,
        "time_stride": 16,
    },
}


class ForestThreatAST(nn.Module):
    """
    AI Systems Design Layer 4 & 5:
    Configurable Audio Spectrogram Transformer Backbone for 3 Variants:
      1. AST-AudioSet-10x10
      2. AST-SpeechCommands-V2
      3. AST-FastStride-16x16
    """

    def __init__(self, num_classes: int, variant_name: str = "AST-AudioSet-10x10"):
        super().__init__()
        if variant_name not in AST_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant_name}'. Choose from: {list(AST_VARIANTS.keys())}"
            )

        self.variant_name = variant_name
        spec = AST_VARIANTS[variant_name]

        # Configure architecture (enforces 1024 frames & custom patch strides)
        config = ASTConfig.from_pretrained(
            spec["checkpoint"],
            max_length=1024,
            num_mel_bins=128,
            frequency_stride=spec["frequency_stride"],
            time_stride=spec["time_stride"],
        )

        # ignore_mismatched_sizes=True interpolates positional embeddings for SpeechCommands & 16x16 stride
        self.ast = ASTForAudioClassification.from_pretrained(
            spec["checkpoint"],
            config=config,
            ignore_mismatched_sizes=True,
        )
        hidden_size = self.ast.classifier.dense.in_features
        self.ast.classifier.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, input_values):
        return self.ast(input_values=input_values).logits


def compute_multilabel_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5
):
    """
    Calculates Multi-Label Element-wise Accuracy, Exact Subset Accuracy,
    Macro Precision, and Mean Average Precision (mAP).
    """
    y_pred = (y_probs >= threshold).astype(int)
    y_true_int = y_true.astype(int)

    # Element-wise binary accuracy across all 4 threat classes
    element_acc = float((y_pred == y_true_int).mean())
    # Exact match subset accuracy (all 4 labels in a sample predicted accurately)
    subset_acc = float(np.all(y_pred == y_true_int, axis=1).mean())
    # Macro Precision across the 4 classes
    precision = float(
        precision_score(y_true_int, y_pred, average="macro", zero_division=0)
    )

    # Mean Average Precision (mAP) per class (safe against batches missing a rare class)
    ap_per_class = []
    for c in range(y_true_int.shape[1]):
        if y_true_int[:, c].sum() > 0:
            ap_per_class.append(
                average_precision_score(y_true_int[:, c], y_probs[:, c])
            )
    map_score = float(np.mean(ap_per_class)) if ap_per_class else 0.0

    return {
        "accuracy": element_acc,
        "subset_accuracy": subset_acc,
        "precision": precision,
        "mAP": map_score,
    }


def train_ast_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: str,
    model_name: str = "AST-Model",
    num_epochs: int = 5,
    lr: float = 1e-5,
    weight_decay: float = 1e-4,
    save_path: str = "trained_models/ecoast_ast_model.pth",
    log_dir: str = "runs",
):
    """
    Executes the Mixed Precision training loop and logs Loss, Accuracy,
    Precision, and mAP curves directly to TensorBoard.
    """
    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True
    )
    writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))

    print(f"\n{'=' * 70}")
    print(
        f"Training [{model_name}] on {device.upper()} | TensorBoard: {log_dir}/{model_name}"
    )
    print(f"{'=' * 70}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(device) if device == "cuda" else None

    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_probs = []

        loop = tqdm(
            train_loader,
            desc=f"[{model_name}] Epoch [{epoch + 1}/{num_epochs}]",
            leave=True,
        )

        for batch in loop:
            input_values = batch["input_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True).float()

            optimizer.zero_grad()

            if device == "cuda":
                with autocast(device_type="cuda"):
                    logits = model(input_values)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_values)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss
            global_step += 1

            # Store predictions for epoch-level accuracy, precision, and mAP
            probs = torch.sigmoid(logits.detach()).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.detach().cpu().numpy())

            # Log step loss to TensorBoard
            writer.add_scalar("Batch/Train_Loss", batch_loss, global_step)
            loop.set_postfix(avg_loss=f"{running_loss / (loop.n + 1):.4f}")

        # Calculate Epoch Metrics
        epoch_loss = running_loss / len(train_loader)
        y_true = np.vstack(all_labels)
        y_probs = np.vstack(all_probs)
        metrics = compute_multilabel_metrics(y_true, y_probs)

        # Log Epoch Curves to TensorBoard (All 3 models share these exact tag names for overlay)
        step_idx = epoch + 1
        writer.add_scalar("Epoch/Loss", epoch_loss, step_idx)
        writer.add_scalar("Epoch/Accuracy", metrics["accuracy"] * 100.0, step_idx)
        writer.add_scalar(
            "Epoch/Subset_Accuracy", metrics["subset_accuracy"] * 100.0, step_idx
        )
        writer.add_scalar("Epoch/Precision_Macro", metrics["precision"], step_idx)
        writer.add_scalar("Epoch/mAP", metrics["mAP"], step_idx)

        print(
            f" -> Epoch {step_idx} Summary | Loss: {epoch_loss:.4f} | "
            f"Acc: {metrics['accuracy'] * 100:.2f}% | Precision: {metrics['precision']:.4f} | mAP: {metrics['mAP']:.4f}"
        )

    writer.flush()
    writer.close()

    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint saved: '{save_path}'")

    return model


@torch.no_grad()
def log_pretrained_model_to_tensorboard(
    model: nn.Module,
    checkpoint_path: str,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    model_name: str = "AST-AudioSet-10x10",
    num_epochs: int = 5,
    log_dir: str = "runs",
    recorded_losses: list = None,
):
    """
    Loads your already-trained Model 1 checkpoint ('ast_best_model.pth'), evaluates its
    Loss, Accuracy, Precision, and mAP on the dataset, and writes its baseline curve into
    TensorBoard so you don't have to retrain Model 1 from scratch.
    """
    print(
        f"\nLoading already-trained checkpoint '{checkpoint_path}' for [{model_name}]..."
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    all_labels, all_probs = [], []

    loop = tqdm(
        data_loader, desc=f"Evaluating [{model_name}] for TensorBoard", leave=True
    )
    for batch in loop:
        input_values = batch["input_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True).float()

        if device == "cuda":
            with autocast(device_type="cuda"):
                logits = model(input_values)
                loss = criterion(logits, labels)
        else:
            logits = model(input_values)
            loss = criterion(logits, labels)

        running_loss += loss.item()
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    eval_loss = running_loss / len(data_loader)
    metrics = compute_multilabel_metrics(np.vstack(all_labels), np.vstack(all_probs))

    writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))
    for epoch in range(1, num_epochs + 1):
        # If you pass the 5 epoch losses from your first run, it plots the exact decay curve;
        # otherwise it plots the converged evaluation loss across the 5 epochs as a reference line.
        loss_val = (
            recorded_losses[epoch - 1]
            if (recorded_losses and len(recorded_losses) >= epoch)
            else eval_loss
        )
        writer.add_scalar("Epoch/Loss", loss_val, epoch)
        writer.add_scalar("Epoch/Accuracy", metrics["accuracy"] * 100.0, epoch)
        writer.add_scalar(
            "Epoch/Subset_Accuracy", metrics["subset_accuracy"] * 100.0, epoch
        )
        writer.add_scalar("Epoch/Precision_Macro", metrics["precision"], epoch)
        writer.add_scalar("Epoch/mAP", metrics["mAP"], epoch)

    writer.flush()
    writer.close()

    print(
        f"Logged [{model_name}] to TensorBoard | Final Loss: {eval_loss:.4f} | "
        f"Acc: {metrics['accuracy'] * 100:.2f}% | Precision: {metrics['precision']:.4f} | mAP: {metrics['mAP']:.4f}"
    )
    return metrics
