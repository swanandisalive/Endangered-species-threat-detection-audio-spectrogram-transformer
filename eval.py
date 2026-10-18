"""
evaluation.py
Evaluates the trained AST-FastStride-16x16 model on the test split.
Outputs Classification Report, Mean Average Precision (mAP), and saves a Confusion Matrix plot.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from pre_processing_pipeline import (
    IDX_TO_CLASS,
    NUM_CLASSES,
    build_ast_faststride_16x16_model,
    build_dataset_splits,
    get_dataloaders,
    set_seed,
)
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize


def evaluate_model():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 2. Load Test Data
    print("[INFO] Loading test dataset...")
    _, _, test_df = build_dataset_splits(
        fsd50k_root="./data", max_samples_per_class=700
    )
    print(f"[INFO] Test set size: {len(test_df)}")

    _, _, test_loader, _ = get_dataloaders(
        # Pass test_df to all just to satisfy the function signature,
        # but we only care about test_loader here.
        train_df=test_df,
        val_df=test_df,
        test_df=test_df,
        batch_size=16,
        num_workers=2,
    )

    # 3. Load Trained Model
    print("[INFO] Loading trained model weights...")
    model = build_ast_faststride_16x16_model().to(device)

    try:
        model.load_state_dict(
            torch.load(
                "best_ast_faststride_model.pth", map_location=device, weights_only=True
            )
        )
        print("[INFO] Successfully loaded best_ast_faststride_model.pth")
    except FileNotFoundError:
        print("[ERROR] Model weights not found! Run train.py first.")
        return

    model.eval()

    # 4. Inference Loop
    all_targets = []
    all_preds = []
    all_probs = []

    print("[INFO] Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_values=inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 5. Calculate Metrics
    class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(
        classification_report(
            all_targets, all_preds, target_names=class_names, zero_division=0
        )
    )

    # Calculate Mean Average Precision (mAP)
    # Binarize targets for multi-class AP calculation
    targets_binarized = label_binarize(all_targets, classes=range(NUM_CLASSES))

    # Calculate AP for each class, then macro-average for mAP
    ap_scores = []
    for i in range(NUM_CLASSES):
        # Handle cases where a class might not be present in the test set
        if np.sum(targets_binarized[:, i]) > 0:
            ap = average_precision_score(targets_binarized[:, i], all_probs[:, i])
            ap_scores.append(ap)
            print(f"AP for {class_names[i]}: {ap:.4f}")

    mAP = np.mean(ap_scores)
    print("-" * 50)
    print(f"Mean Average Precision (mAP): {mAP:.4f}")
    print("=" * 50)

    # 6. Plot Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title("Confusion Matrix: AST-FastStride-16x16", pad=20, fontsize=14)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the plot
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("\n[INFO] Confusion matrix plot saved as 'confusion_matrix.png'")

    # Show the plot if running in a notebook/GUI
    plt.show()


if __name__ == "__main__":
    evaluate_model()
