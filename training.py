"""
train.py
Main training loop for the 7-Class AST-FastStride-16x16 Acoustic Threat Detection Model.
"""

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from preprocessing_pipeline import (
    build_ast_faststride_16x16_model,
    build_dataset_splits,
    get_dataloaders,
    set_seed,
)


def train_model():
    # 1. Setup & Reproducibility
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 2. Parse Data Splits
    print("[INFO] Parsing FSD50K dataset...")
    # Make sure this path matches the 'data' folder from your directory structure
    train_df, val_df, test_df = build_dataset_splits(
        fsd50k_root="./data", max_samples_per_class=700
    )
    print(
        f"[INFO] Data splits created -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}"
    )

    # 3. Build DataLoaders
    print("[INFO] Initializing DataLoaders for 16x16 model...")
    # Set num_workers=0 if you encounter multiprocessing errors on Windows in Jupyter
    train_loader, val_loader, test_loader, extractor = get_dataloaders(
        train_df, val_df, test_df, batch_size=16, num_workers=2
    )

    # 4. Build Model
    print("[INFO] Loading AST-FastStride-16x16 architecture...")
    model = build_ast_faststride_16x16_model().to(device)

    # 5. Optimizer, Loss, and Scheduler
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    # label_smoothing helps prevent the model from becoming overconfident on ambiguous sounds
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    EPOCHS = 10
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 6. Training Loop
    print("[INFO] Starting training...")
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for batch in train_bar:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_values=inputs)
            logits = outputs.logits

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_values=inputs)
                logits = outputs.logits
                loss = criterion(logits, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        print(
            f"Epoch {epoch + 1} Results | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%"
        )

        # Save the best model based on validation accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "best_ast_faststride_model.pth")
            print(
                "[*] New best validation accuracy. Model saved to best_ast_faststride_model.pth"
            )

    print("[INFO] Training complete!")


if __name__ == "__main__":
    train_model()
