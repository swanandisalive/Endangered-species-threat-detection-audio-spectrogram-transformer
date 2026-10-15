import torch
import transformers
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm.auto import (
    tqdm,  # Automatically selects the best progress bar for Jupyter/Terminal
)
from transformers import ASTForAudioClassification

# Force HF to print the true missing package if any issues occur
transformers.logging.set_verbosity_debug()


class ForestThreatAST(nn.Module):
    """
    AI Systems Design Layer 4 & 5:
    Audio Spectrogram Transformer Backbone with Multi-Label Linear Head
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.ast = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            ignore_mismatched_sizes=True,  # Swaps the default 527-class AudioSet head
        )
        hidden_size = self.ast.classifier.dense.in_features
        self.ast.classifier.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, input_values):
        return self.ast(input_values=input_values).logits


def train_ast_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: str,
    num_epochs: int = 5,
    lr: float = 1e-5,
    weight_decay: float = 1e-4,
    save_path: str = "ecoast_ast_model.pth",
):
    """
    Executes the training loop for the AST model using Mixed Precision.
    """
    print(f"Executing AST Fine-Tuning on device: {device.upper()}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(device) if device == "cuda" else None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # tqdm progress bar tracks batches per epoch
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=True)

        for batch in loop:
            input_values = batch["input_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed Precision Context for VRAM Efficiency
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

            running_loss += loss.item()
            loop.set_postfix(avg_loss=f"{running_loss / (loop.n + 1):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"\nTraining Complete! Checkpoint saved as '{save_path}'.")

    return model
