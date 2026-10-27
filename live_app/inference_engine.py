"""
inference_engine.py
Handles AST-FastStride-16x16 initialization, audio preprocessing, and direct 5-class inference.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification

SAMPLE_RATE = 16000
TARGET_DURATION = 10.24
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION)  # 163,840 samples

# 5-class native architecture mapping
CLASS_TO_IDX = {
    "Sawing_and_Chainsaw": 0,
    "Gunshot_and_Explosion": 1,
    "Engine_and_Vehicle": 2,
    "Wood_Chop_and_Break": 3,
    "Human_Vocalization": 4,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_TO_IDX)

DISPLAY_LABELS = {
    "Sawing_and_Chainsaw": "Sawing & Chainsaw",
    "Gunshot_and_Explosion": "Gunshot & Explosion",
    "Engine_and_Vehicle": "Engine & Vehicle",
    "Wood_Chop_and_Break": "Wood Chop & Break",
    "Human_Vocalization": "Human Vocalization",
}

HIGH_PRIORITY_THREATS = {
    "Sawing_and_Chainsaw",
    "Gunshot_and_Explosion",
    "Engine_and_Vehicle",
    "Wood_Chop_and_Break",
}


def build_ast_faststride_16x16_model():
    """Builds the 16x16 non-overlapping patch AST architecture (514 tokens) for 5 classes."""
    checkpoint = "MIT/ast-finetuned-audioset-10-10-0.4593"
    pretrained_model = ASTForAudioClassification.from_pretrained(checkpoint)

    config = ASTConfig.from_pretrained(
        checkpoint,
        patch_size=16,
        frequency_stride=16,
        time_stride=16,
        max_length=1024,
        num_mel_bins=128,
    )
    config.num_labels = NUM_CLASSES
    config.id2label = IDX_TO_CLASS
    config.label2id = CLASS_TO_IDX

    model = ASTForAudioClassification(config)

    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()
    for k, v in pretrained_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            model_dict[k] = v

    # Interpolate Positional Embeddings (12x101 -> 8x64)
    orig_pos = pretrained_dict[
        "audio_spectrogram_transformer.embeddings.position_embeddings"
    ]
    cls_dist = orig_pos[:, :2, :]
    grid_2d = orig_pos[:, 2:, :].transpose(1, 2).reshape(1, -1, 12, 101)
    interp_grid = F.interpolate(
        grid_2d, size=(8, 64), mode="bilinear", align_corners=False
    )
    interp_tokens = interp_grid.flatten(2).transpose(1, 2)

    model_dict["audio_spectrogram_transformer.embeddings.position_embeddings"] = (
        torch.cat([cls_dist, interp_tokens], dim=1)
    )
    model.load_state_dict(model_dict)
    return model


class LiveASTPredictor:
    def __init__(
        self,
        weights_path: str = "trained_models/5_classes_best_ast_faststride_model.pth",
        confidence_threshold: float = 0.35,
        rms_silence_threshold: float = 0.008,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.rms_silence_threshold = rms_silence_threshold

        print(
            f"[INFO] Initializing AST-FastStride-16x16 (5 classes) on {self.device}..."
        )
        self.extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.model = build_ast_faststride_16x16_model()

        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{path.resolve()}'. "
                f"Please ensure '{path.name}' is located inside '{path.parent}'."
            )

        state_dict = torch.load(str(path), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("[INFO] 5-class model loaded and ready for live inference!")

    def preprocess_waveform(self, audio_np: np.ndarray) -> tuple[torch.Tensor, float]:
        """Applies exact training pipeline preprocessing: Pad/Crop, RMS norm, Pre-emphasis."""
        waveform = torch.from_numpy(audio_np).float()

        # Compute raw RMS for silence gating before normalization
        raw_rms = float(torch.sqrt(torch.mean(waveform.pow(2)) + 1e-8).item())

        num_samples = waveform.shape[0]
        if num_samples > TARGET_SAMPLES:
            waveform = waveform[-TARGET_SAMPLES:]  # Keep latest 10.24s
        elif num_samples < TARGET_SAMPLES:
            pad = TARGET_SAMPLES - num_samples
            waveform = F.pad(waveform, (pad, 0), mode="constant", value=0.0)

        # RMS Normalization + Pre-emphasis
        waveform = waveform - waveform.mean()
        rms = torch.sqrt(torch.mean(waveform.pow(2)) + 1e-8)
        waveform = torch.tanh(waveform * (0.1 / rms))
        waveform = torch.cat([waveform[:1], waveform[1:] - 0.97 * waveform[:-1]], dim=0)

        inputs = self.extractor(
            waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        return inputs["input_values"].to(self.device), raw_rms

    @torch.no_grad()
    def predict(self, audio_np: np.ndarray) -> dict:
        input_values, raw_rms = self.preprocess_waveform(audio_np)

        # 1. Check Silence Gate
        if raw_rms < self.rms_silence_threshold:
            return {
                "status": "quiet",
                "predicted_class": None,
                "display_label": "Listening... (Quiet)",
                "confidence": 0.0,
                "is_threat": False,
                "rms": round(raw_rms, 4),
                "probabilities": {
                    DISPLAY_LABELS[IDX_TO_CLASS[i]]: 0.0 for i in range(NUM_CLASSES)
                },
            }

        # 2. Run AST-16x16 Forward Pass
        outputs = self.model(input_values=input_values)
        logits = outputs.logits[0]  # Shape: (5,)
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        top_idx = int(np.argmax(probs))
        predicted_class = IDX_TO_CLASS[top_idx]
        top_prob = float(probs[top_idx])

        ui_probs = {
            DISPLAY_LABELS[IDX_TO_CLASS[i]]: round(float(probs[i]), 4)
            for i in range(NUM_CLASSES)
        }

        # 3. Suppress low-confidence noise
        if top_prob < self.confidence_threshold:
            return {
                "status": "suppressed",
                "predicted_class": None,
                "display_label": "Ambient / Unsure Sound",
                "confidence": round(top_prob, 4),
                "is_threat": False,
                "rms": round(raw_rms, 4),
                "probabilities": ui_probs,
            }

        # 4. Valid Threat / Active Class Detected
        return {
            "status": "detected",
            "predicted_class": predicted_class,
            "display_label": DISPLAY_LABELS[predicted_class],
            "confidence": round(top_prob, 4),
            "is_threat": predicted_class in HIGH_PRIORITY_THREATS,
            "rms": round(raw_rms, 4),
            "probabilities": ui_probs,
        }
