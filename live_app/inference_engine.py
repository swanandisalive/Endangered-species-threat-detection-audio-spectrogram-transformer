"""
inference_engine.py
Handles AST-FastStride-16x16 initialization, audio preprocessing, and 5-class threat filtering.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification

SAMPLE_RATE = 16000
TARGET_DURATION = 10.24
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION)  # 163,840 samples

# Full 7-class mapping required to load your saved state_dict
CLASS_TO_IDX = {
    "Background_Nature": 0,
    "Sawing_and_Chainsaw": 1,
    "Gunshot_and_Explosion": 2,
    "Engine_and_Vehicle": 3,
    "Wood_Chop_and_Break": 4,
    "Human_Vocalization": 5,
    "Dog_Bark": 6,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_TO_IDX)

# Classes to suppress from triggering predictions
IGNORED_INDICES = {0, 6}  # 0: Background_Nature, 6: Dog_Bark
ACTIVE_INDICES = [1, 2, 3, 4, 5]

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
    """Builds the 16x16 non-overlapping patch AST architecture (514 tokens)."""
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
        weights_path: str = "trained_models/best_ast_faststride_model.pth",
        confidence_threshold: float = 0.35,
        rms_silence_threshold: float = 0.008,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.rms_silence_threshold = rms_silence_threshold

        print(f"[INFO] Initializing AST-FastStride-16x16 on {self.device}...")
        self.extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.model = build_ast_faststride_16x16_model()

        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{path.resolve()}'. "
                "Place 'best_ast_faststride_model.pth' inside 'trained_models/'."
            )

        state_dict = torch.load(str(path), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("[INFO] Model loaded and ready for live inference!")

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

        # Exact RMS Normalization + Pre-emphasis from preprocessing_pipeline.py
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
                    DISPLAY_LABELS[IDX_TO_CLASS[i]]: 0.0 for i in ACTIVE_INDICES
                },
            }

        # 2. Run AST-16x16 Forward Pass
        outputs = self.model(input_values=input_values)
        logits = outputs.logits[0]  # Shape: (7,)
        raw_probs = F.softmax(logits, dim=-1).cpu().numpy()

        top_raw_idx = int(np.argmax(raw_probs))

        # Extract probabilities for the 5 active target classes
        active_probs_raw = {
            IDX_TO_CLASS[i]: float(raw_probs[i]) for i in ACTIVE_INDICES
        }

        # Best target class among the 5 allowed classes
        best_active_cls = max(active_probs_raw, key=active_probs_raw.get)
        best_active_prob = active_probs_raw[best_active_cls]

        # Re-normalize probabilities across the 5 target classes for clean UI bars
        active_sum = sum(active_probs_raw.values()) + 1e-8
        ui_probs = {
            DISPLAY_LABELS[cls]: round(
                (prob / active_sum) if top_raw_idx not in IGNORED_INDICES else prob, 4
            )
            for cls, prob in active_probs_raw.items()
        }

        # 3. Suppress Background_Nature (0) and Dog_Bark (6) or low-confidence noise
        if (
            top_raw_idx in IGNORED_INDICES
            or best_active_prob < self.confidence_threshold
        ):
            return {
                "status": "suppressed",
                "predicted_class": None,
                "display_label": "Ambient / Ignored Sound",
                "confidence": round(best_active_prob, 4),
                "is_threat": False,
                "rms": round(raw_rms, 4),
                "probabilities": ui_probs,
            }

        # 4. Valid Target Class Detected
        return {
            "status": "detected",
            "predicted_class": best_active_cls,
            "display_label": DISPLAY_LABELS[best_active_cls],
            "confidence": round(ui_probs[DISPLAY_LABELS[best_active_cls]], 4),
            "is_threat": best_active_cls in HIGH_PRIORITY_THREATS,
            "rms": round(raw_rms, 4),
            "probabilities": ui_probs,
        }
