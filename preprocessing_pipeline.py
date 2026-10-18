"""
pre_processing_pipeline.py
Dedicated 16x16 AST Pipeline for 7-Class Forest Acoustic Threat Detection.
Built for easy import into Jupyter Notebooks.
Supports both raw FSD50K and pre-split shared dataset directories.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification

# =====================================================================
# CONFIGURATION & 7-CLASS TAXONOMY
# =====================================================================
SEED = 42
SAMPLE_RATE = 16000
NUM_MEL_BINS = 128
TARGET_DURATION = 10.24  # Required for AST 1024 frames

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

# Exact FSD50K vocabulary strings
FSD50K_EXACT_LABELS = {
    "Sawing_and_Chainsaw": {"Sawing", "Chainsaw"},
    "Gunshot_and_Explosion": {
        "Gunshot_and_gunfire",
        "Explosion",
        "Artillery_fire",
        "Machine_gun",
        "Fusillade",
    },
    "Engine_and_Vehicle": {
        "Engine",
        "Idling",
        "Accelerating_and_revving_and_vroom",
        "Medium_engine_(mid_frequency)",
        "Heavy_engine_(low_frequency)",
        "Light_engine_(high_frequency)",
        "Motor_vehicle_(road)",
        "Truck",
        "Motorcycle",
    },
    "Wood_Chop_and_Break": {"Chop", "Wood", "Crack", "Splinter", "Whack_and_thwack"},
    "Human_Vocalization": {
        "Speech",
        "Male_speech_and_man_speaking",
        "Female_speech_and_woman_speaking",
        "Conversation",
        "Shout",
        "Screaming",
        "Yell",
    },
    "Dog_Bark": {"Bark", "Dog", "Growling", "Howl", "Yip"},
    "Background_Nature": {
        "Bird",
        "Bird_vocalization_and_bird_call_and_bird_song",
        "Frog",
        "Insect",
        "Rain",
        "Thunder",
        "Wind",
        "Rustling_leaves",
        "Stream",
    },
}


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# DATASET LOADING FUNCTIONS
# =====================================================================
def load_dataset_splits(
    dataset_dir: str, metadata_file: str = "dataset_metadata.csv"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads pre-split dataset from a shared directory containing audio/ and dataset_metadata.csv.
    Use this when working with the exported/shared dataset zip.
    """
    root_path = Path(dataset_dir)
    csv_path = root_path / metadata_file
    audio_dir = root_path / "audio"

    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {csv_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found at: {audio_dir}")

    df = pd.read_csv(csv_path)

    # Reconstruct the absolute path required by FastStrideASTDataset
    df["filepath"] = df["filename"].apply(lambda fname: str(audio_dir / fname))

    # Verify that files are present
    missing_files = [fp for fp in df["filepath"] if not Path(fp).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} audio files specified in metadata are missing on disk."
        )

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    return train_df, val_df, test_df


def resolve_clean_fsd50k_class(raw_labels: set) -> int | None:
    matched_threats = [
        cls
        for cls in FSD50K_EXACT_LABELS
        if raw_labels.intersection(FSD50K_EXACT_LABELS[cls])
    ]

    if set(matched_threats) == {"Sawing_and_Chainsaw", "Engine_and_Vehicle"}:
        return CLASS_TO_IDX["Sawing_and_Chainsaw"]
    if len(matched_threats) > 1 and "Background_Nature" not in matched_threats:
        return None
    if len(matched_threats) == 1:
        return CLASS_TO_IDX[matched_threats[0]]
    if "Domestic_animals_and_pets" in raw_labels:
        return None
    if raw_labels.intersection(FSD50K_EXACT_LABELS["Background_Nature"]):
        return CLASS_TO_IDX["Background_Nature"]
    return None


def build_dataset_splits(
    fsd50k_root: str, max_samples_per_class: int = 700
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parses raw FSD50K CSVs and returns Train, Validation, and Test DataFrames based on available files."""
    fsd_path = Path(fsd50k_root)

    def load_split(gt_csv, audio_dir):
        records = []
        if gt_csv.exists() and audio_dir.exists():
            df = pd.read_csv(gt_csv)
            for _, row in df.iterrows():
                lbls = {l.strip() for l in str(row["labels"]).split(",")}
                target_idx = resolve_clean_fsd50k_class(lbls)
                if target_idx is not None:
                    wav_path = audio_dir / f"{row['fname']}.wav"
                    if wav_path.exists():
                        records.append(
                            {
                                "filepath": str(wav_path),
                                "label": target_idx,
                                "class_name": IDX_TO_CLASS[target_idx],
                            }
                        )
        return records

    dev_records = load_split(
        fsd_path / "FSD50K.ground_truth" / "dev.csv", fsd_path / "FSD50K.dev_audio"
    )
    eval_records = load_split(
        fsd_path / "FSD50K.ground_truth" / "eval.csv", fsd_path / "FSD50K.eval_audio"
    )

    full_df = pd.DataFrame(dev_records + eval_records)
    if full_df.empty:
        raise ValueError(
            f"No audio files matched in {fsd50k_root}. Check your path variable."
        )

    balanced_dfs = []
    for cls_idx in CLASS_TO_IDX.values():
        cls_df = full_df[full_df["label"] == cls_idx]
        if len(cls_df) > max_samples_per_class:
            cls_df = cls_df.sample(n=max_samples_per_class, random_state=SEED)
        balanced_dfs.append(cls_df)
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    # Split 70% Train, 15% Validation, 15% Test
    train_val, test_df = train_test_split(
        balanced_df, test_size=0.15, stratify=balanced_df["label"], random_state=SEED
    )
    train_df, val_df = train_test_split(
        train_val,
        test_size=(0.15 / 0.85),
        stratify=train_val["label"],
        random_state=SEED,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# =====================================================================
# PYTORCH DATASET CLASS
# =====================================================================
class FastStrideASTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_extractor, is_train: bool = True):
        self.df = df
        self.feature_extractor = feature_extractor
        self.target_samples = int(TARGET_DURATION * SAMPLE_RATE)
        self.is_train = is_train

        self.bg_files = self.df[self.df["label"] == CLASS_TO_IDX["Background_Nature"]][
            "filepath"
        ].tolist()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=20)
        self.time_mask = T.TimeMasking(time_mask_param=24)
        self._resamplers = {}

    def __len__(self):
        return len(self.df)

    def _load_and_process(self, filepath: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(filepath)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            if sr not in self._resamplers:
                self._resamplers[sr] = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = self._resamplers[sr](waveform)

        waveform = waveform.squeeze(0)
        num_samples = waveform.shape[0]

        # Smart Energy Crop
        if num_samples > self.target_samples:
            chunk_size = 1600
            num_chunks = num_samples // chunk_size
            if num_chunks > 1:
                chunk_energy = (
                    waveform[: num_chunks * chunk_size]
                    .view(num_chunks, chunk_size)
                    .pow(2)
                    .mean(dim=1)
                )
                peak_center = int(torch.argmax(chunk_energy).item()) * chunk_size + (
                    chunk_size // 2
                )
                ideal_start = peak_center - (self.target_samples // 2)
                if self.is_train:
                    ideal_start += random.randint(
                        -int(self.target_samples * 0.2), int(self.target_samples * 0.2)
                    )
                start = max(0, min(num_samples - self.target_samples, ideal_start))
            else:
                start = 0
            waveform = waveform[start : start + self.target_samples]
        elif num_samples < self.target_samples:
            pad = self.target_samples - num_samples
            left = random.randint(0, pad) if self.is_train else pad // 2
            waveform = F.pad(waveform, (left, pad - left), mode="constant", value=0.0)

        # RMS Norm + Pre-emphasis
        waveform = waveform - waveform.mean()
        rms = torch.sqrt(torch.mean(waveform.pow(2)) + 1e-8)
        waveform = torch.tanh(waveform * (0.1 / rms))
        waveform = torch.cat([waveform[:1], waveform[1:] - 0.97 * waveform[:-1]], dim=0)
        return waveform

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        waveform = self._load_and_process(row["filepath"])

        # Background Mixing
        if (
            self.is_train
            and row["label"] != CLASS_TO_IDX["Background_Nature"]
            and self.bg_files
            and random.random() < 0.5
        ):
            bg_wave = self._load_and_process(random.choice(self.bg_files))
            snr = 10 ** (random.uniform(5.0, 20.0) / 20.0)
            scale = waveform.norm(p=2) / (bg_wave.norm(p=2) * snr + 1e-8)
            waveform = torch.tanh(waveform + scale * bg_wave)

        inputs = self.feature_extractor(
            waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        input_values = inputs["input_values"].squeeze(0)

        if self.is_train and random.random() < 0.5:
            spec = input_values.transpose(0, 1).unsqueeze(0)
            spec = self.time_mask(self.freq_mask(spec))
            input_values = spec.squeeze(0).transpose(0, 1)

        return {
            "input_values": input_values.float(),
            "labels": torch.tensor(row["label"], dtype=torch.long),
        }


# =====================================================================
# DATALOADER GENERATOR
# =====================================================================
def get_dataloaders(train_df, val_df, test_df, batch_size=16, num_workers=2):
    """Creates PyTorch DataLoaders with Weighted Random Sampling for training."""
    extractor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    train_ds = FastStrideASTDataset(train_df, extractor, is_train=True)
    val_ds = FastStrideASTDataset(val_df, extractor, is_train=False)
    test_ds = FastStrideASTDataset(test_df, extractor, is_train=False)

    weights = [1.0 / train_df["label"].value_counts()[lbl] for lbl in train_df["label"]]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights), num_samples=len(train_df), replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, extractor


# =====================================================================
# 16x16 FAST STRIDE MODEL BUILDER (CLEANED)
# =====================================================================
def build_ast_faststride_16x16_model():
    """Initializes the AST model with 16x16 non-overlapping patches (514 tokens)."""
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

    # Completely overwrite label mappings to match the 7-class taxonomy
    config.num_labels = NUM_CLASSES
    config.id2label = IDX_TO_CLASS
    config.label2id = CLASS_TO_IDX

    # Delete old cached label attributes if they exist in the config object
    if hasattr(config, "_name_or_path"):
        pass
    config.id2label = IDX_TO_CLASS
    config.label2id = CLASS_TO_IDX

    model = ASTForAudioClassification(config)

    # Copy weights
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
