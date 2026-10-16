import glob
import os

import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ASTFeatureExtractor

# --- CONFIGURATION CONSTANTS ---
DEFAULT_TARGET_SR = 16000
DEFAULT_DURATION_SEC = 10
DEFAULT_TARGET_SAMPLES = DEFAULT_TARGET_SR * DEFAULT_DURATION_SEC  # 160,000 samples

TARGET_CLASSES = ["chainsaw", "gunshot", "engine", "background"]


# =====================================================================
# 1. AUDIO STANDARDIZATION & SPLITTING FUNCTIONS
# =====================================================================


def standardize_audio(
    input_path: str,
    output_path: str,
    target_sr: int = DEFAULT_TARGET_SR,
    target_duration_sec: int = DEFAULT_DURATION_SEC,
) -> bool:
    """
    Standardizes raw input audio into 16 kHz Mono WAV format (160,000 samples).
    Uses soundfile directly as a fallback if torchaudio load fails.
    """
    try:
        try:
            waveform, sr = torchaudio.load(input_path)
        except Exception:
            data, sr = sf.read(input_path, dtype="float32")
            waveform = torch.from_numpy(data)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T

        target_length = target_sr * target_duration_sec

        # Downmix Stereo to Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16 kHz
        if sr != target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        # Pad or Truncate to 10 seconds
        current_samples = waveform.shape[-1]
        if current_samples < target_length:
            padding = target_length - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :target_length]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, waveform, target_sr)
        return True
    except Exception as e:
        print(f"\n[Error processing '{input_path}']: {e}")
        return False


def split_records(
    records: list,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, (
        "Ratios must sum to 1.0"
    )

    val_plus_test = val_ratio + test_ratio
    relative_test_ratio = test_ratio / val_plus_test

    train_records, temp_records = train_test_split(
        records, test_size=val_plus_test, random_state=random_seed
    )
    val_records, test_records = train_test_split(
        temp_records, test_size=relative_test_ratio, random_state=random_seed
    )

    return train_records, val_records, test_records


def _collect_raw_records(
    dataset_dir: str = "dataset", output_dir: str = "split_dataset"
):
    raw_records = []

    # Collect Gunshot Audio Files
    gunshot_dir = os.path.join(dataset_dir, "gunshot")
    if os.path.exists(gunshot_dir):
        gunshot_files = glob.glob(
            os.path.join(gunshot_dir, "**", "*.wav"), recursive=True
        ) + glob.glob(os.path.join(gunshot_dir, "**", "*.mp3"), recursive=True)
        for fpath in gunshot_files:
            fname = os.path.basename(fpath)
            if not fname.startswith(("gunshot_dataset_", "FSD50K_")):
                raw_records.append(
                    {
                        "raw_path": fpath,
                        "labels": "gunshot",
                        "source": "gunshot_dataset",
                    }
                )

    # Collect FSD50K Audio Files
    print("Indexing raw audio files for FSD50K mapping...")
    search_dirs = [dataset_dir, output_dir]
    wav_lookup = {}
    for sdir in search_dirs:
        if os.path.exists(sdir):
            for p in glob.glob(os.path.join(sdir, "**", "*.wav"), recursive=True):
                fname = os.path.basename(p)
                if not fname.startswith(("gunshot_dataset_", "FSD50K_")):
                    wav_lookup[fname] = p

    fsd50k_dev_csv = os.path.join(dataset_dir, "FSD50K.ground_truth", "dev.csv")
    fsd50k_eval_csv = os.path.join(dataset_dir, "FSD50K.ground_truth", "eval.csv")

    for csv_path in [fsd50k_dev_csv, fsd50k_eval_csv]:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                fname = str(row["fname"])
                if not fname.endswith(".wav"):
                    fname += ".wav"

                if fname in wav_lookup:
                    raw_records.append(
                        {
                            "raw_path": wav_lookup[fname],
                            "labels": str(row["labels"]),
                            "source": "FSD50K",
                        }
                    )

    return raw_records


def process_and_export_split(
    records: list, split_name: str, output_dir: str = "split_dataset"
) -> pd.DataFrame:
    split_audio_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_audio_dir, exist_ok=True)

    exported_rows = []
    print(
        f"\nStandardizing audio files for split: '{split_name.upper()}' ({len(records)} samples)..."
    )

    for idx, item in enumerate(tqdm(records, desc=f"Processing {split_name}")):
        out_fname = f"{item['source']}_{split_name}_{idx:05d}.wav"
        out_path = os.path.join(split_audio_dir, out_fname)

        success = standardize_audio(item["raw_path"], out_path)
        if success:
            exported_rows.append(
                {
                    "fname": out_fname,
                    "labels": item["labels"],
                    "source": item["source"],
                    "file_path": out_path,
                }
            )

    df_meta = pd.DataFrame(
        exported_rows, columns=["fname", "labels", "source", "file_path"]
    )
    csv_out_path = os.path.join(output_dir, f"{split_name}_metadata.csv")
    df_meta.to_csv(csv_out_path, index=False)
    print(f"Exported {len(df_meta)} rows to metadata: {csv_out_path}")
    return df_meta


def run_full_pipeline(
    dataset_dir: str = "dataset",
    output_dir: str = "split_dataset",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    print("==================================================")
    print("      EcoAST TRAIN / VAL / TEST PIPELINE          ")
    print("==================================================")

    raw_records = _collect_raw_records(dataset_dir, output_dir)
    print(f"Collected total of {len(raw_records)} raw audio records.")

    if not raw_records:
        raise ValueError("No audio files were found. Please check dataset directories.")

    train_recs, val_recs, test_recs = split_records(
        raw_records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
    )

    df_train = process_and_export_split(train_recs, "train", output_dir)
    df_val = process_and_export_split(val_recs, "val", output_dir)
    df_test = process_and_export_split(test_recs, "test", output_dir)

    print("\n" + "=" * 50)
    print("             PIPELINE EXPORT COMPLETE             ")
    print("=" * 50)
    return df_train, df_val, df_test


# =====================================================================
# 2. PYTORCH DATASET & DATALOADER FACTORY (REUSABLE)
# =====================================================================

# Inside preprocessing_pipeline.py


class ForestThreatDataset(Dataset):
    """
    Unified Dataset class that accepts any Hugging Face AST model checkpoint name.
    """

    def __init__(
        self,
        csv_path: str,
        target_classes: list = TARGET_CLASSES,
        model_checkpoint: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    ):
        self.df = pd.read_csv(csv_path)
        self.target_classes = target_classes
        # Dynamically load the feature extractor matching the chosen model variant
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_checkpoint)

    def _get_multi_hot_label(self, raw_label: str) -> torch.Tensor:
        multi_hot = torch.zeros(len(self.target_classes), dtype=torch.float32)
        raw_label_str = str(raw_label).lower()

        # Explicit Priority: Chainsaw > Gunshot > Engine > Background
        if "chainsaw" in raw_label_str:
            multi_hot[self.target_classes.index("chainsaw")] = 1.0
        elif "gunshot" in raw_label_str or "gun" in raw_label_str:
            multi_hot[self.target_classes.index("gunshot")] = 1.0
        elif "engine" in raw_label_str:
            multi_hot[self.target_classes.index("engine")] = 1.0
        elif "background" in self.target_classes:
            multi_hot[self.target_classes.index("background")] = 1.0

        return multi_hot

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]

        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception:
            data, sr = sf.read(file_path, dtype="float32")
            waveform = torch.from_numpy(data).unsqueeze(0)

        inputs = self.feature_extractor(
            waveform.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt"
        )

        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": self._get_multi_hot_label(row["labels"]),
        }


def get_dataloaders(
    output_dir: str = "split_dataset",
    batch_size: int = 8,
    num_workers: int = 0,
    target_classes: list = TARGET_CLASSES,
    model_checkpoint: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
):
    """
    Factory function instantiating DataLoaders for any requested AST model backbone.
    """
    train_csv = os.path.join(output_dir, "train_metadata.csv")
    val_csv = os.path.join(output_dir, "val_metadata.csv")
    test_csv = os.path.join(output_dir, "test_metadata.csv")

    train_ds = ForestThreatDataset(train_csv, target_classes, model_checkpoint)
    val_ds = ForestThreatDataset(val_csv, target_classes, model_checkpoint)
    test_ds = ForestThreatDataset(test_csv, target_classes, model_checkpoint)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    run_full_pipeline()
