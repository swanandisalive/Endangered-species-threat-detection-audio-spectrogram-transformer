# 🌿 Real-Time Endangered Species & Threat Detection from Rainforest Audio (SDG 15)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-early%20development-orange)

Acoustic monitoring of vulnerable rainforest ecosystems using a fine-tuned **Audio Spectrogram Transformer (AST)**. The system listens to forest audio and flags **illegal deforestation** (chainsaws, engines) and **poaching** (gunshots) in near real time, so rangers can respond before damage is done.

This project supports **UN Sustainable Development Goal 15: Life on Land**.

## 👥 Team Members
* Swanand Deshpande
* Manav Lakhani
* Jash Mewada
* Shravan More

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Why This Matters](#-why-this-matters)
3. [How It Works](#-how-it-works)
4. [Repository Structure](#-repository-structure)
5. [Getting Started](#-getting-started)
6. [Data](#-data)
7. [Training](#-training)
8. [Evaluation](#-evaluation)
9. [Inference](#-inference)
10. [Roadmap](#-roadmap)
11. [Contributing](#-contributing)
12. [Team](#-team)
13. [References](#-references)
14. [License](#-license)

---

## 🎯 Project Overview

| Item | Detail |
|---|---|
| **Task** | Multi-label audio classification (several sound events can occur in one clip) |
| **Model** | Hugging Face Audio Spectrogram Transformer (AST), fine-tuned with PyTorch |
| **Data source** | Rainforest Connection (RFCx) Arbimon platform |
| **Augmentation** | SpecAugment (time and frequency masking) to improve robustness to background noise |
| **Primary metrics** | PR-AUC and False Positive Rate (FPR) at 90% True Positive Rate |

## 🌍 Why This Matters

Rainforests are noisy: rain, wind, insects, birds, and frogs all overlap. Rangers cannot be everywhere, but a solar-powered recorder can. Reliable, low-false-alarm detection of chainsaws and gunshots turns passive recordings into actionable alerts. That is why we track **FPR at a fixed 90% recall** alongside PR-AUC: a detector that cries wolf gets ignored.

## 🧠 How It Works

```
Raw audio (.wav/.flac)
   │  resample to 16 kHz, mono, fixed-length clips
   ▼
Log-Mel filterbank spectrogram   (AST feature extractor)
   │  SpecAugment during training (time + frequency masking)
   ▼
Audio Spectrogram Transformer    (pretrained on AudioSet, fine-tuned here)
   │  classification head with one sigmoid output per class
   ▼
Per-class probabilities  →  thresholding  →  alerts
```

Key design choices:

- **Multi-label, not multi-class.** Each class gets an independent sigmoid, trained with binary cross-entropy (`BCEWithLogitsLoss`), so a clip can contain a gunshot *and* an engine.
- **Transfer learning.** AST is initialised from an AudioSet-pretrained checkpoint, which helps a lot when labelled rainforest data is scarce.
- **Threshold tuning matters.** Operating thresholds should be chosen per class on the validation set (for example, to hit 90% TPR), not left at 0.5.

## 📁 Repository Structure

The repository is in early development. The current layout is:

```
.
├── docs/
│   └── papers/          # Reference papers and background reading
├── .gitignore
├── LICENSE              # MIT
├── README.md
└── requirements.txt     # Python dependencies
```

Planned layout as code lands (please follow it when adding files):

```
.
├── data/                # NOT committed. Raw and processed audio live here locally
│   ├── raw/
│   └── processed/
├── notebooks/           # Exploration and EDA only; no core logic
├── src/
│   ├── data/            # Datasets, loading, resampling, label parsing
│   ├── models/          # AST wrapper, classification head
│   ├── train.py         # Fine-tuning entry point
│   ├── evaluate.py      # PR-AUC, FPR@90%TPR, per-class reports
│   └── infer.py         # Run the model on new audio or a live stream
├── configs/             # YAML experiment configs
├── tests/
└── docs/
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or newer
- Git
- A CUDA-capable GPU is strongly recommended for training (CPU works for inference and small tests)
- `ffmpeg` / `libsndfile` for audio decoding (see below)

### 1. Clone the repository

```bash
git clone https://github.com/swanandisalive/Endangered-species-threat-detection-audio-spectrogram-transformer.git
cd Endangered-species-threat-detection-audio-spectrogram-transformer
```

### 2. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you need GPU support, install the PyTorch build matching your CUDA version first by following the selector at <https://pytorch.org/get-started/locally/>, then run the command above.

System audio libraries, if missing:

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg libsndfile1

# macOS
brew install ffmpeg libsndfile
```

### 4. Verify the install

```bash
python -c "import torch, transformers, torchaudio; print(torch.__version__, transformers.__version__, torchaudio.__version__)"
```

## 🎧 Data

Audio and annotations come from the **RFCx Arbimon** platform (<https://arbimon.org>).

> ⚠️ **Never commit audio files or large datasets.** `data/` should stay in `.gitignore`. Share datasets through a drive link or object store and document the source here.

Guidelines for working with data:

- **Sample rate:** resample everything to **16 kHz mono**, which is what the pretrained AST expects.
- **Clip length:** use fixed-length windows (AST's default input is about 10 seconds); pad or crop consistently.
- **Labels:** stored as a multi-hot vector, one column per class.
- **Splits:** split by **recording site / device / time**, not by random clip. Random clip splits leak the acoustic environment between train and test and inflate scores.
- **Class imbalance:** threat sounds are rare compared to background. Consider weighted loss, oversampling, or hard-negative mining, and always report per-class metrics.

Target classes (to be finalised by the team):

| Class | Category |
|---|---|
| Chainsaw | Illegal logging |
| Vehicle / engine | Illegal logging / access |
| Gunshot | Poaching |
| Endangered species vocalisations | Species presence (list TBD) |
| Background (rain, wind, insects, etc.) | Negative class |

## 🏋️ Training

> The training scripts are not yet in the repo. This section documents the intended setup so everyone builds toward the same interface.

Intended invocation:

```bash
python src/train.py --config configs/ast_base.yaml
```

Baseline setup:

| Setting | Starting point |
|---|---|
| Pretrained checkpoint | `MIT/ast-finetuned-audioset-10-10-0.4593` (Hugging Face) |
| Loss | `BCEWithLogitsLoss` (multi-label) |
| Optimizer | AdamW, low learning rate (around 1e-5 to 5e-5) |
| Augmentation | SpecAugment: time masking + frequency masking |
| Precision | Mixed precision (fp16/bf16) where supported |
| Seeds | Fixed and logged for every run |

Minimal model setup for reference:

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ckpt = "MIT/ast-finetuned-audioset-10-10-0.4593"
extractor = ASTFeatureExtractor.from_pretrained(ckpt)
model = ASTForAudioClassification.from_pretrained(
    ckpt,
    num_labels=NUM_CLASSES,
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True,   # replace the 527-class AudioSet head
)
```

## 📊 Evaluation

We report the following on a held-out, site-disjoint test set:

- **PR-AUC (average precision)**, per class and macro-averaged. It is preferred over ROC-AUC because positives are rare.
- **FPR at 90% TPR.** Pick the threshold where recall reaches 90%, then report the false positive rate at that point. Lower is better. This reflects real-world alert fatigue.
- Per-class precision, recall, and confusion analysis (what gets mistaken for a chainsaw?).

Please include a results table in your PR whenever a change affects model behaviour:

| Run / commit | Classes | Macro PR-AUC | FPR @ 90% TPR | Notes |
|---|---|---|---|---|
| _baseline_ | | | | |

## 🔮 Inference

Planned usage (not yet implemented):

```bash
# Single file
python src/infer.py --audio path/to/clip.wav --checkpoint path/to/model

# Near real-time on a stream or folder of incoming clips
python src/infer.py --stream --window 10 --hop 5 --checkpoint path/to/model
```

Real-time notes: use overlapping sliding windows, smooth predictions over consecutive windows to cut one-off false alarms, and keep per-class thresholds in the config rather than in code.

## 🗺️ Roadmap

- [ ] Finalise class list and labelling scheme
- [ ] Data download and preprocessing pipeline from Arbimon
- [ ] Site-disjoint train/val/test splits
- [ ] Baseline AST fine-tune with SpecAugment
- [ ] Evaluation script (PR-AUC, FPR@90%TPR, per-class reports)
- [ ] Threshold calibration per class
- [ ] Sliding-window real-time inference
- [ ] Model card and documented limitations
- [ ] Edge/low-power deployment study

## 🤝 Contributing

We welcome contributions from everyone on the team and beyond.

### Workflow

1. **Check Issues first** to avoid duplicate work. Open one if your idea is new, or comment to claim an existing one.
2. **Branch from `main`** with a clear name:
   - `feat/<short-description>`
   - `fix/<short-description>`
   - `docs/<short-description>`
   - `exp/<short-description>` for experiments
3. **Make small, focused commits** with clear messages (for example, `feat: add SpecAugment transform`).
4. **Open a Pull Request** against `main` with:
   - what changed and why
   - how you tested it
   - a results table if model behaviour changed
5. **Get at least one review** before merging. Do not push directly to `main`.

### Code standards

- Python 3.10+, formatted with `black` and linted with `ruff` (or `flake8`)
- Type hints and short docstrings on public functions
- No hard-coded absolute paths; use configs or CLI arguments
- Set and log random seeds so results are reproducible
- Keep notebooks for exploration; move reusable logic into `src/`

### Do not commit

- Audio files, spectrograms, or datasets
- Model checkpoints and large artifacts (use releases or external storage)
- API keys, tokens, or credentials
- Personal paths or machine-specific config

### Reporting issues

Please include: what you expected, what happened, steps to reproduce, your OS, Python, and PyTorch versions, and relevant logs.

## 📚 References

- Y. Gong, Y.-A. Chung, J. Glass. *AST: Audio Spectrogram Transformer.* Interspeech 2021. <https://arxiv.org/abs/2104.01778>
- D. S. Park et al. *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.* Interspeech 2019. <https://arxiv.org/abs/1904.08779>
- Rainforest Connection (RFCx) Arbimon: <https://arbimon.org>
- Hugging Face AST documentation: <https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer>
- UN SDG 15, Life on Land: <https://sdgs.un.org/goals/goal15>

More background papers are collected in [`docs/papers/`](docs/papers).

## 📄 License

Released under the [MIT License](LICENSE).

## 🙏 Acknowledgements

Thanks to Rainforest Connection and the Arbimon community for making bioacoustic data accessible, and to the authors of AST and SpecAugment.




