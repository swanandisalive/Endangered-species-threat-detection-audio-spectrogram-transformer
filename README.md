```markdown
# 🌿 EcoAST: Real-Time Endangered Species & Threat Detection from Rainforest Audio (SDG 15)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active%20prototyping-orange)

Acoustic monitoring of vulnerable rainforest ecosystems using a fine-tuned **Audio Spectrogram Transformer (AST)**. The system processes continuous ambient forest soundscapes to flag **illegal deforestation** (chainsaws, engine activity) and **poaching** (gunshots) alongside **endangered species calls** in near real time, enabling rapid intervention by rangers before ecological damage occurs.

This project directly supports **UN Sustainable Development Goal 15: Life on Land** and **SDG 13: Climate Action**.

---

## 👥 Team Members
- Swanand Deshpande
- Manav Lakhani
- Jash Mewada
- Shravan More

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Why This Matters](#-why-this-matters)
3. [System Architecture & Design](#-system-architecture--design)
4. [Data Strategy & Selection](#-data-strategy--selection)
5. [Repository Structure](#-repository-structure)
6. [Getting Started](#-getting-started)
7. [Implementation & Pipeline](#-implementation--pipeline)
8. [Evaluation & Mathematical Threshold Calibration](#-evaluation--mathematical-threshold-calibration)
9. [Inference & Deployment Considerations](#-inference--deployment-considerations)
10. [Roadmap](#-roadmap)
11. [Contributing](#-contributing)
12. [Team & References](#-team--references)
13. [License](#-license)

---

## 🎯 Project Overview

| Item | Detail |
|---|---|
| **Task** | Multi-label audio event detection (polyphonic overlap of background noise, wildlife, and human threats) |
| **Model** | Audio Spectrogram Transformer (`MIT/ast-finetuned-audioset-10-10-0.4593`), fine-tuned with PyTorch |
| **Data Sources** | RFCx Arbimon, Zenodo FSD50K, Kaggle Gunshot Dataset, Hugging Face `ashraq/esc50` |
| **Preprocessing & Augmentation** | 16 kHz Mono WAV resampling, Log-Mel filterbank extraction (128 bands), SpecAugment (time & frequency masking), dynamic SNR signal mixing |
| **Loss & Calibration** | Multi-label Binary Cross-Entropy (`BCEWithLogitsLoss`) with class-specific $F_\beta$-score threshold tuning ($F_2$ for threats, $F_1$ for bioacoustics) |
| **Primary Metrics** | PR-AUC (Macro & Per-Class), FPR at 90% True Positive Rate (TPR), $F_2$-score for threat alerts |

---

## 🌍 Why This Matters

Real-world natural habitats are heavily polyphonic: rain, wind, insects, birds, and frogs overlap continuously. Threat signals like gunshots (short transient impulses) and chainsaws (sustained mechanical noise) occur sparsely within this loud ambient acoustic backdrop.

Rangers cannot patrol millions of hectares physically, but solar-powered acoustic nodes can monitor continuously. However, if a detection system generates excessive false alarms, rangers will ignore it ("alert fatigue"). By tracking **FPR at a fixed 90% recall** and applying class-specific threshold optimization ($F_2$-optimization for high recall on threats), EcoAST minimizes missed poaching events while maintaining high precision on false positives.

---

## 🧠 System Architecture & Design

We structure EcoAST around a **5-Layer Modular AI Systems Architecture**:


```

┌─────────────────────────────────────────────────────────┐
│ Layer 1: Audio Ingestion & Streaming                    │
│ Stream datasets dynamically (Hugging Face / Local WAVs) │
└───────────────────────────┬─────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│ Layer 2: Synthetic Polyphonic Audio Synthesizer         │
│ Overlay threat signals onto rainforest background at    │
│ randomized Signal-to-Noise Ratios (SNR 0–10 dB)        │
└───────────────────────────┬─────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│ Layer 3: Audio Normalization & Feature Extractor        │
│ Resample to 16 kHz Mono, 10s pad/truncation (160k)     │
│ Compute 128-band Log-Mel Spectrograms via AST Extractor │
└───────────────────────────┬─────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│ Layer 4: Audio Spectrogram Transformer (AST) Backbone   │
│ Transfer learning from AudioSet weights; linear head    │
│ projection to N target classes                          │
└───────────────────────────┬─────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│ Layer 5: Multi-Label Loss & Threshold Inference Engine  │
│ Independent Sigmoids, BCEWithLogitsLoss, and class-   │
│ tuned F_beta optimal thresholding                       │
└─────────────────────────────────────────────────────────┘

```

---

## 🎧 Data Strategy & Selection

Instead of re-training on millions of raw audio clips, we initialize AST with weights pretrained on **AudioSet** (527 classes) and fine-tune using a targeted hybrid strategy:

1. **Target Threat Events (Foreground):**
   - **Chainsaws & Machinery:** Zenodo FSD50K (Record 4060432) and ESC-50 (`chainsaw`, `engine`).
   - **Gunshots & Poaching:** Kaggle Gunshot Audio Dataset (Emrah Aydemir) for transient impulse profiles.
2. **Bioacoustics & Species Calls (Target Wildlife):**
   - **Endangered Species & Amphibians/Birds:** Rainforest Connection (RFCx) Arbimon platform & DCASE bioacoustic datasets.
3. **Ambient Background Canvas:**
   - **Rainforest Noise & Weather:** RFCx continuous background recordings (wind, rain, insects) used as the baseline audio canvas during synthetic polyphony generation.
4. **Rapid Prototyping:**
   - Hugging Face `ashraq/esc50` for local iteration and verification of loss convergence.

---

## 📁 Repository Structure


```

.
├── docs/
│   └── papers/                # 8 foundational literature survey research papers
├── data/                      # Local raw and processed audio (GITIGNORED)
│   ├── raw/
│   └── processed/
├── notebooks/                 # Exploratory data analysis & prototyping
│   └── Deep_Learning_Lab.ipynb
├── src/
│   ├── data/                  # Dynamic polyphonic audio synthesizer & data loaders
│   ├── models/                # AST wrapper and multi-label classification head
│   ├── utils/                 # Threshold calibration & F_beta evaluation tools
│   ├── train.py               # Fine-tuning entry point
│   ├── evaluate.py            # PR-AUC, FPR@90%TPR, per-class evaluation
│   └── infer.py               # Real-time sliding window inference
├── configs/                   # Experiment YAML configurations
├── .gitignore
├── LICENSE                    # MIT
├── README.md                  # Project documentation
└── requirements.txt           # Dependency environment configuration

```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.x with CUDA / MPS support
- `ffmpeg` and `libsndfile` for audio decoding

### 1. Installation

```bash
git clone [https://github.com/swanandisalive/Endangered-species-threat-detection-audio-spectrogram-transformer.git](https://github.com/swanandisalive/Endangered-species-threat-detection-audio-spectrogram-transformer.git)
cd Endangered-species-threat-detection-audio-spectrogram-transformer

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

```

System dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg libsndfile1

# macOS
brew install ffmpeg libsndfile

```

---

## 🏋️ Implementation & Pipeline

### 1. Polyphonic Audio Synthesizer (`src/data/synthesizer.py`)

Handles dynamic audio mixing at varying Signal-to-Noise Ratios (SNR) to simulate real-world forest polyphony:

```python
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Audio
from transformers import ASTFeatureExtractor
import random

class ForestAudioSynthesizer(Dataset):
    def __init__(self, target_classes=["chainsaw", "fireworks", "rooster", "crackling_fire"], target_duration=10, sample_rate=16000):
        self.sample_rate = sample_rate
        self.target_length = target_duration * sample_rate
        self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
        self.esc50 = load_dataset("ashraq/esc50", split="train").cast_column("audio", Audio(sampling_rate=self.sample_rate, mono=True))
        self.target_classes = target_classes
        self.filtered_samples = [item for item in self.esc50 if item["category"] in self.target_classes]

    def _pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[-1] < self.target_length:
            return torch.nn.functional.pad(waveform, (0, self.target_length - waveform.shape[-1]))
        return waveform[:self.target_length]

    def _mix_signals(self, foreground: torch.Tensor, background: torch.Tensor, snr_db: float = 5.0) -> torch.Tensor:
        fg_power = foreground.pow(2).mean()
        bg_power = background.pow(2).mean()
        if bg_power == 0 or fg_power == 0:
            return foreground + background
        desired_bg_power = fg_power / (10 ** (snr_db / 10))
        scaling_factor = torch.sqrt(desired_bg_power / (bg_power + 1e-8))
        return foreground + (background * scaling_factor)

    def __len__(self):
        return len(self.filtered_samples)

    def __getitem__(self, idx):
        fg_item = self.filtered_samples[idx]
        fg_waveform = self._pad_or_truncate(torch.tensor(fg_item["audio"]["array"], dtype=torch.float32))
        
        bg_item = random.choice(self.esc50)
        bg_waveform = self._pad_or_truncate(torch.tensor(bg_item["audio"]["array"], dtype=torch.float32))

        snr = random.uniform(0.0, 10.0)
        mixed_waveform = self._mix_signals(fg_waveform, bg_waveform, snr_db=snr)

        multi_hot_label = torch.zeros(len(self.target_classes), dtype=torch.float32)
        multi_hot_label[self.target_classes.index(fg_item["category"])] = 1.0

        inputs = self.feature_extractor(mixed_waveform.numpy(), sampling_rate=self.sample_rate, return_tensors="pt")
        return {"input_values": inputs.input_values.squeeze(0), "labels": multi_hot_label}

```

### 2. Fine-Tuned AST Backbone (`src/models/ast_model.py`)

Adapts the Audio Spectrogram Transformer for multi-label binary output classification:

```python
import torch
import torch.nn as nn
from transformers import ASTForAudioClassification

class ForestThreatAST(nn.Module):
    def __init__(self, num_classes):
        super(ForestThreatAST, self).__init__()
        self.ast = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            ignore_mismatched_sizes=True
        )
        hidden_size = self.ast.classifier.dense.in_features
        self.ast.classifier.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, input_values):
        return self.ast(input_values=input_values).logits

```

---

## 📊 Evaluation & Mathematical Threshold Calibration

In production deployment, static `0.50` decision thresholds fail due to class imbalance and differing cost profiles. We tune operational thresholds per class using the $F_\beta$-score:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$

* **Threat Classes (Gunshots / Chainsaws):** Optimized with $\beta = 2.0$ ($F_2$-score) to penalize False Negatives heavily and maximize Recall.
* **Bioacoustics & Species Calls:** Optimized with $\beta = 1.0$ ($F_1$-score) for balanced precision-recall trade-offs.

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

def find_optimal_thresholds(y_true, y_probs, target_classes):
    optimal_thresholds = {}
    for i, class_name in enumerate(target_classes):
        precisions, recalls, thresholds = precision_recall_curve(y_true[:, i], y_probs[:, i])
        beta = 2.0 if class_name in ['chainsaw', 'gunshot'] else 1.0
        f_beta = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls + 1e-8)
        
        best_idx = np.argmax(f_beta)
        optimal_thresholds[class_name] = thresholds[best_idx]
        print(f"[{class_name}] Optimal Threshold: {thresholds[best_idx]:.3f} (Max F{beta}: {f_beta[best_idx]:.3f})")
    return optimal_thresholds

```

---

## 🗺️ Roadmap

* [x] Complete comprehensive literature survey across 8 foundational audio & bioacoustic papers
* [x] Define multi-label AI Systems Design architecture & data strategy
* [x] Implement dynamic Polyphonic Audio Synthesizer prototype with SNR signal mixing
* [x] Build Hugging Face AST multi-label PyTorch backbone wrapper (`ForestThreatAST`)
* [x] Develop mathematical $F_\beta$-score threshold calibration logic
* [ ] Finalize site-disjoint dataset splitting (train/val/test by geographic device location)
* [ ] Train end-to-end AST model with SpecAugment on blended RFCx + Zenodo + Gunshot datasets
* [ ] Build sliding-window inference pipeline for continuous audio streams
* [ ] Perform quantization/pruning feasibility study for edge deployment (Raspberry Pi/ESP32)

---

## 📚 References

1. Gong, Y., Chung, Y. A., & Glass, J. (2021). *AST: Audio Spectrogram Transformer.* IEEE Interspeech 2021. [arXiv:2104.01778](https://arxiv.org/abs/2104.01778)
2. Park, D. S., et al. (2019). *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.* IEEE Interspeech 2019. [arXiv:1904.08779](https://arxiv.org/abs/1904.08779)
3. Kong, Q., et al. (2020). *PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition.* IEEE/ACM TASLP. [arXiv:1912.10211](https://arxiv.org/abs/1912.10211)
4. Colonna, J. G., et al. (2016). *A Framework for Chainsaw Detection using One-Class Kernel and WASN into Amazon Rainforest.* IEEE SMC.
5. Dufourq, E., et al. (2021). *Passive acoustic monitoring of animal populations with transfer learning.* Ecological Indicators.
6. Rainforest Connection (RFCx) Arbimon: [arbimon.org](https://arbimon.org)

---

## 📄 License

Distributed under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```
