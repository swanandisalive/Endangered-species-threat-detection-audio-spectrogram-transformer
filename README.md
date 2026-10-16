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



## Trained model links
<a href="https://drive.google.com/drive/folders/1stzPrd-3p-bQXTVcSlXFV1bTPMGbNJvj?usp=drive_link" target="_blank">
  <img src="https://img.shields.io/badge/Google%20Drive-View%20Files-blue?style=for-the-badge&logo=googledrive&logoColor=white" alt="Google Drive"/>
</a>

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
| **Data Sources** | Zenodo FSD50K, UrbanSound8K |
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
