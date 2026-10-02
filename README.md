# Real-Time Endangered Species & Threat Identification (SDG 15)

Monitoring vulnerable rainforest ecosystems using an **Audio Spectrogram Transformer (AST)** to detect illegal deforestation (chainsaws, engines) and poaching (gunshots) in real-time.

## 🎯 Project Overview
* **Task**: Multi-label audio classification.
* **Architecture**: Fine-tuned Hugging Face AST (Audio Spectrogram Transformer) using PyTorch.
* **Data Source**: Rainforest Connection (RFCx) Arbimon platform.
* **Data Augmentation**: SpecAugment (time and frequency masking) to mitigate background noise.
* **Evaluation Metrics**: PR-AUC (Precision-Recall Area Under Curve) and False Positive Rate (FPR) at 90% True Positive Rate.

## 👥 Team Members
* [Member 1 Name]
* [Member 2 Name]
* [Member 3 Name]
* [Member 4 Name]

## 🚀 Getting Started
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/endangered-species-sound-detection.git](https://github.com/your-username/endangered-species-sound-detection.git)
   cd endangered-species-sound-detection
