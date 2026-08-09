# Hybrid Bone Cancer Classification (ICIMCPS-2026)

This repository contains the official implementation of the **Hybrid CNN + Swin Transformer + Mamba Architecture with Adaptive Attention Fusion** for high-precision bone cancer (Normal vs. Cancer) classification, targeted for **ICIMCPS-2026**.

## 📌 Architecture Overview

The framework combines three complementary feature extraction backbones:
1. **CNN Branch**: Captures localized structural features, edges, textures, and lesion boundaries.
2. **Swin Transformer Branch**: Captures global context and spatial dependencies using shifted windows.
3. **Mamba / SSM Branch**: Efficient sequential and long-range feature modeling via State Space Models.

Features from all three branches are projected into a unified embedding space and fused using an **Adaptive Attention Fusion** mechanism prior to refinement and classification.

```text
[Input Image (640x640)] 
       │
       ├─► CNN Branch ──────────► [Feature Vector (256-D)] ──┐
       ├─► Swin-Tiny Branch ────► [Feature Vector (256-D)] ──┼─► [Adaptive Attention Fusion] ──► [Feature Refinement] ──► [Classifier] ──► Normal / Cancer
       └─► Mamba Branch ────────► [Feature Vector (256-D)] ──┘
```

## 📁 Repository Structure

```text
BONE CANCER ICIMCPS/
├── DATASET/                   # Labeled dataset (train, valid, test splits)
├── src/                       # Source code directory
│   ├── models/                # Backbone architectures & fusion module
│   ├── training/              # Training scripts & utilities
│   ├── evaluation/            # Metrics, ablation study & evaluation scripts
│   └── utils/                 # Dataset loaders, preprocessing & visualization
├── configs/                   # Model and training hyperparameter configs
├── checkpoints/               # Model weights & saved checkpoints
├── ICIMCPS_2026_CODING_AGENT_HANDOFF(1).md  # Detailed research PRD & specification
├── requirements.txt           # Project dependencies
└── README.md                  # Project overview and setup instructions
```

## 📊 Dataset Summary

- **Total Images**: 8,811 (JPG, 640×640 RGB)
- **Splits**:
  - **Train**: 7,057 images (3,976 Normal / 3,081 Cancer)
  - **Validation**: 882 images (484 Normal / 398 Cancer)
  - **Test**: 872 images (488 Normal / 384 Cancer)
- **Task**: Binary Classification (`0: Cancer`, `1: Normal`)

## 🚀 Setup & Usage

### 1. Installation

```bash
git clone <repository-url>
cd BONE-CANCER-ICIMCPS
pip install -r requirements.txt
```

### 2. Dataset Verification

Run the dataset audit script to verify dataset integrity and split distribution:

```bash
python -m src.utils.dataset_audit
```

## 📜 License

This project is developed for ICIMCPS-2026 research work.
