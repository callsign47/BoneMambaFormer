# BoneMambaFormer: Adaptive Attention Fusion of MobileNetV2, Swin-Tiny, and Mamba Backbones for Leakage-Clean Bone Cancer Classification (ICIMCPS-2026)

Official PyTorch implementation of **BoneMambaFormer**, a tri-branch hybrid deep learning architecture for high-precision bone cancer classification (Normal vs. Cancer) targeting **ICIMCPS-2026**.

---

## 📌 Architecture Overview

BoneMambaFormer integrates three complementary feature extraction backbones into a unified 256-D embedding space, fused via a **Dynamic Softmax Attention Fusion** module:

1. **CNN Branch (MobileNetV2 / ResNet-18)**: Extracts high-resolution localized spatial features, micro-textures, and lesion boundary contours.
2. **Vision Transformer Branch (Swin-Tiny)**: Captures multi-scale global context and non-local spatial dependencies via Shifted Windows.
3. **State Space Model Branch (Pure Mamba / SSM)**: Models selective sequential long-range spatial context with linear computational complexity.

```text
                                  +-----------------------+
                                  | Input Radiograph      |
                                  | (224 x 224 x 3 RGB)   |
                                  +-----------+-----------+
                                              |
                +-----------------------------+-----------------------------+
                |                             |                             |
                v                             v                             v
  +--------------------------+  +--------------------------+  +--------------------------+
  | MobileNetV2 Conv Branch  |  | Swin-Tiny Hierarchical   |  | Mamba Selective State    |
  | Local Feature Extractor  |  | Vision Transformer       |  | Space Sequential Model   |
  +-------------+------------+  +-------------+------------+  +-------------+------------+
                |                             |                             |
                v                             v                             v
  +--------------------------+  +--------------------------+  +--------------------------+
  | Linear Projection (256D) |  | Linear Projection (256D) |  | Linear Projection (256D) |
  +-------------+------------+  +-------------+------------+  +-------------+------------+
                | (B, 256)                    | (B, 256)                    | (B, 256)
                +-----------------------------+-----------------------------+
                                              |
                                              v  Stacked Features (B, 3, 256)
                               +-------------------------------+
                               | Dynamic Softmax Branch        |
                               | Attention Module (256D -> 64D)|
                               +---------------+---------------+
                                               | Attention Weights alpha = [a_cnn, a_swin, a_mamba]
                                               v  Fused Feature (B, 256)
                               +-------------------------------+
                               | Classification MLP Head       |
                               | FC(256->128) + LN + GELU + FC |
                               +---------------+---------------+
                                               |
                                               v
                               +-------------------------------+
                               | Class Probabilities           |
                               | [Cancer (0), Normal (1)]      |
                               +-------------------------------+
```

---

## 📊 Five-Model Benchmark Matrix (Untouched Test Set: 695 Samples)

Evaluated on the leakage-clean `derived_clean` untouched test split ($383$ Cancer, $312$ Normal):

| Model Architecture | Params | Best Val Acc | Test Acc | Macro F1 | ROC-AUC | PR-AUC | Latency / Sample |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1: MobileNetV2 (CNN)** | 2.55M | 97.85% | 97.55% | 0.9753 | 0.9988 | 0.9986 | 29.40 ms |
| **M2: Swin-Tiny (ViT)** | 27.75M | 98.57% | 98.13% | 0.9811 | 0.9991 | 0.9990 | 31.19 ms |
| **M3: Mamba (SSM)** | 0.66M | 85.39% | 85.76% | 0.8573 | 0.9468 | 0.9443 | 27.93 ms |
| **M4: Attention Fusion (CNN+Swin)** | 31.04M | 98.57% | **98.85%** | **0.9884** | 0.9992 | 0.9991 | 31.32 ms |
| **M5: BoneMambaFormer (Hybrid)** | 31.04M | **99.00%** | 97.99% | 0.9797 | **0.9993** | **0.9992** | 35.15 ms |

---

## 🔬 Controlled Ablation Study Summary (Phase 9)

| Configuration | Test Accuracy | Macro F1 | ROC-AUC | PR-AUC | Kappa ($\kappa$) | Key Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Hybrid Baseline** | **98.42%** | **0.9840** | **0.9990** | **0.9987** | **0.9680** | Full joint architecture with dynamic attention gating. |
| Equal Weighting ($\alpha=1/3$) | 98.71% | 0.9869 | 0.9990 | 0.9988 | 0.9738 | Replaces dynamic attention with fixed uniform weights. |
| Swin + Mamba (w/o CNN) | 98.71% | 0.9869 | 0.9989 | 0.9987 | 0.9738 | Removes local convolutional feature branch. |
| CNN + Swin (w/o Mamba) | 98.71% | 0.9869 | 0.9992 | 0.9990 | 0.9738 | Removes Mamba SSM sequential representation branch. |
| CNN + Mamba (w/o Swin) | **69.35%** | **0.6331** | **0.9578** | **0.9530** | **0.3392** | **Catastrophic 29.07% drop** when Swin Transformer is removed. |

---

## 🛠 Project Phase Execution Commands (Phases 1–10)

### 1. Installation & Environment Setup
```bash
git clone https://github.com/callsign47/BoneMambaFormer.git
cd BoneMambaFormer
pip install -r requirements.txt
```

### 2. Phase 2: Dataset Integrity Audit & Cross-Split Leakage Check
```bash
# Run dataset structure and hash verification
python -m src.utils.dataset_audit

# Run cross-split MD5 duplicate hash audit
python -m src.check_cross_split_leak
```

### 3. Phase 3: CNN Backbone Training & Evaluation (MobileNetV2 / ResNet-18)
```bash
# Train CNN backbone
python -m src.training.train_cnn

# Evaluate CNN baseline & export HDF5 (.h5) container
python -m src.evaluation.evaluate_cnn
```

### 4. Phase 4: Swin-Tiny Vision Transformer Baseline Training & Evaluation
```bash
# Train Swin-Tiny backbone
python -m src.training.train_swin

# Evaluate Swin-Tiny baseline & export HDF5 (.h5) container
python -m src.evaluation.evaluate_swin
```

### 5. Phase 5: Pure Mamba / SSM Baseline Training & Evaluation
```bash
# Train Pure Mamba backbone
python -m src.training.train_mamba

# Evaluate Mamba baseline & export HDF5 (.h5) container
python -m src.evaluation.evaluate_mamba
```

### 6. Phase 6: Adaptive Attention Fusion Module Training & Evaluation
```bash
# Train Adaptive Attention Fusion module
python -m src.training.train_attention_fusion

# Evaluate Attention Fusion model & export HDF5 (.h5) container
python -m src.evaluation.evaluate_attention_fusion
```

### 7. Phase 7 & 8: BoneMambaFormer Fused Hybrid Training & Master Evaluation
```bash
# Train end-to-end fused hybrid model
python -m src.training.train_hybrid

# Run comprehensive master evaluation on untouched test set
python -m src.evaluation.evaluate_hybrid
```

### 8. Phase 9: Comprehensive 5-Model Ablation Study
```bash
# Execute controlled ablation study across all 5 configurations
python -m src.evaluation.run_ablation_study

# Verify ablation metrics consistency
python -m src.evaluation.verify_ablations
```

### 9. Phase 10: Evidence Package & Master Technical Report Generation
```bash
# Generate frozen JSON evidence package and LaTeX master table
python -m src.evaluation.generate_paper_evidence_package

# Build Godfather-level 47-section master markdown report
python build_final_master_report.py
```

---

## 📁 Repository Directory Structure

```text
BoneMambaFormer/
├── DATASET/                                   # Dataset directory (train, valid, test splits)
├── src/
│   ├── models/                                # Model definitions
│   │   ├── cnn.py                             # ResNet-18 & MobileNetV2 CNN backbones
│   │   ├── swin.py                            # Swin-Tiny Vision Transformer backbone
│   │   ├── mamba.py                           # Pure Mamba State Space Model backbone
│   │   ├── attention_fusion.py                # Softmax Adaptive Attention Fusion module
│   │   └── hybrid.py                          # BoneMambaFormer fused architecture
│   ├── training/                              # Training pipelines for all phases
│   │   ├── train_cnn.py
│   │   ├── train_swin.py
│   │   ├── train_mamba.py
│   │   ├── train_attention_fusion.py
│   │   └── train_hybrid.py
│   ├── evaluation/                            # Evaluation and analysis tools
│   │   ├── evaluate_cnn.py
│   │   ├── evaluate_swin.py
│   │   ├── evaluate_mamba.py
│   │   ├── evaluate_attention_fusion.py
│   │   ├── evaluate_hybrid.py
│   │   ├── run_ablation_study.py
│   │   ├── verify_ablations.py
│   │   └── generate_paper_evidence_package.py
│   └── utils/                                 # Datasets, transforms, audit & HDF5 export
│       ├── dataset.py
│       ├── preprocessing.py
│       ├── augmentation.py
│       ├── dataset_audit.py
│       └── export_h5.py
├── checkpoints/                               # Saved PyTorch (.pth) & HDF5 (.h5) models
├── results/                                   # Master metrics JSON, CSVs, and LaTeX tables
├── figures/                                   # Publication-ready plots (CM, ROC, PR, Ablation)
├── reports/                                   # Phase 2–10 Markdown technical reports
│   ├── phase2_data_pipeline_report.md
│   ├── phase3_cnn_training_report.md
│   ├── phase4_swin_training_report.md
│   ├── phase5_mamba_training_report.md
│   ├── phase6_attention_fusion_report.md
│   ├── phase7_hybrid_training_report.md
│   ├── phase8_final_evaluation_report.md
│   ├── phase9_ablation_analysis_report.md
│   ├── phase10_paper_evidence_package.md
│   └── FINAL_MASTER_TECHNICAL_REPORT.md      # 47-section master conference report
├── build_final_master_report.py              # Master Markdown report generator
├── generate_docx_report.py                    # Master DOCX report generator
├── ICIMCPS_2026_CODING_AGENT_HANDOFF(1).md    # Research PRD & technical handoff document
├── requirements.txt                           # Project python dependencies
└── README.md                                  # Repository overview and execution guide
```

---

## 📜 License & Citation

Developed for **ICIMCPS-2026** research.
```bibtex
@inproceedings{bonemambaformer2026,
  title={BoneMambaFormer: Adaptive Softmax Attention Fusion of MobileNetV2, Swin-Tiny, and Mamba Backbones for Leakage-Clean Bone Cancer Classification},
  author={ICIMCPS-2026 Research Team},
  booktitle={International Conference on Intelligent Modeling, Computing, and Pattern Recognition (ICIMCPS-2026)},
  year={2026}
}
```
