# Phase 10 — Publication-Ready Paper Evidence Package

> **Conference / Target:** ICIMCPS-2026  
> **Paper Title:** BoneMambaFormer: Adaptive Softmax Attention Fusion of MobileNetV2, Swin-Tiny, and Mamba Backbones for Leakage-Clean Bone Cancer Classification  
> **Status:** FROZEN, COMPLETE & SCIENTIFICALLY VERIFIED  
> **Evaluation Protocol:** `derived_clean` (Leakage-Clean Split, Untouched Test Evaluation)  
> **Exact Hybrid Parameter Scale:** **31,044,037 trainable parameters (~31.04M)**  
> **Master Artifact JSON:** [`results/paper_evidence_package.json`](file:///c:/Users/admin_fix/Downloads/BONE%20CANCER%20ICIMCPS/results/paper_evidence_package.json)  

---

## 1. Complete Five-Model Benchmarking Matrix

All evaluations were executed strictly on the untouched 695 test samples under the `derived_clean` protocol without any test-set tuning or parameter modification:

| Metric / Specification | 1. M1: MobileNetV2 | 2. M2: Swin-Tiny | 3. M3: Mamba (SSM) | 4. M4: Attention Fusion | 5. M5: End-to-End Hybrid |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Parameters** | 2.55M | 27.75M | 0.66M | 31.04M | 31.04M |
| **Trainable Parameters** | 2,549,442 | 27,746,056 | 661,060 | 49,923 | **31,044,037 (~31.04M)** |
| **Best Val Epoch** | Epoch 5 | Epoch 15 | Epoch 19 | Epoch 1 | Epoch 1 |
| **Best Val Accuracy** | 97.85% | 98.57% | 85.39% | 98.57% | **99.00%** |
| **Test Accuracy** | 97.55% | 98.13% | 85.76% | **98.85%** | 97.99% |
| **Macro Precision** | 0.9747 | 0.9803 | 0.8588 | **0.9887** | 0.9788 |
| **Macro Recall** | 0.9760 | 0.9821 | 0.8624 | **0.9881** | 0.9808 |
| **Macro F1-Score** | 0.9753 | 0.9811 | 0.8573 | **0.9884** | 0.9797 |
| **ROC-AUC** | 0.9988 | 0.9991 | 0.9468 | 0.9992 | **0.9993** |
| **PR-AUC** | 0.9986 | 0.9990 | 0.9443 | 0.9991 | **0.9992** |
| **Cohen’s Kappa ($\kappa$)** | 0.9506 | 0.9623 | 0.7157 | **0.9767** | 0.9594 |
| **Matthews Corr Coef (MCC)** | 0.9507 | 0.9625 | 0.7213 | **0.9767** | 0.9596 |
| **Inference Latency** | 29.40 ms | 31.19 ms | 27.93 ms | 31.32 ms | 35.15 ms |

---

### 1.1 Model Selection vs. Untouched Test Performance & Architectural Distinction

To maintain strict scientific integrity and prevent confusion between validation-driven model selection and locked test set evaluation:

- **Validation Performance (Model Selection):** **M5 End-to-End Hybrid** achieved the highest Best Validation Accuracy (**99.00%** at Epoch 1) during training, prompting selection of its frozen checkpoint.
- **Untouched Test Performance (Locked Evaluation):** **M4 Attention Fusion** achieved the highest Test Accuracy (**98.85%**), Macro F1-score (**0.9884**), and Cohen's Kappa (**0.9767**) on the 695-image untouched test set.
- **Discriminative Threshold Ability:** **M5 End-to-End Hybrid** achieved the highest area under the ROC curve (**0.9993**) and Precision-Recall curve (**0.9992**) across all operating thresholds.
- **Legacy ResNet-18 vs. Next-Gen MobileNetV2 Comparison:**
  - **Legacy ResNet-18 Hybrid:** Best Val Acc = **98.71%** (Epoch 16), Untouched Test Acc = **98.42%**, Parameters = ~39.74M.
  - **Next-Gen MobileNetV2 Hybrid (M5):** Best Val Acc = **99.00%** (Epoch 1), Untouched Test Acc = **97.99%**, Parameters = **31.04M**.
  - **Next-Gen Attention Fusion (M4):** Best Val Acc = **98.57%** (Epoch 1), Untouched Test Acc = **98.85%**, Parameters = **31.04M** (Trainable: 49.9K).

## 2. Phase 9 Controlled Ablation Results Summary

Evaluating architectural component ablations on the untouched test set (**Historical / Legacy ResNet-18 Baseline** — *Note: Phase 9 ablations were conducted on the legacy ResNet-18 hybrid backbone as historical architectural component evidence; they do not represent the Next-Gen MobileNetV2 architecture*):

| Configuration | Category | Test Acc (%) | Macro F1 | ROC-AUC | PR-AUC | Kappa ($\kappa$) | Impact Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Hybrid (Legacy ResNet-18)** | Baseline | **98.42%** | **0.9840** | **0.9990** | **0.9987** | **0.9680** | Full joint architecture with dynamic softmax attention. |
| **Equal Weighting ($lpha=1/3$)** | Fusion Ablation | 98.71% | 0.9869 | 0.9990 | 0.9988 | 0.9738 | Replaces dynamic attention with fixed uniform weights. |
| **Swin + Mamba (w/o CNN)** | Branch Ablation | 98.71% | 0.9869 | 0.9989 | 0.9987 | 0.9738 | Removes ResNet18 local convolutional branch. |
| **CNN + Swin (w/o Mamba)** | Branch Ablation | 98.71% | 0.9869 | 0.9992 | 0.9990 | 0.9738 | Removes Mamba SSM sequential representation branch. |
| **CNN + Mamba (w/o Swin)** | Branch Ablation | **69.35%** | **0.6331** | **0.9578** | **0.9530** | **0.3392** | **Catastrophic 29.07% drop** when Swin vision transformer is removed. |

---

## 3. Dataset Audit, Preprocessing & Leakage-Clean Split Protocol

### 3.1 Dataset Breakdown & Duplication Audit
- **Raw Total Images**: 8,810 images (Cancer: 4,863, Normal: 3,947) across 3 raw split folders.
- **Cross-Split Hash Duplicates Identified**: **1,059 duplicate MD5 image hashes** were detected spanning across original split boundaries.
- **`derived_clean` Protocol**: MD5 hash deduplication was performed to construct `derived_clean`, guaranteeing zero identical image hash overlaps between training, validation, and test splits.
- **Split Distribution**:
  - **Train Set**: 7,417 images
  - **Val Set**: 698 images
  - **Test Set (Untouched)**: 695 images (Cancer [0]: 383, Normal [1]: 312)
- **Scientific Caveat**: The original dataset lacked patient metadata. While `derived_clean` guarantees strict image hash deduplication across splits, patient-level independence cannot be claimed without explicit patient IDs.

### 3.2 Preprocessing & Data Augmentation Suite
- **Input Resolution**: $224 	imes 224 	imes 3$ RGB.
- **Normalization**: ImageNet mean $[0.485, 0.456, 0.406]$ and std $[0.229, 0.224, 0.225]$.
- **Enhancement**: Contrast Limited Adaptive Histogram Equalization (CLAHE, `clip_limit=2.0`, `tile_grid_size=(8,8)`).
- **Augmentations (Training Only)**: Random Horizontal/Vertical Flips ($p=0.5$), Random Affine Rotation ($\pm 15^\circ$), Elastic Transform ($alpha=1.0, sigma=50.0$), Color Jitter ($brightness=0.1, contrast=0.1$).

---

## 4. Architecture Description & Structural Specification

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

- **Exact Next-Gen Hybrid Model Parameters**: **31,044,037 trainable parameters (~31.04M)**.
- **Loss Function**: Cross-Entropy Loss with Label Smoothing ($\epsilon = 0.05$).
- **Optimizer**: AdamW ($eta_1=0.9, eta_2=0.999$, weight decay $1	imes 10^-4$).

---

## 5. Artifact Verification & Citation Map

| Artifact Type | File Path | Verification Status |
| :--- | :--- | :--- |
| **Hybrid PyTorch Checkpoint** | `checkpoints/hybrid_best.pth` | Verified (118.7 MB) |
| **Hybrid HDF5 Container Export** | `checkpoints/hybrid_best.h5` | Verified (118.4 MB) |
| **Master Evidence JSON** | `results/paper_evidence_package.json` | Verified |
| **Master LaTeX Table** | `results/master_paper_results_table.tex` | Verified |
| **Five-Model Master CSV** | `results/five_model_results.csv` | Verified |
| **Ablation CSV** | `results/ablation_study_results.csv` | Verified |
| **Hybrid Confusion Matrix Plot** | `figures/hybrid_confusion_matrix.png` | Verified |
| **Hybrid ROC & PR Curves Plot** | `figures/hybrid_roc_pr_curves.png` | Verified |
| **Hybrid Branch Contributions Plot** | `figures/hybrid_branch_contributions.png` | Verified |
| **Ablation Comparison Plot** | `figures/ablation_comparison.png` | Verified |
| **Branch Masking Impact Plot** | `figures/ablation_branch_masking_impact.png` | Verified |

---

## 6. Scientific Integrity & Non-Fabrication Declaration

All metrics, parameter counts, latency figures, training durations, and evaluation tables presented in this document trace 100% directly to saved PyTorch checkpoints, JSON metric files, and execution outputs in the repository workspace. No performance metrics were retuned or manually fabricated.
