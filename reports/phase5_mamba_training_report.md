# Phase 5 — Mamba Standalone Training & Mandatory Deliverables Final Report

> **Project:** ICIMCPS-2026 Bone Cancer Model  
> **Phase:** Phase 5 (Mamba Standalone Baseline Model Training, Evaluation & Verification)  
> **Status:** COMPLETED & FULLY VERIFIED  
> **Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU  
> **Split Protocol:** `derived_clean` (Leakage-Clean Split)  
> **Report Path:** `reports/phase5_mamba_training_report.md`

---

## 1. Executive Summary

Phase 5 Mamba standalone baseline model training and all mandatory PRD deliverables have been finalized. The architecture consists of a Pure PyTorch Selective State Space Model (`BoneCancerMamba`) with patch embedding (16x16 patch size -> 196 tokens), 1D positional encoding, 2 sequential Selective SSM blocks, LayerNorm + GELU 256-D feature projection, and a 2-class linear classification head (`cancer = 0`, `normal = 1`).

The model was trained on the `derived_clean` split ($7,417$ training images, $698$ validation images, $695$ untouched test images) using the exact same preprocessing and augmentation protocol as Phase 3 CNN and Phase 4 Swin-Tiny. The best checkpoint was selected strictly based on validation performance (Epoch 19: **85.39% Val Acc**).

---

## 2. Final Evaluation Metrics (Untouched Test Set: 695 Samples)

| Metric Category | Metric | Score / Value |
|---|---|---|
| **Overall Performance** | **Test Accuracy** | **85.76%** ($596 / 695$ correct) |
| | **Macro F1-Score** | **0.8573** |
| | **Macro Precision** | **0.8588** |
| | **Macro Recall** | **0.8624** |
| **Statistical Agreement** | **Cohen’s Kappa ($\kappa$)** | **0.7157** |
| | **Matthews Correlation Coef (MCC)** | **0.7213** |
| **Discriminative Ability** | **ROC-AUC Score** | **0.9468** |
| | **PR-AUC Score (Avg Precision)** | **0.9443** |
| **Timing & Latency** | **Total Training Duration** | **88.06 minutes** (5283.8 seconds) |
| | **Inference Latency** | **29.78 ms / sample** |
| | **Inference Throughput** | **33.6 FPS** (CUDA) |

---

## 3. Detailed Confusion Matrix Breakdown (Positive Class = Cancer [0])

```text
                           Predicted Cancer (0)    Predicted Normal (1)
True Cancer (0) (n=383)           312 (TP)                71 (FN)
True Normal (1) (n=312)             28 (FP)               284 (TN)
```

- **True Positives (TP)**: **312** (True Cancer correctly predicted as Cancer)
- **False Negatives (FN)**: **71** (True Cancer incorrectly predicted as Normal)
- **False Positives (FP)**: **28** (True Normal incorrectly predicted as Cancer)
- **True Negatives (TN)**: **284** (True Normal correctly predicted as Normal)

| Class Name | Label | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |
|---|---|---|---|---|---|---|
| **Cancer** | `0` | **91.76%** | **81.46%** | 91.03% | **0.8631** | 383 |
| **Normal** | `1` | **80.00%** | **91.03%** | 81.46% | **0.8516** | 312 |
| **Macro Average** | — | **85.88%** | **86.24%** | **86.24%** | **0.8573** | 695 |

---

## 4. HDF5 (`.h5`) Container Export & Verification

A real HDF5 model container file `checkpoints/mamba_best.h5` was generated from `checkpoints/mamba_best.pth` and verified:
- **Export Path:** [`checkpoints/mamba_best.h5`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/checkpoints/mamba_best.h5)
- **Container Size:** **2.43 MB**
- **Verification Status:** **PASSED** ($0$ numerical discrepancies against PyTorch `.pth` checkpoint).

---

## 5. Mandatory Phase 5 Artifact Inventory

All mandatory artifacts specified in the PRD are generated, saved, and verified in the workspace:

1. **PyTorch Checkpoint:** [`checkpoints/mamba_best.pth`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/checkpoints/mamba_best.pth)
2. **HDF5 Model Export:** [`checkpoints/mamba_best.h5`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/checkpoints/mamba_best.h5)
3. **Publication-Ready Confusion Matrix Plot:** [`figures/mamba_confusion_matrix.png`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/figures/mamba_confusion_matrix.png)
4. **ROC Curve Plot:** [`figures/mamba_roc_curve.png`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/figures/mamba_roc_curve.png)
5. **Precision-Recall Curve Plot:** [`figures/mamba_pr_curve.png`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/figures/mamba_pr_curve.png)
6. **Standalone Classification Report (TXT):** [`results/mamba_classification_report.txt`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/results/mamba_classification_report.txt)
7. **Standalone Classification Report (Markdown):** [`reports/mamba_classification_report.md`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/reports/mamba_classification_report.md)
8. **Master Metrics JSON:** [`results/mamba_evaluation_results.json`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/results/mamba_evaluation_results.json)
9. **Phase 5 Master Report:** [`reports/phase5_mamba_training_report.md`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/reports/phase5_mamba_training_report.md)
