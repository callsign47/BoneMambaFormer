# Phase 4 — Swin-Tiny Training & Mandatory Deliverables Final Report

> **Project:** ICIMCPS-2026 Bone Cancer Model  
> **Phase:** Phase 4 (Swin-Tiny Model Training, Evaluation & Verification)  
> **Status:** COMPLETED & FULLY VERIFIED  
> **Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU  
> **Split Protocol:** `derived_clean` (Leakage-Clean Split)  
> **Report Path:** `reports/phase4_swin_training_report.md`

---

## 1. Executive Summary

Phase 4 Swin-Tiny training and all mandatory PRD deliverables have been finalized. The architecture consists of a Swin-Tiny (`swin_t`) backbone pretrained on ImageNet (utilizing Shifted Windows for hierarchical visual feature extraction), followed by a LayerNorm + GELU 256-D feature projection layer and a 2-class linear classification head (`cancer = 0`, `normal = 1`).

The model was trained on the `derived_clean` split ($7,417$ training images, $698$ validation images, $695$ untouched test images) using the exact same preprocessing and augmentation protocol as Phase 3 CNN. The best checkpoint was selected strictly based on validation performance (Epoch 15: **98.57% Val Acc**).

---

## 2. Final Evaluation Metrics (Untouched Test Set: 695 Samples)

| Metric Category | Metric | Score / Value |
|---|---|---|
| **Overall Performance** | **Test Accuracy** | **98.13%** ($682 / 695$ correct) |
| | **Macro F1-Score** | **0.9811** |
| | **Macro Precision** | **0.9803** |
| | **Macro Recall** | **0.9821** |
| **Statistical Agreement** | **Cohen’s Kappa ($\kappa$)** | **0.9623** |
| | **Matthews Correlation Coef (MCC)** | **0.9625** |
| **Discriminative Ability** | **ROC-AUC Score** | **0.9991** |
| | **PR-AUC Score (Avg Precision)** | **0.9990** |
| **Timing & Latency** | **Total Training Duration** | **101.13 minutes** (6067.8 seconds) |
| | **Inference Latency** | **28.66 ms / sample** |
| | **Inference Throughput** | **34.9 FPS** (CUDA) |

---

## 3. Detailed Confusion Matrix Breakdown (Positive Class = Cancer [0])

```text
                           Predicted Cancer (0)    Predicted Normal (1)
True Cancer (0) (n=383)           373 (TP)                10 (FN)
True Normal (1) (n=312)             3 (FP)               309 (TN)
```

- **True Positives (TP)**: **373** (True Cancer correctly predicted as Cancer)
- **False Negatives (FN)**: **10** (True Cancer incorrectly predicted as Normal)
- **False Positives (FP)**: **3** (True Normal incorrectly predicted as Cancer)
- **True Negatives (TN)**: **309** (True Normal correctly predicted as Normal)

| Class Name | Label | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |
|---|---|---|---|---|---|---|
| **Cancer** | `0` | **99.20%** | **97.39%** | 99.04% | **0.9829** | 383 |
| **Normal** | `1` | **96.87%** | **99.04%** | 97.39% | **0.9794** | 312 |
| **Macro Average** | — | **98.03%** | **98.21%** | **98.21%** | **0.9811** | 695 |

---

## 4. HDF5 (`.h5`) Container Export & Verification

A real HDF5 model container file `checkpoints/swin_best.h5` was generated from `checkpoints/swin_best.pth` and verified:
- **Export Path:** [`checkpoints/swin_best.h5`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/checkpoints/swin_best.h5)
- **Container Size:** **98.92 MB**
- **Verification Status:** **PASSED** ($0$ numerical discrepancies against PyTorch `.pth` checkpoint).

---

## 5. Mandatory Phase 4 Artifact Inventory

All mandatory artifacts specified in the PRD are generated, saved, and verified in the workspace:

1. **PyTorch Checkpoint:** [`checkpoints/swin_best.pth`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/checkpoints/swin_best.pth)
2. **HDF5 Model Export:** [`checkpoints/swin_best.h5`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/checkpoints/swin_best.h5)
3. **Publication-Ready Confusion Matrix Plot:** [`figures/swin_confusion_matrix.png`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/figures/swin_confusion_matrix.png)
4. **ROC Curve Plot:** [`figures/swin_roc_curve.png`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/figures/swin_roc_curve.png)
5. **Precision-Recall Curve Plot:** [`figures/swin_pr_curve.png`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/figures/swin_pr_curve.png)
6. **Standalone Classification Report (TXT):** [`results/swin_classification_report.txt`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/results/swin_classification_report.txt)
7. **Standalone Classification Report (Markdown):** [`reports/swin_classification_report.md`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/reports/swin_classification_report.md)
8. **Master Metrics JSON:** [`results/swin_evaluation_results.json`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/results/swin_evaluation_results.json)
9. **Phase 4 Master Report:** [`reports/phase4_swin_training_report.md`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/reports/phase4_swin_training_report.md)
