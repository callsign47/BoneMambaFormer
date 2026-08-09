# Phase 3 — CNN Training & Mandatory Deliverables Final Report

> **Project:** ICIMCPS-2026 Bone Cancer Model  
> **Phase:** Phase 3 (CNN Model Training, Deliverables & Verification)  
> **Status:** COMPLETED & FULLY VERIFIED  
> **Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU  
> **Split Protocol:** `derived_clean` (Leakage-Clean Split)  
> **Report Path:** `reports/phase3_cnn_training_report.md`

---

## 1. Executive Summary

Phase 3 CNN training and all mandatory PRD deliverables have been finalized. The architecture consists of a ResNet-18 backbone pretrained on ImageNet, followed by a LayerNorm + GELU 256-D feature projection layer and a 2-class linear classification head (`cancer = 0`, `normal = 1`).

The model was trained on the `derived_clean` split ($7,417$ training images, $698$ validation images, $695$ untouched test images). The best checkpoint was selected strictly based on validation performance (Epoch 7: **97.71% Val Acc**).

---

## 2. Final Evaluation Metrics (Untouched Test Set: 695 Samples)

| Metric Category | Metric | Score / Value |
|---|---|---|
| **Overall Performance** | **Test Accuracy** | **97.27%** ($676 / 695$ correct) |
| | **Macro F1-Score** | **0.9724** |
| | **Macro Precision** | **0.9722** |
| | **Macro Recall** | **0.9725** |
| **Statistical Agreement** | **Cohen’s Kappa ($\kappa$)** | **0.9448** (Near-perfect agreement) |
| | **Matthews Correlation Coef (MCC)** | **0.9448** |
| **Discriminative Ability** | **ROC-AUC Score** | **0.9977** |
| | **PR-AUC Score (Avg Precision)** | **0.9972** |
| **Timing & Latency** | **Total Training Duration** | **28.64 minutes** ($1,718.5$ seconds) |
| | **Inference Latency** | **29.38 ms / sample** |
| | **Inference Throughput** | **34.0 FPS** (CUDA) |

---

## 3. Detailed Confusion Matrix Breakdown (Positive Class = Cancer [0])

```text
                           Predicted Cancer (0)    Predicted Normal (1)
True Cancer (0) (n=383)           373 (TP)                10 (FN)
True Normal (1) (n=312)             9 (FP)               303 (TN)
```

- **True Positives (TP)**: **373** (True Cancer correctly predicted as Cancer)
- **False Negatives (FN)**: **10** (True Cancer incorrectly predicted as Normal)
- **False Positives (FP)**: **9** (True Normal incorrectly predicted as Cancer)
- **True Negatives (TN)**: **303** (True Normal correctly predicted as Normal)

| Class Name | Label | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |
|---|---|---|---|---|---|---|
| **Cancer** | `0` | **97.64%** | **97.39%** | 97.12% | **0.9752** | 383 |
| **Normal** | `1` | **96.81%** | **97.12%** | 97.39% | **0.9696** | 312 |
| **Macro Average** | — | **97.22%** | **97.25%** | **97.25%** | **0.9724** | 695 |

---

## 4. HDF5 (`.h5`) Container Export & Verification

A real HDF5 model container file `checkpoints/cnn_best.h5` was generated from `checkpoints/cnn_best.pth` and verified:
- **Export Path:** [`checkpoints/cnn_best.h5`](file:///c:/Users/admin_fix/Downloads/BONE%20CANCER%20ICIMCPS/checkpoints/cnn_best.h5)
- **Container Size:** **40.79 MB**
- **Stored Layers/Datasets:** $126$ tensor keys ($11,318,486$ parameters)
- **Verification Status:** **PASSED** ($0$ numerical discrepancies against PyTorch `.pth` checkpoint).

---

## 5. Mandatory Phase 3 Artifact Inventory

All mandatory artifacts specified in the PRD are generated, saved, and verified in the workspace:

1. **PyTorch Checkpoint:** [`checkpoints/cnn_best.pth`](file:///c:/Users/admin_fix/Downloads/BONE%20CANCER%20ICIMCPS/checkpoints/cnn_best.pth)
2. **HDF5 Model Export:** [`checkpoints/cnn_best.h5`](file:///c:/Users/admin_fix/Downloads/BONE%20CANCER%20ICIMCPS/checkpoints/cnn_best.h5)
3. **Publication-Ready Confusion Matrix Plot:** [`figures/cnn_confusion_matrix.png`](file:///c:/Users/admin_fix/Downloads/BONE%20CANCER%20ICIMCPS/figures/cnn_confusion_matrix.png)
4. **ROC Curve Plot:** [`figures/cnn_roc_curve.png`](file:///c:/Users/admin_fix/Downloads/BONE%20CANCER%20ICIMCPS/figures/cnn_roc_curve.png)
5. **Precision-Recall Curve Plot:** [`figures/cnn_pr_curve.png`](file:///c:/Users/admin_fix/Downloads/BONE%20CANCER%20ICIMCPS/figures/cnn_pr_curve.png)
6. **Standalone Classification Report (TXT):** [`results/cnn_classification_report.txt`](file:///c:/Users/admin_fix/Downloads/BONE%20CANCER%20ICIMCPS/results/cnn_classification_report.txt)
7. **Standalone Classification Report (Markdown):** [`reports/cnn_classification_report.md`](file:///c:/Users/admin_fix/Downloads/BONE%20CANCER%20ICIMCPS/reports/cnn_classification_report.md)
8. **Master Metrics JSON:** [`results/cnn_evaluation_results.json`](file:///c:/Users/admin_fix/Downloads/BONE%20CANCER%20ICIMCPS/results/cnn_evaluation_results.json)
9. **Phase 3 Master Report:** [`reports/phase3_cnn_training_report.md`](file:///c:/Users/admin_fix/Downloads/BONE%20CANCER%20ICIMCPS/reports/phase3_cnn_training_report.md)
