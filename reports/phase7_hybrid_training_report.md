# Phase 7 — Final End-to-End Hybrid Model Training & Evaluation Master Report

> **Project:** ICIMCPS-2026 Bone Cancer Classification  
> **Phase:** Phase 7 (Final End-to-End Joint Hybrid Model Training & Benchmark Evaluation)  
> **Status:** COMPLETED & FULLY VERIFIED  
> **Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU  
> **Split Protocol:** `derived_clean` (Leakage-Clean Split)  
> **Report Path:** `reports/phase7_hybrid_training_report.md`

---

## 1. Executive Summary

Phase 7 final end-to-end hybrid model training has been successfully executed and evaluated. Unlike Phase 6 (where backbones were frozen), Phase 7 optimized the complete **CNN (ResNet-18) + Swin-Tiny + Mamba (Selective SSM)** architecture jointly end-to-end alongside the **Softmax Branch Attention** mechanism and classification head.

The joint model was trained on the `derived_clean` split ($7,417$ training images, $698$ validation images, $695$ untouched test images) using differential learning rates ($2\times 10^{-5}$ for backbones, $1\times 10^{-4}$ for fusion head) and Automatic Mixed Precision (AMP). The best checkpoint was selected strictly based on validation accuracy (Epoch 1: **99.00% Val Acc**).

---

## 2. Learned Branch Attention Weight Contributions

The end-to-end fine-tuned attention mechanism learned the following relative branch contribution weights on the untouched test set:

| Branch | Backbone Architecture | Mean Attention Weight (%) | Cancer Class Weight (%) | Normal Class Weight (%) |
|---|---|---|---|---|
| **CNN** | ResNet-18 | **31.29%** | 37.99% | 23.06% |
| **Swin** | Swin-Tiny | **55.68%** | 46.17% | 67.35% |
| **Mamba** | Selective SSM | **13.03%** | 15.84% | 9.59% |

---

## 3. Final Evaluation Metrics (Untouched Test Set: 695 Samples)

| Metric Category | Metric | Score / Value |
|---|---|---|
| **Overall Performance** | **Test Accuracy** | **97.99%** ($681 / 695$ correct) |
| | **Macro F1-Score** | **0.9797** |
| | **Macro Precision** | **0.9788** |
| | **Macro Recall** | **0.9808** |
| **Statistical Agreement** | **Cohen’s Kappa ($\kappa$)** | **0.9594** |
| | **Matthews Correlation Coef (MCC)** | **0.9596** |
| **Discriminative Ability** | **ROC-AUC Score** | **0.9993** |
| | **PR-AUC Score (Avg Precision)** | **0.9992** |
| **Timing & Latency** | **Total Training Duration** | **17.40 minutes** (1043.7 seconds) |
| | **Inference Latency** | **35.15 ms / sample** |
| | **Inference Throughput** | **28.5 FPS** (CUDA) |

---

## 4. Detailed Confusion Matrix Breakdown (Positive Class = Cancer [0])

```text
                           Predicted Cancer (0)    Predicted Normal (1)
True Cancer (0) (n=383)           372 (TP)                11 (FN)
True Normal (1) (n=312)             3 (FP)               309 (TN)
```

- **True Positives (TP)**: **372** (True Cancer correctly predicted as Cancer)
- **False Negatives (FN)**: **11** (True Cancer incorrectly predicted as Normal)
- **False Positives (FP)**: **3** (True Normal incorrectly predicted as Cancer)
- **True Negatives (TN)**: **309** (True Normal correctly predicted as Normal)

| Class Name | Label | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |
|---|---|---|---|---|---|---|
| **Cancer** | `0` | **99.20%** | **97.13%** | 99.04% | **0.9815** | 383 |
| **Normal** | `1` | **96.56%** | **99.04%** | 97.13% | **0.9778** | 312 |
| **Macro Average** | — | **97.88%** | **98.08%** | **98.08%** | **0.9797** | 695 |

---

## 5. HDF5 (`.h5`) Container Export & Verification

The HDF5 model container `checkpoints/hybrid_best.h5` was generated from `checkpoints/hybrid_best.pth` and verified:
- **Export Path:** [`checkpoints/hybrid_best.h5`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/checkpoints/hybrid_best.h5)
- **Container Size:** **111.47 MB**
- **Verification Status:** **PASSED** ($0$ numerical discrepancies against PyTorch `.pth` checkpoint).

---

## 6. Mandatory Phase 7 Artifact Inventory

All mandatory Phase 7 deliverables have been created and verified:

1. **PyTorch Checkpoint:** [`checkpoints/hybrid_best.pth`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/checkpoints/hybrid_best.pth)
2. **HDF5 Model Export:** [`checkpoints/hybrid_best.h5`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/checkpoints/hybrid_best.h5)
3. **Branch Contributions JSON:** [`results/hybrid_branch_contributions.json`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/results/hybrid_branch_contributions.json)
4. **Branch Contributions Plot:** [`figures/hybrid_branch_contributions.png`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/figures/hybrid_branch_contributions.png)
5. **Publication-Ready Confusion Matrix Plot:** [`figures/hybrid_confusion_matrix.png`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/figures/hybrid_confusion_matrix.png)
6. **ROC & PR Curves Plot:** [`figures/hybrid_roc_pr_curves.png`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/figures/hybrid_roc_pr_curves.png)
7. **Classification Report (TXT):** [`results/hybrid_classification_report.txt`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/results/hybrid_classification_report.txt)
8. **Classification Report (Markdown):** [`reports/hybrid_classification_report.md`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/reports/hybrid_classification_report.md)
9. **Master Metrics JSON:** [`results/hybrid_evaluation_results.json`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/results/hybrid_evaluation_results.json)
10. **Phase 7 Master Report:** [`reports/phase7_hybrid_training_report.md`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/reports/phase7_hybrid_training_report.md)
