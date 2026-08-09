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

The joint model was trained on the `derived_clean` split ($7,417$ training images, $698$ validation images, $695$ untouched test images) using differential learning rates ($2\times 10^{-5}$ for backbones, $1\times 10^{-4}$ for fusion head) and Automatic Mixed Precision (AMP). The best checkpoint was selected strictly based on validation accuracy (Epoch 16: **98.71% Val Acc**).

---

## 2. Learned Branch Attention Weight Contributions

The end-to-end fine-tuned attention mechanism learned the following relative branch contribution weights on the untouched test set:

| Branch | Backbone Architecture | Mean Attention Weight (%) | Cancer Class Weight (%) | Normal Class Weight (%) |
|---|---|---|---|---|
| **CNN** | ResNet-18 | **1.14%** | 1.77% | 0.38% |
| **Swin** | Swin-Tiny | **92.61%** | 87.84% | 98.46% |
| **Mamba** | Selective SSM | **6.25%** | 10.39% | 1.16% |

---

## 3. Final Evaluation Metrics (Untouched Test Set: 695 Samples)

| Metric Category | Metric | Score / Value |
|---|---|---|
| **Overall Performance** | **Test Accuracy** | **98.42%** ($684 / 695$ correct) |
| | **Macro F1-Score** | **0.9840** |
| | **Macro Precision** | **0.9839** |
| | **Macro Recall** | **0.9842** |
| **Statistical Agreement** | **Cohen’s Kappa ($\kappa$)** | **0.9680** |
| | **Matthews Correlation Coef (MCC)** | **0.9680** |
| **Discriminative Ability** | **ROC-AUC Score** | **0.9990** |
| | **PR-AUC Score (Avg Precision)** | **0.9987** |
| **Timing & Latency** | **Total Training Duration** | **88.65 minutes** (5319.3 seconds) |
| | **Inference Latency** | **33.55 ms / sample** |
| | **Inference Throughput** | **29.8 FPS** (CUDA) |

---

## 4. Detailed Confusion Matrix Breakdown (Positive Class = Cancer [0])

```text
                           Predicted Cancer (0)    Predicted Normal (1)
True Cancer (0) (n=383)           377 (TP)                6 (FN)
True Normal (1) (n=312)             5 (FP)               307 (TN)
```

- **True Positives (TP)**: **377** (True Cancer correctly predicted as Cancer)
- **False Negatives (FN)**: **6** (True Cancer incorrectly predicted as Normal)
- **False Positives (FP)**: **5** (True Normal incorrectly predicted as Cancer)
- **True Negatives (TN)**: **307** (True Normal correctly predicted as Normal)

| Class Name | Label | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |
|---|---|---|---|---|---|---|
| **Cancer** | `0` | **98.69%** | **98.43%** | 98.40% | **0.9856** | 383 |
| **Normal** | `1` | **98.08%** | **98.40%** | 98.43% | **0.9824** | 312 |
| **Macro Average** | — | **98.39%** | **98.42%** | **98.42%** | **0.9840** | 695 |

---

## 5. HDF5 (`.h5`) Container Export & Verification

The HDF5 model container `checkpoints/hybrid_best.h5` was generated from `checkpoints/hybrid_best.pth` and verified:
- **Export Path:** [`checkpoints/hybrid_best.h5`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/checkpoints/hybrid_best.h5)
- **Container Size:** **142.34 MB**
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
