# Phase 6 — Attention Fusion Training & Evaluation Master Report

> **Project:** ICIMCPS-2026 Bone Cancer Classification  
> **Phase:** Phase 6 (Attention Fusion Mechanism Integration & Baseline Model Synergy Evaluation)  
> **Status:** COMPLETED & FULLY VERIFIED  
> **Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU  
> **Split Protocol:** `derived_clean` (Leakage-Clean Split)  
> **Report Path:** `reports/phase6_attention_fusion_report.md`

---

## 1. Executive Summary

Phase 6 Attention Fusion has been successfully implemented, trained, and evaluated. In accordance with the experimental design, the standalone CNN (`BoneCancerCNN`), Swin-Tiny (`BoneCancerSwin`), and Mamba (`BoneCancerMamba`) branches were frozen, and their 256-D feature representations extracted. An Adaptive Softmax Branch Attention mechanism was trained alongside a 2-class classification head to dynamic weight and fuse the multi-modal representations ($f_{fused} = \sum_{k} \alpha_k f_k$).

The fusion model was trained on the `derived_clean` split ($7,417$ training images, $698$ validation images, $695$ untouched test images). The best checkpoint was selected strictly based on validation performance (Epoch 1: **98.57% Val Acc**).

---

## 2. Learned Branch Attention Weight Contributions

The adaptive attention mechanism learned the relative diagnostic contribution of each branch on the test set:

| Branch | Backbone Architecture | Mean Attention Weight (%) | Cancer Class Weight (%) | Normal Class Weight (%) |
|---|---|---|---|---|
| **CNN** | ResNet-18 | **29.74%** | 38.77% | 18.65% |
| **Swin** | Swin-Tiny | **52.75%** | 38.90% | 69.74% |
| **Mamba** | Selective SSM | **17.52%** | 22.33% | 11.61% |

---

## 3. Final Evaluation Metrics (Untouched Test Set: 695 Samples)

| Metric Category | Metric | Score / Value |
|---|---|---|
| **Overall Performance** | **Test Accuracy** | **98.85%** ($687 / 695$ correct) |
| | **Macro F1-Score** | **0.9884** |
| | **Macro Precision** | **0.9887** |
| | **Macro Recall** | **0.9881** |
| **Statistical Agreement** | **Cohen’s Kappa ($\kappa$)** | **0.9767** |
| | **Matthews Correlation Coef (MCC)** | **0.9767** |
| **Discriminative Ability** | **ROC-AUC Score** | **0.9992** |
| | **PR-AUC Score (Avg Precision)** | **0.9991** |
| **Timing & Latency** | **Total Training Duration** | **0.11 minutes** (6.6 seconds) |
| | **Inference Latency** | **31.32 ms / sample** |
| | **Inference Throughput** | **31.9 FPS** (CUDA) |

---

## 4. Detailed Confusion Matrix Breakdown (Positive Class = Cancer [0])

```text
                           Predicted Cancer (0)    Predicted Normal (1)
True Cancer (0) (n=383)           380 (TP)                3 (FN)
True Normal (1) (n=312)             5 (FP)               307 (TN)
```

- **True Positives (TP)**: **380** (True Cancer correctly predicted as Cancer)
- **False Negatives (FN)**: **3** (True Cancer incorrectly predicted as Normal)
- **False Positives (FP)**: **5** (True Normal incorrectly predicted as Cancer)
- **True Negatives (TN)**: **307** (True Normal correctly predicted as Normal)

| Class Name | Label | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |
|---|---|---|---|---|---|---|
| **Cancer** | `0` | **98.70%** | **99.22%** | 98.40% | **0.9896** | 383 |
| **Normal** | `1` | **99.03%** | **98.40%** | 99.22% | **0.9871** | 312 |
| **Macro Average** | — | **98.87%** | **98.81%** | **98.81%** | **0.9884** | 695 |

---

## 5. HDF5 (`.h5`) Container Export & Verification

The HDF5 model container `checkpoints/attention_fusion_best.h5` was generated from `checkpoints/attention_fusion_best.pth` and verified:
- **Export Path:** [`checkpoints/attention_fusion_best.h5`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/checkpoints/attention_fusion_best.h5)
- **Container Size:** **111.47 MB**
- **Verification Status:** **PASSED** ($0$ numerical discrepancies against PyTorch `.pth` checkpoint).

---

## 6. Mandatory Phase 6 Artifact Inventory

All mandatory Phase 6 deliverables have been created and verified:

1. **PyTorch Checkpoint:** [`checkpoints/attention_fusion_best.pth`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/checkpoints/attention_fusion_best.pth)
2. **HDF5 Model Export:** [`checkpoints/attention_fusion_best.h5`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/checkpoints/attention_fusion_best.h5)
3. **Branch Contributions JSON:** [`results/attention_fusion_branch_contributions.json`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/results/attention_fusion_branch_contributions.json)
4. **Branch Contributions Plot:** [`figures/attention_fusion_branch_contributions.png`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/figures/attention_fusion_branch_contributions.png)
5. **Publication-Ready Confusion Matrix Plot:** [`figures/attention_fusion_confusion_matrix.png`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/figures/attention_fusion_confusion_matrix.png)
6. **ROC & PR Curves Plot:** [`figures/attention_fusion_roc_pr_curves.png`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/figures/attention_fusion_roc_pr_curves.png)
7. **Classification Report (TXT):** [`results/attention_fusion_classification_report.txt`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/results/attention_fusion_classification_report.txt)
8. **Classification Report (Markdown):** [`reports/attention_fusion_classification_report.md`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/reports/attention_fusion_classification_report.md)
9. **Master Metrics JSON:** [`results/attention_fusion_evaluation_results.json`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/results/attention_fusion_evaluation_results.json)
10. **Phase 6 Master Report:** [`reports/phase6_attention_fusion_report.md`](file:///C:/Users/admin_fix/Downloads/BONE CANCER ICIMCPS/reports/phase6_attention_fusion_report.md)
