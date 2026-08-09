# Attention Fusion Classification Report (Derived Clean Test Set)

- **Model**: BoneCancerAttentionFusion (Frozen CNN + Swin + Mamba + Softmax Branch Attention)
- **Evaluated Checkpoint**: `C:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\checkpoints\attention_fusion_best.pth`
- **HDF5 Export**: `C:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\checkpoints\attention_fusion_best.h5`
- **Test Set Size**: 695 samples

## Branch Contribution Weights

| Branch | Architecture | Overall Weight (%) | Cancer Class Weight (%) | Normal Class Weight (%) |
|---|---|---|---|---|
| **CNN** | ResNet-18 | **4.48%** | 6.48% | 2.02% |
| **Swin** | Swin-Tiny | **78.27%** | 64.54% | 95.11% |
| **Mamba** | Selective SSM | **17.26%** | 28.98% | 2.86% |

## Confusion Matrix & Terminology (Positive Class = Cancer [0])

| Term | Full Name | Definition | Sample Count |
|---|---|---|---|
| **TP** | True Positive | True Cancer predicted as Cancer | **379** |
| **FN** | False Negative | True Cancer predicted as Normal | **4** |
| **FP** | False Positive | True Normal predicted as Cancer | **4** |
| **TN** | True Negative | True Normal predicted as Normal | **308** |

## Performance Summary

| Metric | Score |
|---|---|
| **Test Accuracy** | **98.85%** |
| **Macro Precision** | **0.9884** |
| **Macro Recall** | **0.9884** |
| **Macro F1-Score** | **0.9884** |
| **Cohen's Kappa** | **0.9767** |
| **Matthews Correlation Coefficient (MCC)** | **0.9767** |
| **ROC-AUC** | **0.9978** |
| **PR-AUC** | **0.9971** |
| **Inference Latency** | **40.98 ms / sample** |
| **Inference Throughput** | **24.4 FPS** |

## Per-Class Breakdown

| Class | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |
|---|---|---|---|---|---|
| **Cancer (0)** | 0.9896 | 0.9896 | 0.9872 | 0.9896 | 383 |
| **Normal (1)** | 0.9872 | 0.9872 | 0.9896 | 0.9872 | 312 |
