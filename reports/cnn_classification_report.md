# CNN Classification Report (Derived Clean Test Set)

- **Model**: BoneCancerCNN (MobileNetV2 Backbone + 256D Feature Projection)
- **Evaluated Checkpoint**: `C:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\checkpoints\cnn_best.pth`
- **HDF5 Export**: `C:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\checkpoints\cnn_best.h5`
- **Test Set Size**: 695 samples

## Confusion Matrix & Terminology (Positive Class = Cancer [0])

| Term | Full Name | Definition | Sample Count |
|---|---|---|---|
| **TP** | True Positive | True Cancer predicted as Cancer | **372** |
| **FN** | False Negative | True Cancer predicted as Normal | **11** |
| **FP** | False Positive | True Normal predicted as Cancer | **6** |
| **TN** | True Negative | True Normal predicted as Normal | **306** |

## Performance Summary

| Metric | Score |
|---|---|
| **Test Accuracy** | **97.55%** |
| **Macro Precision** | **0.9747** |
| **Macro Recall** | **0.9760** |
| **Macro F1-Score** | **0.9753** |
| **Cohen's Kappa** | **0.9506** |
| **Matthews Correlation Coefficient (MCC)** | **0.9507** |
| **ROC-AUC** | **0.9988** |
| **PR-AUC** | **0.9986** |
| **Inference Latency** | **29.40 ms / sample** |
| **Inference Throughput** | **34.0 FPS** |

## Per-Class Breakdown

| Class | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |
|---|---|---|---|---|---|
| **Cancer (0)** | 0.9841 | 0.9713 | 0.9808 | 0.9777 | 383 |
| **Normal (1)** | 0.9653 | 0.9808 | 0.9713 | 0.9730 | 312 |
