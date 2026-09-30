# Swin-Tiny Classification Report (Derived Clean Test Set)

- **Model**: BoneCancerSwin (Swin-Tiny Backbone + 256D Feature Projection)
- **Evaluated Checkpoint**: `C:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\checkpoints\swin_best.pth`
- **HDF5 Export**: `C:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\checkpoints\swin_best.h5`
- **Test Set Size**: 695 samples

## Confusion Matrix & Terminology (Positive Class = Cancer [0])

| Term | Full Name | Definition | Sample Count |
|---|---|---|---|
| **TP** | True Positive | True Cancer predicted as Cancer | **373** |
| **FN** | False Negative | True Cancer predicted as Normal | **10** |
| **FP** | False Positive | True Normal predicted as Cancer | **3** |
| **TN** | True Negative | True Normal predicted as Normal | **309** |

## Performance Summary

| Metric | Score |
|---|---|
| **Test Accuracy** | **98.13%** |
| **Macro Precision** | **0.9803** |
| **Macro Recall** | **0.9821** |
| **Macro F1-Score** | **0.9811** |
| **Cohen's Kappa** | **0.9623** |
| **Matthews Correlation Coefficient (MCC)** | **0.9625** |
| **ROC-AUC** | **0.9991** |
| **PR-AUC** | **0.9990** |
| **Inference Latency** | **31.19 ms / sample** |
| **Inference Throughput** | **32.1 FPS** |

## Per-Class Breakdown

| Class | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |
|---|---|---|---|---|---|
| **Cancer (0)** | 0.9920 | 0.9739 | 0.9904 | 0.9829 | 383 |
| **Normal (1)** | 0.9687 | 0.9904 | 0.9739 | 0.9794 | 312 |
