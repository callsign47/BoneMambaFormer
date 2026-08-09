# Mamba Classification Report (Derived Clean Test Set)

- **Model**: BoneCancerMamba (Selective SSM Mamba Backbone + 256D Feature Projection)
- **Evaluated Checkpoint**: `C:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\checkpoints\mamba_best.pth`
- **HDF5 Export**: `C:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\checkpoints\mamba_best.h5`
- **Test Set Size**: 695 samples

## Confusion Matrix & Terminology (Positive Class = Cancer [0])

| Term | Full Name | Definition | Sample Count |
|---|---|---|---|
| **TP** | True Positive | True Cancer predicted as Cancer | **312** |
| **FN** | False Negative | True Cancer predicted as Normal | **71** |
| **FP** | False Positive | True Normal predicted as Cancer | **28** |
| **TN** | True Negative | True Normal predicted as Normal | **284** |

## Performance Summary

| Metric | Score |
|---|---|
| **Test Accuracy** | **85.76%** |
| **Macro Precision** | **0.8588** |
| **Macro Recall** | **0.8624** |
| **Macro F1-Score** | **0.8573** |
| **Cohen's Kappa** | **0.7157** |
| **Matthews Correlation Coefficient (MCC)** | **0.7213** |
| **ROC-AUC** | **0.9468** |
| **PR-AUC** | **0.9443** |
| **Inference Latency** | **29.78 ms / sample** |
| **Inference Throughput** | **33.6 FPS** |

## Per-Class Breakdown

| Class | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |
|---|---|---|---|---|---|
| **Cancer (0)** | 0.9176 | 0.8146 | 0.9103 | 0.8631 | 383 |
| **Normal (1)** | 0.8000 | 0.9103 | 0.8146 | 0.8516 | 312 |
