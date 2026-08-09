# CNN Classification Report (Derived Clean Test Set)

- **Model**: BoneCancerCNN (ResNet18 Backbone + 256D Feature Projection)
- **Evaluated Checkpoint**: `checkpoints/cnn_best.pth`
- **HDF5 Export**: `checkpoints/cnn_best.h5`
- **Test Set Size**: 695 samples

## Confusion Matrix & Terminology (Positive Class = Cancer [0])

| Term | Full Name | Clinical Definition | Sample Count |
|---|---|---|---|
| **TP** | True Positive | True Cancer correctly predicted as Cancer | **373** |
| **FN** | False Negative | True Cancer incorrectly predicted as Normal | **10** |
| **FP** | False Positive | True Normal incorrectly predicted as Cancer | **9** |
| **TN** | True Negative | True Normal correctly predicted as Normal | **303** |

## Overall Performance Metrics

| Metric | Score |
|---|---|
| **Test Accuracy** | **97.27%** ($676 / 695$ correct) |
| **Macro Precision** | **0.9722** |
| **Macro Recall** | **0.9725** |
| **Macro F1-Score** | **0.9724** |
| **Cohen's Kappa ($\kappa$)** | **0.9448** |
| **Matthews Correlation Coefficient (MCC)** | **0.9448** |
| **ROC-AUC** | **0.9977** |
| **PR-AUC** | **0.9972** |
| **Inference Latency** | **29.38 ms / sample** |
| **Inference Throughput** | **34.0 FPS** |

## Per-Class Breakdown

| Class | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |
|---|---|---|---|---|---|
| **Cancer (Class 0)** | **0.9764** | **0.9739** | **0.9712** | **0.9752** | 383 |
| **Normal (Class 1)** | **0.9681** | **0.9712** | **0.9739** | **0.9696** | 312 |
| **Macro Average** | **0.9722** | **0.9725** | **0.9725** | **0.9724** | 695 |
