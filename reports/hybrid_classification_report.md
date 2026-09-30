# End-to-End Hybrid Model Classification Report (Derived Clean Test Set)

- **Model**: BoneCancerHybridModel (Joint End-to-End CNN + Swin + Mamba + Softmax Branch Attention)
- **Evaluated Checkpoint**: `C:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\checkpoints\hybrid_best.pth`
- **HDF5 Export**: `C:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\checkpoints\hybrid_best.h5`
- **Test Set Size**: 695 samples

## Branch Contribution Weights

| Branch | Architecture | Overall Weight (%) | Cancer Class Weight (%) | Normal Class Weight (%) |
|---|---|---|---|---|
| **CNN** | ResNet-18 | **31.29%** | 37.99% | 23.06% |
| **Swin** | Swin-Tiny | **55.68%** | 46.17% | 67.35% |
| **Mamba** | Selective SSM | **13.03%** | 15.84% | 9.59% |

## Confusion Matrix & Terminology (Positive Class = Cancer [0])

| Term | Full Name | Definition | Sample Count |
|---|---|---|---|
| **TP** | True Positive | True Cancer predicted as Cancer | **372** |
| **FN** | False Negative | True Cancer predicted as Normal | **11** |
| **FP** | False Positive | True Normal predicted as Cancer | **3** |
| **TN** | True Negative | True Normal predicted as Normal | **309** |

## Performance Summary

| Metric | Score |
|---|---|
| **Test Accuracy** | **97.99%** |
| **Macro Precision** | **0.9788** |
| **Macro Recall** | **0.9808** |
| **Macro F1-Score** | **0.9797** |
| **Cohen's Kappa** | **0.9594** |
| **Matthews Correlation Coefficient (MCC)** | **0.9596** |
| **ROC-AUC** | **0.9993** |
| **PR-AUC** | **0.9992** |
| **Inference Latency** | **35.15 ms / sample** |
| **Inference Throughput** | **28.5 FPS** |

## Per-Class Breakdown

| Class | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |
|---|---|---|---|---|---|
| **Cancer (0)** | 0.9920 | 0.9713 | 0.9904 | 0.9815 | 383 |
| **Normal (1)** | 0.9656 | 0.9904 | 0.9713 | 0.9778 | 312 |
