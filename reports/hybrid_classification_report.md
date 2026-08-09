# End-to-End Hybrid Model Classification Report (Derived Clean Test Set)

- **Model**: BoneCancerHybridModel (Joint End-to-End CNN + Swin + Mamba + Softmax Branch Attention)
- **Evaluated Checkpoint**: `C:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\checkpoints\hybrid_best.pth`
- **HDF5 Export**: `C:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\checkpoints\hybrid_best.h5`
- **Test Set Size**: 695 samples

## Branch Contribution Weights

| Branch | Architecture | Overall Weight (%) | Cancer Class Weight (%) | Normal Class Weight (%) |
|---|---|---|---|---|
| **CNN** | ResNet-18 | **1.14%** | 1.77% | 0.38% |
| **Swin** | Swin-Tiny | **92.61%** | 87.84% | 98.46% |
| **Mamba** | Selective SSM | **6.25%** | 10.39% | 1.16% |

## Confusion Matrix & Terminology (Positive Class = Cancer [0])

| Term | Full Name | Definition | Sample Count |
|---|---|---|---|
| **TP** | True Positive | True Cancer predicted as Cancer | **377** |
| **FN** | False Negative | True Cancer predicted as Normal | **6** |
| **FP** | False Positive | True Normal predicted as Cancer | **5** |
| **TN** | True Negative | True Normal predicted as Normal | **307** |

## Performance Summary

| Metric | Score |
|---|---|
| **Test Accuracy** | **98.42%** |
| **Macro Precision** | **0.9839** |
| **Macro Recall** | **0.9842** |
| **Macro F1-Score** | **0.9840** |
| **Cohen's Kappa** | **0.9680** |
| **Matthews Correlation Coefficient (MCC)** | **0.9680** |
| **ROC-AUC** | **0.9990** |
| **PR-AUC** | **0.9987** |
| **Inference Latency** | **33.55 ms / sample** |
| **Inference Throughput** | **29.8 FPS** |

## Per-Class Breakdown

| Class | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |
|---|---|---|---|---|---|
| **Cancer (0)** | 0.9869 | 0.9843 | 0.9840 | 0.9856 | 383 |
| **Normal (1)** | 0.9808 | 0.9840 | 0.9843 | 0.9824 | 312 |
