# Attention Fusion Classification Report (Derived Clean Test Set)

- **Model**: BoneCancerAttentionFusion (Frozen CNN + Swin + Mamba + Softmax Branch Attention)
- **Evaluated Checkpoint**: `C:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\checkpoints\attention_fusion_best.pth`
- **HDF5 Export**: `C:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\checkpoints\attention_fusion_best.h5`
- **Test Set Size**: 695 samples

## Branch Contribution Weights

| Branch | Architecture | Overall Weight (%) | Cancer Class Weight (%) | Normal Class Weight (%) |
|---|---|---|---|---|
| **CNN** | ResNet-18 | **29.74%** | 38.77% | 18.65% |
| **Swin** | Swin-Tiny | **52.75%** | 38.90% | 69.74% |
| **Mamba** | Selective SSM | **17.52%** | 22.33% | 11.61% |

## Confusion Matrix & Terminology (Positive Class = Cancer [0])

| Term | Full Name | Definition | Sample Count |
|---|---|---|---|
| **TP** | True Positive | True Cancer predicted as Cancer | **380** |
| **FN** | False Negative | True Cancer predicted as Normal | **3** |
| **FP** | False Positive | True Normal predicted as Cancer | **5** |
| **TN** | True Negative | True Normal predicted as Normal | **307** |

## Performance Summary

| Metric | Score |
|---|---|
| **Test Accuracy** | **98.85%** |
| **Macro Precision** | **0.9887** |
| **Macro Recall** | **0.9881** |
| **Macro F1-Score** | **0.9884** |
| **Cohen's Kappa** | **0.9767** |
| **Matthews Correlation Coefficient (MCC)** | **0.9767** |
| **ROC-AUC** | **0.9992** |
| **PR-AUC** | **0.9991** |
| **Inference Latency** | **31.32 ms / sample** |
| **Inference Throughput** | **31.9 FPS** |

## Per-Class Breakdown

| Class | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |
|---|---|---|---|---|---|
| **Cancer (0)** | 0.9870 | 0.9922 | 0.9840 | 0.9896 | 383 |
| **Normal (1)** | 0.9903 | 0.9840 | 0.9922 | 0.9871 | 312 |
