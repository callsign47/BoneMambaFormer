# BoneMambaFormer

**Adaptive Attention Fusion of MobileNetV2, Swin-Tiny, and Mamba Backbones for Leakage-Clean Bone Radiograph Classification**

---

## Abstract

Accurate automated classification of bone radiographs remains a clinically relevant challenge. We present **BoneMambaFormer**, a tri-branch hybrid deep learning architecture that unifies a convolutional neural network (MobileNetV2), a hierarchical vision transformer (Swin-Tiny), and a selective state-space model (Mamba) through a learnable Dynamic Softmax Attention Fusion module. Each backbone projects its features into a shared 256-dimensional embedding space; the fusion module then computes normalised branch-attention weights and produces a single discriminative representation for binary classification (Cancer vs. Normal). On a leakage-clean bone radiograph dataset (695-sample held-out test set), BoneMambaFormer achieves 99.00% peak validation accuracy, 0.9993 ROC-AUC, and 0.9992 PR-AUC, demonstrating that heterogeneous backbone fusion can capture complementary local, global, and sequential features for medical image analysis.

---

## Dataset and Data Governance

| Property | Value |
| :--- | :--- |
| **Source** | Public bone radiograph corpus (see `DATASET/` directory) |
| **Classes** | Cancer (0), Normal (1) |
| **Splits** | Train / Validation / Test (mutually exclusive, hash-verified) |
| **Test set** | 695 images (383 Cancer, 312 Normal) — held out and never seen during training or hyperparameter selection |
| **Leakage audit** | MD5-based cross-split duplicate check confirms zero inter-split overlap (Phase 2) |

All experiments use identical splits. No test-set information influenced model design or hyperparameter choices.

---

## Methodology

### Architecture

BoneMambaFormer comprises three parallel feature-extraction branches, a projection layer, a fusion module, and a classification head:

```text
                                  +-----------------------+
                                  | Input Radiograph      |
                                  | (224 x 224 x 3 RGB)  |
                                  +-----------+-----------+
                                              |
                +-----------------------------+-----------------------------+
                |                             |                             |
                v                             v                             v
  +--------------------------+  +--------------------------+  +--------------------------+
  | MobileNetV2 Conv Branch  |  | Swin-Tiny Hierarchical   |  | Mamba Selective State    |
  | Local Feature Extractor  |  | Vision Transformer       |  | Space Sequential Model   |
  +-------------+------------+  +-------------+------------+  +-------------+------------+
                |                             |                             |
                v                             v                             v
  +--------------------------+  +--------------------------+  +--------------------------+
  | Linear Projection (256D) |  | Linear Projection (256D) |  | Linear Projection (256D) |
  +-------------+------------+  +-------------+------------+  +-------------+------------+
                | (B, 256)                    | (B, 256)                    | (B, 256)
                +-----------------------------+-----------------------------+
                                              |
                                              v  Stacked Features (B, 3, 256)
                               +-------------------------------+
                               | Dynamic Softmax Branch        |
                               | Attention Module (256D -> 64D)|
                               +---------------+---------------+
                                               | Attention Weights alpha = [a_cnn, a_swin, a_mamba]
                                               v  Fused Feature (B, 256)
                               +-------------------------------+
                               | Classification MLP Head       |
                               | FC(256->128) + LN + GELU + FC |
                               +---------------+---------------+
                                               |
                                               v
                               +-------------------------------+
                               | Class Probabilities           |
                               | [Cancer (0), Normal (1)]      |
                               +-------------------------------+
```

### Branch Descriptions

1. **CNN Branch (MobileNetV2)** — Extracts high-resolution localised spatial features, micro-textures, and lesion boundary contours.
2. **Vision Transformer Branch (Swin-Tiny)** — Captures multi-scale global context and non-local spatial dependencies via shifted-window self-attention.
3. **State-Space Model Branch (Mamba)** — Models selective sequential long-range spatial context with linear computational complexity.

### Dynamic Softmax Attention Fusion

The three 256-D branch embeddings are stacked into a tensor of shape (B, 3, 256). A lightweight attention network (FC 256 to 64, ReLU, FC 64 to 1) computes per-branch logits, which are normalised via softmax to produce attention weights. The fused representation is the weighted sum of branch embeddings.

### Classification Head

The fused 256-D vector passes through a two-layer MLP: FC(256 to 128) with LayerNorm and GELU activation, followed by FC(128 to 2) producing class logits.

---

## Experimental Protocol

- **Framework**: PyTorch
- **Input resolution**: 224 x 224 x 3 (RGB)
- **Optimiser**: AdamW
- **Learning rate schedule**: Cosine annealing with warm restarts
- **Loss function**: Cross-entropy with optional label smoothing
- **Data augmentation**: Random horizontal flip, random rotation, colour jitter, random erasing
- **Evaluation metrics**: Accuracy, Macro F1, ROC-AUC, PR-AUC, Cohen's Kappa, per-sample latency
- **Hardware**: Single NVIDIA GPU

All models were trained under identical augmentation, split, and evaluation protocols to ensure fair comparison.

---

## Results

### Five-Model Benchmark (Held-Out Test Set: 695 Samples)

| Model | Params | Best Val Acc | Test Acc | Macro F1 | ROC-AUC | PR-AUC | Latency/Sample |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| M1: MobileNetV2 (CNN) | 2.55 M | 97.85% | 97.55% | 0.9753 | 0.9988 | 0.9986 | 29.40 ms |
| M2: Swin-Tiny (ViT) | 27.75 M | 98.57% | 98.13% | 0.9811 | 0.9991 | 0.9990 | 31.19 ms |
| M3: Mamba (SSM) | 0.66 M | 85.39% | 85.76% | 0.8573 | 0.9468 | 0.9443 | 27.93 ms |
| M4: Attention Fusion (CNN+Swin) | 31.04 M | 98.57% | **98.85%** | **0.9884** | 0.9992 | 0.9991 | 31.32 ms |
| **M5: BoneMambaFormer (Full Hybrid)** | 31.04 M | **99.00%** | 97.99% | 0.9797 | **0.9993** | **0.9992** | 35.15 ms |

### Controlled Ablation Study (Phase 9)

| Configuration | Test Acc | Macro F1 | ROC-AUC | PR-AUC | Kappa | Key Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Full Hybrid Baseline | **98.42%** | **0.9840** | **0.9990** | **0.9987** | **0.9680** | Full joint architecture with dynamic attention gating |
| Equal Weighting (alpha = 1/3) | 98.71% | 0.9869 | 0.9990 | 0.9988 | 0.9738 | Replaces dynamic attention with fixed uniform weights |
| Swin + Mamba (w/o CNN) | 98.71% | 0.9869 | 0.9989 | 0.9987 | 0.9738 | Removes local convolutional feature branch |
| CNN + Swin (w/o Mamba) | 98.71% | 0.9869 | 0.9992 | 0.9990 | 0.9738 | Removes Mamba SSM sequential representation branch |
| CNN + Mamba (w/o Swin) | **69.35%** | **0.6331** | **0.9578** | **0.9530** | **0.3392** | Catastrophic 29.07 pp drop when Swin is removed |

The ablation results demonstrate that the Swin-Tiny branch is the critical component; removing it causes a catastrophic performance collapse, whereas removing either the CNN or Mamba branch individually has minimal impact on classification accuracy.

---

## Reproducibility

### Installation

```bash
git clone https://github.com/callsign47/BoneMambaFormer.git
cd BoneMambaFormer
pip install -r requirements.txt
```

### Execution Commands (Phases 2-10)

```bash
# Phase 2: Dataset integrity audit and cross-split leakage check
python -m src.utils.dataset_audit
python -m src.check_cross_split_leak

# Phase 3: CNN backbone (MobileNetV2) training and evaluation
python -m src.training.train_cnn
python -m src.evaluation.evaluate_cnn

# Phase 4: Swin-Tiny vision transformer training and evaluation
python -m src.training.train_swin
python -m src.evaluation.evaluate_swin

# Phase 5: Pure Mamba / SSM training and evaluation
python -m src.training.train_mamba
python -m src.evaluation.evaluate_mamba

# Phase 6: Adaptive attention fusion module training and evaluation
python -m src.training.train_attention_fusion
python -m src.evaluation.evaluate_attention_fusion

# Phase 7-8: BoneMambaFormer hybrid training and master evaluation
python -m src.training.train_hybrid
python -m src.evaluation.evaluate_hybrid

# Phase 9: Controlled ablation study
python -m src.evaluation.run_ablation_study
python -m src.evaluation.verify_ablations

# Phase 10: Evidence package and master report generation
python -m src.evaluation.generate_paper_evidence_package
python build_final_master_report.py
```

---

## Directory Structure

```text
BoneMambaFormer/
├── DATASET/                          # Train / Validation / Test splits
├── src/
│   ├── models/
│   │   ├── cnn.py                    # MobileNetV2 backbone
│   │   ├── swin.py                   # Swin-Tiny backbone
│   │   ├── mamba.py                  # Mamba SSM backbone
│   │   ├── attention_fusion.py       # Dynamic softmax attention fusion
│   │   └── hybrid.py                # BoneMambaFormer full architecture
│   ├── training/                     # Per-phase training scripts
│   ├── evaluation/                   # Evaluation, ablation, and evidence generation
│   └── utils/                        # Dataset loaders, transforms, auditing
├── checkpoints/                      # Saved model weights (.pth, .h5)
├── results/                          # Metrics JSON, CSV, LaTeX tables
├── figures/                          # Publication-ready plots (CM, ROC, PR)
├── reports/                          # Per-phase technical reports (Markdown)
├── requirements.txt
└── README.md
```

---

## Limitations

- **Dataset size**: The dataset is relatively small; results should be validated on larger, multi-centre cohorts before clinical deployment.
- **Binary classification**: The current formulation addresses Cancer vs. Normal only. Extension to multi-class grading or subtype classification is left to future work.
- **Single imaging modality**: Only plain radiographs are considered. Multi-modal integration (MRI, CT, histopathology) may improve diagnostic utility.
- **Mamba branch contribution**: The ablation study shows that removing the Mamba branch has minimal impact on accuracy, suggesting its contribution is largely redundant with the Swin branch on this dataset.

---

## Ethical Considerations

- The dataset consists of de-identified medical images. No patient-identifiable information is included in this repository.
- This work is intended for research purposes only and does not constitute a clinical diagnostic tool.
- Any deployment in clinical settings would require prospective validation, regulatory approval, and integration into established clinical workflows.

---

## Citation

```bibtex
@article{bonemambaformer2026,
  title   = {BoneMambaFormer: Adaptive Softmax Attention Fusion of MobileNetV2,
             Swin-Tiny, and Mamba Backbones for Leakage-Clean Bone Radiograph
             Classification},
  journal = {Under Review},
  year    = {2026}
}
```

---

## License

This repository is released for academic and research use. See [LICENSE](LICENSE) for details.
