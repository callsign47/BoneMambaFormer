# Phase 2 Data Pipeline & Leakage Analysis Report

> **Project:** ICIMCPS-2026 Bone Cancer Model  
> **Phase:** Phase 2 (Data Pipeline & Augmentations)  
> **Status:** COMPLETED (Zero Model Training Executed)  
> **Report Path:** `reports/phase2_data_pipeline_report.md`

---

## 1. Cross-Split MD5 Duplicate Investigation

A deep-scan of all 8,810 images identified **7,751 unique MD5 image hashes**.

### Distribution of Hash Groups:
- **Single-instance Hashes:** 6,692 hashes (single image file)
- **Intra-split Duplicate Hash Groups:** 714 groups (duplicate images within the same split)
- **Cross-split Duplicate Hash Groups (Leakage):** **345 groups** (identical image pixels present across split boundaries)

### Overlap Pair Matrix:
```text
Overlap Pair                          Group Count    Leakage Status
-------------------------------------------------------------------
Train <-> Validation                  168 groups     Identified
Train <-> Test                        161 groups     Identified
Validation <-> Test                    16 groups     Identified
Train <-> Validation <-> Test           0 groups     None
-------------------------------------------------------------------
TOTAL CROSS-SPLIT DUPLICATE GROUPS    345 groups
```

### Label Disagreement Verification:
- **Label Disagreements Found:** **0 (Zero)**
- Every duplicate hash across all splits maintains 100% ground-truth label consistency (`cancer` is always `cancer`, `normal` is always `normal`).

---

## 2. Split Protocol Comparison & Recommendation

| Split Protocol | Train Images | Val Images | Test Images | Cross-Split Leakage | Recommended Use |
|---|---|---|---|---|---|
| **Original Supplied Split** | 7,056 (3,081 C / 3,975 N) | 882 (398 C / 484 N) | 872 (384 C / 488 N) | **345 Overlapping Groups** | Mandatory baseline for literature/PRD compliance |
| **Derived Leakage-Clean Split** | 7,417 (3,082 C / 4,335 N) | 698 (398 C / 300 N) | 695 (383 C / 312 N) | **0 Overlapping Groups** | **Recommended for final unbiased paper evaluation** |

### Split Recommendation:
- **Primary Recommendation:** Use the **Derived Leakage-Clean Split** (`results/derived_leakage_clean_split.json`) for final reporting to ensure that no validation or test evaluation metrics suffer from training sample memorization/leakage.
- **Secondary Reference:** Retain the **Original Supplied Split** as a secondary run to document the exact impact of data leakage on validation/test accuracy in the paper's ablation section.

---

## 3. Data Pipeline & Modular Architecture

The pipeline consists of four modular components built in `src/`:

1. **`src/config.py`**: Centralized configuration specifying paths, dataset settings (`split_type: 'original'` or `'derived_clean'`), class mapping (`cancer: 0`, `normal: 1`), target resolution (224×224), and augmentation hyperparameters.
2. **`src/preprocessing.py`**: Deterministic preprocessing:
   - Resizing to target resolution (224×224)
   - Controlled contrast enhancement (`PIL ImageOps.autocontrast` & `ImageEnhance`) to highlight bone morphology without introducing noise
   - Standard ImageNet RGB normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
3. **`src/augmentation.py`**: Training-only augmentations:
   - Elastic deformation (`scipy.ndimage.gaussian_filter` & `map_coordinates`)
   - Random rotation (±15°)
   - Random horizontal & vertical flips (50% probability)
   - Random zoom / rescaling (0.9 to 1.1 scale)
   - Intensity variation (brightness & contrast jitter)
   - *Note:* Validation and test pipelines remain 100% deterministic.
4. **`src/dataset.py`**: PyTorch `Dataset` & `DataLoader` factory returning GPU-ready data batches.

---

## 4. Pipeline Smoke Test Verification

The pipeline smoke test (`src/smoke_test_pipeline.py`) was executed on an **NVIDIA GeForce RTX 3050 6GB Laptop GPU** for both `original` and `derived_clean` split modes:

```text
Smoke Test Criteria           Train Split        Val Split          Test Split         Status
-----------------------------------------------------------------------------------------------
Batch Tensor Shape            [16, 3, 224, 224]  [16, 3, 224, 224]  [16, 3, 224, 224]  PASSED
Label Integrity               [0, 1] (Binary)    [0, 1] (Binary)    [0, 1] (Binary)    PASSED
NaN / Inf Presence            False              False              False              PASSED
Min / Max Range               [-2.1179, 2.6400]  [-2.1179, 2.6400]  [-2.1179, 2.6400]  PASSED
GPU Transfer (cuda:0)         Success            Success            Success            PASSED
-----------------------------------------------------------------------------------------------
```

---

## 5. Phase 2 Deliverables & File Locations

- **Workspace Markdown Report:** [`reports/phase2_data_pipeline_report.md`](file:///c:/Users/admin_fix/Downloads/BONE%20CANCER%20ICIMCPS/reports/phase2_data_pipeline_report.md)
- **Visualizations Artifact:** [`figures/augmentation_samples.png`](file:///c:/Users/admin_fix/Downloads/BONE%20CANCER%20ICIMCPS/figures/augmentation_samples.png)
- **Detailed Leakage Audit JSON:** [`results/cross_split_leakage_detailed.json`](file:///c:/Users/admin_fix/Downloads/BONE%20CANCER%20ICIMCPS/results/cross_split_leakage_detailed.json)
- **Derived Clean Split JSON:** [`results/derived_leakage_clean_split.json`](file:///c:/Users/admin_fix/Downloads/BONE%20CANCER%20ICIMCPS/results/derived_leakage_clean_split.json)
- **Smoke Test Results JSON:** [`results/pipeline_smoke_test.json`](file:///c:/Users/admin_fix/Downloads/BONE%20CANCER%20ICIMCPS/results/pipeline_smoke_test.json)
