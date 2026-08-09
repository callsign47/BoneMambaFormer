# CODING AGENT HANDOFF --- BONE CANCER CONFERENCE MODEL

## 0. READ THIS FIRST

You are the primary coding agent for a time-constrained research
implementation.

Your job is to **build the conference model correctly from the current
dataset and coordinator-provided specification**, not to redesign the
research direction.

The project is targeting **ICIMCPS-2026**. The current paper submission
deadline shown in the supplied conference material is **15 August
2026**. The internal engineering target is to have the model and core
experiments substantially complete by **12--13 August 2026**.

The coordinator-provided figure is the architectural reference.

The current dataset is the only confirmed new-project asset.

The previous BoneMambaFormer code/results are **legacy work**. They may
be inspected and reused only after verification. Do not silently treat
old results as results of this new experiment.

------------------------------------------------------------------------

# 1. NON-NEGOTIABLE RULES

## 1.1 Scientific integrity

Never:

-   fabricate results
-   fabricate metrics
-   fabricate training times
-   fabricate dataset properties
-   claim patient-level separation
-   claim an experiment was performed when it was not
-   report validation accuracy as test accuracy
-   tune repeatedly against the test set
-   leak labels into a self-supervised objective
-   claim DINOv2, MAE, SimCLR, or BYOL was used unless it is actually
    implemented and trained
-   claim SOTA without a defensible literature comparison
-   claim clinical validation
-   invent cancer subtypes that do not exist in the supplied dataset
-   silently modify the dataset split
-   silently change preprocessing after results have been generated
-   overwrite previous experimental results without preserving them

If something is unknown, write:

``` text
TBD
```

or:

``` text
NEEDS VERIFICATION
```

Do not guess.

## 1.2 Priority

Use this priority order:

``` text
Correctness
    >
Reproducibility
    >
Complete experimental evidence
    >
Implementation simplicity
    >
Raw accuracy
    >
Architectural sophistication
```

The deadline is a reason to choose a simpler valid implementation. It is
not a reason to cut scientific corners.

## 1.3 Source hierarchy

When making project decisions, use this order:

1.  Coordinator-provided figure
2.  Coordinator-provided implementation proposal
3.  Actual supplied dataset
4.  Explicit instructions in this handoff
5.  Existing legacy code only after verification

Do not let legacy implementation silently override the coordinator
specification.

------------------------------------------------------------------------

# 2. PROJECT OBJECTIVE

Build a hybrid bone-cancer classification model based on:

``` text
CNN
+
Swin Transformer
+
Mamba
        ↓
Adaptive Attention Fusion
        ↓
Feature Refinement
        ↓
Fully Connected Classification
        ↓
Normal / Cancer
```

The coordinator-provided workflow additionally includes:

``` text
Image Preprocessing
        ↓
Medical Image Augmentation
        ↓
Medical Self-Supervised Pretraining
        ↓
Domain-Specific Medical Encoder
        ↓
Transfer Learning
        ↓
Labeled Dataset
        ↓
Supervised Fine-Tuning
        ↓
Hybrid Feature Extraction
        ↓
Adaptive Attention Fusion
        ↓
Feature Refinement
        ↓
Classification
        ↓
Evaluation
```

The coordinator's figure lists:

-   DINOv2
-   MAE
-   SimCLR
-   BYOL

under self-supervised pretraining.

**Important:** the supplied material does not establish that all four
must be implemented. Do not implement all four merely because they
appear in the figure. Choose the simplest valid interpretation unless
the coordinator explicitly provides a mandatory method.

------------------------------------------------------------------------

# 3. CURRENT DATASET

The dataset is already classified and split.

Expected structure:

``` text
train/
├── normal/
└── cancer/

val/
├── normal/
└── cancer/

test/
├── normal/
└── cancer/
```

Supplied archive names:

``` text
train_sorted-20251125T062947Z-1-001
valid_sorted-20251125T062929Z-1-001
test_sorted-20251125T062917Z-1-001
```

## 3.1 Verified counts

``` text
TOTAL:       8,811

TRAIN:       7,057
VALIDATION:    882
TEST:          872

NORMAL:      4,948
CANCER:      3,863
```

Breakdown:

  Split          Normal   Cancer   Total
  ------------ -------- -------- -------
  Train           3,976    3,081   7,057
  Validation        484      398     882
  Test              488      384     872
  Total           4,948    3,863   8,811

## 3.2 Image properties

Confirmed:

-   JPG
-   640 × 640
-   RGB

## 3.3 Classification task

The actual supplied dataset is:

``` text
Class 0: cancer
Class 1: normal
```

This is a **binary classification problem**.

Do NOT create:

-   osteosarcoma
-   Ewing sarcoma
-   chondrosarcoma
-   other cancer

as additional classes unless a new dataset containing those labels is
explicitly supplied.

The coordinator figure visually shows multiple cancer categories, but
the supplied dataset currently contains only `normal` and `cancer`.

------------------------------------------------------------------------

# 4. DATASET INTEGRITY

Patient-level independence is currently:

``` text
UNKNOWN
```

No patient metadata has been supplied.

Do not claim that the split is patient-independent.

Before final paper results, perform all feasible checks without patient
metadata:

-   corrupted images
-   unreadable files
-   unexpected extensions
-   unexpected folders
-   exact file duplicates
-   image hash duplicates
-   filename duplicates
-   perceptual/near-duplicate checks if feasible

If patient metadata later becomes available, perform patient-level
verification.

Do not block the entire implementation waiting for unavailable patient
metadata.

------------------------------------------------------------------------

# 5. FIRST TASK: INSPECT THE EXISTING WORKSPACE

Before writing new code:

1.  Inspect the repository/workspace.
2.  Identify existing source code.
3.  Identify legacy BoneMambaFormer code.
4.  Identify checkpoints.
5.  Identify existing dataset loaders.
6.  Identify installed Python/PyTorch/CUDA environment.
7.  Identify available GPU.
8.  Identify existing requirements files.
9.  Identify whether the dataset is already mounted/copied.
10. Do not modify or delete legacy work before understanding it.

Create a short internal map:

``` text
CURRENT WORKSPACE
├── new implementation area
├── legacy implementation
├── checkpoints
├── data
└── documentation
```

If the workspace is empty, create the project structure specified below.

------------------------------------------------------------------------

# 6. LEGACY WORK

Previous project work may contain:

-   CNN implementation
-   Swin-Tiny implementation
-   custom Mamba implementation
-   Adaptive Attention Fusion
-   projection layers
-   training utilities
-   evaluation utilities

Previous reported results included:

``` text
Swin-Tiny:
99.32% peak validation accuracy

Mamba:
91.04% peak validation accuracy
```

These are **legacy results**.

Do not place them into new experiment result tables.

Do not call them results of the current model.

You may reuse code after verifying:

-   input assumptions
-   labels
-   preprocessing
-   augmentation
-   output dimensions
-   training behavior
-   compatibility with the new pipeline

------------------------------------------------------------------------

# 7. REQUIRED PROJECT STRUCTURE

If no suitable structure already exists, use:

``` text
BONE_CANCER_CONFERENCE/
│
├── data/
│   ├── train/
│   │   ├── normal/
│   │   └── cancer/
│   ├── val/
│   │   ├── normal/
│   │   └── cancer/
│   └── test/
│       ├── normal/
│       └── cancer/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── augmentation.py
│   │
│   ├── models/
│   │   ├── cnn.py
│   │   ├── swin.py
│   │   ├── mamba.py
│   │   ├── fusion.py
│   │   └── hybrid.py
│   │
│   ├── training/
│   │   ├── train_single.py
│   │   ├── train_hybrid.py
│   │   └── checkpoint.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluate.py
│   │   ├── ablation.py
│   │   └── attention_analysis.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── visualization.py
│
├── configs/
│   ├── cnn.yaml
│   ├── swin.yaml
│   ├── mamba.yaml
│   └── hybrid.yaml
│
├── checkpoints/
├── results/
├── figures/
├── reports/
├── requirements.txt
├── README.md
└── PRD.md
```

Do not create unnecessary modules simply to match this tree. Simplicity
is preferred.

------------------------------------------------------------------------

# 8. DATA PIPELINE

Implement the data pipeline first.

## 8.1 Requirements

The loader must:

-   load train/validation/test separately
-   map classes deterministically
-   expose labels
-   support transforms
-   support batching
-   support shuffling only for training
-   preserve validation/test separation

Expected label mapping:

``` python
cancer = 0
normal = 1
```

Keep the mapping in one configuration location.

Do not duplicate class mappings throughout the code.

## 8.2 Dataset audit script

Create a script that reports:

-   total images
-   per-split count
-   per-class count
-   image dimensions
-   image modes
-   corrupted files
-   duplicate hashes
-   unexpected files

Also create representative image grids.

------------------------------------------------------------------------

# 9. PREPROCESSING

The coordinator's figure specifies:

``` text
Resize
Normalization
Denoising
Contrast Enhancement
```

Implement these as a controlled pipeline.

## 9.1 Training

``` text
Image
→ deterministic preprocessing
→ random augmentation
→ model
```

## 9.2 Validation

``` text
Image
→ deterministic preprocessing
→ model
```

## 9.3 Test

``` text
Image
→ deterministic preprocessing
→ model
```

No random augmentation in validation/test.

## 9.4 Important

Do not automatically apply an aggressive denoiser or contrast
transformation simply because it appears in the diagram.

The transformation must preserve diagnostic morphology.

Visually inspect representative examples before freezing it.

------------------------------------------------------------------------

# 10. AUGMENTATION

Coordinator-specified candidate operations:

-   Rotation
-   Flip
-   Zoom
-   Intensity variation
-   Elastic transform

Use only reasonable medical-image ranges.

Do not invent extreme values.

All augmentation parameters must be stored in configuration.

Example:

``` yaml
augmentation:
  rotation: TBD
  flip_probability: TBD
  zoom: TBD
  intensity_variation: TBD
  elastic_transform: TBD
```

Replace TBD with actual values after implementation/testing.

Validation and test must remain deterministic.

------------------------------------------------------------------------

# 11. SELF-SUPERVISED LEARNING

The coordinator figure includes:

``` text
DINOv2
MAE
SimCLR
BYOL
```

Do not automatically implement all four.

## Decision rule

Select the simplest valid method that:

-   fits the available hardware
-   can actually be trained
-   produces a useful encoder
-   can be completed before the deadline
-   can be described honestly

If the coordinator explicitly instructs a particular method later,
follow that instruction.

## Label handling

During SSL:

``` text
image
→ SSL objective
```

The class label must not be used by the SSL loss.

The fact that the source dataset is labeled does not prevent
self-supervised learning.

------------------------------------------------------------------------

# 12. DOMAIN-SPECIFIC MEDICAL ENCODER

The coordinator workflow includes a domain-specific medical encoder.

Implement only what can be justified by the selected SSL strategy.

Document:

-   backbone
-   pretraining method
-   output dimension
-   frozen layers
-   trainable layers
-   transfer procedure

Do not use the phrase "domain-specific" as an unsupported claim. The
actual adaptation mechanism must exist in code.

------------------------------------------------------------------------

# 13. TRANSFER LEARNING

Clearly separate:

``` text
Pretraining
↓
Encoder initialization
↓
Transfer
↓
Supervised fine-tuning
```

If a pretrained Swin model is used, record:

-   source weights
-   model variant
-   pretrained dataset if known
-   whether all layers or selected layers are fine-tuned

------------------------------------------------------------------------

# 14. CNN BRANCH

Purpose:

``` text
Local feature extraction
```

Expected information:

-   edges
-   texture
-   local lesion patterns
-   boundaries

Use a lightweight architecture appropriate for the GPU.

Do not choose a huge CNN.

The CNN must expose a feature vector before classification.

Preferred conceptual API:

``` python
features = cnn.forward_features(x)
```

------------------------------------------------------------------------

# 15. SWIN BRANCH

Use **Swin-Tiny** as the practical transformer branch unless the
coordinator specifies another model.

Reason:

-   computationally manageable
-   already validated technically in legacy work
-   appropriate for the available GPU

However, the new run must use the new locked preprocessing/training
protocol.

Do not reuse the old 99.32% number as the new baseline.

The branch must expose its feature representation before classification.

------------------------------------------------------------------------

# 16. MAMBA BRANCH

Implement a lightweight Mamba/state-space branch.

The previous custom implementation may be reused if:

-   it works correctly
-   it is compatible with the current input pipeline
-   it is stable
-   it produces the required feature representation

The branch must expose its feature vector before classification.

Do not claim "official Mamba" if the implementation is custom.

Use precise terminology such as:

``` text
custom Mamba/state-space feature extractor
```

if appropriate.

------------------------------------------------------------------------

# 17. COMMON FEATURE DIMENSION

Preferred target:

``` text
CNN
→ 256-D

Swin
→ 256-D

Mamba
→ 256-D
```

Use projection layers to create compatible representations.

Do not force this dimension if it causes unnecessary problems. If
changed, document why.

------------------------------------------------------------------------

# 18. ADAPTIVE ATTENTION FUSION

This is a core contribution.

Given:

``` text
f_cnn
f_swin
f_mamba
```

generate:

``` text
w_cnn
w_swin
w_mamba
```

with:

``` text
sum(weights) = 1
```

using softmax.

Conceptually:

``` text
f_fused =
    w_cnn  * f_cnn
  + w_swin * f_swin
  + w_mamba * f_mamba
```

The exact implementation must reflect the actual code.

## Mandatory checks

After implementation verify:

-   output shape
-   no NaNs
-   gradients exist
-   weights are finite
-   weights sum to approximately 1
-   batch processing works
-   backward pass works

------------------------------------------------------------------------

# 19. FEATURE REFINEMENT

After fusion:

``` text
fused feature
→ refinement layer
→ classifier
```

Keep this lightweight.

Do not introduce unnecessary deep blocks.

------------------------------------------------------------------------

# 20. CLASSIFIER

The current task is binary:

``` text
cancer
normal
```

Use two output logits if using cross-entropy.

Do not create a five-class classifier.

------------------------------------------------------------------------

# 21. HARDWARE

Known available environment from previous work:

``` text
GPU:
NVIDIA RTX 3050 Laptop GPU

VRAM:
approximately 6 GB
```

Design accordingly.

Use where appropriate:

-   AMP
-   small batch size
-   gradient accumulation
-   frozen pretrained layers
-   partial unfreezing
-   lightweight models

Do not assume more VRAM.

------------------------------------------------------------------------

# 22. TRAINING

Every run must record:

-   seed
-   model
-   image size
-   batch size
-   epochs
-   optimizer
-   learning rate
-   scheduler
-   weight decay
-   loss
-   augmentation
-   normalization
-   checkpoint criterion
-   PyTorch version
-   CUDA version
-   GPU
-   training time
-   peak memory if feasible

## Checkpointing

Save the best validation checkpoint.

Define "best" explicitly, preferably using validation F1 or validation
loss/accuracy according to the final experimental protocol.

Do not select checkpoints using test performance.

------------------------------------------------------------------------

# 23. TEST SET PROTOCOL

The test set is:

``` text
872 images
```

Do not use it during iterative development.

Final protocol:

``` text
Train
↓
Validation-based model selection
↓
Freeze final configuration
↓
ONE final test evaluation
```

If multiple final test evaluations are performed for legitimate
experimental reasons, record all of them and explain why.

Never cherry-pick the best test result.

------------------------------------------------------------------------

# 24. BASELINE EXPERIMENTS

At minimum:

``` text
CNN
Swin
Mamba
Full Hybrid
```

Preferred full ablation:

``` text
CNN
Swin
Mamba
CNN + Swin
CNN + Mamba
Swin + Mamba
CNN + Swin + Mamba
```

If time becomes critically constrained, prioritize:

``` text
CNN
Swin
Mamba
Swin + Mamba
Full Hybrid
```

But report honestly which experiments were completed.

------------------------------------------------------------------------

# 25. METRICS

Generate:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   ROC-AUC
-   Confusion matrix

Prefer per-class metrics where appropriate.

Create a single result table:

  Model            Accuracy   Precision   Recall    F1   ROC-AUC
  -------------- ---------- ----------- -------- ----- ---------
  CNN                   TBD         TBD      TBD   TBD       TBD
  Swin                  TBD         TBD      TBD   TBD       TBD
  Mamba                 TBD         TBD      TBD   TBD       TBD
  CNN + Swin            TBD         TBD      TBD   TBD       TBD
  CNN + Mamba           TBD         TBD      TBD   TBD       TBD
  Swin + Mamba          TBD         TBD      TBD   TBD       TBD
  Full Hybrid           TBD         TBD      TBD   TBD       TBD

Never fill TBD values manually.

The table should be generated from saved result files where practical.

------------------------------------------------------------------------

# 26. ATTENTION ANALYSIS

If the fusion mechanism exposes weights, save them.

Required analysis:

-   mean CNN weight
-   mean Swin weight
-   mean Mamba weight
-   distribution
-   optional class-wise distribution

Do not assume that higher weight means a branch is universally "better."

Weights are evidence of the fusion mechanism's behavior, not standalone
proof of feature quality.

------------------------------------------------------------------------

# 27. TRAINING CURVES

Save:

-   training loss
-   validation loss
-   training accuracy
-   validation accuracy
-   optionally validation F1

Generate plots after training.

Do not manually redraw numerical results.

------------------------------------------------------------------------

# 28. CONFUSION MATRIX

Generate final test confusion matrix.

Clearly label:

``` text
Predicted
Actual
Normal
Cancer
```

Use raw counts and, if useful, normalized values.

Do not use a confusion matrix from validation while labeling it as test.

------------------------------------------------------------------------

# 29. ROC-AUC

For binary classification:

-   obtain prediction probabilities
-   compute ROC curve
-   compute AUC
-   save the curve
-   record threshold-independent AUC

Do not compute AUC from hard class predictions if probability outputs
are available.

------------------------------------------------------------------------

# 30. RESULT STORAGE

Use a structured results directory.

Example:

``` text
results/
├── dataset_audit.json
├── experiment_registry.csv
├── cnn/
├── swin/
├── mamba/
├── fusion/
├── ablation/
└── final/
```

Each experiment should contain:

``` text
config.yaml
metrics.json
history.csv
checkpoint_info.json
```

Do not overwrite previous runs.

Use timestamp or unique experiment IDs.

------------------------------------------------------------------------

# 31. EXPERIMENT REGISTRY

Maintain a simple registry:

  ID       Model    Config     Seed   Best Val   Test Status
  -------- -------- -------- ------ ---------- ------ --------
  EXP001   CNN      ...          42        TBD    TBD ...
  EXP002   Swin     ...          42        TBD    TBD ...
  EXP003   Mamba    ...          42        TBD    TBD ...
  EXP004   Hybrid   ...          42        TBD    TBD ...

This is mandatory for traceability.

------------------------------------------------------------------------

# 32. FAILURE HANDLING

If training fails:

1.  save the error
2.  identify likely cause
3.  make the smallest justified change
4.  rerun
5.  record the new configuration

Common expected issues:

-   CUDA OOM
-   tensor shape mismatch
-   NaN
-   augmentation incompatibility
-   unsupported operator
-   model output dimension mismatch

Do not hide failures.

------------------------------------------------------------------------

# 33. PERFORMANCE AND MEMORY

Before full hybrid training:

Run a memory smoke test.

Measure if possible:

``` text
allocated VRAM
reserved VRAM
peak VRAM
batch size
input resolution
AMP status
```

If OOM:

1.  enable AMP
2.  reduce batch size
3.  freeze branches
4.  use gradient accumulation
5.  reduce feature dimension
6.  reduce input resolution only if scientifically acceptable

Do not immediately remove a required architectural component.

------------------------------------------------------------------------

# 34. SIMPLE IMPLEMENTATION STRATEGY

The project is deadline constrained.

Therefore:

## Prefer

``` text
Pretrained Swin-Tiny
+
Lightweight CNN
+
Lightweight/custom Mamba
+
Simple projection heads
+
Small attention fusion module
+
Small refinement head
```

## Avoid

-   giant backbones
-   multiple unnecessary SSL models
-   complex custom attention stacks
-   unnecessary distributed training
-   complicated data pipelines
-   unnecessary dependencies
-   architecture changes that cannot be evaluated before submission

------------------------------------------------------------------------

# 35. SELF-SUPERVISED LEARNING DECISION GATE

Before implementing DINOv2/MAE/SimCLR/BYOL, stop and determine:

``` text
Is one specific SSL method explicitly required?
        │
        ├── YES → implement that method
        │
        └── NO
             ↓
Can one simple SSL method be implemented,
trained, validated, and documented within deadline?
        │
        ├── YES → select one and document it
        │
        └── NO → do not fabricate SSL;
                  implement the defensible downstream
                  architecture and explicitly document
                  the deviation from the conceptual figure
```

Do not spend the deadline implementing four SSL systems unless
explicitly required.

------------------------------------------------------------------------

# 36. DEVELOPMENT ORDER

Execute in this exact dependency order.

## Phase 1 --- Dataset

``` text
Inspect
→ audit
→ verify counts
→ verify classes
→ verify files
```

## Phase 2 --- Data pipeline

``` text
Loader
→ preprocessing
→ augmentation
→ visualization
```

## Phase 3 --- Single model

``` text
CNN
→ train
→ validate
```

## Phase 4 --- Transformer

``` text
Swin
→ train
→ validate
```

## Phase 5 --- Mamba

``` text
Mamba
→ train
→ validate
```

## Phase 6 --- Fusion

``` text
CNN feature
+
Swin feature
+
Mamba feature
→ projection
→ attention fusion
→ refinement
→ classifier
```

## Phase 7 --- Hybrid training

``` text
smoke test
→ memory test
→ short run
→ full run
```

## Phase 8 --- Evaluation

``` text
validation
→ freeze
→ test
→ metrics
```

## Phase 9 --- Ablation

``` text
single branches
→ pairwise
→ full hybrid
```

## Phase 10 --- Paper evidence

``` text
tables
→ figures
→ attention analysis
→ methodology
→ discussion
```

------------------------------------------------------------------------

# 37. DO NOT START WITH THE FULL MODEL

First prove:

``` text
DataLoader works
```

Then:

``` text
CNN forward/backward works
```

Then:

``` text
Swin forward/backward works
```

Then:

``` text
Mamba forward/backward works
```

Then:

``` text
Fusion forward/backward works
```

Then full training.

This minimizes debugging complexity.

------------------------------------------------------------------------

# 38. CODE QUALITY

Use:

-   clear names
-   type hints where useful
-   deterministic seeds
-   centralized configuration
-   reusable dataset class
-   reusable training loop
-   reusable evaluation code
-   logging

Avoid:

-   hardcoded absolute paths
-   duplicated preprocessing
-   duplicated class mappings
-   magic numbers
-   hidden global state

------------------------------------------------------------------------

# 39. PATH CONFIGURATION

Do not hardcode the user's personal absolute dataset path.

Use configuration:

``` yaml
data:
  train: ...
  val: ...
  test: ...
```

The code must work when the project is moved to another machine.

------------------------------------------------------------------------

# 40. ENVIRONMENT

Before installing anything:

1.  inspect existing environment
2.  inspect Python version
3.  inspect PyTorch
4.  inspect CUDA availability
5.  inspect GPU
6.  inspect installed packages

Do not unnecessarily reinstall PyTorch/CUDA.

Do not break the existing environment without reason.

------------------------------------------------------------------------

# 41. GIT / CHECKPOINT SAFETY

Before major architectural changes:

-   preserve working code
-   commit if Git is available
-   use meaningful commit messages
-   never commit secrets
-   never commit large datasets unless explicitly intended
-   keep checkpoints outside Git where appropriate

------------------------------------------------------------------------

# 42. PAPER-SAFE TERMINOLOGY

Use:

``` text
proposed model
hybrid model
CNN branch
Swin Transformer branch
Mamba/state-space branch
adaptive attention fusion
binary bone cancer classification
```

Avoid unsupported terms such as:

``` text
state-of-the-art
clinically validated
patient-independent
diagnostic-grade
medical-grade
SOTA
```

unless evidence actually supports them.

------------------------------------------------------------------------

# 43. EXPECTED FINAL ARCHITECTURE

The target implementation should approximately become:

``` text
                 INPUT IMAGE
                      │
                      ↓
          PREPROCESSING + AUGMENTATION
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
        CNN         SWIN        MAMBA
          │           │           │
          ↓           ↓           ↓
       Feature      Feature      Feature
          │           │           │
          ↓           ↓           ↓
       256-D        256-D        256-D
          │           │           │
          └───────────┼───────────┘
                      ↓
          ADAPTIVE ATTENTION FUSION
                      │
                      ↓
              FUSED REPRESENTATION
                      │
                      ↓
              FEATURE REFINEMENT
                      │
                      ↓
                FC CLASSIFIER
                      │
                      ↓
              NORMAL / CANCER
```

------------------------------------------------------------------------

# 44. FINAL ACCEPTANCE CRITERIA

The implementation is acceptable only if:

-   [ ] Dataset loads correctly.
-   [ ] Dataset counts are verified.
-   [ ] Classes are binary Normal/Cancer.
-   [ ] Train/validation/test are separated.
-   [ ] Preprocessing is deterministic where required.
-   [ ] Augmentation is training-only.
-   [ ] CNN works.
-   [ ] Swin works.
-   [ ] Mamba works.
-   [ ] Feature dimensions are compatible.
-   [ ] Fusion works.
-   [ ] Fusion weights are valid.
-   [ ] Hybrid model trains.
-   [ ] Checkpoints are saved.
-   [ ] Validation metrics are recorded.
-   [ ] Test set remains untouched until final evaluation.
-   [ ] Final test metrics are generated.
-   [ ] Ablation results are generated to the feasible extent.
-   [ ] All results are reproducible from saved configurations.
-   [ ] No result is fabricated.
-   [ ] No unsupported claim is introduced.

------------------------------------------------------------------------

# 45. WHAT TO REPORT BACK AFTER EACH MAJOR STEP

After completing a major task, provide a concise engineering report
containing:

``` text
TASK:
STATUS:

FILES CREATED:
- ...

FILES MODIFIED:
- ...

COMMANDS RUN:
- ...

RESULT:
- ...

METRICS:
- ...

GPU / VRAM:
- ...

ERRORS:
- ...

FIXES:
- ...

EXPERIMENT ID:
- ...

NEXT REQUIRED STEP:
- ...
```

Do not merely say "done."

------------------------------------------------------------------------

# 46. FIRST ACTION RIGHT NOW

Do **not** begin by writing the hybrid model.

Start with:

``` text
1. Inspect workspace
2. Locate dataset
3. Audit dataset
4. Verify counts
5. Verify image properties
6. Check for corruption
7. Check duplicate hashes
8. Generate sample image grid
9. Report findings
```

Expected dataset baseline:

``` text
8,811 total
7,057 train
882 validation
872 test

4,948 normal
3,863 cancer

640×640 RGB JPG
```

After the audit passes, implement the preprocessing/augmentation
pipeline.

Only then proceed to the model branches.

------------------------------------------------------------------------

# 47. CURRENT STATE

``` text
PROJECT:
ICIMCPS-2026 Conference Model

DATE:
9 August 2026

DEADLINE:
15 August 2026

INTERNAL MODEL TARGET:
12–13 August 2026

AVAILABLE ASSET:
Dataset only

DATASET:
8,811 images

TASK:
Binary Normal vs Cancer

TRAIN:
7,057

VALIDATION:
882

TEST:
872

CNN:
Not implemented for this new experiment

SWIN:
Not implemented for this new experiment

MAMBA:
Not implemented for this new experiment

FUSION:
Not implemented for this new experiment

SSL:
Not yet selected

PATIENT-LEVEL SPLIT:
Unknown

TEST EVALUATION:
Not performed

ABLATION:
Not performed

PAPER:
Not started

IMMEDIATE TASK:
Dataset audit → preprocessing → augmentation
```

------------------------------------------------------------------------

# 48. FINAL INSTRUCTION TO THE CODING AGENT

Do not optimize this project for an impressive-looking architecture.

Optimize it for a result that can survive technical questioning.

If a simpler implementation is sufficient, use it.

If a coordinator requirement is ambiguous, identify the ambiguity
instead of inventing an interpretation.

If a component cannot be completed correctly before the deadline, do not
fake it.

If the final model performs worse than a baseline, report it.

If the fusion does not improve performance, report it.

If a preprocessing choice changes results, record it.

The goal is a **real, reproducible, technically defensible conference
experiment**, completed within the available deadline.

---

# 49. LOCKED FIVE-MODEL TRAINING SET

The project must train and evaluate **exactly these five model configurations**:

1. **CNN**
2. **Swin-Tiny**
3. **Mamba**
4. **Fusion**
5. **Hybrid**

The term **"all 5"** refers to these five configurations and must be used consistently throughout the code, experiment registry, results, checkpoints, exports, and paper tables.

Do not replace these five with arbitrary pairwise ablations.

## 49.1 Definitions

### Model 1 — CNN

Standalone CNN branch trained for the binary Normal/Cancer task.

### Model 2 — Swin-Tiny

Standalone Swin-Tiny branch trained for the binary Normal/Cancer task.

### Model 3 — Mamba

Standalone Mamba/state-space branch trained for the binary Normal/Cancer task.

### Model 4 — Fusion

The **Fusion** configuration must represent the explicitly defined feature-fusion experiment used by the project.

Its exact branch composition must match the implemented architecture and be recorded in the configuration.

Do not label a model "Fusion" unless the code actually performs the intended feature fusion.

### Model 5 — Hybrid

The **Hybrid** configuration is the final proposed CNN + Swin-Tiny + Mamba architecture with Adaptive Attention Fusion, followed by feature refinement and classification.

The Hybrid model is the primary proposed method for the conference paper.

---

# 50. MANDATORY FIVE-MODEL EXPERIMENT REGISTRY

Create a centralized registry containing exactly:

```text
CNN
Swin-Tiny
Mamba
Fusion
Hybrid
```

Example conceptual structure:

```python
MODEL_REGISTRY = {
    "cnn": ...,
    "swin_tiny": ...,
    "mamba": ...,
    "fusion": ...,
    "hybrid": ...,
}
```

Every model must have:

- unique experiment ID
- model name
- configuration
- checkpoint directory
- result directory
- training log
- evaluation output
- native checkpoint
- HDF5 export

Do not duplicate model-specific evaluation logic unnecessarily.

The same evaluation framework should consume each registered model.

---

# 51. MANDATORY TRAINING DELIVERABLES FOR ALL FIVE

Every one of the five models must be actually trained.

Required:

```text
CNN
Swin-Tiny
Mamba
Fusion
Hybrid
```

For each:

- [ ] Training completed
- [ ] Validation performed
- [ ] Best validation checkpoint selected
- [ ] Native checkpoint saved
- [ ] `.h5` export generated
- [ ] `.h5` export verified
- [ ] Training time recorded
- [ ] Inference time recorded
- [ ] Test metrics generated after final configuration freeze
- [ ] Confusion matrix image generated
- [ ] Classification report generated
- [ ] ROC-AUC or PR-AUC generated
- [ ] Cohen's Kappa generated
- [ ] MCC generated

No model is considered complete without its complete artifact package.

---

# 52. MANDATORY METRICS FOR ALL FIVE

For each of:

```text
CNN
Swin-Tiny
Mamba
Fusion
Hybrid
```

calculate and save:

1. Overall Accuracy
2. Precision
3. Recall
4. F1-score
5. Confusion Matrix
6. Classification Report
7. ROC-AUC
8. PR-AUC where applicable/useful
9. Cohen's Kappa
10. Matthews Correlation Coefficient (MCC)
11. Training Time
12. Inference Time

The paper-ready comparison must contain the actual measured values.

No manually entered values.

---

# 53. PAPER-READY FIVE-MODEL RESULT TABLE

Generate a machine-readable result table and a paper-ready version:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Cohen's Kappa | MCC | Training Time | Inference Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Swin-Tiny | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Mamba | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Fusion | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Hybrid | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

The table must be generated from saved experiment outputs where possible.

Do not manually type metrics into the final table.

---

# 54. HDF5 (`.h5`) REQUIREMENT

A trained `.h5` artifact is a **mandatory deliverable for each of the five models**.

Required:

```text
CNN       → cnn_best.h5
Swin-Tiny → swin_tiny_best.h5
Mamba     → mamba_best.h5
Fusion    → fusion_best.h5
Hybrid    → hybrid_best.h5
```

## 54.1 Native checkpoint

Because the implementation may use PyTorch, also save the native PyTorch checkpoint:

```text
CNN:
  cnn_best.pth
  cnn_best.h5

Swin-Tiny:
  swin_tiny_best.pth
  swin_tiny_best.h5

Mamba:
  mamba_best.pth
  mamba_best.h5

Fusion:
  fusion_best.pth
  fusion_best.h5

Hybrid:
  hybrid_best.pth
  hybrid_best.h5
```

The native checkpoint is the primary restoration artifact if the model is implemented in PyTorch.

The `.h5` is an additional required export.

## 54.2 HDF5 integrity requirement

Do not create an empty, placeholder, or fake `.h5` file.

The exported HDF5 must contain the actual trained model weights/parameters in a meaningful format.

The export process must be documented.

After export, perform a verification step:

```text
trained model
      ↓
export .h5
      ↓
reload/inspect .h5
      ↓
verify expected weight tensors / parameter data exist
```

If direct framework-native HDF5 serialization is not supported for the implemented PyTorch model, use a technically valid HDF5 weight export representation and document exactly what the `.h5` contains.

Do not claim that the `.h5` is a directly executable Keras model unless it actually is one.

---

# 55. CONFUSION MATRIX IMAGE REQUIREMENT

A confusion matrix image is mandatory for **each of the five models**.

Required files:

```text
figures/confusion_matrix/
├── confusion_matrix_cnn.png
├── confusion_matrix_swin_tiny.png
├── confusion_matrix_mamba.png
├── confusion_matrix_fusion.png
└── confusion_matrix_hybrid.png
```

Each image must be publication-ready.

## 55.1 Required title

Use explicit titles:

```text
Confusion Matrix — CNN

Confusion Matrix — Swin-Tiny

Confusion Matrix — Mamba

Confusion Matrix — Fusion

Confusion Matrix — Hybrid
```

## 55.2 Required content

Each image must include:

- title
- actual class labels
- predicted class labels
- Normal
- Cancer
- cell counts
- readable axis labels
- consistent formatting
- high-resolution PNG

Recommended:

- 300 DPI
- consistent figure dimensions
- consistent font sizes
- consistent class ordering

The class ordering must remain consistent across all five matrices.

---

# 56. CLASSIFICATION REPORT REQUIREMENT

Generate a classification report for every model.

Required files:

```text
reports/classification/
├── cnn.txt
├── swin_tiny.txt
├── mamba.txt
├── fusion.txt
└── hybrid.txt
```

The report should contain, as appropriate:

- precision
- recall
- F1-score
- support
- macro average
- weighted average

Do not overwrite one model's report with another.

---

# 57. ROC / PR FIGURE REQUIREMENT

For every model, generate at least the ROC curve and ROC-AUC.

Recommended structure:

```text
figures/roc/
├── roc_cnn.png
├── roc_swin_tiny.png
├── roc_mamba.png
├── roc_fusion.png
└── roc_hybrid.png
```

Titles:

```text
ROC Curve — CNN
ROC Curve — Swin-Tiny
ROC Curve — Mamba
ROC Curve — Fusion
ROC Curve — Hybrid
```

If PR-AUC is calculated, also generate PR curves:

```text
figures/pr/
├── pr_cnn.png
├── pr_swin_tiny.png
├── pr_mamba.png
├── pr_fusion.png
└── pr_hybrid.png
```

---

# 58. TRAINING TIME

Training time is mandatory for all five models.

Measure actual wall-clock training duration.

Record:

```text
model
start_time
end_time
total_training_seconds
total_training_minutes
epochs_completed
```

Do not estimate training time from epoch duration unless clearly marked as estimated.

Preferred:

```text
training_time_seconds
```

as the primary stored value.

---

# 59. INFERENCE TIME

Inference time is mandatory for all five models.

Measure actual inference time on the same evaluation environment.

The measurement protocol must be consistent across models.

Recommended procedure:

1. load the final checkpoint
2. move model to evaluation mode
3. warm up GPU if applicable
4. synchronize CUDA where applicable
5. measure inference over a defined number of test samples/batches
6. exclude unrelated preprocessing/file I/O where possible
7. report:
   - total inference time
   - average inference time per image

Store both:

```text
total_inference_seconds
average_inference_ms_per_image
```

Do not compare times obtained under different batch sizes or hardware conditions without documenting the difference.

---

# 60. COHEN'S KAPPA

Calculate Cohen's Kappa from final test predictions.

Save:

```text
cohens_kappa
```

in the model's metrics JSON.

Do not calculate it from validation predictions and label it as test Kappa.

---

# 61. MCC

Calculate Matthews Correlation Coefficient from final binary test predictions.

Save:

```text
mcc
```

in the model's metrics JSON.

MCC must be computed consistently for all five models.

---

# 62. METRICS JSON SCHEMA

Each model should produce something conceptually similar to:

```json
{
  "model": "hybrid",
  "dataset": {
    "train": 7057,
    "validation": 882,
    "test": 872
  },
  "accuracy": null,
  "precision": null,
  "recall": null,
  "f1": null,
  "roc_auc": null,
  "pr_auc": null,
  "cohens_kappa": null,
  "mcc": null,
  "training_time_seconds": null,
  "inference_time_seconds": null,
  "inference_ms_per_image": null
}
```

`null` is appropriate before the experiment is actually completed.

Never replace null/TBD with fabricated numbers.

---

# 63. ATTENTION FUSION OUTPUTS

The Hybrid model contains Adaptive Attention Fusion.

If the Fusion configuration also uses the same adaptive fusion mechanism, export its actual learned attention weights as well.

Required artifacts for every model that contains Adaptive Attention Fusion:

```text
attention/
├── attention_weights.csv
├── attention_weights.json
├── attention_weights.png
└── classwise_attention.png
```

At minimum capture:

```text
CNN weight
Swin-Tiny weight
Mamba weight
```

for the relevant model.

## 63.1 Visualization titles

Use:

```text
Adaptive Attention Fusion Weights — Fusion

Adaptive Attention Fusion Weights — Hybrid
```

and, if class-wise analysis is generated:

```text
Class-wise Adaptive Attention Fusion Weights — Fusion

Class-wise Adaptive Attention Fusion Weights — Hybrid
```

Only generate the Fusion-specific files if Fusion actually implements adaptive attention.

Do not create fake attention visualizations for models that do not contain attention fusion.

---

# 64. FINAL ARTIFACT STRUCTURE

The final project should contain an artifact structure approximately like:

```text
artifacts/
│
├── models/
│   ├── cnn/
│   │   ├── cnn_best.pth
│   │   └── cnn_best.h5
│   │
│   ├── swin_tiny/
│   │   ├── swin_tiny_best.pth
│   │   └── swin_tiny_best.h5
│   │
│   ├── mamba/
│   │   ├── mamba_best.pth
│   │   └── mamba_best.h5
│   │
│   ├── fusion/
│   │   ├── fusion_best.pth
│   │   └── fusion_best.h5
│   │
│   └── hybrid/
│       ├── hybrid_best.pth
│       └── hybrid_best.h5
│
├── metrics/
│   ├── cnn.json
│   ├── swin_tiny.json
│   ├── mamba.json
│   ├── fusion.json
│   └── hybrid.json
│
├── reports/
│   ├── cnn.txt
│   ├── swin_tiny.txt
│   ├── mamba.txt
│   ├── fusion.txt
│   └── hybrid.txt
│
├── confusion_matrix/
│   ├── confusion_matrix_cnn.png
│   ├── confusion_matrix_swin_tiny.png
│   ├── confusion_matrix_mamba.png
│   ├── confusion_matrix_fusion.png
│   └── confusion_matrix_hybrid.png
│
├── roc/
│   ├── roc_cnn.png
│   ├── roc_swin_tiny.png
│   ├── roc_mamba.png
│   ├── roc_fusion.png
│   └── roc_hybrid.png
│
├── pr/
│   ├── pr_cnn.png
│   ├── pr_swin_tiny.png
│   ├── pr_mamba.png
│   ├── pr_fusion.png
│   └── pr_hybrid.png
│
├── attention/
│   ├── fusion/
│   └── hybrid/
│
└── comparison/
    ├── five_model_results.csv
    └── five_model_results.json
```

---

# 65. FIVE-MODEL TRAINING CHECKLIST

Before declaring the training stage complete:

## CNN

- [ ] Trained
- [ ] Best checkpoint saved
- [ ] `.pth` saved
- [ ] `.h5` saved and verified
- [ ] Test evaluation completed
- [ ] Metrics saved
- [ ] Classification report saved
- [ ] Confusion matrix image saved
- [ ] ROC curve saved
- [ ] PR curve saved if applicable
- [ ] Training time saved
- [ ] Inference time saved

## Swin-Tiny

- [ ] Trained
- [ ] Best checkpoint saved
- [ ] `.pth` saved
- [ ] `.h5` saved and verified
- [ ] Test evaluation completed
- [ ] Metrics saved
- [ ] Classification report saved
- [ ] Confusion matrix image saved
- [ ] ROC curve saved
- [ ] PR curve saved if applicable
- [ ] Training time saved
- [ ] Inference time saved

## Mamba

- [ ] Trained
- [ ] Best checkpoint saved
- [ ] `.pth` saved
- [ ] `.h5` saved and verified
- [ ] Test evaluation completed
- [ ] Metrics saved
- [ ] Classification report saved
- [ ] Confusion matrix image saved
- [ ] ROC curve saved
- [ ] PR curve saved if applicable
- [ ] Training time saved
- [ ] Inference time saved

## Fusion

- [ ] Trained
- [ ] Best checkpoint saved
- [ ] `.pth` saved
- [ ] `.h5` saved and verified
- [ ] Test evaluation completed
- [ ] Metrics saved
- [ ] Classification report saved
- [ ] Confusion matrix image saved
- [ ] ROC curve saved
- [ ] PR curve saved if applicable
- [ ] Training time saved
- [ ] Inference time saved
- [ ] Attention weights saved if Fusion contains adaptive attention

## Hybrid

- [ ] Trained
- [ ] Best checkpoint saved
- [ ] `.pth` saved
- [ ] `.h5` saved and verified
- [ ] Test evaluation completed
- [ ] Metrics saved
- [ ] Classification report saved
- [ ] Confusion matrix image saved
- [ ] ROC curve saved
- [ ] PR curve saved if applicable
- [ ] Training time saved
- [ ] Inference time saved
- [ ] Attention weights saved
- [ ] Attention visualization saved

---

# 66. UPDATED DEFINITION OF DONE

The conference model implementation is **not done** when the Hybrid model merely trains.

It is done only when:

```text
CNN
Swin-Tiny
Mamba
Fusion
Hybrid
```

have all been trained and have complete, traceable evaluation artifacts.

Minimum final package:

```text
5 trained models
+
5 native checkpoints
+
5 HDF5 exports
+
5 metric records
+
5 classification reports
+
5 confusion-matrix PNGs
+
5 ROC curves
+
5 training-time measurements
+
5 inference-time measurements
+
required PR-AUC measurements/plots
+
required Kappa measurements
+
required MCC measurements
+
attention-fusion artifacts for models that contain adaptive fusion
+
one final five-model comparison table
```

No placeholder files.

No fake values.

No missing model simply because it performed poorly.

---

# 67. UPDATED EXECUTION ORDER

The deadline remains critical.

Use this order:

```text
1. Dataset audit
2. Preprocessing
3. Augmentation
4. CNN implementation
5. CNN training + evaluation
6. Swin-Tiny implementation
7. Swin-Tiny training + evaluation
8. Mamba implementation
9. Mamba training + evaluation
10. Fusion implementation
11. Fusion training + evaluation
12. Hybrid implementation
13. Hybrid training + evaluation
14. Generate all five confusion matrices
15. Generate all five ROC/PR outputs
16. Generate Kappa + MCC
17. Record training/inference time
18. Export all five .pth
19. Export and verify all five .h5
20. Generate attention-fusion artifacts
21. Generate final five-model comparison table
22. Freeze experimental results
23. Begin paper writing
```

Do not begin paper claims until the corresponding result artifacts exist.

---

# 68. FINAL INSTRUCTION

The **five-model requirement is now locked**:

```text
CNN
Swin-Tiny
Mamba
Fusion
Hybrid
```

Every one must reach the same final evaluation pipeline.

The final goal is not merely:

> "the Hybrid model works."

The goal is:

> "We have five actually trained, reproducible model configurations, each with a verified checkpoint/export, complete quantitative evaluation, publication-ready confusion matrix, timing measurements, and a directly comparable result record."

Only then should the paper's comparative claims be written.

