# NEXT-GEN PRD --- BoneMambaFormer

## MobileNetV2 + Swin-Tiny + Mamba + Adaptive Softmax Attention Fusion

**Project:** ICIMCPS-2026 Bone Cancer Classification\
**Purpose:** Authoritative execution specification for the next revision
of the already-developed pipeline.\
**Primary revision:** Replace the legacy ResNet-18 CNN branch with
**MobileNetV2**, correct model-selection/overfitting behavior with
validation-based early stopping and best-checkpoint restoration, then
regenerate all downstream evidence affected by the change.

------------------------------------------------------------------------

# 1. Executive Directive

This is **not a new project from scratch**.

The existing dataset audit, preprocessing pipeline, Swin-Tiny branch,
Mamba branch, Attention Fusion, End-to-End Hybrid, Phase 8 evaluation,
Phase 9 ablation, and Phase 10 evidence/report work already exist.

The Next-Gen revision makes only the changes that are now justified and
required:

1.  Replace **ResNet-18 → MobileNetV2** for the CNN branch.
2.  Retrain the standalone CNN baseline as MobileNetV2.
3.  Retrain Attention Fusion using MobileNetV2 + Swin-Tiny + Mamba.
4.  Apply proper validation-based early stopping/checkpoint restoration
    to Attention Fusion.
5.  Retrain the End-to-End Hybrid using MobileNetV2 + Swin-Tiny +
    Mamba + Adaptive Softmax Attention.
6.  Apply proper validation-based early stopping/checkpoint restoration
    to the Hybrid.
7.  Re-run Phase 8 final evaluation for the five final configurations.
8.  Re-run Phase 9 ablations with MobileNetV2 as the CNN branch.
9.  Recalculate parameter counts, metrics, figures, tables, and all
    affected report evidence.
10. Preserve legacy ResNet-18 results for traceability, but **never mix
    them into the final Next-Gen five-model benchmark**.

Do not restart already validated components unnecessarily.

------------------------------------------------------------------------

# 2. Source-of-Truth Rules

The existing technical handoff establishes the following priorities:

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

When something is unknown:

``` text
TBD
```

or:

``` text
NEEDS VERIFICATION
```

Never fabricate a metric, parameter count, checkpoint, training time, or
architectural property.

The original handoff also requires preserving working code/checkpoints
before major architectural changes and recommends minimal justified
changes when failures occur.

------------------------------------------------------------------------

# 3. Final Next-Gen Architecture

``` text
                         Bone Image
                             │
                             ▼
                Preprocessing + Augmentation
                             │
                             ▼
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
  MobileNetV2            Swin-Tiny             Mamba / SSM
   CNN branch          Transformer branch      State-space branch
       │                     │                     │
       ▼                     ▼                     ▼
  Feature projection    Feature projection    Feature projection
       │                     │                     │
       └─────────────────────┼─────────────────────┘
                             ▼
                Adaptive Softmax Attention
                         Fusion Module
                             │
                             ▼
                    Feature Refinement
                             │
                             ▼
                    Binary Classifier
                             │
                             ▼
                     Normal / Cancer
```

The existing project architecture is conceptually:

``` text
CNN + Swin Transformer + Mamba
        ↓
Adaptive Attention Fusion
        ↓
Feature Refinement
        ↓
Fully Connected Classification
        ↓
Normal / Cancer
```

This is retained; only the CNN backbone changes from ResNet-18 to
MobileNetV2. The original handoff explicitly defines the binary
classification objective and this hybrid architecture.

------------------------------------------------------------------------

# 4. Why MobileNetV2

The coordinator explicitly requested **MobileNetV2** instead of the
previous ResNet-18 CNN branch.

Therefore MobileNetV2 is the canonical CNN branch for the Next-Gen
pipeline.

The change must be treated as an architectural revision, not as an
assumed accuracy improvement.

Correct scientific statement:

> MobileNetV2 is introduced as the revised lightweight CNN branch, and
> its standalone and downstream fusion performance are empirically
> evaluated under the locked validation/test protocol.

Do **not** claim that MobileNetV2 is superior until the new experiment
proves it.

------------------------------------------------------------------------

# 5. Dataset Protocol

The existing dataset audit reports:

-   Original images are RGB JPEG.
-   Original resolution is 640 × 640.
-   Binary labels:
    -   Class 0 = Cancer
    -   Class 1 = Normal.
-   The original supplied split contains approximately 8.8k images.
-   A leakage-clean `derived_clean` protocol was subsequently
    constructed using image-hash deduplication.
-   Current `derived_clean` evidence reports:
    -   Train: 7,417
    -   Validation: 698
    -   Untouched Test: 695
    -   Test: 383 Cancer, 312 Normal.

Use the **already-finalized data pipeline and derived_clean protocol**.

Do not redesign the dataset merely because the CNN backbone changes.

## Test-set rule

The test set is never used for:

-   hyperparameter tuning
-   architecture selection
-   early stopping
-   threshold tuning
-   branch selection
-   repeated optimization

The final test evaluation occurs only after architecture and checkpoint
selection are frozen.

------------------------------------------------------------------------

# 6. Preprocessing

The finalized evidence package uses:

``` text
224 × 224 × 3 RGB
```

with the established preprocessing and augmentation pipeline.

Keep this pipeline unchanged unless an actual MobileNetV2 compatibility
requirement makes a minimal documented adjustment necessary.

Do not introduce a new preprocessing pipeline simply to improve a
result.

------------------------------------------------------------------------

# 7. Final Five Models

The final benchmark must contain exactly:

  -----------------------------------------------------------------------
  ID                Final model       Architecture      Training mode
  ----------------- ----------------- ----------------- -----------------
  M1                MobileNetV2       MobileNetV2       Standalone

  M2                Swin-Tiny         Swin-Tiny         Standalone

  M3                Mamba             Mamba / SSM       Standalone

  M4                Attention Fusion  MobileNetV2 +     Frozen branches +
                                      Swin-Tiny + Mamba trainable fusion

  M5                End-to-End Hybrid MobileNetV2 +     Joint end-to-end
                                      Swin-Tiny +       
                                      Mamba + adaptive  
                                      attention         
  -----------------------------------------------------------------------

**ResNet-18 is not a final model.**

It may remain as a historical baseline.

------------------------------------------------------------------------

# 8. M1 --- MobileNetV2 Standalone

## Objective

Replace the legacy CNN baseline with MobileNetV2.

## Requirements

-   Use MobileNetV2.
-   Preserve the existing binary classification interface.
-   Preserve the established feature projection convention where
    compatible.
-   Use the existing train/validation protocol.
-   Select the best checkpoint from validation performance.
-   Generate the same evidence categories as the other standalone
    models.

## Required outputs

-   best checkpoint
-   training history
-   validation history
-   confusion matrix
-   ROC curve
-   PR curve
-   classification report
-   Accuracy
-   Precision
-   Recall
-   F1
-   ROC-AUC
-   PR-AUC
-   Cohen's Kappa
-   MCC
-   training time
-   inference latency
-   throughput
-   total parameters
-   trainable parameters
-   best validation epoch
-   best validation accuracy

------------------------------------------------------------------------

# 9. M2 --- Swin-Tiny

Swin-Tiny is a carried-forward component.

Do not redesign it.

If its existing checkpoint and interface are compatible with the new
MobileNetV2-based fusion pipeline, reuse them.

Retrain only if necessary for compatibility/reproducibility or if an
actual implementation issue requires it.

If retraining occurs, document why.

------------------------------------------------------------------------

# 10. M3 --- Mamba / SSM

Mamba is also carried forward.

Do not redesign the Mamba architecture.

Retrain only if required by compatibility, reproducibility, or an actual
dependency introduced by the Next-Gen pipeline.

------------------------------------------------------------------------

# 11. M4 --- Attention Fusion

## Architecture

``` text
MobileNetV2 ──┐
Swin-Tiny ────┼──> projected branch features
Mamba ────────┘
                    ↓
             Softmax Branch Attention
                    ↓
              Weighted Feature Fusion
                    ↓
               Feature Refinement
                    ↓
                 Classifier
```

The existing Attention Fusion implementation freezes the backbone
branches and trains the fusion head.

The previous fusion head had 49,923 trainable parameters while the
overall model scale was approximately 40M due to the three underlying
branches.

The Next-Gen implementation should preserve this conceptual strategy
unless an actual interface issue requires a justified modification.

------------------------------------------------------------------------

# 12. Attention Fusion --- Overfitting Correction

The previous Attention Fusion training graph showed clear
training/validation divergence:

-   training loss approached zero
-   training accuracy approached 100%
-   validation accuracy peaked early
-   validation loss continued increasing

The previous best validation point was **Epoch 8 at 98.85% validation
accuracy**.

Therefore the Next-Gen Attention Fusion run must use validation-based
model selection.

## Required checkpoint policy

``` text
monitor = val_accuracy
mode = max
save_best_only = True
restore_best_weights = True
patience = 3
```

Recommended moderate regularization:

``` text
Dropout = 0.30
AdamW weight_decay = 1e-4
```

These are starting values, not guaranteed optimal values.

Do not tune them against the test set.

## Key principle

The final Attention Fusion model is the **best validation checkpoint**,
not the last epoch.

------------------------------------------------------------------------

# 13. Attention Fusion Training Curves

The final training figure must show:

-   training loss
-   validation loss
-   training accuracy
-   validation accuracy
-   best validation epoch
-   selected checkpoint/early stopping marker when practical

The figure must remain scientifically honest.

Do not smooth, crop, rescale, or manipulate the graph to hide
overfitting.

If validation performance stops improving while training performance
continues improving, report this explicitly.

------------------------------------------------------------------------

# 14. M5 --- End-to-End Hybrid

## Final architecture

``` text
                    Input Image
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     MobileNetV2     Swin-Tiny      Mamba
          │             │             │
          └─────────────┼─────────────┘
                        ▼
              Adaptive Softmax Attention
                        ▼
                 Feature Refinement
                        ▼
                    Classifier
                        ▼
                 Normal / Cancer
```

The three branches, attention module, refinement component, and
classifier are jointly optimized end-to-end.

------------------------------------------------------------------------

# 15. Hybrid Parameter Count

The legacy ResNet-18 Hybrid contained exactly:

``` text
39,737,093 trainable parameters
≈ 40M
```

That number is **not valid for the MobileNetV2 revision** because the
CNN backbone has changed.

The new exact parameter count must be calculated from the actual
Next-Gen model.

## Mandatory final report statement

The final report must include:

> **The End-to-End Hybrid Model is trained on approximately 40M
> trainable parameters.**

If the measured MobileNetV2 model has a materially different total,
report the exact measured number and do not force the old 40M figure.

Never copy the old ResNet-18 parameter count into the new report.

------------------------------------------------------------------------

# 16. Hybrid --- Overfitting Correction

The legacy Hybrid run reached approximately **98.71% best validation
accuracy**.

That old epoch/result must be treated as historical reference only.

The new MobileNetV2 Hybrid must determine its own best epoch.

Required policy:

``` text
monitor = val_accuracy
mode = max
save_best_only = True
restore_best_weights = True
patience = 3
```

Do not hardcode the old best epoch.

Do not assume the new model will reach 98.71%.

The new experiment is authoritative.

------------------------------------------------------------------------

# 17. Early Stopping Policy

Early stopping is a **model-selection mechanism**, not merely a speed
optimization.

For every newly retrained model:

1.  Track validation performance every epoch.
2.  Save the best checkpoint.
3.  Stop after the patience window is exhausted.
4.  Restore the best checkpoint.
5.  Record the best epoch.
6.  Record best validation accuracy/loss.
7.  Freeze the selected configuration.
8.  Only then evaluate the untouched test set.

Correct:

``` text
Training
   ↓
Validation monitoring
   ↓
Best checkpoint
   ↓
Early stopping
   ↓
Restore best weights
   ↓
Freeze configuration
   ↓
Untouched test evaluation
```

Incorrect:

``` text
Train fixed number of epochs
   ↓
take final epoch
   ↓
evaluate test
```

------------------------------------------------------------------------

# 18. Test Evaluation Freeze

Once the final configuration is selected:

``` text
Architecture
      ↓
Hyperparameters
      ↓
Checkpoint
      ↓
Threshold
      ↓
FREEZE
      ↓
Test evaluation
```

After the test run starts, do not modify:

-   architecture
-   learning rate
-   regularization
-   threshold
-   checkpoint
-   augmentation
-   feature dimensions

If a new architecture change is required after test evaluation, it
starts a new development cycle.

------------------------------------------------------------------------

# 19. Phase 8 --- Final Evaluation

The original development order defines Phase 8 as:

``` text
validation
→ freeze
→ test
→ metrics
```

The Next-Gen Phase 8 must evaluate:

1.  MobileNetV2
2.  Swin-Tiny
3.  Mamba
4.  Attention Fusion
5.  End-to-End Hybrid

## Required metrics for every model

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Macro Precision
-   Macro Recall
-   Macro F1
-   ROC-AUC
-   PR-AUC
-   Cohen's Kappa
-   MCC
-   confusion matrix
-   training time
-   inference time
-   average inference latency
-   throughput/FPS
-   total parameters
-   trainable parameters
-   best validation epoch
-   best validation accuracy
-   generalization gap

------------------------------------------------------------------------

# 20. Required Final Figures

## MobileNetV2

-   confusion matrix
-   ROC curve
-   PR curve
-   training/validation loss
-   training/validation accuracy

## Swin-Tiny

-   confusion matrix
-   ROC curve
-   PR curve
-   training/validation curves

## Mamba

-   confusion matrix
-   ROC curve
-   PR curve
-   training/validation curves

## Attention Fusion

-   confusion matrix
-   combined ROC + PR
-   training/validation curves
-   overall learned branch contribution
-   class-specific branch contribution where supported

## End-to-End Hybrid

-   confusion matrix
-   combined ROC + PR
-   training/validation curves
-   overall learned branch contribution
-   class-specific branch contribution where supported

The existing report structure already follows this five-model figure
philosophy. The Next-Gen evidence must preserve the coverage while
replacing ResNet-18 with MobileNetV2.

------------------------------------------------------------------------

# 21. Attention Interpretability

For Attention Fusion and Hybrid, report:

``` text
MobileNetV2 contribution
Swin-Tiny contribution
Mamba contribution
```

Include:

-   overall mean contribution
-   class-specific contribution
-   relevant distribution statistics if available

Use the phrase:

> learned attention contribution

Do not call a high attention weight definitive causal importance.

------------------------------------------------------------------------

# 22. Phase 9 --- Next-Gen Ablation

The existing Phase 9 design is:

``` text
single branches
→ pairwise combinations
→ full hybrid
```

The Next-Gen ablation must replace the legacy CNN branch with
MobileNetV2.

## Required configurations

### Standalone

1.  MobileNetV2
2.  Swin-Tiny
3.  Mamba

### Pairwise

4.  MobileNetV2 + Swin-Tiny
5.  MobileNetV2 + Mamba
6.  Swin-Tiny + Mamba

### Full

7.  MobileNetV2 + Swin-Tiny + Mamba + Adaptive Attention

### Fusion control

Where the existing implementation supports it:

8.  Equal weighting
9.  Learned adaptive attention

Do not add arbitrary ablations just to increase the number of
experiments.

------------------------------------------------------------------------

# 23. Ablation Questions

Phase 9 must answer:

1.  Does each branch contribute useful information?
2.  Which branches are complementary?
3.  Does adaptive attention improve over equal weighting?
4.  Which branch receives the greatest learned contribution?
5.  What happens when a branch is removed?
6.  Does the final architecture justify its additional complexity?

The previous ablation results involving ResNet-18 are historical and
must be regenerated.

For example, the old finding that removing Swin caused a \~29% accuracy
drop belongs to the ResNet-18 experiment and cannot be copied into the
Next-Gen report without rerunning the ablation.

------------------------------------------------------------------------

# 24. Ablation Fairness

Every ablation must use:

-   the same finalized dataset protocol
-   the same preprocessing
-   the same validation-selection logic
-   the same metric definitions
-   the same untouched test protocol

Never compare:

``` text
best-checkpoint model
```

against:

``` text
arbitrary final-epoch model
```

and call it a fair architecture comparison.

------------------------------------------------------------------------

# 25. Phase 10 --- Final Evidence

The final evidence stage must contain:

## Dataset

-   dataset description
-   class distribution
-   split distribution
-   preprocessing
-   augmentation
-   duplicate/leakage audit
-   derived_clean construction
-   patient-level limitation

## Architecture

-   complete architecture diagram
-   MobileNetV2 branch
-   Swin-Tiny branch
-   Mamba branch
-   feature projections
-   Adaptive Softmax Attention
-   feature refinement
-   binary classifier

## Training

-   optimizer
-   learning rate
-   weight decay
-   batch size
-   epochs
-   early stopping
-   checkpoint selection
-   augmentation
-   hardware
-   random seed

## Results

-   five-model benchmark
-   all metrics
-   all confusion matrices
-   all ROC curves
-   all PR curves
-   all training curves
-   attention analysis
-   ablation analysis
-   computational efficiency

## Discussion

-   strengths
-   weaknesses
-   overfitting behavior
-   generalization gap
-   branch contribution
-   limitations
-   future work

------------------------------------------------------------------------

# 26. Old Results vs Next-Gen Results

This distinction is mandatory.

## Legacy

``` text
ResNet-18 + Swin-Tiny + Mamba
```

## Next-Gen

``` text
MobileNetV2 + Swin-Tiny + Mamba
```

Old results may be preserved for historical comparison, but the final
benchmark must use only the Next-Gen configuration.

Never create a table such as:

``` text
Old ResNet18 standalone
+
New MobileNetV2 fusion
+
Old Swin
+
New Hybrid
```

and call it one final five-model benchmark.

All downstream models must use the same finalized CNN branch.

------------------------------------------------------------------------

# 27. Parameter Audit

Recalculate all parameters after MobileNetV2 integration.

Required:

``` text
Total parameters
Trainable parameters
Non-trainable parameters
MobileNetV2 parameters
Swin-Tiny parameters
Mamba parameters
Fusion/refinement/head parameters
```

Do not reuse the old 39,737,093 value.

The exact count must come from the actual model object.

------------------------------------------------------------------------

# 28. Artifact Policy

For each newly trained model, preserve:

``` text
checkpoints/
    <model>_best.pth

results/
    <model>/
        metrics.json
        classification_report.txt
        training_history.json

figures/
    confusion_matrix/
    roc/
    pr/
    training/
    attention/
    ablation/
```

Follow the existing directory conventions wherever possible.

Do not create a second parallel project structure.

------------------------------------------------------------------------

# 29. Script Policy

**Do not create unnecessary scripts.**

Reuse the existing:

-   dataset implementation
-   preprocessing
-   training loop
-   evaluation framework
-   plotting utilities
-   model interfaces
-   report evidence utilities

Modify existing scripts where practical.

New scripts are permitted only when genuinely necessary, such as:

-   MobileNetV2 implementation if no compatible module exists
-   required migration/configuration support
-   final Markdown compilation
-   DOCX generation
-   report compilation

Do not create multiple near-identical scripts such as:

``` text
phase8_final.py
phase8_final2.py
phase8_fixed.py
phase8_really_final.py
```

Keep the repository clean.

------------------------------------------------------------------------

# 30. Version-Control Safety

Before the MobileNetV2 migration:

1.  Preserve working code.
2.  Preserve ResNet-18 checkpoints.
3.  Preserve old results.
4.  If Git is available, commit the working state.
5.  Use a meaningful commit message.
6.  Keep large datasets/checkpoints out of Git unless explicitly
    intended.

The old ResNet-18 experiment must remain recoverable.

------------------------------------------------------------------------

# 31. Failure Handling

If anything fails:

``` text
1. Save the error.
2. Identify the likely cause.
3. Make the smallest justified change.
4. Rerun.
5. Record the changed configuration.
```

Expected classes of failure include:

-   CUDA OOM
-   tensor shape mismatch
-   NaN
-   augmentation incompatibility
-   unsupported operator
-   output-dimension mismatch

Do not hide failures.

------------------------------------------------------------------------

# 32. RTX 3050 6GB Constraint

Before full Hybrid training:

``` text
memory smoke test
→ peak VRAM
→ batch size
→ input resolution
→ AMP status
```

If OOM occurs, use this order:

1.  enable AMP
2.  reduce batch size
3.  freeze branches where scientifically appropriate
4.  gradient accumulation
5.  reduce feature dimension
6.  reduce input resolution only if scientifically acceptable

Do not immediately remove a required architectural component.

------------------------------------------------------------------------

# 33. Reproducibility

Keep:

-   deterministic seed
-   centralized configuration
-   reusable dataset class
-   reusable training loop
-   reusable evaluation code
-   logging

Record for each experiment:

``` text
Python version
PyTorch version
CUDA version
GPU
seed
batch size
learning rate
weight decay
optimizer
scheduler
epochs
early stopping patience
input resolution
augmentation
checkpoint
best validation epoch
best validation accuracy
```

------------------------------------------------------------------------

# 34. Final Architecture Figure Requirement

The final report must contain a clean architecture figure equivalent to:

``` text
                           INPUT
                      Bone Radiograph
                            │
                            ▼
                 Preprocessing / Augmentation
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
     MobileNetV2        Swin-Tiny          Mamba
     CNN Branch       Transformer Branch   SSM Branch
          │                 │                 │
          ▼                 ▼                 ▼
      Projection        Projection        Projection
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                Adaptive Softmax Attention
                            │
                            ▼
                   Weighted Feature Fusion
                            │
                            ▼
                     Feature Refinement
                            │
                            ▼
                     Binary Classifier
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
                Cancer             Normal
```

Before final report generation, verify this diagram against the actual
code.

------------------------------------------------------------------------

# 35. Exact Execution Order

Execute progressively and do not skip dependency checks.

``` text
1. Inspect existing project
        ↓
2. Verify existing branch interfaces
        ↓
3. Preserve legacy ResNet18 state
        ↓
4. Replace CNN branch with MobileNetV2
        ↓
5. Smoke-test MobileNetV2 forward/backward
        ↓
6. Train MobileNetV2 standalone
        ↓
7. Select best MobileNetV2 validation checkpoint
        ↓
8. Integrate MobileNetV2 into Attention Fusion
        ↓
9. Train Attention Fusion with early stopping
        ↓
10. Select best Attention Fusion checkpoint
        ↓
11. Integrate MobileNetV2 into End-to-End Hybrid
        ↓
12. Memory smoke test
        ↓
13. Train Hybrid with early stopping
        ↓
14. Select best Hybrid checkpoint
        ↓
15. Freeze all five final configurations
        ↓
16. Run Phase 8 untouched test evaluation
        ↓
17. Run Next-Gen Phase 9 ablations
        ↓
18. Regenerate final figures/tables
        ↓
19. Generate final Markdown evidence/report
        ↓
20. Perform final consistency audit
```

------------------------------------------------------------------------

# 36. Final Consistency Audit

## Architecture

-   [ ] MobileNetV2 is actually instantiated.
-   [ ] No final model accidentally uses ResNet-18.
-   [ ] Swin-Tiny remains correct.
-   [ ] Mamba remains correct.
-   [ ] Adaptive Softmax Attention exists.
-   [ ] Feature refinement exists.
-   [ ] Binary classifier exists.

## Training

-   [ ] Early stopping implemented.
-   [ ] Best checkpoint saved.
-   [ ] Best checkpoint restored.
-   [ ] Attention Fusion uses validation-based selection.
-   [ ] Hybrid uses validation-based selection.
-   [ ] No test-set tuning.

## Evaluation

-   [ ] Five final models evaluated.
-   [ ] Same test protocol.
-   [ ] Same metric definitions.
-   [ ] Confusion matrices generated.
-   [ ] ROC generated.
-   [ ] PR generated.
-   [ ] Training curves generated.
-   [ ] Attention plots generated.
-   [ ] Parameter counts recalculated.
-   [ ] Training/inference efficiency recalculated.

## Ablation

-   [ ] MobileNetV2 is the CNN branch.
-   [ ] All required pairwise combinations are evaluated.
-   [ ] Full hybrid is evaluated.
-   [ ] Equal-weight vs adaptive attention is evaluated if supported.
-   [ ] No old ResNet-18 ablation result is presented as a Next-Gen
    result.

## Report

-   [ ] Architecture figure updated.
-   [ ] Five-model table updated.
-   [ ] Parameter table updated.
-   [ ] Training details updated.
-   [ ] Early stopping explained.
-   [ ] Overfitting discussed honestly.
-   [ ] Leakage-clean protocol explained.
-   [ ] Limitations included.
-   [ ] \~40M Hybrid parameter statement included where numerically
    accurate.
-   [ ] Legacy vs Next-Gen results clearly separated.

------------------------------------------------------------------------

# 37. Definition of Done

Next-Gen is complete only when:

``` text
MobileNetV2
    +
Swin-Tiny
    +
Mamba
    +
Adaptive Softmax Attention
    +
Feature Refinement
    ↓
Final Hybrid
    ↓
Best validation checkpoint
    ↓
Untouched test evaluation
    ↓
Phase 9 ablation
    ↓
Complete evidence package
    ↓
Final Markdown report
```

The final five-model benchmark must be internally consistent:

``` text
1. MobileNetV2
2. Swin-Tiny
3. Mamba
4. Attention Fusion
5. End-to-End Hybrid
```

No metric may be copied from the legacy ResNet-18 experiment into the
final Next-Gen benchmark.

No test-set result may drive a new architecture or hyperparameter
decision.

No unnecessary scripts should be created.

No overfitting should be hidden from the report.

The objective is not to manufacture a higher number. The objective is to
produce a cleaner, reproducible, scientifically defensible final
architecture and evidence package.

------------------------------------------------------------------------

# 38. Historical Reference Values

The previous ResNet-18-based experiment reported:

-   Attention Fusion best validation epoch: **Epoch 8**
-   Attention Fusion best validation accuracy: **98.85%**
-   Hybrid best validation accuracy: approximately **98.71%**
-   Legacy End-to-End Hybrid trainable parameters: **39,737,093
    (\~40M)**

These values are historical references only.

They must not be copied into the Next-Gen final benchmark after
MobileNetV2 migration.

The new MobileNetV2, Attention Fusion, Hybrid, Phase 8, and Phase 9
values must be generated from the new experiments.

------------------------------------------------------------------------

# 39. Final Agent Command

**Execute this PRD progressively.**

Do not restart the entire project.

Do not redesign validated components.

Do not introduce unrelated improvements.

Do not chase raw accuracy at the expense of reproducibility.

First make the CNN branch MobileNetV2.

Then retrain and select the MobileNetV2 checkpoint.

Then rebuild the dependent Attention Fusion.

Then rebuild the dependent End-to-End Hybrid.

Use early stopping and best-checkpoint restoration.

Then freeze.

Then evaluate.

Then ablate.

Then regenerate evidence.

Then produce the final Markdown report.

The final scientific story must be:

``` text
Lightweight CNN representation
        +
Hierarchical Transformer representation
        +
State-space representation
        ↓
Adaptive learned fusion
        ↓
Feature refinement
        ↓
Binary bone-cancer classification
```

with validation-controlled model selection and a single untouched final
test evaluation.
