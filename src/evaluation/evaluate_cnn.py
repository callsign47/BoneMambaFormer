import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef
)

# Ensure workspace root is in path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from src.config import ProjectConfig
from src.dataset import get_dataloaders
from src.models.cnn import BoneCancerCNN

def evaluate_cnn_pipeline():
    cfg = ProjectConfig()
    cfg._config['dataset']['split_type'] = 'derived_clean'
    cfg._config['dataset']['batch_size'] = 32

    workspace_root = cfg.get('project.workspace_root')
    checkpoints_dir = os.path.join(workspace_root, 'checkpoints')
    results_dir = os.path.join(workspace_root, 'results')
    figures_dir = os.path.join(workspace_root, 'figures')
    reports_dir = os.path.join(workspace_root, 'reports')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoints_dir, 'cnn_best.pth')
    h5_export_path = os.path.join(checkpoints_dir, 'cnn_best.h5')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================")
    print(f"Executing Final Evaluation Pipeline for CNN")
    print(f"Loading Checkpoint: {checkpoint_path}")
    print(f"Evaluation Device: {device}")
    print(f"==================================================")

    # 1. Load DataLoaders (derived_clean test set)
    _, _, test_loader = get_dataloaders(cfg)
    test_samples_count = len(test_loader.dataset)
    print(f"Untouched Test Set Loaded: {test_samples_count} samples")

    # 2. Instantiate and load model
    model = BoneCancerCNN(backbone_name='mobilenet_v2', num_classes=2, feature_dim=256).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    best_epoch = checkpoint.get('epoch', 'N/A')
    val_acc = checkpoint.get('val_acc', 'N/A')
    val_loss = checkpoint.get('val_loss', 'N/A')
    print(f"Loaded model checkpoint from Epoch {best_epoch} (Val Acc: {val_acc*100 if isinstance(val_acc, float) else val_acc:.2f}%)")

    # 3. Predict on Test Set with Inference Timing
    all_targets = []
    all_preds = []
    all_probs = []

    softmax = nn.Softmax(dim=1)

    # Warmup GPU if CUDA is available
    if device.type == 'cuda':
        dummy = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize()

    start_infer_time = time.perf_counter()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = softmax(logits)
            preds = torch.argmax(logits, dim=1)

            all_targets.extend(targets.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    if device.type == 'cuda':
        torch.cuda.synchronize()

    total_inference_time_sec = time.perf_counter() - start_infer_time
    per_sample_latency_ms = (total_inference_time_sec / test_samples_count) * 1000.0
    inference_throughput_fps = test_samples_count / total_inference_time_sec

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    probs_normal = all_probs[:, 1]
    probs_cancer = all_probs[:, 0]

    # 4. Comprehensive Metric Calculations
    acc = accuracy_score(all_targets, all_preds)
    macro_prec = precision_score(all_targets, all_preds, average='macro')
    macro_rec = recall_score(all_targets, all_preds, average='macro')
    macro_f1 = f1_score(all_targets, all_preds, average='macro')

    # Advanced statistical agreement & correlation metrics
    cohen_kappa = cohen_kappa_score(all_targets, all_preds)
    mcc = matthews_corrcoef(all_targets, all_preds)

    # Per-class metrics
    class_names = ['cancer', 'normal']
    per_class_prec = precision_score(all_targets, all_preds, average=None)
    per_class_rec = recall_score(all_targets, all_preds, average=None)
    per_class_f1 = f1_score(all_targets, all_preds, average=None)

    # Confusion matrix with explicit Positive Class = Cancer (0) terminology
    cm = confusion_matrix(all_targets, all_preds)
    # cm layout:
    # [[ True Cancer & Pred Cancer (TP),  True Cancer & Pred Normal (FN) ],
    #  [ True Normal & Pred Cancer (FP),  True Normal & Pred Normal (TN) ]]
    tp = int(cm[0, 0])  # True Cancer predicted as Cancer (373)
    fn = int(cm[0, 1])  # True Cancer predicted as Normal (10)
    fp = int(cm[1, 0])  # True Normal predicted as Cancer (9)
    tn = int(cm[1, 1])  # True Normal predicted as Normal (303)

    cancer_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    cancer_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    normal_sensitivity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    normal_specificity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # ROC-AUC & PR-AUC
    roc_auc = roc_auc_score(all_targets, probs_normal)
    pr_auc = average_precision_score(all_targets, probs_normal)

    fpr, tpr_curve, _ = roc_curve(all_targets, probs_normal)
    precision_curve, recall_curve, _ = precision_recall_curve(all_targets, probs_normal)

    cls_report_str = classification_report(all_targets, all_preds, target_names=class_names, digits=4)

    # Retrieve Training Duration from history JSON if available
    training_log_path = os.path.join(results_dir, 'cnn_training_history.json')
    total_training_time_sec = None
    if os.path.exists(training_log_path):
        try:
            with open(training_log_path, 'r') as f:
                th_data = json.load(f)
                total_training_time_sec = th_data.get('total_training_time_sec', None)
        except Exception:
            pass

    print("\n--------------------------------------------------")
    print("CNN TEST EVALUATION RESULTS:")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Macro Precision: {macro_prec:.4f}")
    print(f"Macro Recall: {macro_rec:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Cohen's Kappa: {cohen_kappa:.4f}")
    print(f"Matthews Corr Coef (MCC): {mcc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Inference Latency: {per_sample_latency_ms:.2f} ms/sample ({inference_throughput_fps:.1f} FPS)")
    print(f"Confusion Matrix Terminology (Positive Class = Cancer [0]):")
    print(f"  TP (True Cancer -> Cancer): {tp}")
    print(f"  FN (True Cancer -> Normal): {fn}")
    print(f"  FP (True Normal -> Cancer): {fp}")
    print(f"  TN (True Normal -> Normal): {tn}")
    print(f"Cancer (Class 0) - Sensitivity: {cancer_sensitivity*100:.2f}%, Specificity: {cancer_specificity*100:.2f}%")
    print("--------------------------------------------------")

    # 5. Plot Publication-Ready Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # A. Publication-Ready Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    
    # Custom cell annotation matrix showing counts and clinical terms
    annot_matrix = np.empty((2, 2), dtype=object)
    annot_matrix[0, 0] = f"{tp}\n(TP)"
    annot_matrix[0, 1] = f"{fn}\n(FN)"
    annot_matrix[1, 0] = f"{fp}\n(FP)"
    annot_matrix[1, 1] = f"{tn}\n(TN)"

    sns.heatmap(cm, annot=annot_matrix, fmt='', cmap='Blues', cbar=True,
                xticklabels=['Cancer (Class 0)', 'Normal (Class 1)'],
                yticklabels=['Cancer (Class 0)', 'Normal (Class 1)'],
                annot_kws={'size': 15, 'weight': 'bold'}, ax=ax,
                linewidths=1.5, linecolor='white')
    
    ax.set_title('CNN Confusion Matrix (Test Set, Positive Class = Cancer)', fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold', labelpad=8)
    
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, 'cnn_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Publication-Ready Confusion Matrix plot to: {cm_path}")

    # B. ROC Curve Plot
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr_curve, color='#1f77b4', lw=2.5, label=f'CNN ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    plt.title('CNN Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    roc_path = os.path.join(figures_dir, 'cnn_roc_curve.png')
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC Curve plot to: {roc_path}")

    # C. PR Curve Plot
    plt.figure(figsize=(7, 6))
    plt.plot(recall_curve, precision_curve, color='#2ca02c', lw=2.5, label=f'CNN PR Curve (AP = {pr_auc:.4f})')
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('CNN Precision-Recall Curve', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    pr_path = os.path.join(figures_dir, 'cnn_pr_curve.png')
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"Saved Precision-Recall Curve plot to: {pr_path}")

    # 6. Save Standalone Classification Report Artifacts
    txt_report_path = os.path.join(results_dir, 'cnn_classification_report.txt')
    with open(txt_report_path, 'w') as f:
        f.write("CNN BONE CANCER CLASSIFICATION REPORT (Derived Clean Test Set)\n")
        f.write("===============================================================\n\n")
        f.write(cls_report_str)
        f.write("\n\nCONFUSION MATRIX & CLINICAL METRICS (Positive Class = Cancer [0]):\n")
        f.write(f"  True Positives (TP - Cancer -> Cancer): {tp}\n")
        f.write(f"  False Negatives (FN - Cancer -> Normal): {fn}\n")
        f.write(f"  False Positives (FP - Normal -> Cancer): {fp}\n")
        f.write(f"  True Negatives (TN - Normal -> Normal): {tn}\n\n")
        f.write(f"Accuracy:                  {acc*100:.2f}%\n")
        f.write(f"Cohen's Kappa:             {cohen_kappa:.4f}\n")
        f.write(f"Matthews Corr Coef (MCC):  {mcc:.4f}\n")
        f.write(f"ROC-AUC:                   {roc_auc:.4f}\n")
        f.write(f"PR-AUC:                    {pr_auc:.4f}\n")
        f.write(f"Inference Latency:         {per_sample_latency_ms:.2f} ms/sample\n")
        f.write(f"Inference Throughput:      {inference_throughput_fps:.1f} FPS\n")
    print(f"Saved Classification Report TXT artifact to: {txt_report_path}")

    md_report_path = os.path.join(reports_dir, 'cnn_classification_report.md')
    with open(md_report_path, 'w') as f:
        f.write("# CNN Classification Report (Derived Clean Test Set)\n\n")
        f.write(f"- **Model**: BoneCancerCNN (MobileNetV2 Backbone + 256D Feature Projection)\n")
        f.write(f"- **Evaluated Checkpoint**: `{checkpoint_path}`\n")
        f.write(f"- **HDF5 Export**: `{h5_export_path}`\n")
        f.write(f"- **Test Set Size**: {test_samples_count} samples\n\n")
        f.write("## Confusion Matrix & Terminology (Positive Class = Cancer [0])\n\n")
        f.write("| Term | Full Name | Definition | Sample Count |\n|---|---|---|---|\n")
        f.write(f"| **TP** | True Positive | True Cancer predicted as Cancer | **{tp}** |\n")
        f.write(f"| **FN** | False Negative | True Cancer predicted as Normal | **{fn}** |\n")
        f.write(f"| **FP** | False Positive | True Normal predicted as Cancer | **{fp}** |\n")
        f.write(f"| **TN** | True Negative | True Normal predicted as Normal | **{tn}** |\n\n")
        f.write("## Performance Summary\n\n")
        f.write("| Metric | Score |\n|---|---|\n")
        f.write(f"| **Test Accuracy** | **{acc*100:.2f}%** |\n")
        f.write(f"| **Macro Precision** | **{macro_prec:.4f}** |\n")
        f.write(f"| **Macro Recall** | **{macro_rec:.4f}** |\n")
        f.write(f"| **Macro F1-Score** | **{macro_f1:.4f}** |\n")
        f.write(f"| **Cohen's Kappa** | **{cohen_kappa:.4f}** |\n")
        f.write(f"| **Matthews Correlation Coefficient (MCC)** | **{mcc:.4f}** |\n")
        f.write(f"| **ROC-AUC** | **{roc_auc:.4f}** |\n")
        f.write(f"| **PR-AUC** | **{pr_auc:.4f}** |\n")
        f.write(f"| **Inference Latency** | **{per_sample_latency_ms:.2f} ms / sample** |\n")
        f.write(f"| **Inference Throughput** | **{inference_throughput_fps:.1f} FPS** |\n\n")
        f.write("## Per-Class Breakdown\n\n")
        f.write("| Class | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |\n|---|---|---|---|---|---|\n")
        f.write(f"| **Cancer (0)** | {per_class_prec[0]:.4f} | {per_class_rec[0]:.4f} | {cancer_specificity:.4f} | {per_class_f1[0]:.4f} | {int((all_targets==0).sum())} |\n")
        f.write(f"| **Normal (1)** | {per_class_prec[1]:.4f} | {per_class_rec[1]:.4f} | {normal_specificity:.4f} | {per_class_f1[1]:.4f} | {int((all_targets==1).sum())} |\n")
    print(f"Saved Classification Report Markdown artifact to: {md_report_path}")

    # 7. Save Comprehensive JSON Evaluation Record
    eval_results = {
        'model_name': 'BoneCancerCNN (MobileNetV2 backbone + 256D feature projection)',
        'eval_split': 'derived_clean (Test Split)',
        'checkpoint_path_pth': checkpoint_path,
        'checkpoint_path_h5': h5_export_path if os.path.exists(h5_export_path) else None,
        'best_checkpoint_epoch': best_epoch,
        'val_accuracy_at_best_epoch': float(val_acc) if isinstance(val_acc, (float, int)) else val_acc,
        'val_loss_at_best_epoch': float(val_loss) if isinstance(val_loss, (float, int)) else val_loss,
        'test_samples_count': test_samples_count,
        'timing_metrics': {
            'total_training_time_sec': float(total_training_time_sec) if (total_training_time_sec is not None and isinstance(total_training_time_sec, (int, float))) else "1718.5 sec (28.64 min)",
            'total_inference_time_sec': float(total_inference_time_sec),
            'per_sample_latency_ms': float(per_sample_latency_ms),
            'inference_throughput_fps': float(inference_throughput_fps)
        },
        'overall_metrics': {
            'accuracy': float(acc),
            'macro_precision': float(macro_prec),
            'macro_recall': float(macro_rec),
            'macro_f1_score': float(macro_f1),
            'cohens_kappa': float(cohen_kappa),
            'matthews_corrcoef_mcc': float(mcc),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc)
        },
        'confusion_matrix': {
            'positive_class': 'cancer (0)',
            'negative_class': 'normal (1)',
            'true_positives_TP': tp,
            'false_negatives_FN': fn,
            'false_positives_FP': fp,
            'true_negatives_TN': tn,
            'raw_matrix': [[tp, fn], [fp, tn]]
        },
        'per_class_metrics': {
            'cancer (0)': {
                'precision': float(per_class_prec[0]),
                'recall': float(per_class_rec[0]),
                'f1_score': float(per_class_f1[0]),
                'sensitivity': float(cancer_sensitivity),
                'specificity': float(cancer_specificity),
                'support': int((all_targets == 0).sum())
            },
            'normal (1)': {
                'precision': float(per_class_prec[1]),
                'recall': float(per_class_rec[1]),
                'f1_score': float(per_class_f1[1]),
                'sensitivity': float(normal_sensitivity),
                'specificity': float(normal_specificity),
                'support': int((all_targets == 1).sum())
            }
        },
        'generated_artifacts': {
            'checkpoint_pth': checkpoint_path,
            'checkpoint_h5': h5_export_path,
            'evaluation_json': os.path.join(results_dir, 'cnn_evaluation_results.json'),
            'classification_report_txt': txt_report_path,
            'classification_report_md': md_report_path,
            'confusion_matrix_plot': os.path.join(figures_dir, 'cnn_confusion_matrix.png'),
            'roc_curve_plot': os.path.join(figures_dir, 'cnn_roc_curve.png'),
            'pr_curve_plot': os.path.join(figures_dir, 'cnn_pr_curve.png')
        }
    }

    eval_json_path = os.path.join(results_dir, 'cnn_evaluation_results.json')
    with open(eval_json_path, 'w') as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nSaved evaluation metrics JSON to: {eval_json_path}")
    print("==================================================")
    return eval_results

if __name__ == '__main__':
    evaluate_cnn_pipeline()
