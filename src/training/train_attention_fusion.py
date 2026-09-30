import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure workspace root is in path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from src.config import ProjectConfig
from src.dataset import get_dataloaders
from src.models.attention_fusion import BoneCancerAttentionFusion

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preextract_features(model, dataloader, device):
    """
    Pre-extracts 256-D feature representations from frozen backbones (CNN, Swin, Mamba)
    for all samples in dataloader. This eliminates redundant backbone forward passes
    during fusion head training.
    """
    model.eval()
    all_feats = []
    all_targets = []
    
    print(f"Extracting branch features for {len(dataloader.dataset)} samples...", flush=True)
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            feats = model.extract_branch_features(inputs) # (B, 3, 256)
            all_feats.append(feats.cpu())
            all_targets.append(targets)
            
    all_feats = torch.cat(all_feats, dim=0)     # (N, 3, 256)
    all_targets = torch.cat(all_targets, dim=0) # (N,)
    
    print(f"Features extracted in {time.time() - start_time:.2f}s | Shape: {all_feats.shape}", flush=True)
    return all_feats, all_targets

def train_one_epoch_cached(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_feats, targets in dataloader:
        batch_feats = batch_feats.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        # Direct pass through attention fusion mechanism & classifier head
        fused_feature, _ = model.attention_fusion(batch_feats)
        outputs = model.classifier(fused_feature)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_feats.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    return epoch_loss, epoch_acc, epoch_f1

def validate_cached(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_feats, targets in dataloader:
            batch_feats = batch_feats.to(device)
            targets = targets.to(device)

            fused_feature, _ = model.attention_fusion(batch_feats)
            outputs = model.classifier(fused_feature)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_feats.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    val_rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return val_loss, val_acc, val_prec, val_rec, val_f1

def plot_training_curves(history, figures_dir):
    epochs = [e['epoch'] for e in history]
    t_loss = [e['train_loss'] for e in history]
    v_loss = [e['val_loss'] for e in history]
    t_acc = [e['train_acc'] * 100 for e in history]
    v_acc = [e['val_acc'] * 100 for e in history]

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, t_loss, 'o-', color='#1f77b4', linewidth=2, label='Train Loss')
    ax1.plot(epochs, v_loss, 's--', color='#ff7f0e', linewidth=2, label='Val Loss')
    ax1.set_title('Attention Fusion Loss Curves (Derived Clean Split)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, t_acc, 'o-', color='#2ca02c', linewidth=2, label='Train Accuracy')
    ax2.plot(epochs, v_acc, 's--', color='#d62728', linewidth=2, label='Val Accuracy')
    ax2.set_title('Attention Fusion Accuracy Curves (Derived Clean Split)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    curves_path = os.path.join(figures_dir, 'attention_fusion_training_curves.png')
    plt.savefig(curves_path, dpi=300)
    plt.close()
    return curves_path

def run_attention_fusion_training(num_epochs=25, batch_size=32, lr=5e-4, weight_decay=1e-4, seed=42, patience=3):
    import numpy as np
    set_seed(seed)
    
    cfg = ProjectConfig()
    cfg._config['dataset']['split_type'] = 'derived_clean'
    cfg._config['dataset']['batch_size'] = batch_size
    cfg._config['project']['seed'] = seed

    workspace_root = cfg.get('project.workspace_root')
    checkpoints_dir = os.path.join(workspace_root, 'checkpoints')
    results_dir = os.path.join(workspace_root, 'results')
    figures_dir = os.path.join(workspace_root, 'figures')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================", flush=True)
    print(f"Starting Attention Fusion Training (M4, MobileNetV2 + Swin-Tiny + Mamba)", flush=True)
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)
    print(f"Split Protocol: derived_clean (Leakage-Clean)", flush=True)
    print(f"Input Resolution: 224x224", flush=True)
    print(f"Class Mapping: cancer=0, normal=1", flush=True)
    print(f"Batch Size: {batch_size}, LR: {lr}, Epochs: {num_epochs}, Patience: {patience}", flush=True)
    print(f"==================================================", flush=True)

    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    print(f"Dataset Loaded | Train Samples: {len(train_loader.dataset)}, Val Samples: {len(val_loader.dataset)}", flush=True)
    print(f"Test Set retained untouched: {len(test_loader.dataset)} samples", flush=True)

    cnn_ckpt = os.path.join(checkpoints_dir, 'cnn_best.pth')
    swin_ckpt = os.path.join(checkpoints_dir, 'swin_best.pth')
    mamba_ckpt = os.path.join(checkpoints_dir, 'mamba_best.pth')

    model = BoneCancerAttentionFusion(
        cnn_ckpt_path=cnn_ckpt,
        swin_ckpt_path=swin_ckpt,
        mamba_ckpt_path=mamba_ckpt,
        cnn_backbone_name='mobilenet_v2',
        feature_dim=256,
        num_classes=2,
        dropout_rate=0.2,
        freeze_backbones=True
    ).to(device)

    # Calculate parameter stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\nModel Parameter Stats:")
    print(f"  - Total Parameters:     {total_params:,}")
    print(f"  - Frozen Parameters:    {frozen_params:,} (MobileNetV2 + Swin + Mamba Backbones)")
    print(f"  - Trainable Parameters: {trainable_params:,} (Attention Fusion + Classifier Head)\n", flush=True)

    # Pre-extract features for train and val loaders
    train_feats, train_targets = preextract_features(model, train_loader, device)
    val_feats, val_targets = preextract_features(model, val_loader, device)

    cached_train_loader = DataLoader(TensorDataset(train_feats, train_targets), batch_size=batch_size, shuffle=True)
    cached_val_loader = DataLoader(TensorDataset(val_feats, val_targets), batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    early_stopping_triggered = False
    best_checkpoint_path = os.path.join(checkpoints_dir, 'attention_fusion_best.pth')

    history = []
    start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc, train_f1 = train_one_epoch_cached(model, cached_train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate_cached(model, cached_val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        is_best = False
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            is_best = True
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'val_f1': val_f1,
                'config': cfg.to_dict()
            }, best_checkpoint_path)
        else:
            patience_counter += 1

        epoch_record = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'train_f1': float(train_f1),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'val_precision': float(val_prec),
            'val_recall': float(val_rec),
            'val_f1': float(val_f1),
            'lr': float(current_lr),
            'epoch_time_sec': float(epoch_time),
            'is_best': is_best
        }
        history.append(epoch_record)

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_time*1000:.1f}ms) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f} | "
              f"LR: {current_lr:.6f} {'[BEST]' if is_best else f'[Patience: {patience_counter}/{patience}]'}", flush=True)

        if patience_counter >= patience:
            print(f"\n[EARLY STOPPING] Triggered at epoch {epoch}. Best epoch was Epoch {best_epoch} with Val Acc: {best_val_acc*100:.2f}%", flush=True)
            early_stopping_triggered = True
            break

        gpu_stats = {}
        if torch.cuda.is_available():
            gpu_stats = {
                'device_name': torch.cuda.get_device_name(0),
                'max_memory_allocated_mb': torch.cuda.max_memory_allocated(device) / (1024**2),
                'max_memory_reserved_mb': torch.cuda.max_memory_reserved(device) / (1024**2)
            }

        plot_training_curves(history, figures_dir)

    training_summary = {
        'model_name': 'BoneCancerAttentionFusion (Frozen MobileNetV2 + Swin + Mamba + Softmax Branch Attention)',
        'split_type': 'derived_clean',
        'config': cfg.to_dict(),
        'hyperparameters': {
            'epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'seed': seed,
            'patience': patience,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR'
        },
        'parameter_counts': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params
        },
        'best_checkpoint': {
            'epoch': best_epoch,
            'checkpoint_path': best_checkpoint_path,
            'val_accuracy': float(best_val_acc),
            'val_loss': float(best_val_loss),
            'val_f1': float(history[best_epoch-1]['val_f1']),
            'val_precision': float(history[best_epoch-1]['val_precision']),
            'val_recall': float(history[best_epoch-1]['val_recall'])
        },
        'early_stopping_triggered': early_stopping_triggered,
        'total_training_time_sec': time.time() - start_time,
        'total_training_time_min': (time.time() - start_time) / 60.0,
        'gpu_usage': gpu_stats,
        'epoch_history': history
    }

    log_path = os.path.join(results_dir, 'attention_fusion_training_history.json')
    with open(log_path, 'w') as f:
        json.dump(training_summary, f, indent=2)

    # Restore best checkpoint weights
    if os.path.exists(best_checkpoint_path):
        ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Restored best model weights from Epoch {best_epoch} ({best_checkpoint_path})", flush=True)

    # Extract attention weights on validation set for branch contribution visualization
    model.eval()
    val_attn_list = []
    with torch.no_grad():
        for batch_feats, _ in cached_val_loader:
            batch_feats = batch_feats.to(device)
            _, attn_w = model.attention_fusion(batch_feats) # (B, 3, 1)
            val_attn_list.extend(attn_w.squeeze(-1).cpu().numpy())
    
    val_attn = np.array(val_attn_list) # (N_val, 3)
    mean_attn = np.mean(val_attn, axis=0) # [MobileNetV2, Swin, Mamba]
    
    branch_names = ['CNN (MobileNetV2)', 'Swin-Tiny', 'Mamba (SSM)']
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    bars = ax.bar(branch_names, mean_attn * 100, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Mean Attention Weight (%)', fontsize=12, fontweight='bold')
    ax.set_title('M4 Attention Fusion Branch Contribution Weights (Val Set)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(mean_attn * 100) * 1.25])
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')

    for bar, val in zip(bars, mean_attn * 100):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{val:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    branch_plot_path = os.path.join(figures_dir, 'attention_fusion_branch_contributions.png')
    plt.savefig(branch_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Attention Branch Contributions plot to: {branch_plot_path}", flush=True)

    total_training_time = time.time() - start_time
    print("\n==================================================", flush=True)
    print("Attention Fusion Training Phase COMPLETED Successfully!", flush=True)
    print(f"Total Time: {total_training_time/60.0:.2f} minutes ({total_training_time:.2f} s)", flush=True)
    print(f"Best Checkpoint: Epoch {best_epoch} | Val Acc: {best_val_acc*100:.2f}% | Val Loss: {best_val_loss:.4f}", flush=True)
    print(f"Early Stopping Triggered: {early_stopping_triggered}", flush=True)
    print(f"Saved Checkpoint to: {best_checkpoint_path}", flush=True)
    print(f"Saved Training History to: {log_path}", flush=True)
    print("Saved Training Curves Plot to: figures/attention_fusion_training_curves.png", flush=True)
    print("==================================================", flush=True)

    return training_summary

if __name__ == '__main__':
    run_attention_fusion_training(num_epochs=25, batch_size=32, lr=5e-4)
