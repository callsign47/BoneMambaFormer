import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure workspace root is in path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from src.config import ProjectConfig
from src.dataset import get_dataloaders
from src.models.cnn import BoneCancerCNN

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
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
    ax1.set_title('CNN Loss Curves (Derived Clean Split)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, t_acc, 'o-', color='#2ca02c', linewidth=2, label='Train Accuracy')
    ax2.plot(epochs, v_acc, 's--', color='#d62728', linewidth=2, label='Val Accuracy')
    ax2.set_title('CNN Accuracy Curves (Derived Clean Split)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    curves_path = os.path.join(figures_dir, 'cnn_training_curves.png')
    plt.savefig(curves_path, dpi=300)
    plt.close()
    return curves_path

def run_cnn_training(num_epochs=20, batch_size=32, lr=1e-4, weight_decay=1e-4, seed=42, patience=3):
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
    print(f"Starting Standalone MobileNetV2 CNN Training (M1)", flush=True)
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

    model = BoneCancerCNN(backbone_name='mobilenet_v2', num_classes=2, feature_dim=256, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = os.path.join(checkpoints_dir, 'cnn_best.pth')

    history = []
    start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)
        
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

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f} | "
              f"LR: {current_lr:.6f} {'[BEST]' if is_best else f'[Patience: {patience_counter}/{patience}]'}", flush=True)

        if patience_counter >= patience:
            print(f"\n[EARLY STOPPING] Triggered at epoch {epoch}. Best epoch was Epoch {best_epoch} with Val Acc: {best_val_acc*100:.2f}%", flush=True)
            break

        # Incremental write of training summary & curves after every epoch
        gpu_stats = {}
        if torch.cuda.is_available():
            gpu_stats = {
                'device_name': torch.cuda.get_device_name(0),
                'max_memory_allocated_mb': torch.cuda.max_memory_allocated(device) / (1024**2),
                'max_memory_reserved_mb': torch.cuda.max_memory_reserved(device) / (1024**2)
            }

        training_summary = {
            'model_name': 'BoneCancerCNN (MobileNetV2 backbone + 256D projection)',
            'split_type': 'derived_clean',
            'config': cfg.to_dict(),
            'hyperparameters': {
                'epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': lr,
                'weight_decay': weight_decay,
                'seed': seed,
                'optimizer': 'AdamW',
                'scheduler': 'CosineAnnealingLR'
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
            'total_training_time_sec': time.time() - start_time,
            'total_training_time_min': (time.time() - start_time) / 60.0,
            'gpu_usage': gpu_stats,
            'epoch_history': history
        }

        log_path = os.path.join(results_dir, 'cnn_training_history.json')
        with open(log_path, 'w') as f:
            json.dump(training_summary, f, indent=2)

        plot_training_curves(history, figures_dir)

    total_training_time = time.time() - start_time
    print("\n==================================================", flush=True)
    print("CNN Training Phase COMPLETED Successfully!", flush=True)
    print(f"Total Time: {total_training_time/60.0:.2f} minutes", flush=True)
    print(f"Best Checkpoint: Epoch {best_epoch} | Val Acc: {best_val_acc*100:.2f}% | Val Loss: {best_val_loss:.4f}", flush=True)
    print(f"Saved Checkpoint to: {best_checkpoint_path}", flush=True)
    print(f"Saved Training History to: {log_path}", flush=True)
    print("Saved Training Curves Plot to: figures/cnn_training_curves.png", flush=True)
    print("==================================================", flush=True)

    return training_summary

if __name__ == '__main__':
    run_cnn_training(num_epochs=20, batch_size=32, lr=1e-4)
