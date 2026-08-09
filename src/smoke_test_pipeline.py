import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ProjectConfig
from src.dataset import get_dataloaders

def run_smoke_test():
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 80)
    print("STARTING DATA PIPELINE SMOKE TEST")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    smoke_test_report = {
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'splits_tested': {}
    }

    # Test both Original and Derived Clean Split configurations
    for split_mode in ['original', 'derived_clean']:
        print(f"\n--- Testing Split Mode: {split_mode.upper()} ---")
        cfg = ProjectConfig()
        cfg._config['dataset']['split_type'] = split_mode
        cfg._config['dataset']['batch_size'] = 16
        cfg._config['dataset']['num_workers'] = 0 # 0 for synchronous test execution
        
        train_loader, val_loader, test_loader = get_dataloaders(cfg)
        
        mode_results = {}
        
        for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            # Fetch first batch
            images, labels = next(iter(loader))
            
            # 1. Shapes
            img_shape = list(images.shape)
            lbl_shape = list(labels.shape)
            
            # 2. Labels validation
            unique_labels = sorted(labels.unique().tolist())
            valid_labels = all(lbl in [0, 1] for lbl in unique_labels)
            
            # 3. Normalization Stats
            img_min = float(images.min())
            img_max = float(images.max())
            img_mean = float(images.mean())
            img_std = float(images.std())
            
            # 4. NaN / Inf check
            has_nan = bool(torch.isnan(images).any() or torch.isnan(labels.float()).any())
            has_inf = bool(torch.isinf(images).any() or torch.isinf(labels.float()).any())
            
            # 5. GPU Transfer
            try:
                images_gpu = images.to(device)
                labels_gpu = labels.to(device)
                gpu_transfer_success = True
                gpu_tensor_device = str(images_gpu.device)
            except Exception as e:
                gpu_transfer_success = False
                gpu_tensor_device = str(e)

            print(f"  [{name.upper()}] Batch Image Shape: {img_shape}, Label Shape: {lbl_shape}")
            print(f"          Labels Present: {unique_labels} (Valid: {valid_labels})")
            print(f"          Min: {img_min:.4f}, Max: {img_max:.4f}, Mean: {img_mean:.4f}, Std: {img_std:.4f}")
            print(f"          NaNs: {has_nan}, Infs: {has_inf}, GPU Transfer: {gpu_transfer_success} ({gpu_tensor_device})")
            
            mode_results[name] = {
                'total_batches': len(loader),
                'total_samples': len(loader.dataset),
                'batch_image_shape': img_shape,
                'batch_label_shape': lbl_shape,
                'unique_labels': unique_labels,
                'valid_labels_check': valid_labels,
                'stats': {
                    'min': img_min,
                    'max': img_max,
                    'mean': img_mean,
                    'std': img_std
                },
                'nan_check_passed': not has_nan,
                'inf_check_passed': not has_inf,
                'gpu_transfer_passed': gpu_transfer_success,
                'gpu_device': gpu_tensor_device
            }
            
        smoke_test_report['splits_tested'][split_mode] = mode_results

    out_file = os.path.join(workspace_root, 'results', 'pipeline_smoke_test.json')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(smoke_test_report, f, indent=2)
    print("\n" + "=" * 80)
    print(f"SMOKE TEST COMPLETE! Results saved to {out_file}")
    print("=" * 80)

if __name__ == '__main__':
    run_smoke_test()
