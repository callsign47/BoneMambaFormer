import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Ensure workspace root is in path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from src.config import ProjectConfig
from src.dataset import get_dataloaders
from src.models.attention_fusion import BoneCancerAttentionFusion
from src.evaluation.run_ablation_study import AblationWrapper

def verify_ablation_independence():
    cfg = ProjectConfig()
    cfg._config['dataset']['split_type'] = 'derived_clean'
    cfg._config['dataset']['batch_size'] = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints_dir = os.path.join(cfg.get('project.workspace_root'), 'checkpoints')
    hybrid_ckpt_path = os.path.join(checkpoints_dir, 'hybrid_best.pth')

    _, _, test_loader = get_dataloaders(cfg)

    hybrid = BoneCancerAttentionFusion(feature_dim=256, num_classes=2, freeze_backbones=False).to(device)
    hybrid_ckpt = torch.load(hybrid_ckpt_path, map_location=device)
    hybrid.load_state_dict(hybrid_ckpt['model_state_dict'])
    hybrid.eval()

    modes = ['dynamic', 'equal_weights', 'no_cnn', 'no_mamba', 'no_swin']
    mode_outputs = {}

    softmax = nn.Softmax(dim=1)

    for mode in modes:
        wrapper = AblationWrapper(hybrid, ablation_mode=mode).to(device)
        wrapper.eval()

        all_probs = []
        all_preds = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                logits = wrapper(inputs)
                probs = softmax(logits)
                preds = torch.argmax(logits, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        mode_outputs[mode] = {
            'probs': np.array(all_probs),
            'preds': np.array(all_preds)
        }

    print("==================================================")
    print("SANITY CHECK: VERIFYING ABLATION INDEPENDENCE & MASKING")
    print("==================================================")

    # 1. Compare prediction agreement between modes
    for i, mode1 in enumerate(modes):
        for mode2 in modes[i+1:]:
            preds1 = mode_outputs[mode1]['preds']
            preds2 = mode_outputs[mode2]['preds']
            prob_diff = np.max(np.abs(mode_outputs[mode1]['probs'] - mode_outputs[mode2]['probs']))
            mean_prob_diff = np.mean(np.abs(mode_outputs[mode1]['probs'] - mode_outputs[mode2]['probs']))
            matching_preds = np.sum(preds1 == preds2)
            total = len(preds1)
            print(f"{mode1:15s} vs {mode2:15s} -> Matching Preds: {matching_preds}/{total} ({matching_preds/total*100:.2f}%), Max Prob Diff: {prob_diff:.6f}, Mean Prob Diff: {mean_prob_diff:.6f}")

    # 2. Check individual feature magnitudes to confirm branch features are active
    sample_x, _ = next(iter(test_loader))
    sample_x = sample_x.to(device)
    with torch.no_grad():
        stacked_feat = hybrid.extract_branch_features(sample_x) # (B, 3, 256)
        norm_cnn = torch.norm(stacked_feat[:, 0, :], dim=1).mean().item()
        norm_swin = torch.norm(stacked_feat[:, 1, :], dim=1).mean().item()
        norm_mamba = torch.norm(stacked_feat[:, 2, :], dim=1).mean().item()

        print("\nBranch Feature Norms (L2):")
        print(f"  CNN Branch Feature Norm:   {norm_cnn:.4f}")
        print(f"  Swin Branch Feature Norm:  {norm_swin:.4f}")
        print(f"  Mamba Branch Feature Norm: {norm_mamba:.4f}")

    print("==================================================")

if __name__ == '__main__':
    verify_ablation_independence()
