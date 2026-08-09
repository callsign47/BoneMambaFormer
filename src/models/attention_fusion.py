import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from src.models.cnn import BoneCancerCNN
from src.models.swin import BoneCancerSwin
from src.models.mamba import BoneCancerMamba

class SoftmaxBranchAttention(nn.Module):
    """
    Learns dynamic attention weights across CNN, Swin, and Mamba feature branches.
    Given branch features of shape (B, 3, D), produces normalized attention weights alpha (B, 3, 1)
    and computes the weighted combination of features:
    fused_feature = sum_k (alpha_k * f_k)
    """
    def __init__(self, feature_dim=256, hidden_dim=64):
        super(SoftmaxBranchAttention, self).__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, branch_features):
        """
        branch_features: Tensor of shape (B, 3, feature_dim)
        Returns:
            fused_feature: Tensor of shape (B, feature_dim)
            weights: Tensor of shape (B, 3, 1) - normalized attention weights [CNN, Swin, Mamba]
        """
        # Calculate unnormalized attention score for each branch feature vector
        attn_scores = self.attn_mlp(branch_features) # (B, 3, 1)
        attn_weights = F.softmax(attn_scores, dim=1) # (B, 3, 1) - sum over 3 branches = 1.0
        
        # Fused feature as weighted sum across branches
        fused_feature = (branch_features * attn_weights).sum(dim=1) # (B, feature_dim)
        return fused_feature, attn_weights

class BoneCancerAttentionFusion(nn.Module):
    """
    Phase 6 Attention Fusion Model.
    Combines frozen standalone backbones (CNN, Swin-Tiny, Mamba)
    with trainable Softmax Branch Attention and Classification Head.
    
    Branch indexing:
      0: CNN (ResNet18)
      1: Swin-Tiny
      2: Mamba (Selective SSM)
    """
    def __init__(
        self,
        cnn_ckpt_path=None,
        swin_ckpt_path=None,
        mamba_ckpt_path=None,
        feature_dim=256,
        num_classes=2,
        dropout_rate=0.2,
        freeze_backbones=True
    ):
        super(BoneCancerAttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.freeze_backbones = freeze_backbones

        # Initialize backbone branches
        self.cnn_branch = BoneCancerCNN(pretrained=False, feature_dim=feature_dim, num_classes=num_classes)
        self.swin_branch = BoneCancerSwin(pretrained=False, feature_dim=feature_dim, num_classes=num_classes)
        self.mamba_branch = BoneCancerMamba(img_size=224, patch_size=32, d_model=128, depth=2, feature_dim=feature_dim, num_classes=num_classes)

        # Load pretrained standalone checkpoints if provided
        if cnn_ckpt_path and os.path.exists(cnn_ckpt_path):
            ckpt = torch.load(cnn_ckpt_path, map_location='cpu')
            self.cnn_branch.load_state_dict(ckpt.get('model_state_dict', ckpt))
            print(f"Loaded CNN checkpoint from: {cnn_ckpt_path}")

        if swin_ckpt_path and os.path.exists(swin_ckpt_path):
            ckpt = torch.load(swin_ckpt_path, map_location='cpu')
            self.swin_branch.load_state_dict(ckpt.get('model_state_dict', ckpt))
            print(f"Loaded Swin checkpoint from: {swin_ckpt_path}")

        if mamba_ckpt_path and os.path.exists(mamba_ckpt_path):
            ckpt = torch.load(mamba_ckpt_path, map_location='cpu')
            self.mamba_branch.load_state_dict(ckpt.get('model_state_dict', ckpt))
            print(f"Loaded Mamba checkpoint from: {mamba_ckpt_path}")

        # Freeze standalone backbones if requested
        if freeze_backbones:
            for branch in [self.cnn_branch, self.swin_branch, self.mamba_branch]:
                for param in branch.parameters():
                    param.requires_grad = False
                branch.eval()

        # Attention Fusion Mechanism
        self.attention_fusion = SoftmaxBranchAttention(feature_dim=feature_dim, hidden_dim=64)

        # Final Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def train(self, mode=True):
        super().train(mode)
        # Force frozen backbones to remain in eval mode to preserve batchnorm/dropout state
        if self.freeze_backbones:
            self.cnn_branch.eval()
            self.swin_branch.eval()
            self.mamba_branch.eval()
        return self

    def extract_branch_features(self, x):
        """
        Extract 256-D feature representations from each backbone branch.
        """
        if self.freeze_backbones:
            with torch.no_grad():
                f_cnn = self.cnn_branch.forward_features(x)     # (B, 256)
                f_swin = self.swin_branch.forward_features(x)   # (B, 256)
                f_mamba = self.mamba_branch.forward_features(x) # (B, 256)
        else:
            f_cnn = self.cnn_branch.forward_features(x)
            f_swin = self.swin_branch.forward_features(x)
            f_mamba = self.mamba_branch.forward_features(x)

        # Stack into shape (B, 3, 256)
        stacked_features = torch.stack([f_cnn, f_swin, f_mamba], dim=1)
        return stacked_features

    def forward(self, x, return_attention=False):
        """
        Forward pass returning logits and optionally attention weights.
        """
        stacked_features = self.extract_branch_features(x)
        fused_feature, attn_weights = self.attention_fusion(stacked_features)
        logits = self.classifier(fused_feature)

        if return_attention:
            return logits, attn_weights
        return logits

if __name__ == '__main__':
    print("Testing BoneCancerAttentionFusion architecture...")
    model = BoneCancerAttentionFusion(freeze_backbones=True)
    x = torch.randn(4, 3, 224, 224)
    logits, attn_weights = model(x, return_attention=True)
    print("Attention Fusion Test PASSED!")
    print("Input shape:", x.shape)
    print("Logits shape:", logits.shape)
    print("Attention Weights shape:", attn_weights.shape)
    print("Sample Attention Weights (CNN, Swin, Mamba):\n", attn_weights.squeeze(-1))
