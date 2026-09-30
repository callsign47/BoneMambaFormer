import os
import sys
import torch
import torch.nn as nn

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from src.models.cnn import BoneCancerCNN
from src.models.swin import BoneCancerSwin
from src.models.mamba import BoneCancerMamba
from src.models.attention_fusion import BoneCancerAttentionFusion

def run_mobilenet_v2_smoke_test():
    print("=" * 80)
    print("MOBILENET_V2 BACKBONE & BRANCH INTERFACE SMOKE TEST (STEP 1 & 2)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    x = torch.randn(4, 3, 224, 224).to(device)
    labels = torch.tensor([0, 1, 0, 1]).to(device)
    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # TEST 1: MobileNetV2 Standalone Backbone & Feature Projection
    # ---------------------------------------------------------
    print("\n--- TEST 1: MobileNetV2 Standalone (M1) ---")
    cnn_mobilenet = BoneCancerCNN(backbone_name='mobilenet_v2', pretrained=True, feature_dim=256, num_classes=2).to(device)
    
    # Feature extraction check
    features = cnn_mobilenet.forward_features(x)
    assert features.shape == (4, 256), f"Expected feature shape (4, 256), got {features.shape}"
    print(f"[PASSED] forward_features shape: {features.shape}")

    # Forward pass check
    logits = cnn_mobilenet(x)
    assert logits.shape == (4, 2), f"Expected logits shape (4, 2), got {logits.shape}"
    print(f"[PASSED] forward logits shape: {logits.shape}")

    # Backward pass smoke test
    loss = criterion(logits, labels)
    loss.backward()
    print(f"[PASSED] Forward & Backward pass loss: {loss.item():.4f}")

    # Parameter Audit for MobileNetV2 Standalone
    total_mobilenet_params = sum(p.numel() for p in cnn_mobilenet.parameters())
    trainable_mobilenet_params = sum(p.numel() for p in cnn_mobilenet.parameters() if p.requires_grad)
    backbone_only_params = sum(p.numel() for p in cnn_mobilenet.backbone.parameters())
    proj_params = sum(p.numel() for p in cnn_mobilenet.projection.parameters())
    head_params = sum(p.numel() for p in cnn_mobilenet.classifier.parameters())
    
    print(f"  MobileNetV2 Backbone Params: {backbone_only_params:,}")
    print(f"  Projection Layer Params:      {proj_params:,}")
    print(f"  Classifier Head Params:       {head_params:,}")
    print(f"  Total Standalone Params:      {total_mobilenet_params:,}")

    # ---------------------------------------------------------
    # TEST 2: Comparison with Legacy ResNet-18
    # ---------------------------------------------------------
    print("\n--- TEST 2: Legacy ResNet-18 Comparison ---")
    cnn_resnet = BoneCancerCNN(backbone_name='resnet18', pretrained=False, feature_dim=256, num_classes=2).to(device)
    total_resnet_params = sum(p.numel() for p in cnn_resnet.parameters())
    print(f"  Legacy ResNet-18 Standalone Params:   {total_resnet_params:,}")
    print(f"  Next-Gen MobileNetV2 Standalone Params: {total_mobilenet_params:,}")
    print(f"  Parameter Reduction: {((total_resnet_params - total_mobilenet_params) / total_resnet_params) * 100:.2f}%")

    # ---------------------------------------------------------
    # TEST 3: Attention Fusion (M4) with MobileNetV2 (Frozen Backbones)
    # ---------------------------------------------------------
    print("\n--- TEST 3: Attention Fusion (M4) with MobileNetV2 (Frozen Backbones) ---")
    attn_fusion_frozen = BoneCancerAttentionFusion(
        cnn_backbone_name='mobilenet_v2',
        feature_dim=256,
        num_classes=2,
        freeze_backbones=True
    ).to(device)

    logits_af, weights_af = attn_fusion_frozen(x, return_attention=True)
    assert logits_af.shape == (4, 2), f"Expected Attention Fusion logits (4, 2), got {logits_af.shape}"
    assert weights_af.shape == (4, 3, 1), f"Expected attention weights (4, 3, 1), got {weights_af.shape}"
    print(f"[PASSED] Attention Fusion Logits shape: {logits_af.shape}")
    print(f"[PASSED] Attention Weights shape: {weights_af.shape}")
    print(f"  Sample Normalized Weights [MobileNetV2, Swin-Tiny, Mamba]:\n  {weights_af[0].squeeze().detach().cpu().numpy()}")

    loss_af = criterion(logits_af, labels)
    loss_af.backward()
    print(f"[PASSED] Frozen Attention Fusion backward pass loss: {loss_af.item():.4f}")

    total_af_params = sum(p.numel() for p in attn_fusion_frozen.parameters())
    trainable_af_params = sum(p.numel() for p in attn_fusion_frozen.parameters() if p.requires_grad)
    print(f"  Total Attention Fusion Params:     {total_af_params:,}")
    print(f"  Trainable Attention Fusion Params: {trainable_af_params:,}")

    # ---------------------------------------------------------
    # TEST 4: End-to-End Hybrid (M5) with MobileNetV2 (Joint Optimization)
    # ---------------------------------------------------------
    print("\n--- TEST 4: End-to-End Hybrid (M5) with MobileNetV2 (Joint Optimization) ---")
    hybrid_e2e = BoneCancerAttentionFusion(
        cnn_backbone_name='mobilenet_v2',
        feature_dim=256,
        num_classes=2,
        freeze_backbones=False
    ).to(device)

    logits_hy, weights_hy = hybrid_e2e(x, return_attention=True)
    loss_hy = criterion(logits_hy, labels)
    loss_hy.backward()
    print(f"[PASSED] Joint Hybrid forward & backward pass loss: {loss_hy.item():.4f}")

    total_hy_params = sum(p.numel() for p in hybrid_e2e.parameters())
    trainable_hy_params = sum(p.numel() for p in hybrid_e2e.parameters() if p.requires_grad)
    print(f"  Total Hybrid Params:     {total_hy_params:,}")
    print(f"  Trainable Hybrid Params: {trainable_hy_params:,}")

    # ---------------------------------------------------------
    # SUMMARY REPORT
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1 & STEP 2 VERIFICATION SUMMARY")
    print("=" * 80)
    print("1. MobileNetV2 CNN Backbone: VERIFIED")
    print("2. 256-D Common Projection Interface: VERIFIED")
    print("3. Forward & Backward Pass (Gradients): VERIFIED")
    print("4. Attention Fusion (M4) Integration: VERIFIED")
    print("5. End-to-End Hybrid (M5) Integration: VERIFIED")
    print("6. Exact Parameter Counts Audited:")
    print(f"   - M1 (MobileNetV2 Standalone): {total_mobilenet_params:,} params")
    print(f"   - M4 (Attention Fusion Frozen): {trainable_af_params:,} trainable / {total_af_params:,} total")
    print(f"   - M5 (End-to-End Hybrid Joint): {trainable_hy_params:,} trainable / {total_hy_params:,} total")
    print("=" * 80)

    # Save verification JSON artifact
    results_dir = os.path.join(workspace_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    verification_path = os.path.join(results_dir, 'mobilenet_v2_interface_verification.json')
    
    import json
    with open(verification_path, 'w') as f:
        json.dump({
            'status': 'PASSED',
            'device': str(device),
            'mobilenet_v2_standalone': {
                'backbone_params': backbone_only_params,
                'projection_params': proj_params,
                'classifier_params': head_params,
                'total_params': total_mobilenet_params
            },
            'legacy_resnet18_standalone_params': total_resnet_params,
            'attention_fusion_m4': {
                'total_params': total_af_params,
                'trainable_params': trainable_af_params
            },
            'end_to_end_hybrid_m5': {
                'total_params': total_hy_params,
                'trainable_params': trainable_hy_params
            }
        }, f, indent=2)
    print(f"Verification artifact saved to: {verification_path}")
    print("=" * 80)

if __name__ == '__main__':
    run_mobilenet_v2_smoke_test()
