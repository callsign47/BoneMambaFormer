import torch
import torch.nn as nn
import torchvision.models as models

class BoneCancerCNN(nn.Module):
    """
    Lightweight CNN Backbone for Bone Cancer Radiograph Classification.
    Provides local feature extraction (edges, texture, boundaries) and
    projects features to a common 256-D space for future Adaptive Attention Fusion.
    
    Default backbone: ResNet-18 (ImageNet pretrained)
    Target resolution: 224x224
    Class mapping: cancer=0, normal=1
    """
    def __init__(self, backbone_name='resnet18', num_classes=2, feature_dim=256, pretrained=True, dropout_rate=0.2):
        super(BoneCancerCNN, self).__init__()
        self.backbone_name = backbone_name
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        if backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.backbone = base_model
        elif backbone_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.backbone = base_model
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Feature Projection to 256-D common representation
        self.projection = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate)
        )

        # Final Classifier Head
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward_features(self, x):
        """
        Extract 256-D feature representation before classification.
        Returns:
            features: Tensor of shape (B, feature_dim)
        """
        bb_feat = self.backbone(x)
        proj_feat = self.projection(bb_feat)
        return proj_feat

    def forward(self, x):
        """
        Forward pass returning logits.
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits

if __name__ == '__main__':
    model = BoneCancerCNN()
    x = torch.randn(4, 3, 224, 224)
    feat = model.forward_features(x)
    logits = model(x)
    print("CNN Architecture Test PASSED!")
    print("Input shape:", x.shape)
    print("Features shape:", feat.shape)
    print("Logits shape:", logits.shape)
