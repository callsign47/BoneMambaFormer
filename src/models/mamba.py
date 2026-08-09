import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def selective_ssm_recurrence(
    x_conv: torch.Tensor,
    dA_all: torch.Tensor,
    dB_x_all: torch.Tensor,
    C_exp: torch.Tensor,
    D: torch.Tensor
) -> torch.Tensor:
    """
    TorchScript JIT Compiled Selective Scan Recurrence for State Space Model (SSM).
    Compiles the sequence scanning loop into optimized C++/CUDA kernel execution.
    """
    B, L, d_inner, d_state = dA_all.shape
    state = torch.zeros(B, d_inner, d_state, device=dA_all.device, dtype=dA_all.dtype)
    ys = torch.empty(B, L, d_inner, device=dA_all.device, dtype=dA_all.dtype)
    for t in range(L):
        state = state * dA_all[:, t] + dB_x_all[:, t]
        state = torch.nan_to_num(torch.clamp(state, min=-50.0, max=50.0), nan=0.0, posinf=50.0, neginf=-50.0)
        yt = (state * C_exp[:, t]).sum(dim=-1) + x_conv[:, t] * D
        yt = torch.nan_to_num(torch.clamp(yt, min=-50.0, max=50.0), nan=0.0, posinf=50.0, neginf=-50.0)
        ys[:, t] = yt
    return ys


class PurePyTorchMambaBlock(nn.Module):
    """
    Pure PyTorch implementation of Selective State Space Model (SSM) block.
    Guarantees portable execution across all hardware platforms without CUDA extensions.
    Implements input-dependent gating, 1D convolution, state projections, and residual connection.
    """
    def __init__(self, d_model: int = 128, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        # Projection into expanded dimension (in_proj)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # 1D Depthwise Convolution for local sequence context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )

        # Selective SSM Parameter Projections (dt, B, C)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)
        self.dt_proj = nn.Linear(1, self.d_inner)

        # State Space Matrices (A, D)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, L, d_model)
        """
        B, L, _ = x.shape

        # Dual projection (gated activation branch)
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1)  # (B, L, d_inner) each

        # 1D Depthwise Conv along sequence dimension L
        x_conv = x_branch.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # Trim padding
        x_conv = F.silu(x_conv.transpose(1, 2))  # (B, L, d_inner)

        # Selective gating
        ssm_params = self.x_proj(x_conv)  # (B, L, 2*d_state + 1)
        dt_param = ssm_params[:, :, :1]
        B_param = ssm_params[:, :, 1:self.d_state+1]
        C_param = ssm_params[:, :, self.d_state+1:]

        dt = F.softplus(self.dt_proj(dt_param)).float()  # (B, L, d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        x_conv_f32 = x_conv.float()
        B_param_f32 = B_param.float()
        C_param_f32 = C_param.float()

        # Precompute SSM discretization parameters
        dA_all = torch.exp(torch.clamp(dt.unsqueeze(-1) * A.view(1, 1, self.d_inner, self.d_state), max=0.0))
        dB_x_all = x_conv_f32.unsqueeze(-1) * (dt.unsqueeze(-1) * B_param_f32.unsqueeze(2))
        C_exp = C_param_f32.unsqueeze(2)

        # Selective scan recurrence
        y = selective_ssm_recurrence(x_conv_f32, dA_all, dB_x_all, C_exp, self.D.float()).to(x.dtype)

        # Gated multiplicative output
        out = y * F.silu(z_branch)
        out = self.out_proj(out)  # (B, L, d_model)
        return out


class BoneCancerMamba(nn.Module):
    """
    Selective State Space Model (Mamba) Backbone for Bone Cancer Radiograph Classification.
    Provides long-range sequential context modeling by patchifying input images (224x224)
    into 196 tokens (16x16 patches) and processing them sequentially via SSM blocks.
    Projects features to a common 256-D space for standalone baseline evaluation
    and future Adaptive Attention Fusion.

    Default configuration:
      - Patch size: 32x32 -> 49 sequence tokens
      - Model dimension: 128
      - Depth: 2 Mamba blocks
      - Projection dim: 256-D
      - Target resolution: 224x224
      - Class mapping: cancer=0, normal=1
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 32,
        in_chans: int = 3,
        d_model: int = 128,
        depth: int = 2,
        num_classes: int = 2,
        feature_dim: int = 256,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        num_patches = (img_size // patch_size) ** 2  # (224 // 32)^2 = 7x7 = 49 patches

        # Patch Embedding Layer
        self.patch_embed = nn.Conv2d(
            in_chans, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Stack of Selective SSM Mamba Blocks
        self.blocks = nn.ModuleList([
            PurePyTorchMambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Feature Projection Head to 256-D common representation
        self.projection = nn.Sequential(
            nn.Linear(d_model, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate)
        )

        # Final Classifier Head
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 256-D feature representation before classification.
        Returns:
            features: Tensor of shape (B, feature_dim)
        """
        B = x.shape[0]
        # Patchify -> (B, d_model, H_p, W_p) -> (B, L, d_model)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        # Pass through Mamba SSM blocks with residual connections
        for block in self.blocks:
            x = x + block(x)

        x = self.norm(x)
        # Global Mean Pooling over sequence tokens
        seq_pooled = x.mean(dim=1)  # (B, d_model)

        # Project to 256-D common feature space
        proj_feat = self.projection(seq_pooled)  # (B, feature_dim)
        return proj_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning logits.
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits


if __name__ == '__main__':
    print("Testing BoneCancerMamba architecture...")
    model = BoneCancerMamba(img_size=224, patch_size=16, d_model=128, depth=2, feature_dim=256, num_classes=2)
    x = torch.randn(4, 3, 224, 224)
    feat = model.forward_features(x)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    print("BoneCancerMamba Architecture Test PASSED!")
    print("Input shape:", x.shape)
    print("Features shape:", feat.shape)
    print("Logits shape:", logits.shape)
    print("Backward pass successful!")
