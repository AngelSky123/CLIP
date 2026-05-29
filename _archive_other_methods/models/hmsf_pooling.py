"""
Hierarchical Multi-Scale Feature Pooling (HMSF).

DROP-IN REPLACEMENT for LocalFeaturePooling in models/local_encoder.py.

Signature is identical to LocalFeaturePooling:
    __init__(in_channels, out_channels)
    forward(z_local: (B, T, C, H, W)) -> (B, T, out_channels)

ORIGINAL LocalFeaturePooling:
    AdaptiveAvgPool2d(1) + Linear(C -> out_channels)
    → collapses entire (114, 10) spatial plane to a single vector per frame.

HMSF version:
    Coarse:  AdaptiveAvgPool2d(1)  -> C * 1   features  (whole plane)
    Medium:  AdaptiveAvgPool2d(2)  -> C * 4   features  (4 regions)
    Fine:    AdaptiveAvgPool2d(4)  -> C * 16  features  (16 regions)
    Concat + Linear -> out_channels

This preserves joint-scale detail that the single-pool version discards.

USAGE — TWO OPTIONS:

Option A: Edit your existing models/local_encoder.py:
    Replace the LocalFeaturePooling class body with the one from this file
    (keep the class name).

Option B (recommended): Import this class and patch full_model.py:
    In models/full_model.py:
        # BEFORE:
        from .local_encoder import LocalSpatioTemporalEncoder, LocalFeaturePooling
        # AFTER:
        from .local_encoder import LocalSpatioTemporalEncoder
        from .hmsf_pooling import HierarchicalLocalFeaturePooling as LocalFeaturePooling

This way the original file stays untouched and you can A/B test easily.
"""
import torch
import torch.nn as nn


class HierarchicalLocalFeaturePooling(nn.Module):
    """Drop-in replacement for LocalFeaturePooling with multi-scale pooling.

    Args:
        in_channels:  C of input (B, T, C, H, W).  Typically 64.
        out_channels: dim of output (B, T, out_channels). Typically 128.
        coarse_size:  AdaptiveAvgPool2d output size for coarse branch. Default 1.
        medium_size:  for medium branch. Default 2.
        fine_size:    for fine branch. Default 4.
        dropout:      optional dropout after projection. Default 0.0.

    Output shape and dtype are IDENTICAL to LocalFeaturePooling, so
    no downstream module (GlobalTemporalModeler etc.) needs to change.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 coarse_size: int = 1,
                 medium_size: int = 2,
                 fine_size: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.coarse_size = coarse_size
        self.medium_size = medium_size
        self.fine_size = fine_size

        self.pool_coarse = nn.AdaptiveAvgPool2d(coarse_size)
        self.pool_medium = nn.AdaptiveAvgPool2d(medium_size)
        self.pool_fine = nn.AdaptiveAvgPool2d(fine_size)

        total_dim = in_channels * (coarse_size ** 2
                                   + medium_size ** 2
                                   + fine_size ** 2)
        # Match LocalFeaturePooling's projection: Linear + LayerNorm + GELU
        layers = [
            nn.Linear(total_dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.proj = nn.Sequential(*layers)

    def forward(self, z_local: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_local: (B, T, C, H, W)
        Returns:
            (B, T, out_channels)
        """
        B, T, C, H, W = z_local.shape
        x = z_local.reshape(B * T, C, H, W)

        f_c = self.pool_coarse(x).flatten(1)   # (B*T, C * coarse^2)
        f_m = self.pool_medium(x).flatten(1)   # (B*T, C * medium^2)
        f_f = self.pool_fine(x).flatten(1)     # (B*T, C * fine^2)

        f = torch.cat([f_c, f_m, f_f], dim=1)
        f = self.proj(f)
        return f.reshape(B, T, -1)


if __name__ == "__main__":
    # Sanity check: same I/O signature as LocalFeaturePooling
    # Original: in_channels=64, out_channels=128
    pool = HierarchicalLocalFeaturePooling(in_channels=64, out_channels=128)
    x = torch.randn(2, 64, 64, 114, 10)  # (B=2, T=64, C=64, H=114, W=10)
    y = pool(x)
    print(f"input  : {x.shape}")
    print(f"output : {y.shape}")  # expect (2, 64, 128)
    print(f"params : {sum(p.numel() for p in pool.parameters() if p.requires_grad)/1e3:.1f}K")
    # Compare with original
    print()
    print("Original LocalFeaturePooling for reference:")
    orig = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 128),
        nn.LayerNorm(128),
        nn.GELU(),
    )
    n_orig = sum(p.numel() for p in orig.parameters())
    print(f"params : {n_orig/1e3:.1f}K  (HMSF adds {(sum(p.numel() for p in pool.parameters()) - n_orig)/1e3:.1f}K)")