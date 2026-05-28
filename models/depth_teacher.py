"""
Depth -> 3D pose teacher (Step A of depth cross-modal distillation).

Trained from scratch on source envs (E01-E03) with GT pose supervision. Once
trained and frozen, its per-frame feature (z_global, B,T,128) is the distillation
target the CSI encoder's z_global is aligned to (Step B).

Design notes:
  * Single-channel depth CNN (from scratch — no ImageNet; depth->pose is a strong
    signal and avoids the modality-transfer pitfalls we already hit).
  * Outputs z_global with the SAME dim (128) as the CSI GlobalTemporalModeler, so
    feature-level distillation is a direct alignment, no shape juggling.
  * Reuses GlobalTemporalModeler for temporal modeling so teacher and student
    share the same temporal feature geometry.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .global_encoder import GlobalTemporalModeler


class _ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
    def forward(self, x):
        return F.gelu(self.bn(self.conv(x)))


class DepthEncoder(nn.Module):
    """Per-frame single-channel depth -> feature vector.

    (N, 1, H, W) -> (N, out_dim). Five stride-2 stages: 112 -> 56 -> 28 -> 14 -> 7.
    """
    def __init__(self, out_dim=128, widths=(32, 64, 128, 128), chunk_size=16):
        super().__init__()
        self.chunk_size = chunk_size
        chans = [1] + list(widths)
        self.stem = nn.Sequential(
            nn.Conv2d(1, chans[1], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(chans[1]), nn.GELU(),
        )
        blocks = []
        for i in range(1, len(widths)):
            blocks.append(_ConvBlock(chans[i], chans[i + 1], stride=2))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(widths[-1], out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def _encode(self, x):                       # (N,1,H,W) -> (N,out_dim)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)

    def forward(self, depth):                   # (B,T,1,H,W) -> (B,T,out_dim)
        B, T, C, H, W = depth.shape
        outs = []
        for t0 in range(0, T, self.chunk_size):
            t1 = min(t0 + self.chunk_size, T)
            n = t1 - t0
            chunk = depth[:, t0:t1].reshape(B * n, C, H, W)
            outs.append(self._encode(chunk).reshape(B, n, -1))
        return torch.cat(outs, dim=1)


class DepthPoseTeacher(nn.Module):
    """Depth -> z_global -> 3D pose. z_global is the distillation target."""
    def __init__(self, global_dim=128, num_joints=17, seq_len=64,
                 num_transformer_layers=3, num_heads=4,
                 tcn_channels=(128, 128), tcn_kernel_size=3, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints
        self.encoder = DepthEncoder(out_dim=global_dim)
        self.global_modeler = GlobalTemporalModeler(
            in_dim=global_dim, global_dim=global_dim,
            num_transformer_layers=num_transformer_layers, num_heads=num_heads,
            tcn_channels=list(tcn_channels), tcn_kernel_size=tcn_kernel_size,
            dropout=dropout, max_seq_len=seq_len + 50)
        self.pose_head = nn.Sequential(
            nn.Linear(global_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, num_joints * 3))

    def forward(self, depth):                   # (B,T,1,H,W)
        feat = self.encoder(depth)              # (B,T,gd)
        z_global = self.global_modeler(feat)    # (B,T,gd)
        B, T, _ = z_global.shape
        pose = self.pose_head(z_global).reshape(B, T, self.num_joints, 3)
        return {'p_final': pose, 'z_global': z_global}


if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    m = DepthPoseTeacher(global_dim=128, seq_len=64)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    x = torch.randn(2, 8, 1, 112, 112)
    out = m(x)
    print(f"depth {tuple(x.shape)} -> pose {tuple(out['p_final'].shape)} "
          f"z_global {tuple(out['z_global'].shape)} | params {n/1e6:.2f}M")
    out['p_final'].sum().backward()
    print("backward OK")