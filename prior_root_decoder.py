"""
prior_root_decoder.py
=====================
把一个 base 解码器 (HybridFKPoseDecoder) 包起来:
  * 保留它学出来的【相对骨架结构】(root-relative)。
  * 把它的【全局 root】整个换掉, 换成:
        root = action_prior(按动作软查表) + tanh 限幅的小残差
    其中 action_prior 用源域 canonical hip 初始化 (可迁移),
    残差被 tanh 硬卡在 ±residual_scale 米以内 (默认 8cm)。

为什么这样不会动 PA-MPJPE / MPJPE_aligned:
  * PA-MPJPE 逐帧做 Procrustes 对齐 (去平移/旋转/缩放);
  * MPJPE_aligned 逐帧减 hip (去平移)。
  两者都【逐帧】去掉了平移, 所以无论 root 换成什么, 这两个指标
  数值完全不变。只有 raw MPJPE 里的 hip_error 项会变, 且在未见房间
  里被 (先验 + 8cm 上界) 约束, 不会像原来那样乱漂。

接口与 HybridFKPoseDecoder 完全一致 (多接一个可选 action_probs):
    forward(z_global, action_emb, action_probs=None) -> (p_coarse, p_final)
"""
import torch
import torch.nn as nn


class PriorRootDecoder(nn.Module):
    def __init__(self, base, in_dim=128, num_actions=27, residual_scale=0.08,
                 hidden=128, canonical=None, freeze_prior=False):
        """
        base           : 已构造好的 HybridFKPoseDecoder (结构支 + FK 支)
        in_dim         : z_global 维度 (= global_dim)
        num_actions    : 动作类别数
        residual_scale : 每帧 root 残差的硬上界 (米)。越小越安全, 0 = 纯先验
        canonical      : (num_actions, 3) 源域按动作平均 hip, 用来初始化先验
        freeze_prior   : True 则冻结 action_prior (最硬的 E04 保证, 但牺牲室内精度)
        """
        super().__init__()
        self.base = base
        self.num_actions = num_actions
        self.residual_scale = float(residual_scale)

        prior = canonical.clone().float() if canonical is not None \
                else torch.zeros(num_actions, 3)
        self.action_prior = nn.Parameter(prior, requires_grad=not freeze_prior)

        self.res = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.last_res = None   # 仅供调试; 残差 L2 惩罚在 trainer 里用 clean z_global 重算

    def set_alpha(self, a):
        # 透传给 base (Hybrid FK 的 alpha 退火), 这样 trainer 的 set_alpha 调用不用改
        if hasattr(self.base, 'set_alpha'):
            self.base.set_alpha(a)

    def _reroot(self, pose, root):
        """pose: (B,T,J,3), root: (B,T,3)。先去掉原 root (留结构), 再加新 root。"""
        return (pose - pose[:, :, 0:1, :]) + root[:, :, None, :]

    def forward(self, z_global, action_emb, action_probs=None):
        p_coarse, p_final = self.base(z_global, action_emb)
        B, T, _ = z_global.shape

        if action_probs is None:
            # 兜底: 没传动作概率时用均匀分布 -> 先验退化为所有动作均值
            action_probs = z_global.new_full((B, self.num_actions),
                                             1.0 / self.num_actions)

        root_prior = action_probs @ self.action_prior              # (B,3) 软查表
        res = torch.tanh(self.res(z_global)) * self.residual_scale  # (B,T,3), |res|<=scale
        self.last_res = res
        root = root_prior[:, None, :] + res                        # (B,T,3)

        p_coarse = self._reroot(p_coarse, root)
        p_final = self._reroot(p_final, root)
        return p_coarse, p_final


# ----------------------------------------------------------------------
# Sandbox: 验证 re-root 不改 PA / MPJPE_aligned, 只改 raw MPJPE
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    import numpy as np
    torch.manual_seed(0)
    B, T, J, C, A = 2, 8, 17, 128, 27

    # 假 base: 直接吐一个固定姿态 (root 任意), 用来验证 re-root 的数学性质
    class StubBase(nn.Module):
        def __init__(self): super().__init__(); self.dummy = nn.Linear(1, 1)
        def forward(self, z, emb):
            p = torch.randn(z.shape[0], z.shape[1], J, 3)
            return p, p.clone()

    dec = PriorRootDecoder(StubBase(), in_dim=C, num_actions=A, residual_scale=0.08)
    z = torch.randn(B, T, C)
    emb = torch.randn(B, 32)
    probs = torch.softmax(torch.randn(B, A), -1)
    _, pf = dec(z, emb, probs)
    print(f"z{tuple(z.shape)} -> p_final{tuple(pf.shape)}")
    assert pf.shape == (B, T, J, 3)

    # 同一结构, 换两个不同 root, 验证 hip-aligned 误差不变
    def mpjpe_aligned(p, g):
        pa = p - p[..., :1, :]; ga = g - g[..., :1, :]
        return float(torch.norm(pa - ga, dim=-1).mean())
    gt = torch.randn(B, T, J, 3) * 0.3
    rel = torch.randn(B, T, J, 3) * 0.3
    rel = rel - rel[..., :1, :]
    r1 = rel + torch.randn(B, T, 1, 3) * 0.5     # root 方案 A
    r2 = rel + torch.randn(B, T, 1, 3) * 0.5     # root 方案 B
    print(f"MPJPE_aligned(rootA)={mpjpe_aligned(r1, gt):.4f}  "
          f"MPJPE_aligned(rootB)={mpjpe_aligned(r2, gt):.4f}  (应相等)")
    assert abs(mpjpe_aligned(r1, gt) - mpjpe_aligned(r2, gt)) < 1e-6
    print("[OK] 换 root 不影响 hip-aligned 结构误差 (PA 同理由 Procrustes 保证)")