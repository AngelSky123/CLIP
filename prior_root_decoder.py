"""
prior_root_decoder.py  (v2: 残差头改吃【保尺度特征】)
=====================================================
把一个 base 解码器 (HybridFKPoseDecoder) 包起来:
  * 保留它学出来的【相对骨架结构】(root-relative)。
  * 把它的【全局 root】整个换掉, 换成:
        root = action_prior(按动作软查表) + tanh 限幅的小残差
    其中 action_prior 用源域 canonical hip 初始化 (可迁移),
    残差被 tanh 硬卡在 ±residual_scale 米以内 (默认 0.3m)。

v2 关键改动 (相对 v1):
  残差头 self.res 的输入从 z_global 改成【外部传入的保尺度特征 rawscale_feat】。
  原因: z_global 是主干 (逐天线 min-max + InstanceNorm + MixStyle) 的产物,
  幅度尺度/天线间相对幅度在它到达这里前已被删干净 —— 残差头从 z_global 拿不到
  任何绝对定位线索, 放大 residual_scale 也只是让它从无信息特征里瞎猜 (上次 328->346
  的源域过拟合就是这么来的)。改吃 raw_scale_encoder 输出的保尺度特征后, 残差头
  才【可能】看到位置线索。是否真有效, 看 hip_err 止损线。

  residual_scale 默认 0.08 -> 0.3: E04 hip 缺口 ~329mm, 8cm 残差就算方向对也搬不动;
  0.3m 给足空间, 又不至于像 0.5-1.0m 那样让源域过拟合彻底失控。

为什么这样不会动 PA-MPJPE / MPJPE_aligned (与 v1 完全相同):
  PA 逐帧 Procrustes 去平移, MPJPE_aligned 逐帧减 hip。两者都【逐帧去平移】,
  所以无论 root 换成什么、残差头学好学坏, 这两个指标数值完全不变。只有 raw MPJPE
  的 hip_error 项会变。=> 本实验最坏情况只是 hip 不动, PA (102.x) 一定保住。

接口:
    forward(z_global, action_emb, action_probs=None, rawscale_feat=None)
        -> (p_coarse, p_final)
  rawscale_feat: (B,T,in_dim), 由 full_model 的 raw_scale_encoder 产出。
                 为 None 时残差退化为 0 (纯先验 root), 用于无保尺度输入的兜底。
"""
import torch
import torch.nn as nn


class PriorRootDecoder(nn.Module):
    def __init__(self, base, in_dim=128, num_actions=27, residual_scale=0.3,
                 hidden=128, canonical=None, freeze_prior=False):
        """
        base           : 已构造好的 HybridFKPoseDecoder (结构支 + FK 支)
        in_dim         : z_global 维度 (= global_dim), 也是 rawscale_feat 的维度
        num_actions    : 动作类别数
        residual_scale : 每帧 root 残差的硬上界 (米)。越小越安全, 0 = 纯先验
        canonical      : (num_actions, 3) 源域按动作平均 hip, 用来初始化先验
        freeze_prior   : True 则冻结 action_prior
        """
        super().__init__()
        self.base = base
        self.num_actions = num_actions
        self.residual_scale = float(residual_scale)

        prior = canonical.clone().float() if canonical is not None \
                else torch.zeros(num_actions, 3)
        self.action_prior = nn.Parameter(prior, requires_grad=not freeze_prior)

        # 残差头: 输入是【保尺度特征】(in_dim 维), 不再是 z_global。
        self.res = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.last_res = None   # 仅供调试; 残差 L2 惩罚在 trainer 里用 clean rawscale_feat 重算

    def set_alpha(self, a):
        # 透传给 base (Hybrid FK 的 alpha 退火)
        if hasattr(self.base, 'set_alpha'):
            self.base.set_alpha(a)

    def _reroot(self, pose, root):
        """pose: (B,T,J,3), root: (B,T,3)。先去掉原 root (留结构), 再加新 root。"""
        return (pose - pose[:, :, 0:1, :]) + root[:, :, None, :]

    def forward(self, z_global, action_emb, action_probs=None, rawscale_feat=None):
        p_coarse, p_final = self.base(z_global, action_emb)
        B, T, _ = z_global.shape

        if action_probs is None:
            action_probs = z_global.new_full((B, self.num_actions),
                                             1.0 / self.num_actions)

        root_prior = action_probs @ self.action_prior              # (B,3) 软查表

        if rawscale_feat is not None:
            res = torch.tanh(self.res(rawscale_feat)) * self.residual_scale  # (B,T,3)
        else:
            res = z_global.new_zeros(B, T, 3)
        self.last_res = res

        root = root_prior[:, None, :] + res                        # (B,T,3)
        p_coarse = self._reroot(p_coarse, root)
        p_final = self._reroot(p_final, root)
        return p_coarse, p_final


if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    torch.manual_seed(0)
    B, T, J, C, A = 2, 8, 17, 128, 27

    class StubBase(nn.Module):
        def __init__(self): super().__init__(); self.dummy = nn.Linear(1, 1)
        def set_alpha(self, a): pass
        def forward(self, z, emb):
            p = torch.randn(z.shape[0], z.shape[1], J, 3)
            return p, p.clone()

    dec = PriorRootDecoder(StubBase(), in_dim=C, num_actions=A, residual_scale=0.3)
    z = torch.randn(B, T, C)
    emb = torch.randn(B, 32)
    probs = torch.softmax(torch.randn(B, A), -1)
    rs_feat = torch.randn(B, T, C, requires_grad=True)

    _, pf = dec(z, emb, probs, rs_feat)
    print(f"z{tuple(z.shape)} rawscale_feat{tuple(rs_feat.shape)} -> p_final{tuple(pf.shape)}")
    assert pf.shape == (B, T, J, 3)
    pf.sum().backward()
    assert rs_feat.grad is not None and rs_feat.grad.abs().sum() > 0
    assert dec.res[0].weight.grad is not None
    print("[OK] 梯度回流 rawscale_feat 与残差头")

    assert dec.last_res.abs().max().item() <= 0.3 + 1e-6
    print(f"[OK] 残差 |res|max={dec.last_res.abs().max().item():.3f} <= residual_scale=0.3")

    _, pf_none = dec(z, emb, probs, None)
    assert dec.last_res.abs().max().item() < 1e-9
    print("[OK] rawscale_feat=None -> 零残差 (纯先验 root)")

    def mpjpe_aligned(p, g):
        pa = p - p[..., :1, :]; ga = g - g[..., :1, :]
        return float(torch.norm(pa - ga, dim=-1).mean())
    gt = torch.randn(B, T, J, 3) * 0.3
    rel = torch.randn(B, T, J, 3) * 0.3; rel = rel - rel[..., :1, :]
    r1 = rel + torch.randn(B, T, 1, 3) * 0.5
    r2 = rel + torch.randn(B, T, 1, 3) * 0.5
    assert abs(mpjpe_aligned(r1, gt) - mpjpe_aligned(r2, gt)) < 1e-6
    print("[OK] 换 root 不影响 hip-aligned 结构误差 (PA 由 Procrustes 同理保证)")