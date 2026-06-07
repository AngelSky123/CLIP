"""
mae_dcl_losses.py  (v2, 路A: 逐帧抗塌缩)

== v1 失败原因 ==
v1 把 uniformity/contrastive 都作用在 _pool_time(z) = 对 64 帧时间平均后的
窗口表征上。但对 64 帧平均会抹平动作时序动态, 剩下近似「平均能量」标量,
不同动作/环境平均完都差不多 -> 全挤在一起。再加上 lambda_unif=0.04 太小,
压不过 MAE 重建把特征往一起拉的趋势。结果探针: s_diff 从 0.91 恶化到 0.99。

== v2 改动 ==
1) uniformity / temporal_contrastive / domain_invariant 全部改在【逐帧特征】
   z.reshape(B*T, C) 上算 (不池化), 保住每一帧的判别性, 直接打散塌缩。
2) 权重默认拉大 (调用方用命令行覆盖): unif 推荐 0.5, tcl 0.1, dinv 0.3。
3) uniformity 用更强的形式: 余弦相似度平方 + 方差下界 hinge (VICReg 式),
   直接阻止「所有帧塌成一个点」。

判读这次探针只看一个数: s_diff 能否从 0.99 掉下来 (塌缩是否被打散)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1) Uniformity (逐帧 + 方差下界, 强力抗塌缩)
# ----------------------------------------------------------------------
def uniformity_loss(z, eps=1e-8, var_margin=1.0):
    """逐帧抗塌缩。z: (B,T,C) -> 展平成 (N=B*T, C) 后:
      (a) 余弦相似度平方惩罚 (DT-Pose Eq.3 的逐帧版)
      (b) 方差下界 hinge: 每维标准差低于 var_margin 时惩罚 (VICReg 式),
          直接阻止「所有帧塌成一个点」。
    """
    if z.dim() == 3:
        B, T, C = z.shape
        z = z.reshape(B * T, C)
    N = z.shape[0]
    if N < 2:
        return torch.zeros((), device=z.device)

    # (a) 余弦相似度平方 (子采样控成本)
    if N > 512:
        idx = torch.randperm(N, device=z.device)[:512]
        zc = z[idx]
    else:
        zc = z
    zn = F.normalize(zc, dim=-1, eps=eps)
    sim = zn @ zn.t()
    m = ~torch.eye(zn.shape[0], dtype=torch.bool, device=z.device)
    l_cos = (sim[m] ** 2).mean()

    # (b) 方差下界 hinge (VICReg): 鼓励每个特征维度有 >= var_margin 的标准差
    std = torch.sqrt(z.var(dim=0) + eps)          # (C,)
    l_var = F.relu(var_margin - std).mean()

    return l_cos + l_var


# ----------------------------------------------------------------------
# 2) Temporal contrastive (逐帧版, 同窗口相邻帧正 / 跨窗口帧负)
# ----------------------------------------------------------------------
def temporal_contrastive_loss(z, tau=0.1, max_pairs_per_win=8):
    """z: (B,T,C)。正样本=同窗口相邻帧, 负样本=其他窗口的【逐帧】特征 (不池化)。"""
    B, T, C = z.shape
    if B < 2 or T < 2:
        return torch.zeros((), device=z.device)
    k = min(max_pairs_per_win, T - 1)
    tidx = torch.randperm(T - 1, device=z.device)[:k]
    a = F.normalize(z[:, tidx, :], dim=-1)        # (B,k,C)
    p = F.normalize(z[:, tidx + 1, :], dim=-1)    # (B,k,C)

    # 负样本池: 所有窗口各随机取 1 帧的逐帧特征 (B,C), 比池化更有判别性
    neg_t = torch.randint(0, T, (B,), device=z.device)
    neg_pool = F.normalize(z[torch.arange(B), neg_t], dim=-1)   # (B,C)

    a_flat = a.reshape(B * k, C)
    p_flat = p.reshape(B * k, C)
    pos = (a_flat * p_flat).sum(-1, keepdim=True) / tau          # (BK,1)
    neg = a_flat @ neg_pool.t() / tau                            # (BK,B)
    own = torch.arange(B, device=z.device).repeat_interleave(k)
    neg.scatter_(1, own.unsqueeze(1), float('-inf'))
    logits = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(B * k, dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)


# ----------------------------------------------------------------------
# 3) Domain-invariant (逐帧版, 同动作跨环境拉近 / 不同动作推远)
# ----------------------------------------------------------------------
def domain_invariant_loss(z, env_ids, action_ids, tau=0.1, frames_per_win=4):
    """在逐帧特征上做 supervised contrastive。

    z: (B,T,C); env_ids/action_ids: (B,). 每个窗口随机取 frames_per_win 帧,
    标签继承自所属窗口的 action/env。同 action 跨 env 为正, 不同 action 为负。
    """
    B, T, C = z.shape
    if B < 2:
        return torch.zeros((), device=z.device)
    fpw = min(frames_per_win, T)
    tidx = torch.randperm(T, device=z.device)[:fpw]
    zf = z[:, tidx, :].reshape(B * fpw, C)                       # (N,C)
    e = env_ids.repeat_interleave(fpw)                          # (N,)
    a = action_ids.repeat_interleave(fpw)
    zf = F.normalize(zf, dim=-1)
    N = zf.shape[0]
    sim = zf @ zf.t() / tau                                     # (N,N)
    same_act = a.unsqueeze(0) == a.unsqueeze(1)
    diff_env = e.unsqueeze(0) != e.unsqueeze(1)
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    pos_mask = same_act & diff_env & ~eye
    if pos_mask.sum() == 0:
        return torch.zeros((), device=z.device)
    sim_exp = torch.exp(sim) * (~eye).float()
    denom = sim_exp.sum(dim=1, keepdim=True) + 1e-8
    log_prob = sim - torch.log(denom)
    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return torch.zeros((), device=z.device)
    loss = -(log_prob * pos_mask.float()).sum(dim=1)[valid] / pos_count[valid]
    return loss.mean()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    torch.manual_seed(0)
    B, T, C = 8, 64, 128
    env = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    act = torch.tensor([3, 5, 3, 5, 3, 5, 7, 7])

    z = torch.randn(B, T, C, requires_grad=True)
    print(f"random: unif={uniformity_loss(z).item():.4f} "
          f"tcl={temporal_contrastive_loss(z).item():.4f} "
          f"dinv={domain_invariant_loss(z, env, act).item():.4f}")
    (uniformity_loss(z) + temporal_contrastive_loss(z)
     + domain_invariant_loss(z, env, act)).backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    print("grad -> z: OK")

    zc = (torch.ones(B, T, C) + torch.randn(B, T, C) * 0.001).requires_grad_(True)
    print(f"collapsed: unif={uniformity_loss(zc).item():.4f} (应较大: cos≈1 + var hinge≈1)")
    print("[OK]")