"""
mae_dcl_losses.py — 在你的 MAE 预训练 (Stage 1A) 上叠加 DT-Pose 式的
域一致表征学习 (TC-CL + uniformity), 但【翻译适配】到你的 64 帧窗口架构。

== 为什么不能照搬 DT-Pose ==
DT-Pose 的 TC-CL (论文 Eq.2) 前提是「一个样本=单帧」, 正样本对=同序列相邻两帧。
你的一个样本是 64 帧窗口, z_global 形状 (B, T=64, 128) —— 相邻帧已经在同一个
样本【内部】, 没有「另一个相邻帧样本」可配对。而且你的 GlobalTemporalModeler
已经在窗口内建模了时序。所以直接照抄 Eq.2 在你这里没有对应物。

== 翻译后的三个组件 (都作用在 z_global, 与 MAE 重建并行) ==

1) uniformity_loss(z)  —— DT-Pose Eq.3, 直接可用
   防 dimensional collapse。对治你诊断里 PredStd 低 / E04 退化均值姿态。

2) temporal_contrastive_loss(z)  —— TC-CL 的窗口内翻译版
   正样本: 同一窗口内相邻帧 (z_t, z_{t+1}); 负样本: batch 内其他窗口的帧。
   保留「相邻帧相似、跨序列帧相异」的 motion-discriminative 语义。

3) domain_invariant_loss(z, env, action)  —— 真正针对 cross-env 瓶颈 (★)
   你的 domain shift 是 hip_err 域内212→域外342。这一项直接学环境不变:
   同一【动作】、不同【环境】的窗口特征拉近; 不同动作的推远。
   这比 DT-Pose 原版更直接打你的痛点 (它原版不显式用 env 标签)。
   需要 dataset 返回 env/action —— 你的 MMFiDataset 已经返回了。

接入只改 train_mae.py 的 forward 与 loss 累加, 不改模型结构, 不改 dataset。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pool_time(z):
    """(B,T,C) -> (B,C) 时间平均池化, 得到每个窗口的紧凑表征。"""
    return z.mean(dim=1) if z.dim() == 3 else z


# ----------------------------------------------------------------------
# 1) Uniformity (DT-Pose Eq.3)
# ----------------------------------------------------------------------
def uniformity_loss(z, eps=1e-8):
    """惩罚 batch 内表征两两余弦相似度平方, 强制铺开、防塌缩。
    z: (B,T,C) 或 (B,C)。"""
    e = _pool_time(z)
    B = e.shape[0]
    if B < 2:
        return torch.zeros((), device=e.device)
    e = F.normalize(e, dim=-1, eps=eps)
    sim = e @ e.t()
    mask = ~torch.eye(B, dtype=torch.bool, device=e.device)
    return (sim[mask] ** 2).mean()


# ----------------------------------------------------------------------
# 2) Temporal contrastive (TC-CL 窗口内翻译版)
# ----------------------------------------------------------------------
def temporal_contrastive_loss(z, tau=0.1, max_pairs_per_win=8):
    """同窗口相邻帧为正, 跨窗口帧为负的 InfoNCE。

    z: (B,T,C)。对每个窗口随机取若干相邻帧对 (t, t+1) 作为正样本,
    负样本来自 batch 内【其他窗口】的帧 (avg-pooled 代表)。

    实现: 每个窗口先取一个 anchor 帧 z_t 和它的正样本 z_{t+1};
    负样本池 = 其他窗口的时间平均表征 (B-1 个)。InfoNCE over batch。
    """
    B, T, C = z.shape
    if B < 2 or T < 2:
        return torch.zeros((), device=z.device)
    zt = z[:, :-1, :]                       # (B,T-1,C) anchors
    ztp = z[:, 1:, :]                        # (B,T-1,C) positives
    # 每个窗口随机采若干相邻对
    k = min(max_pairs_per_win, T - 1)
    idx = torch.randperm(T - 1, device=z.device)[:k]
    a = F.normalize(zt[:, idx, :], dim=-1)   # (B,k,C)
    p = F.normalize(ztp[:, idx, :], dim=-1)  # (B,k,C)
    # 负样本: 其他窗口的窗口级表征 (B,C)
    win_repr = F.normalize(_pool_time(z), dim=-1)   # (B,C)

    a_flat = a.reshape(B * k, C)             # (BK,C)
    pos = (a_flat * p.reshape(B * k, C)).sum(-1, keepdim=True) / tau   # (BK,1)
    neg = a_flat @ win_repr.t() / tau        # (BK,B) 与所有窗口表征
    # 屏蔽 anchor 自己所属窗口 (它不算负样本)
    own = torch.arange(B, device=z.device).repeat_interleave(k)       # (BK,)
    neg.scatter_(1, own.unsqueeze(1), float('-inf'))
    logits = torch.cat([pos, neg], dim=1)    # (BK, 1+B)
    labels = torch.zeros(B * k, dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)


# ----------------------------------------------------------------------
# 3) Domain-invariant contrastive (★ 针对 cross-env 瓶颈)
# ----------------------------------------------------------------------
def domain_invariant_loss(z, env_ids, action_ids, tau=0.1):
    """同动作-跨环境拉近, 不同动作推远的 supervised contrastive。

    z: (B,T,C); env_ids,action_ids: (B,) long。
    正样本对 = 同 action 但【不同 env】的窗口 (强制环境不变);
    负样本 = 不同 action 的窗口。
    若 batch 内没有「同动作跨环境」的对, 该项返回 0 (跳过)。
    """
    B = z.shape[0]
    if B < 2:
        return torch.zeros((), device=z.device)
    e = F.normalize(_pool_time(z), dim=-1)         # (B,C)
    sim = e @ e.t() / tau                          # (B,B)
    same_act = action_ids.unsqueeze(0) == action_ids.unsqueeze(1)   # (B,B)
    diff_env = env_ids.unsqueeze(0) != env_ids.unsqueeze(1)
    pos_mask = same_act & diff_env                 # 同动作跨环境 = 正
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    pos_mask = pos_mask & ~eye
    if pos_mask.sum() == 0:
        return torch.zeros((), device=z.device)
    # supervised contrastive: 对每个 anchor, 分母含所有非自身样本
    sim_exp = torch.exp(sim) * (~eye).float()
    denom = sim_exp.sum(dim=1, keepdim=True) + 1e-8
    log_prob = sim - torch.log(denom)
    # 每个 anchor 对其所有正样本取平均 log-prob
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
    z = torch.randn(B, T, C, requires_grad=True)
    env = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    act = torch.tensor([3, 5, 3, 5, 3, 5, 7, 7])   # action 3 跨 env 0/1/2

    lu = uniformity_loss(z)
    lt = temporal_contrastive_loss(z)
    ld = domain_invariant_loss(z, env, act)
    print(f"uniformity        = {lu.item():.4f}")
    print(f"temporal_contrast = {lt.item():.4f}")
    print(f"domain_invariant  = {ld.item():.4f}")
    (lu + lt + ld).backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    print("grad -> z: OK")

    # 塌缩检验
    zc = torch.ones(B, T, C, requires_grad=True)
    print(f"collapsed uniformity = {uniformity_loss(zc).item():.4f} (应≈1)")
    # 无同动作跨环境时 domain_invariant 应为 0
    act2 = torch.arange(B)
    print(f"no cross-env pairs -> domain_inv = {domain_invariant_loss(z, env, act2).item():.4f} (应=0)")
    print("[OK]")