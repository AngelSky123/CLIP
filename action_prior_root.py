"""
action_prior_root.py — 路1: 承认 E04 绝对位置不可测, 优化「不可测时的最优 fallback」。

== 探针给出的事实 ==
线性/MLP 探针均证明: E04 的 hip 绝对位置信息 ≈ 0 (探针误差 316~328 ≈ 均值基线 325)。
更关键: 你端到端 hip_err=341 > 均值基线 325 —— 模型在 E04 上 hip 比【瞎猜训练集均值】
还差, 说明它被无效特征带着【乱跑】。DT-Pose 的 316 大概率是 hip 稳定停在好先验上,
而非真的解出了位置。

== 因此路1不试图解位置 (探针证明不可能), 而是: ==
  A) ActionPriorRoot: hip = 「该动作的可学习先验位置」+「一个被强约束的小残差」。
     先验由预测的动作类别决定 (动作→典型hip位置), 残差只在小范围内微调,
     防止 hip 被无效特征带跑。E04 测不准时, 至少稳定停在「该动作的好先验」上。
  B) 损失层面: hip 残差 L2 惩罚 (限制乱动) + 时序平滑 (压逐帧抖动)。

接口与 RootDecoupledPoseDecoder 一致, 在 full_model 里替换 pose_decoder 即可。
pose = pose_rel(相对骨架, 复用现有结构) + root(动作先验+小残差)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pose_decoder import CoarsePoseHead, SkeletonRefiner


class ActionPriorRoot(nn.Module):
    """hip 全局位置 = 动作条件先验 + 受约束的小残差。

    - action_prior: 可学习查找表 (num_actions, 3), 每个动作一个典型 hip 位置。
      用预测的动作软概率加权 -> 得到该样本的先验 hip。
    - residual head: 从 z_global 回归一个【小】残差 (tanh 限幅 * scale),
      只允许在先验附近微调, 不让 hip 被无效特征带跑。
    - 时序平滑: 对最终 root 序列做 depthwise 平滑。
    """
    def __init__(self, in_dim=128, num_actions=27, residual_scale=0.3,
                 hidden=128, num_phases=8):
        super().__init__()
        self.num_actions = num_actions
        self.num_phases = num_phases           # 窗口内时间相位分桶 (承载hip轨迹)
        self.residual_scale = residual_scale   # 残差最大幅度(米), 限制乱动

        # 动作×时间相位 -> 典型hip位置 的可学习先验 (零初始, 训练/先验损失学到)
        # 形状 (A, P, 3): 每个动作在窗口内每个相位段一个典型 hip 位置
        self.action_prior = nn.Parameter(torch.zeros(num_actions, num_phases, 3))

        # 受约束的小残差头
        self.residual = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        # 时序平滑 (kernel=7, 比之前更强, 因为 hip 该尽量稳)
        self.smooth = nn.Conv1d(3, 3, kernel_size=7, padding=3, groups=3, bias=False)
        with torch.no_grad():
            self.smooth.weight.fill_(1.0 / 7.0)

    def _phase_index(self, T, device):
        """把窗口内 T 帧映射到 [0, num_phases) 的相位桶 (B无关, 返回(T,))。"""
        idx = (torch.arange(T, device=device).float() * self.num_phases / max(T, 1))
        return idx.long().clamp(0, self.num_phases - 1)        # (T,)

    def forward(self, z_global, action_probs):
        """
        z_global:     (B,T,C)
        action_probs: (B, num_actions) 软概率 (来自 action_classifier)
        return root_xyz: (B,T,1,3)
        """
        B, T, _ = z_global.shape
        pidx = self._phase_index(T, z_global.device)          # (T,)
        # 1) 动作×相位先验: 先按动作软加权 -> (B,P,3), 再按每帧相位取 -> (B,T,3)
        #    action_probs:(B,A) , action_prior:(A,P,3)
        prior_ap = torch.einsum('ba,apc->bpc', action_probs, self.action_prior)  # (B,P,3)
        prior = prior_ap[:, pidx, :]                          # (B,T,3) 按相位gather
        # 2) 受限残差: tanh 限幅, 只能在 ±residual_scale 米内微调
        res = torch.tanh(self.residual(z_global)) * self.residual_scale  # (B,T,3)
        root = prior + res                                    # (B,T,3)
        # 3) 时序平滑
        root = self.smooth(root.transpose(1, 2)).transpose(1, 2)
        return root.unsqueeze(2)                              # (B,T,1,3)


class ActionPriorPoseDecoder(nn.Module):
    """相对骨架(复用现有) + 动作先验 root。接口同 PoseDecoder。

    需要 action_probs: 由 full_model 在 forward 里从 action_classifier 取,
    通过 forward 的额外参数传入。为兼容现有 PoseDecoder(z, action_emb) 签名,
    这里约定: 把 action_probs 也通过 action_emb 之外的通道传 —— 见 full_model 接入。
    """
    def __init__(self, in_dim=128, hidden_dim=256, gcn_hidden=128,
                 num_gcn_layers=3, num_joints=17, action_embed_dim=32,
                 num_actions=27, residual_scale=0.3):
        super().__init__()
        self.num_joints = num_joints
        self.coarse_head = CoarsePoseHead(in_dim, hidden_dim, num_joints, action_embed_dim)
        self.root_head = ActionPriorRoot(in_dim, num_actions, residual_scale)
        self.refiner = SkeletonRefiner(in_features=3, hidden_dim=gcn_hidden,
                                       num_layers=num_gcn_layers, num_joints=num_joints)

    def forward(self, z_global, action_emb, action_probs=None):
        pose_rel = self.coarse_head(z_global, action_emb)
        pose_rel = pose_rel - pose_rel[:, :, 0:1, :]          # 强制 root-relative
        if action_probs is None:
            # 兜底: 没传 action_probs 时用均匀分布 (退化成单一先验)
            B = z_global.shape[0]
            action_probs = torch.full((B, self.root_head.num_actions),
                                      1.0 / self.root_head.num_actions,
                                      device=z_global.device)
        root_xyz = self.root_head(z_global, action_probs)     # (B,T,1,3)
        p_coarse = pose_rel + root_xyz
        p_final = self.refiner(p_coarse)
        return p_coarse, p_final


# ----------------------------------------------------------------------
# 配套损失: 限制残差乱动 + 鼓励先验贴近GT动作典型位置
# ----------------------------------------------------------------------
def root_prior_losses(decoder, p_pred, p_gt, action_ids, lambda_res=0.1):
    """
    decoder: ActionPriorPoseDecoder (取 root_head.action_prior 做先验监督)
    p_pred/p_gt: (B,T,J,3); action_ids: (B,) long
    返回 (loss, dict)

    先验监督改为「动作×相位」: 让 action_prior[a, p] 贴近该动作在窗口相位 p 段的
    GT hip 均值。这样先验能承载 hip 在动作过程中的轨迹 (C2 的优势)。
    """
    rh = decoder.root_head
    B, T = p_gt.shape[0], p_gt.shape[1]
    P = rh.num_phases
    pidx = rh._phase_index(T, p_gt.device)                 # (T,)
    gt_hip = p_gt[:, :, 0, :]                              # (B,T,3)

    # 1) 先验监督: 对每个 (样本动作, 相位段) 求 GT hip 均值, 监督 action_prior[a,p]
    #    按相位段聚合 GT (B,P,3)
    gt_phase = torch.zeros(B, P, 3, device=p_gt.device)
    cnt = torch.zeros(B, P, 1, device=p_gt.device)
    gt_phase.index_add_(1, pidx, gt_hip)
    cnt.index_add_(1, pidx, torch.ones(B, T, 1, device=p_gt.device))
    gt_phase = gt_phase / cnt.clamp(min=1.0)               # (B,P,3)
    prior_ap = rh.action_prior[action_ids]                 # (B,P,3)
    l_prior = F.smooth_l1_loss(prior_ap, gt_phase.detach(), beta=0.1)

    # 2) 残差幅度惩罚: 限制 pred hip 偏离「该样本该相位的先验」(防乱跑)
    prior_full = prior_ap[:, pidx, :]                      # (B,T,3) 展开到每帧
    pred_hip = p_pred[:, :, 0, :]                          # (B,T,3)
    l_res = (pred_hip - prior_full.detach()).pow(2).sum(-1).mean()

    total = l_prior + lambda_res * l_res
    return total, {'l_prior': l_prior.item(), 'l_res': l_res.item()}


if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    torch.manual_seed(0)
    B, T, C, J, A = 4, 64, 128, 17, 27
    dec = ActionPriorPoseDecoder(in_dim=C, num_joints=J, num_actions=A)
    z = torch.randn(B, T, C, requires_grad=True)
    emb = torch.randn(B, 32)
    probs = torch.softmax(torch.randn(B, A), -1)
    pc, pf = dec(z, emb, probs)
    print(f"z{tuple(z.shape)} -> p_coarse{tuple(pc.shape)} p_final{tuple(pf.shape)}")
    assert pf.shape == (B, T, J, 3)
    gt = torch.randn(B, T, J, 3) * 0.3
    aid = torch.randint(0, A, (B,))
    l, d = root_prior_losses(dec, pf, gt, aid)
    (pf.sum() + l).backward()
    assert z.grad is not None and dec.root_head.action_prior.grad is not None
    print(f"root_prior_losses: {d}")
    print(f"params: {sum(p.numel() for p in dec.parameters()):,}")
    print("backward OK, grad -> z & action_prior")
    print("[OK]")