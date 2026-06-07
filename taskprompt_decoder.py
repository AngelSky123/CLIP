"""
taskprompt_decoder.py — 给 pose decoder 加 DT-Pose 式 task prompt + uniformity 正则。

两个独立的、低成本的改进，都接在【现有蒸馏训练】里, 不需要重做预训练:

1) TaskPromptCoarseHead —— 替换原 CoarsePoseHead
   原 head: MLP 把 (B,T,C) -> (B,T,17*3), 所有关节共享权重、无 per-joint 先验。
   新 head: 引入可学习 task prompt (J, d), 给每个关节一份独立的位置先验。
   DT-Pose Table 6: task prompt 把 MPJPE 197->174 (PA 几乎不动) ——
   它改善的正是 absolute localization, 因为 prompt 编码了关节的全局位置先验。
   这正对你「结构(MPJPE_a)不差、但绝对定位差」的症结。

2) uniformity_loss —— 加到 z_global 上的反塌陷正则 (DT-Pose Eq.3)
   惩罚 batch 内特征两两余弦相似度的平方, 强制表征「铺开」、不塌缩到一点。
   对治你日志里反复出现的 PredStd 偏低 / E04 退化成均值姿态 (dimensional collapse)。

接入方式见本文件末尾的「接入说明」。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1) Task-prompt coarse head (替换 models.pose_decoder.CoarsePoseHead)
# ----------------------------------------------------------------------
class TaskPromptCoarseHead(nn.Module):
    """动作条件化 + task prompt 的粗姿态头。

    输入: z_global (B,T,C_g) + action_emb (B,D_a)
    输出: P_coarse (B,T,J,3)

    与原 CoarsePoseHead 的差异:
      原版把 (z_global ⊕ action) 直接 MLP 成 J*3, 一次性吐所有关节坐标。
      本版先把条件特征投到 d 维, repeat 成 J 份, 每份【加上该关节专属的
      可学习 prompt】, 再用一个【所有关节共享的】小 MLP 回归各自的 xyz。
      => 每个关节带有独立的位置先验 (prompt), 但回归器共享 (参数高效)。
    """
    def __init__(self, in_dim=128, hidden_dim=256, num_joints=17,
                 action_embed_dim=32, prompt_dim=128):
        super().__init__()
        self.num_joints = num_joints
        self.prompt_dim = prompt_dim

        # 条件特征 (z_global ⊕ action) -> prompt_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(in_dim + action_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, prompt_dim),
        )
        # 可学习 task prompt: 每个关节一份 (J, prompt_dim)
        self.task_prompt = nn.Parameter(torch.zeros(num_joints, prompt_dim))
        nn.init.trunc_normal_(self.task_prompt, std=0.02)

        # 共享的 per-joint 回归器: prompt_dim -> 3
        self.joint_regressor = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
            nn.LayerNorm(prompt_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(prompt_dim, 3),
        )

    def forward(self, z_global, action_emb):
        """
        z_global:   (B, T, C_g)
        action_emb: (B, D_a)
        return:     (B, T, J, 3)
        """
        B, T, _ = z_global.shape
        act = action_emb.unsqueeze(1).expand(-1, T, -1)        # (B,T,D_a)
        cond = torch.cat([z_global, act], dim=-1)               # (B,T,C_g+D_a)
        feat = self.cond_proj(cond)                             # (B,T,prompt_dim)

        # repeat 成 J 份, 加 per-joint prompt
        feat = feat.unsqueeze(2).expand(-1, -1, self.num_joints, -1)  # (B,T,J,d)
        feat = feat + self.task_prompt.view(1, 1, self.num_joints, -1)  # 广播加 prompt

        out = self.joint_regressor(feat)                        # (B,T,J,3)
        return out


# ----------------------------------------------------------------------
# 2) Uniformity 正则 (DT-Pose Eq.3) —— 反 dimensional collapse
# ----------------------------------------------------------------------
def uniformity_loss(z, eps=1e-8):
    """惩罚 batch 内特征两两余弦相似度的平方, 强制表征铺开。

    Args:
        z: (B, T, C) 或 (B, C)。若是 (B,T,C), 先对 T 池化成 (B,C)。
    Returns:
        标量损失。B<2 时返回 0。

    与 DT-Pose Eq.3 一致:  L_unif = mean_{i≠j} (ê_i · ê_j)^2,
    其中 ê 是 L2 归一化后的特征。值越小表示特征越「铺开」、越不塌缩。
    """
    if z.dim() == 3:
        z = z.mean(dim=1)                      # (B,C), 对时间池化
    B = z.shape[0]
    if B < 2:
        return torch.zeros((), device=z.device)
    z = F.normalize(z, dim=-1, eps=eps)        # 单位球面
    sim = z @ z.t()                            # (B,B) 余弦相似度
    mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    return (sim[mask] ** 2).mean()


# ----------------------------------------------------------------------
# Sandbox
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    torch.manual_seed(0)
    B, T, C, J, Da = 4, 64, 128, 17, 32

    print("=" * 60); print("TaskPromptCoarseHead"); print("=" * 60)
    head = TaskPromptCoarseHead(in_dim=C, hidden_dim=256, num_joints=J,
                                action_embed_dim=Da, prompt_dim=128)
    z = torch.randn(B, T, C, requires_grad=True)
    act = torch.randn(B, Da)
    out = head(z, act)
    print(f"z{tuple(z.shape)} act{tuple(act.shape)} -> P_coarse{tuple(out.shape)}")
    assert out.shape == (B, T, J, 3)
    out.sum().backward()
    assert z.grad is not None and head.task_prompt.grad is not None
    print(f"params: {sum(p.numel() for p in head.parameters()):,}")
    print(f"task_prompt shape: {tuple(head.task_prompt.shape)}, grad flows: OK")

    print(); print("=" * 60); print("uniformity_loss"); print("=" * 60)
    # 塌缩特征 (全相同) -> loss 接近 1; 随机特征 -> loss 小
    z_collapse = torch.ones(B, C)
    z_random = torch.randn(B, C)
    print(f"collapsed feats -> unif = {uniformity_loss(z_collapse).item():.4f} (应接近 1)")
    print(f"random feats    -> unif = {uniformity_loss(z_random).item():.4f} (应较小)")
    z3 = torch.randn(B, T, C)
    print(f"(B,T,C) input   -> unif = {uniformity_loss(z3).item():.4f} (自动对T池化)")
    print("\n[ALL OK]")


# ======================================================================
# 接入说明 (改动小, 不碰预训练)
# ======================================================================
#
# --- A. 启用 task prompt: 改 models/full_model.py 里 decoder 的构建 ---
#
# 你现在的 pose_decoder 内部用的是 pose_decoder.PoseDecoder -> CoarsePoseHead。
# 最干净的接法: 在 models/pose_decoder.py 的 PoseDecoder.__init__ 里, 把
#     self.coarse_head = CoarsePoseHead(in_dim, hidden_dim, num_joints, action_embed_dim)
# 换成:
#     from taskprompt_decoder import TaskPromptCoarseHead
#     self.coarse_head = TaskPromptCoarseHead(in_dim, hidden_dim, num_joints, action_embed_dim)
# 其余 (SkeletonRefiner / forward) 完全不动, 接口一致。
#
# 注意: 这会改变 pose_decoder 的权重结构, 从 action_best.pt 加载 backbone 时
# pose_decoder 本来就不在加载列表 (load_pretrained_backbone 只加载 backbone+
# action_classifier), 所以不冲突, decoder 从头训练。
#
# (如果你上一轮已经换成了 RootDecoupledPoseDecoder, 二选一: 要么回退到原
#  PoseDecoder 再加 task prompt, 要么把 RootDecoupled 里的 CoarsePoseHead
#  也换成 TaskPromptCoarseHead。建议先单独测 task prompt, 不要和 root 解耦叠加,
#  否则分不清是哪个起的作用。)
#
# --- B. 启用 uniformity: 在 train_distill_pretrained.py 的 train_one_epoch 里 ---
#
# 在算完 base_loss 之后、加蒸馏项的地方, 加一行:
#     from taskprompt_decoder import uniformity_loss
#     l_unif = uniformity_loss(outputs['z_global'])
#     total = total + args.lambda_unif * l_unif
# 并在 get_args() 里加:
#     p.add_argument('--lambda_unif', type=float, default=0.05)
# 起步 0.05; 若 PredStd 仍低可加到 0.1, 若 PA 变差则降回 0.02。
#
# --- C. 重训命令 (相比上次, 回退 root 解耦相关、加 task prompt + uniformity) ---
#
# python train_distill_pretrained.py \
#     --data_root /home/a123456/PerceptAlign/MMFi \
#     --train_envs E01 E02 E03 --test_env E04 \
#     --pretrain_ckpt checkpoints/stage1b_action/action_best.pt \
#     --teacher_ckpt  checkpoints/depth_teacher_full/teacher_best.pt \
#     --depth_img 112 --depth_clip 5000 \
#     --lambda_feat 0.1 --lambda_out 0.5 \
#     --out_distill_hip_weight 1.0 --lambda_hip 1.0 \
#     --gamma 0.0 --lambda_unif 0.05 \
#     --val_ratio 0.15 \
#     --epochs 50 --batch_size 2 --accumulate_grad 8 \
#     --lr_backbone 1e-4 --lr_head 5e-4 \
#     --use_ema --ema_decay 0.999 \
#     --save_dir ./checkpoints/distill_taskprompt
#
# --- D. 最终评测: 仍用逐帧口径 ---
# python eval_dtpose_faithful.py --ckpt .../best_mpjpe_ema.pth --test_env E04 --variance