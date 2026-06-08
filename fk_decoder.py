"""
fk_decoder.py
=============
Hybrid FK 姿态解码器: 在不替换现有 PoseDecoder 的前提下, 外挂一条
Forward-Kinematics 分支并按 alpha 融合, 专门压低 PA-MPJPE。

设计 (对应优化报告 7.5 Hybrid 形式):
  结构支 (self.base) = 你现有的 PoseDecoder (TaskPromptCoarseHead + SkeletonRefiner)
                       —— 字节级复用, 继承当前 104.73 的强 baseline。
  FK 支   (self.fk)  = z_global -> root + bone_length + bone_direction(单位向量)
                       -> 正运动学合成 -> 姿态。骨架合法性是【构造保证】的。
  融合     p_final = alpha * p_struct + (1-alpha) * p_fk
                     alpha 由训练器逐 epoch 从 1.0 退火到 ~0.4 (注册为 buffer, 随 ckpt 保存,
                     评测时自动用最终 alpha)。

对外接口与 PoseDecoder 完全一致:
    forward(z_global, action_emb) -> (p_coarse, p_final)
因此在 models/__init__.py 里把 self.pose_decoder = PoseDecoder(...) 换成
HybridFKPoseDecoder(...) (参数原样传) 即可, 其余代码一行不改。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# MMFi 17 关节骨架边 (parent, child), joint 0 = hip/root。
# 顺序保证: 每条边的 parent 在更早的边里已作为 child 出现过 -> 可按列表顺序 FK。
EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),       # 两条腿
    (0, 7), (7, 8), (8, 9), (9, 10),                       # 脊柱->头
    (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),  # 两条臂
]


def forward_kinematics(root, bone_dir, bone_len, edges=EDGES, num_joints=17):
    """由 root + 每条骨的方向×长度, 沿运动学树合成绝对关节坐标。
      root     : (B,T,3)
      bone_dir : (B,T,E,3) 单位向量
      bone_len : (B,T,E,1) 正长度 (米)
      return   : (B,T,J,3)
    """
    joints = [None] * num_joints
    joints[0] = root
    for e, (p, c) in enumerate(edges):
        joints[c] = joints[p] + bone_dir[:, :, e, :] * bone_len[:, :, e, :]
    return torch.stack(joints, dim=2)


def decompose_to_fk(pose, edges=EDGES):
    """把绝对姿态拆成 (root, bone_dir, bone_len), 仅用于自检/初始化参考。"""
    root = pose[:, :, 0, :]
    dirs, lens = [], []
    for (p, c) in edges:
        v = pose[:, :, c, :] - pose[:, :, p, :]      # (B,T,3)
        l = v.norm(dim=-1, keepdim=True)             # (B,T,1)
        d = v / l.clamp_min(1e-8)
        dirs.append(d); lens.append(l)
    return root, torch.stack(dirs, 2), torch.stack(lens, 2)


class FKBranch(nn.Module):
    """z_global -> root + bone_dir(单位) + bone_len(softplus, clamp) -> FK 姿态。"""
    def __init__(self, in_dim=128, edges=EDGES, num_joints=17,
                 hidden=256, len_min=0.02, len_max=0.8):
        super().__init__()
        self.edges = edges
        self.num_joints = num_joints
        self.num_bones = len(edges)
        self.len_min, self.len_max = len_min, len_max
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1))
        self.root_head = nn.Linear(hidden, 3)
        self.dir_head = nn.Linear(hidden, self.num_bones * 3)
        self.len_head = nn.Linear(hidden, self.num_bones)

    def forward(self, z):                              # z: (B,T,C)
        B, T, _ = z.shape
        h = self.trunk(z)
        root = self.root_head(h)                       # (B,T,3)
        d = self.dir_head(h).reshape(B, T, self.num_bones, 3)
        d = F.normalize(d, dim=-1)                     # 单位骨向
        l = F.softplus(self.len_head(h)).reshape(B, T, self.num_bones, 1)
        l = l.clamp(self.len_min, self.len_max)        # 正、有界骨长
        return forward_kinematics(root, d, l, self.edges, self.num_joints)


class HybridFKPoseDecoder(nn.Module):
    """结构支(现有 PoseDecoder) + FK 支, alpha 融合。接口同 PoseDecoder。"""
    def __init__(self, *args, in_dim=128, **kwargs):
        super().__init__()
        from models.pose_decoder import PoseDecoder   # 复用你现有结构解码器, 字节级一致
        self.base = PoseDecoder(*args, in_dim=in_dim, **kwargs)
        self.fk = FKBranch(in_dim=in_dim)
        # alpha=1.0 时纯结构支(等价旧模型); 训练器逐 epoch 退火。buffer 随 ckpt 保存。
        self.register_buffer('alpha', torch.tensor(1.0))

    def set_alpha(self, a):
        self.alpha.fill_(float(a))

    def forward(self, z_global, action_emb):
        p_coarse, p_struct = self.base(z_global, action_emb)
        p_fk = self.fk(z_global)
        a = self.alpha
        p_final = a * p_struct + (1.0 - a) * p_fk
        return p_coarse, p_final


# ----------------------------------------------------------------------
# Sandbox (FK 数学 + 分支前向/反向; HybridFK 因依赖 models.* 不在此测)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, J = 2, 8, 17

    # (1) 正运动学可逆性: GT -> 拆解 -> FK 重建, 应数值相等
    gt = torch.randn(B, T, J, 3)
    root, d, l = decompose_to_fk(gt)
    recon = forward_kinematics(root, d, l)
    err = (recon - gt).abs().max().item()
    print(f"[FK 可逆性] 重建最大误差 = {err:.2e} (应 ~0)")
    assert err < 1e-4, "FK 树结构/数学有误"

    # (2) FKBranch 前向 + 反向 + 骨长合法
    fk = FKBranch(in_dim=128)
    z = torch.randn(B, T, 128, requires_grad=True)
    pose = fk(z)
    print(f"[FKBranch] z{tuple(z.shape)} -> pose{tuple(pose.shape)}")
    assert pose.shape == (B, T, J, 3)
    # 检查输出骨长落在 [len_min, len_max]
    _, _, ll = decompose_to_fk(pose)
    print(f"[FKBranch] 骨长范围 = [{ll.min().item():.3f}, {ll.max().item():.3f}] (应 ⊂ [0.02,0.8])")
    pose.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    print("[FKBranch] backward OK, 梯度回流 z_global")
    print("\n[ALL OK]")