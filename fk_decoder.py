"""
fk_decoder.py (v2: root 速度积分模式)
=====================================
在原 Hybrid FK 基础上, 给 FK 支的 root 增加【速度积分】模式, 专打 hip 的【轨迹动态】
(可学、跨域不变的那一块), 不去赌【绝对锚点】(跨房间不可学的那一块)。

root_mode:
  'absolute' (默认, 原行为): root_t = root_head(z_t) 逐帧绝对回归。
  'velocity' (新):
        v_t   = tanh(vel_head(z_t)) * vel_scale        # 每帧位移, 限幅 (相对运动, 跨域不变)
        traj  = cumsum(v) ; traj -= mean_t(traj)       # 绕锚点波动, 去掉系统性长程漂移
        anchor= anchor_head(mean_t z)                  # 单一绝对锚 (B,3); 由退火 L_anchor 拉向 canonical
        root_t= anchor + traj
     => hip 既不被 anchor 摁成常数(有轨迹), 又不让网络去解不可学的绝对映射。
        anchor 是 hip 的均值, 训练器里的 L_anchor 监督的正是均值 hip -> 无需新接线。

接口不变: HybridFKPoseDecoder.forward(z_global, action_emb) -> (p_coarse, p_final)。
仅 __init__ 多两个可选参数 root_mode / vel_scale, 由 full_model 透传 (默认 absolute, 老实验不受影响)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
]


def forward_kinematics(root, bone_dir, bone_len, edges=EDGES, num_joints=17):
    joints = [None] * num_joints
    joints[0] = root
    for e, (p, c) in enumerate(edges):
        joints[c] = joints[p] + bone_dir[:, :, e, :] * bone_len[:, :, e, :]
    return torch.stack(joints, dim=2)


def decompose_to_fk(pose, edges=EDGES):
    root = pose[:, :, 0, :]
    dirs, lens = [], []
    for (p, c) in edges:
        v = pose[:, :, c, :] - pose[:, :, p, :]
        l = v.norm(dim=-1, keepdim=True)
        d = v / l.clamp_min(1e-8)
        dirs.append(d); lens.append(l)
    return root, torch.stack(dirs, 2), torch.stack(lens, 2)


class FKBranch(nn.Module):
    """z_global -> root + bone_dir(单位) + bone_len -> FK 姿态。
    root_mode: 'absolute' (逐帧绝对) | 'velocity' (锚点 + 速度积分轨迹)。"""
    def __init__(self, in_dim=128, edges=EDGES, num_joints=17,
                 hidden=256, len_min=0.02, len_max=0.8,
                 root_mode='absolute', vel_scale=0.12):
        super().__init__()
        assert root_mode in ('absolute', 'velocity'), root_mode
        self.edges = edges
        self.num_joints = num_joints
        self.num_bones = len(edges)
        self.len_min, self.len_max = len_min, len_max
        self.root_mode = root_mode
        self.vel_scale = vel_scale
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1))
        self.dir_head = nn.Linear(hidden, self.num_bones * 3)
        self.len_head = nn.Linear(hidden, self.num_bones)
        if root_mode == 'absolute':
            self.root_head = nn.Linear(hidden, 3)
        else:
            self.vel_head = nn.Linear(hidden, 3)      # 每帧位移
            self.anchor_head = nn.Linear(hidden, 3)   # 单一绝对锚 (对时间池化后)

    def _root(self, h):                                # h: (B,T,hidden) -> (B,T,3)
        if self.root_mode == 'absolute':
            return self.root_head(h)
        v = torch.tanh(self.vel_head(h)) * self.vel_scale        # (B,T,3) 限幅位移
        traj = torch.cumsum(v, dim=1)                            # 积分成轨迹
        traj = traj - traj.mean(dim=1, keepdim=True)             # 去时间均值 -> 绕锚波动, 不整体漂
        anchor = self.anchor_head(h.mean(dim=1))                 # (B,3) 绝对锚 (池化, 不逐帧赌)
        return anchor[:, None, :] + traj                         # (B,T,3)

    def forward(self, z):                              # z: (B,T,C)
        B, T, _ = z.shape
        h = self.trunk(z)
        root = self._root(h)                           # (B,T,3)
        d = F.normalize(self.dir_head(h).reshape(B, T, self.num_bones, 3), dim=-1)
        l = F.softplus(self.len_head(h)).reshape(B, T, self.num_bones, 1)
        l = l.clamp(self.len_min, self.len_max)
        return forward_kinematics(root, d, l, self.edges, self.num_joints)


class HybridFKPoseDecoder(nn.Module):
    """结构支(现有 PoseDecoder) + FK 支, alpha 融合。接口同 PoseDecoder。
    新增 root_mode / vel_scale 透传给 FK 支 (默认 absolute = 原行为)。"""
    def __init__(self, *args, in_dim=128, root_mode='absolute', vel_scale=0.12, **kwargs):
        super().__init__()
        from models.pose_decoder import PoseDecoder
        self.base = PoseDecoder(*args, in_dim=in_dim, **kwargs)
        self.fk = FKBranch(in_dim=in_dim, root_mode=root_mode, vel_scale=vel_scale)
        self.register_buffer('alpha', torch.tensor(1.0))

    def set_alpha(self, a):
        self.alpha.fill_(float(a))

    def forward(self, z_global, action_emb):
        p_coarse, p_struct = self.base(z_global, action_emb)
        p_fk = self.fk(z_global)
        a = self.alpha
        p_final = a * p_struct + (1.0 - a) * p_fk
        return p_coarse, p_final


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, J = 2, 64, 17

    # FK 可逆性 (不变)
    gt = torch.randn(B, T, J, 3)
    root, d, l = decompose_to_fk(gt)
    assert (forward_kinematics(root, d, l) - gt).abs().max() < 1e-4
    print("[FK 可逆性] OK")

    # absolute 模式: 原行为, 前向/反向
    fk_abs = FKBranch(in_dim=128, root_mode='absolute')
    z = torch.randn(B, T, 128, requires_grad=True)
    pose = fk_abs(z); pose.sum().backward()
    assert pose.shape == (B, T, J, 3) and z.grad is not None
    print("[absolute] 前向/反向 OK")

    # velocity 模式
    fk_vel = FKBranch(in_dim=128, root_mode="velocity", vel_scale=0.12).eval()
    z2 = torch.randn(B, T, 128, requires_grad=True)
    pose2 = fk_vel(z2)
    assert pose2.shape == (B, T, J, 3)
    root2 = pose2[:, :, 0, :]                          # (B,T,3) hip 轨迹
    # 性质1: 轨迹去均值后绕锚波动 -> root 的时间均值 == anchor (traj 零均值)
    h = fk_vel.trunk(z2); anchor = fk_vel.anchor_head(h.mean(1))
    err_anchor = (root2.mean(1) - anchor).abs().max().item()
    print(f"[velocity] root 时间均值 == anchor? 最大差 {err_anchor:.2e} (应~0 -> L_anchor 监督的就是 anchor)")
    assert err_anchor < 1e-5
    # 性质2: hip 真的在动 (不是常数) -> 帧间位移非零
    motion = (root2[:, 1:] - root2[:, :-1]).abs().mean().item()
    print(f"[velocity] hip 帧间平均位移 = {motion:.4f} (>0: 有轨迹, 不退化成静止点)")
    assert motion > 1e-4
    # 性质3: 每帧位移受限 (<= vel_scale)
    vmax = (root2[:, 1:] - root2[:, :-1]).abs().max().item()
    print(f"[velocity] 帧间位移上界 ~ {vmax:.3f} (vel_scale={fk_vel.vel_scale})")
    pose2.sum().backward()
    assert z2.grad is not None and z2.grad.abs().sum() > 0
    print("[velocity] 前向/反向 OK, 梯度回流")
    print("\n[ALL OK]")