"""
fk_decoder.py (v3: 轴向分离 root 模式 axis_split)
=================================================
在 v2 (absolute / velocity) 基础上, 新增 root_mode='axis_split', 由缝合诊断的
逐轴 oracle 结论驱动:

  诊断事实 (E04, stitch_axis_probe / z_axis_decompose):
    * x 轴 root 误差 = 大常数 bias(-237) + 高方差 -> 站位平移, 跨域不可观测,
      但【会动】比【锚死】好 (S5: x 用会动的预测, 源域 hip 一致下降 14~40mm)。
    * y/z 轴误差主体是【序列间系统常数】, 锚定到源域先验比让它乱跑稳。
  => 结论: root 的 x 轴该【敢动】(velocity 积分轨迹, 不被 anchor 摁死),
           y/z 轴该【锚定】(absolute + L_anchor 拉向源域 canonical)。

root_mode:
  'absolute' (原): 三轴都逐帧绝对回归。
  'velocity' (v2): 三轴都 锚 + 去均值速度积分。
  'axis_split' (新):
        x 轴: tanh(vel)*vel_scale 逐帧位移 -> cumsum 轨迹 (不去均值, 让 x 自由漂/动);
              x 的绝对位置由 pose loss 自己学, 【不】被 L_anchor 约束 (trainer 侧只锚 y/z)。
        y/z 轴: absolute 逐帧回归 (root_head_yz), 由 L_anchor 拉向源域 canonical。
     => 单模型一次前向就复现 "S5: x敢动 + y/z锚定" 的行为, 无需缝两个模型。

接口不变: HybridFKPoseDecoder.forward(z_global, action_emb) -> (p_coarse, p_final)。
__init__ 多 root_mode/vel_scale (透传给 FK 支), 默认 absolute = 原行为, 老实验不受影响。

注: axis_split 下 L_anchor 必须改成【只约束 y/z】(见 train_distill_pretrained.py 改动),
    否则 x 轴又被锚死, 退化回主线 std=4.5 的摁死状态。
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
    root_mode: 'absolute' | 'velocity' | 'axis_split'。"""
    def __init__(self, in_dim=128, edges=EDGES, num_joints=17,
                 hidden=256, len_min=0.02, len_max=0.8,
                 root_mode='absolute', vel_scale=0.12):
        super().__init__()
        assert root_mode in ('absolute', 'velocity', 'axis_split'), root_mode
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
        elif root_mode == 'velocity':
            self.vel_head = nn.Linear(hidden, 3)      # 每帧位移
            self.anchor_head = nn.Linear(hidden, 3)   # 单一绝对锚 (对时间池化后)
        else:  # axis_split
            # x 轴: 敢动 (velocity 积分, 不去均值 -> 允许自由漂移/移动)
            self.vel_head_x = nn.Linear(hidden, 1)
            # y/z 轴: 锚定 (逐帧 absolute, 由 trainer 的 L_anchor 拉向源域 canonical)
            self.root_head_yz = nn.Linear(hidden, 2)

    def _root(self, h):                                # h: (B,T,hidden) -> (B,T,3)
        if self.root_mode == 'absolute':
            return self.root_head(h)
        if self.root_mode == 'velocity':
            v = torch.tanh(self.vel_head(h)) * self.vel_scale        # (B,T,3) 限幅位移
            traj = torch.cumsum(v, dim=1)                            # 积分成轨迹
            traj = traj - traj.mean(dim=1, keepdim=True)             # 去时间均值 -> 绕锚波动
            anchor = self.anchor_head(h.mean(dim=1))                 # (B,3) 绝对锚
            return anchor[:, None, :] + traj                         # (B,T,3)
        # ---- axis_split ----
        B, T, _ = h.shape
        # x: tanh 限幅每帧位移 -> cumsum 轨迹 (不去均值, 让网络自由学 x 的移动/漂移)
        vx = torch.tanh(self.vel_head_x(h)) * self.vel_scale          # (B,T,1)
        x = torch.cumsum(vx, dim=1)                                   # (B,T,1) x 轨迹
        # y/z: 逐帧 absolute (会被 L_anchor 锚定)
        yz = self.root_head_yz(h)                                     # (B,T,2)
        root = torch.cat([x, yz], dim=-1)                            # (B,T,3) = [x, y, z]
        return root

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

    # absolute 模式: 原行为
    fk_abs = FKBranch(in_dim=128, root_mode='absolute')
    z = torch.randn(B, T, 128, requires_grad=True)
    pose = fk_abs(z); pose.sum().backward()
    assert pose.shape == (B, T, J, 3) and z.grad is not None
    print("[absolute] 前向/反向 OK")

    # velocity 模式
    fk_vel = FKBranch(in_dim=128, root_mode="velocity", vel_scale=0.12).eval()
    z2 = torch.randn(B, T, 128, requires_grad=True)
    pose2 = fk_vel(z2)
    h = fk_vel.trunk(z2); anchor = fk_vel.anchor_head(h.mean(1))
    err_anchor = (pose2[:, :, 0, :].mean(1) - anchor).abs().max().item()
    assert err_anchor < 1e-5
    print("[velocity] root 时间均值 == anchor OK")

    # axis_split 模式 (新)
    fk_ax = FKBranch(in_dim=128, root_mode="axis_split", vel_scale=0.12)
    z3 = torch.randn(B, T, 128, requires_grad=True)
    pose3 = fk_ax(z3)
    assert pose3.shape == (B, T, J, 3)
    root3 = pose3[:, :, 0, :]                              # (B,T,3) hip 轨迹
    # 性质1: x 轴在动 (velocity 积分, 帧间位移非零)
    x_motion = (root3[:, 1:, 0] - root3[:, :-1, 0]).abs().mean().item()
    print(f"[axis_split] x 轴帧间位移 = {x_motion:.4f} (>0: x 敢动)")
    assert x_motion > 1e-5
    # 性质2: x 轴每帧位移受限 (<= vel_scale)
    x_step_max = (root3[:, 1:, 0] - root3[:, :-1, 0]).abs().max().item()
    print(f"[axis_split] x 轴帧间位移上界 ~ {x_step_max:.3f} (vel_scale={fk_ax.vel_scale})")
    assert x_step_max <= fk_ax.vel_scale + 1e-4
    # 性质3: y/z 是 absolute 逐帧 (无 cumsum 约束, 可任意)
    pose3.sum().backward()
    assert z3.grad is not None and z3.grad.abs().sum() > 0
    # 性质4: x head 与 yz head 分离
    assert hasattr(fk_ax, 'vel_head_x') and hasattr(fk_ax, 'root_head_yz')
    assert not hasattr(fk_ax, 'root_head')      # 没有三轴合一的 root_head
    print("[axis_split] x=velocity / y,z=absolute 分离 OK, 前向/反向 OK")
    print("\n[ALL OK]")