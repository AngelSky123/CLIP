"""
structural_losses.py
====================
针对 PA-MPJPE (相对骨架形状) 的结构正则损失。

设计原则
--------
* 三个损失全部作用在【相对骨架】上, 不涉及 hip 全局 xyz。
  => 结构上不可能让 MPJPE 的全局定位主项变差; 最坏 PA/MPJPE 原地不动。
* 结构信息已被验证可跨域迁移 (E04 hip-aligned 误差 ≈ 室内), 所以源域学到的
  骨长/对称先验能迁到 E04 -> 有望真实压低 E04 的 PA-MPJPE。

三个损失
--------
1. bone_length_loss(pred, gt) : 预测骨长 vs GT 骨长 (L1)。需 GT, 用于源域。
2. symmetry_loss(pred)        : 左右同名骨等长 (L1)。无需 GT, 处处可用。
3. temporal_bone_loss(pred)   : 同一窗口内骨长跨帧恒定。无需 GT, 处处可用。

MMFi 17 关节骨架 (joint 0 = hip/root), 边与对称对均按你 losses.py 里的 EDGES 推出。
"""
import torch
import torch.nn.functional as F

# MMFi 17-joint 骨架边 (与项目 losses.py 一致)
EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),       # 两条腿
    (0, 7), (7, 8), (8, 9), (9, 10),                       # 脊柱->头
    (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),  # 两条臂
]

# 左右同名骨对 (镜像, 应等长) —— 由 EDGES 的两侧链推出
SYM_BONE_PAIRS = [
    ((0, 1), (0, 4)), ((1, 2), (4, 5)), ((2, 3), (5, 6)),        # 髋/大腿/小腿
    ((8, 11), (8, 14)), ((11, 12), (14, 15)), ((12, 13), (15, 16)),  # 肩/上臂/前臂
]

_EI = [e[0] for e in EDGES]
_EJ = [e[1] for e in EDGES]


def _bone_lengths(pose):
    """pose: (..., J, 3) -> (..., E) 每条骨的欧氏长度。"""
    return torch.norm(pose[..., _EI, :] - pose[..., _EJ, :], dim=-1)


def bone_length_loss(pred, gt):
    """预测骨长对齐 GT 骨长 (L1)。pred/gt: (..., J, 3)。"""
    return F.l1_loss(_bone_lengths(pred), _bone_lengths(gt))


def symmetry_loss(pred):
    """左右同名骨等长。pred: (..., J, 3)。无需 GT。"""
    total = pred.new_zeros(())
    for a, b in SYM_BONE_PAIRS:
        la = torch.norm(pred[..., a[0], :] - pred[..., a[1], :], dim=-1)
        lb = torch.norm(pred[..., b[0], :] - pred[..., b[1], :], dim=-1)
        total = total + F.l1_loss(la, lb)
    return total / len(SYM_BONE_PAIRS)


def temporal_bone_loss(pred):
    """同窗口内骨长跨帧恒定。pred: (B, T, J, 3)。无需 GT。"""
    if pred.dim() != 4 or pred.shape[1] < 2:
        return pred.new_zeros(())
    bl = _bone_lengths(pred)                 # (B, T, E)
    return (bl[:, 1:] - bl[:, :-1]).abs().mean()


def root_relative_loss(pred, gt):
    """髋(joint0)中心化后的关节位置 L1。直接对齐相对骨架 = PA 量的那个量。
    对 pred/gt 各自减自身 hip, 所以对全局 hip 平移不变, 碰不到定位主项。"""
    pr = pred - pred[..., :1, :]
    gr = gt - gt[..., :1, :]
    return F.l1_loss(pr, gr)


def structural_loss(pred, gt=None, w_bone=1.0, w_sym=0.1, w_temp=0.1, w_rel=3.0):
    """
    组合结构正则。
      pred : (B, T, J, 3) —— 用和 L_pose 同一个 (clean) 预测姿态。
      gt   : (B, T, J, 3) 或 None —— 源域有 GT 时传入; 目标域可只用对称/时序两项。
    返回 (total_scalar, comp_dict)。
    """
    comp = {}
    total = pred.new_zeros(())
    if gt is not None:
        lb = bone_length_loss(pred, gt)
        lr = root_relative_loss(pred, gt)
        comp['bone'] = float(lb.detach())
        comp['rel'] = float(lr.detach())
        total = total + w_bone * lb + w_rel * lr
    ls = symmetry_loss(pred)
    lt = temporal_bone_loss(pred)
    comp['sym'] = float(ls.detach())
    comp['temp'] = float(lt.detach())
    total = total + w_sym * ls + w_temp * lt
    return total, comp


# ----------------------------------------------------------------------
# Sandbox
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, J = 2, 64, 17

    # 完美预测 -> bone/sym 接近 0 的合理性检查
    gt = torch.randn(B, T, J, 3, requires_grad=False)
    pred = gt.clone().requires_grad_(True)
    tot, comp = structural_loss(pred, gt)
    print("[identical pred=gt] total=%.6f  comp=%s" % (float(tot.detach()), comp))
    assert comp['bone'] < 1e-5, "完美预测骨长损失应≈0"
    tot.backward()
    assert pred.grad is not None
    print("  grad flows: OK")

    # 随机预测 -> 三项都 > 0
    pred2 = torch.randn(B, T, J, 3, requires_grad=True)
    tot2, comp2 = structural_loss(pred2, gt)
    print("[random pred] total=%.4f  comp=%s" % (float(tot2.detach()), comp2))
    assert comp2['bone'] > 0 and comp2['sym'] > 0 and comp2['temp'] > 0

    # 对称对正确性: 构造左右严格对称的骨架 -> sym≈0
    sym_pose = torch.randn(B, T, J, 3)
    # 把右侧关节镜像成左侧 (x 取反) 以制造对称
    mirror = {1: 4, 2: 5, 3: 6, 11: 14, 12: 15, 13: 16}
    for r, l in mirror.items():
        sym_pose[..., l, :] = sym_pose[..., r, :] * torch.tensor([-1., 1., 1.])
    # 注: 镜像后同名骨长度严格相等
    print("[mirrored pose] sym=%.6f (应≈0)" % float(symmetry_loss(sym_pose)))

    # GT 缺失 (目标域) -> 只算 sym+temp
    tot3, comp3 = structural_loss(pred2, gt=None)
    print("[no gt] total=%.4f  comp=%s (无 bone 项)" % (float(tot3.detach()), comp3))
    assert 'bone' not in comp3
    print("\n[ALL OK]")


# ======================================================================
# Root anchor (诚实救 MPJPE): 把预测 hip 往【按动作的源域 canonical 位置】拉。
# 动机: 实测模型 E04 hip_err(335) > 预测均值的零信息基线(324), 说明学出的 root
#       比常数还差 (源域过拟合+漂移)。往稳健的源域先验拉 = 不靠偷看 E04、不对齐真值,
#       只是"别让 root 比常数更烂", 可诚实回收 ~10mm MPJPE。不追 316.8 (信息上界)。
# ======================================================================
import os as _os
import glob as _glob
import numpy as _np

_ENV_SUBJECTS = {'E01': range(1, 11), 'E02': range(11, 21),
                 'E03': range(21, 31), 'E04': range(31, 41)}


def build_action_canonical(data_root, train_envs, num_actions=27, hip_joint=0):
    """扫训练集 GT (只读 ground_truth.npy, 不碰 CSI, 很快), 算每个动作的源域平均 hip。
    返回 (num_actions, 3) 的 float tensor; 缺失动作回退为全局平均 hip。
    """
    sums = _np.zeros((num_actions, 3), dtype=_np.float64)
    cnts = _np.zeros((num_actions,), dtype=_np.int64)
    for env in train_envs:
        for sid in _ENV_SUBJECTS[env]:
            subj = f'S{sid:02d}'
            for aid in range(1, num_actions + 1):
                gt_path = _os.path.join(data_root, env, subj, f'A{aid:02d}', 'ground_truth.npy')
                if not _os.path.exists(gt_path):
                    continue
                gt = _np.load(gt_path).astype(_np.float64)   # (F,17,3)
                sums[aid - 1] += gt[:, hip_joint, :].sum(axis=0)
                cnts[aid - 1] += gt.shape[0]
    glob_mean = sums.sum(0) / max(cnts.sum(), 1)
    canon = _np.zeros((num_actions, 3), dtype=_np.float32)
    for a in range(num_actions):
        canon[a] = (sums[a] / cnts[a]) if cnts[a] > 0 else glob_mean
    return torch.from_numpy(canon)


def root_anchor_loss(pred, action_labels, canonical):
    """把 pred 的 hip(joint0) 往按动作源域 canonical 位置拉 (Smooth-L1)。
      pred          : (B,T,J,3)
      action_labels : (B,) long, 取值 0..num_actions-1
      canonical     : (num_actions,3) tensor (与 pred 同 device)
    """
    pred_hip = pred[:, :, 0, :]                         # (B,T,3)
    target = canonical.to(pred.device)[action_labels]  # (B,3)
    target = target[:, None, :].expand_as(pred_hip)    # (B,T,3)
    return F.smooth_l1_loss(pred_hip, target, beta=0.05)