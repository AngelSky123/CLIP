"""
Distillation losses for depth/RGB -> CSI cross-modal distillation.

本版唯一改动: OutputDistillLoss 增加 align_hip 开关 (默认 False, depth 行为不变)。
  align_hip=True (给 RGB 教师): 学生/教师各自减 hip 后再算, 只蒸馏【相对结构】,
  不把单目 RGB 教师那条不可信的绝对 hip (263->319mm) 灌给学生; hip 由 L_pose+L_anchor 负责。
其余 (FeatureDistillLoss / KinematicPriorLoss / sandbox) 一字未动。
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1) Feature-level distillation (z_global)
# ----------------------------------------------------------------------
class DistillProjection(nn.Module):
    """Student-side projection head: z_student (B,T,C) -> (B,T,C)."""
    def __init__(self, in_dim=128, out_dim=128, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class FeatureDistillLoss(nn.Module):
    """Cosine + smooth-L1 alignment of projected student feature to teacher."""
    def __init__(self, cosine_weight=1.0, smoothl1_weight=1.0):
        super().__init__()
        self.cw = cosine_weight
        self.sw = smoothl1_weight

    def forward(self, z_student_proj, z_teacher):
        cos = F.cosine_similarity(z_student_proj, z_teacher, dim=-1)   # (B,T)
        l_cos = (1.0 - cos).mean()
        l_sl1 = F.smooth_l1_loss(z_student_proj, z_teacher)
        total = self.cw * l_cos + self.sw * l_sl1
        return total, {
            'l_distill_feat':     total.item(),
            'l_distill_feat_cos': l_cos.item(),
            'l_distill_feat_sl1': l_sl1.item(),
        }


# ----------------------------------------------------------------------
# 2) Output-level (pose) distillation
# ----------------------------------------------------------------------
class OutputDistillLoss(nn.Module):
    """Smooth-L1 alignment of student pose to teacher pose.

    align_hip=False (默认; depth 教师用, 原行为不变):
        直接对齐绝对关节坐标, joint 0 (hip) 可用 hip_weight 加权。
    align_hip=True (RGB 教师用; 解耦结构蒸馏):
        学生/教师各自减 hip(joint0, 参考点 detach) 后再算 Smooth-L1, 只蒸馏【相对结构】。
        => 教师那条不可信的绝对 hip 不会传给学生; hip 关节(对齐后恒0)从均值里排除,
           hip 的绝对定位由 L_pose 坐标项 + L_anchor 负责, 与本项解耦。
        此时 hip_weight 不起作用 (hip 已被排除)。
    """
    def __init__(self, beta=0.05, hip_weight=1.5, num_joints=17, hip_joint_idx=0,
                 align_hip=False):
        super().__init__()
        self.beta = beta
        self.hip_weight = hip_weight
        self.num_joints = num_joints
        self.hip_joint_idx = hip_joint_idx
        self.align_hip = align_hip
        weights = torch.ones(num_joints)
        weights[hip_joint_idx] = hip_weight
        weights = weights * (num_joints / weights.sum())   # normalize: mean ≈ 1
        self.register_buffer('joint_weights', weights)
        # 对齐模式用: 排除 hip 的 mask (其余关节权 1, sum = num_joints-1)
        amask = torch.ones(num_joints); amask[hip_joint_idx] = 0.0
        self.register_buffer('align_mask', amask)

    def forward(self, p_student, p_teacher):
        """
        p_student: (B,T,J,3) requires grad ; p_teacher: (B,T,J,3) detached
        """
        if p_student.shape != p_teacher.shape:
            raise ValueError(
                f"shape mismatch: p_student={tuple(p_student.shape)} "
                f"vs p_teacher={tuple(p_teacher.shape)}")
        h = self.hip_joint_idx

        if self.align_hip:
            # 各自减 hip (参考点 detach -> hip 本身不从本项收梯度)
            ps = p_student - p_student[..., h:h + 1, :].detach()
            pt = p_teacher - p_teacher[..., h:h + 1, :]
            per = F.smooth_l1_loss(ps, pt, beta=self.beta, reduction='none').mean(dim=-1)  # (B,T,J)
            loss = (per * self.align_mask).sum(dim=-1) / self.align_mask.sum()             # (B,T)
            loss = loss.mean()
            with torch.no_grad():
                diff = ((ps - pt).abs().mean(dim=-1) * self.align_mask).sum() / self.align_mask.sum()
                mean_l1_mm = diff.mean().item() * 1000
        else:
            per = F.smooth_l1_loss(p_student, p_teacher, beta=self.beta, reduction='none')
            per = per.mean(dim=-1)                          # (B,T,J)
            weighted = per * self.joint_weights             # broadcast over (B,T)
            loss = weighted.mean()
            with torch.no_grad():
                mean_l1_mm = (p_student - p_teacher).abs().mean().item() * 1000

        return loss, {
            'l_distill_out':    loss.item(),
            'l_distill_out_mm': mean_l1_mm,
        }


# ----------------------------------------------------------------------
# 3) Kinematic prior (bone-length consistency + bilateral symmetry)
# ----------------------------------------------------------------------
H36M_BONES = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(9,10),
              (8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]

H36M_SYM_PAIRS = [(0,3),(1,4),(2,5),(10,13),(11,14),(12,15)]


class KinematicPriorLoss(nn.Module):
    """Bone-length consistency (vs GT) + bilateral symmetry prior. (未改动)"""
    def __init__(self, bones=None, sym_pairs=None,
                 bone_weight=1.0, sym_weight=0.3, beta=0.02):
        super().__init__()
        self.bones = bones if bones is not None else H36M_BONES
        self.sym_pairs = sym_pairs if sym_pairs is not None else H36M_SYM_PAIRS
        self.bone_weight = bone_weight
        self.sym_weight = sym_weight
        self.beta = beta
        parents = torch.tensor([b[0] for b in self.bones], dtype=torch.long)
        children = torch.tensor([b[1] for b in self.bones], dtype=torch.long)
        self.register_buffer('bone_parents', parents)
        self.register_buffer('bone_children', children)

    def _bone_lengths(self, pose):
        p = pose[..., self.bone_parents, :]
        c = pose[..., self.bone_children, :]
        return torch.linalg.norm(c - p, dim=-1)

    def forward(self, p_pred, p_gt):
        bl_pred = self._bone_lengths(p_pred)
        bl_gt   = self._bone_lengths(p_gt)
        l_bone = F.smooth_l1_loss(bl_pred, bl_gt, beta=self.beta)
        if self.sym_pairs:
            li = torch.tensor([s[0] for s in self.sym_pairs], device=p_pred.device)
            ri = torch.tensor([s[1] for s in self.sym_pairs], device=p_pred.device)
            l_sym = (bl_pred[..., li] - bl_pred[..., ri]).abs().mean()
        else:
            l_sym = torch.zeros((), device=p_pred.device)
        total = self.bone_weight * l_bone + self.sym_weight * l_sym
        with torch.no_grad():
            mean_bone_err_mm = (bl_pred - bl_gt).abs().mean().item() * 1000
        return total, {
            'l_kine':      total.item(),
            'l_kine_bone': l_bone.item(),
            'l_kine_sym':  l_sym.item(),
            'l_kine_bone_mm': mean_bone_err_mm,
        }


# ----------------------------------------------------------------------
# Sandbox tests
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    torch.manual_seed(0)
    B, T, C, J = 2, 64, 128, 17

    print("=" * 60); print("FeatureDistillLoss"); print("=" * 60)
    proj = DistillProjection(C, C)
    floss = FeatureDistillLoss(cosine_weight=1.0, smoothl1_weight=1.0)
    z_student = torch.randn(B, T, C, requires_grad=True)
    z_teacher = torch.randn(B, T, C).detach()
    z_proj = proj(z_student)
    fl, fd = floss(z_proj, z_teacher)
    print(f"loss={fl.item():.4f} | {fd}")
    fl.backward()
    assert z_student.grad is not None and z_student.grad.abs().sum() > 0
    assert z_teacher.grad is None
    print("grad -> student: OK | grad -> teacher: None")

    print(); print("=" * 60); print("OutputDistillLoss (align_hip=False, 原行为)"); print("=" * 60)
    oloss = OutputDistillLoss(beta=0.05, hip_weight=1.5, num_joints=J, hip_joint_idx=0)
    p_student = (torch.randn(B, T, J, 3) * 0.1).requires_grad_(True)
    p_teacher = (torch.randn(B, T, J, 3) * 0.1).detach()
    ol, od = oloss(p_student, p_teacher)
    print(f"loss={ol.item():.4f} | {od}")
    print(f"  mean weight = {oloss.joint_weights.mean().item():.4f} (should be 1.0)")
    ol.backward()
    assert p_student.grad is not None and p_teacher.grad is None
    print("grad -> p_student: OK | grad -> p_teacher: None")
    try:
        oloss(torch.randn(B, T, J, 3), torch.randn(B, T, 16, 3))
        raise AssertionError("expected shape mismatch to raise")
    except ValueError:
        print("shape-mismatch check raises ValueError: OK")
    big_err_hip = torch.zeros(B, T, J, 3); big_err_hip[:, :, 0, :] = 0.5
    big_err_other = torch.zeros(B, T, J, 3); big_err_other[:, :, 1, :] = 0.5
    zero = torch.zeros(B, T, J, 3)
    with torch.no_grad():
        lh, _ = oloss(big_err_hip, zero); lo, _ = oloss(big_err_other, zero)
    ratio = lh.item() / max(lo.item(), 1e-12)
    print(f"  hip-weighted ratio = {ratio:.3f} (should be ~1.5)")
    assert abs(ratio - 1.5) < 0.05

    print(); print("=" * 60); print("OutputDistillLoss (align_hip=True, RGB 解耦)"); print("=" * 60)
    aloss = OutputDistillLoss(beta=0.05, num_joints=J, hip_joint_idx=0, align_hip=True)
    ps = (torch.randn(B, T, J, 3) * 0.1).requires_grad_(True)
    pt = (torch.randn(B, T, J, 3) * 0.1).detach()
    base, _ = aloss(ps, pt)
    with torch.no_grad():
        shifted, _ = aloss(ps + torch.tensor([0.5, -0.3, 0.2]), pt)
    print(f"  全局平移不变: base={base.item():.6f} shifted={shifted.item():.6f}")
    assert abs(base.item() - shifted.item()) < 1e-6
    with torch.no_grad():
        z, _ = aloss(pt + 1.0, pt)
    print(f"  学生=教师+平移 -> loss={z.item():.2e} (应~0: 不传绝对hip)")
    assert z.item() < 1e-6
    base.backward()
    print(f"  hip(joint0) grad = {ps.grad[..., 0, :].abs().sum().item():.2e} (应~0: hip不被本项监督)")
    assert ps.grad[..., 0, :].abs().sum().item() < 1e-6
    print("align_hip 模式: OK")
    print("\n[ALL OK]")