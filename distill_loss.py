"""
Distillation losses for depth -> CSI cross-modal distillation.

Two complementary alignment objectives:

  1) FeatureDistillLoss (z_global alignment, B,T,128)
     - Aligns latent feature geometry through a learnable student projection.
     - Cosine (direction) + smooth-L1 (magnitude). Robust to scale heterogeneity.

  2) OutputDistillLoss (pose alignment, B,T,J,3 in meters)
     - Aligns predicted joint positions directly.
     - Smooth-L1 with beta=5cm (robust to teacher's residual errors).
     - Motivated by the fact that teacher MPJPE=269 << DT-Pose 316.8: the
       teacher's absolute-joint-localization advantage (esp. hip global pos)
       transfers most cleanly through output-level supervision, not feature-
       level alignment. This directly targets the MPJPE gap to DT-Pose.

Teacher features/predictions are always .detach()'d (teacher is frozen).
Both losses scale by their independent lambdas in the training loop.

Sane starting point:
  lambda_feat = 0.1
  lambda_out  = 0.5
"""
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
    """Cosine + smooth-L1 alignment of projected student feature to teacher.

    Args:
        cosine_weight:   weight on (1 - cosine_similarity), direction alignment
        smoothl1_weight: weight on smooth_l1(proj_student, teacher), magnitude
    """
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

    Teacher MPJPE=269 vs CSI baseline MPJPE=345 — the gap is largely absolute
    joint localization (hip global position). Directly distilling teacher's
    predicted pose transfers this geometry more reliably than indirect feature
    alignment, and directly targets the MPJPE gap to DT-Pose (316.8).

    Why smooth-L1 with beta=5cm:
      Teacher predictions are not perfect (27cm avg error). beta=5cm makes the
      loss linear past 5cm, which:
        - downweights teacher's residual errors (won't overfit to teacher noise)
        - still gives a strong pull on within-5cm structure
      Pose is in meters in this codebase.

    Optional hip emphasis (hip_weight): joint 0 (Bot Torso / hip in MMFi)
    gets `hip_weight` * the weight of other joints. Weights are normalized so
    mean weight stays ~1 (keeps lambda_out semantics stable across hip_weight).
    Set hip_weight=1.0 to disable.
    """
    def __init__(self, beta=0.05, hip_weight=1.5, num_joints=17, hip_joint_idx=0):
        super().__init__()
        self.beta = beta
        self.hip_weight = hip_weight
        weights = torch.ones(num_joints)
        weights[hip_joint_idx] = hip_weight
        weights = weights * (num_joints / weights.sum())   # normalize: mean ≈ 1
        self.register_buffer('joint_weights', weights)

    def forward(self, p_student, p_teacher):
        """
        Args:
            p_student: (B,T,J,3) — student pose, requires grad
            p_teacher: (B,T,J,3) — teacher pose, already detached
        """
        if p_student.shape != p_teacher.shape:
            raise ValueError(
                f"shape mismatch: p_student={tuple(p_student.shape)} "
                f"vs p_teacher={tuple(p_teacher.shape)}")
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
    print(f"shapes: z_student{tuple(z_student.shape)} -> proj{tuple(z_proj.shape)}")
    print(f"loss={fl.item():.4f} | {fd}")
    fl.backward()
    assert z_student.grad is not None and z_student.grad.abs().sum() > 0
    assert all(p.grad is not None for p in proj.parameters())
    assert z_teacher.grad is None
    print("grad -> student: OK | grad -> proj: OK | grad -> teacher: None")
    with torch.no_grad():
        l0, _ = floss(z_teacher, z_teacher)
    print(f"loss(teacher,teacher) = {l0.item():.6f} (should be ~0)")

    print(); print("=" * 60); print("OutputDistillLoss"); print("=" * 60)
    oloss = OutputDistillLoss(beta=0.05, hip_weight=1.5, num_joints=J, hip_joint_idx=0)
    p_student = (torch.randn(B, T, J, 3) * 0.1).requires_grad_(True)
    p_teacher = (torch.randn(B, T, J, 3) * 0.1).detach()
    ol, od = oloss(p_student, p_teacher)
    print(f"shapes: p_student{tuple(p_student.shape)} vs p_teacher{tuple(p_teacher.shape)}")
    print(f"loss={ol.item():.4f} | {od}")
    print(f"joint_weights = {oloss.joint_weights.tolist()}")
    print(f"  mean weight = {oloss.joint_weights.mean().item():.4f} (should be 1.0)")
    ol.backward()
    assert p_student.grad is not None and p_student.grad.abs().sum() > 0
    assert p_teacher.grad is None
    print("grad -> p_student: OK | grad -> p_teacher: None")
    with torch.no_grad():
        l0o, _ = oloss(p_teacher, p_teacher)
    print(f"loss(teacher,teacher) = {l0o.item():.6f} (should be ~0)")
    try:
        oloss(torch.randn(B, T, J, 3), torch.randn(B, T, 16, 3))
        raise AssertionError("expected shape mismatch to raise")
    except ValueError as e:
        print(f"shape-mismatch check raises ValueError: OK")

    # Hip weight effect
    big_err_hip = torch.zeros(B, T, J, 3); big_err_hip[:, :, 0, :] = 0.5
    big_err_other = torch.zeros(B, T, J, 3); big_err_other[:, :, 1, :] = 0.5
    zero = torch.zeros(B, T, J, 3)
    with torch.no_grad():
        lh, _ = oloss(big_err_hip, zero)
        lo, _ = oloss(big_err_other, zero)
    ratio = lh.item() / max(lo.item(), 1e-12)
    print(f"loss(hip-err=0.5) = {lh.item():.4f}")
    print(f"loss(joint1-err=0.5) = {lo.item():.4f}")
    print(f"  hip-weighted ratio = {ratio:.3f} (should be ~1.5)")
    assert abs(ratio - 1.5) < 0.05, f"expected ratio ~1.5, got {ratio:.3f}"
    print("hip emphasis: OK")

    print("\n[ALL OK]")