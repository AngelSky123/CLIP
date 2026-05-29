"""
Feature-level distillation loss for Step B (depth -> CSI cross-modal distillation).

INDEPENDENT module — does NOT touch losses.py / train.py. The CSI student's
z_global (B,T,128) is aligned to the frozen depth teacher's z_global (B,T,128)
through a small student-side projection head, using a cosine + smooth-L1 combo.

Why projection head:
  CSI and depth feature spaces are not isomorphic. Forcing z_student == z_teacher
  directly would harm the student's own discriminative features. The projection
  head lets the student keep its native representation while still being
  "projectable" onto the teacher's geometry.

Why cosine + smooth-L1 (not pure MSE):
  z_global dims have heterogeneous scales; pure MSE is dominated by large-scale
  dims. Cosine aligns direction, smooth-L1 aligns magnitude robustly.

Teacher features are always .detach()'d by the caller (teacher is frozen), so no
gradient flows into the teacher.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillProjection(nn.Module):
    """Student-side projection head: z_student (B,T,128) -> (B,T,128)."""
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
    """Cosine + smooth-L1 alignment between projected student feat and teacher feat.

    Args:
        cosine_weight:   weight on (1 - cosine_similarity)
        smoothl1_weight: weight on smooth_l1(proj_student, teacher)
    Both operate per-frame on (B,T,128), averaged over B and T.
    """
    def __init__(self, cosine_weight=1.0, smoothl1_weight=1.0):
        super().__init__()
        self.cw = cosine_weight
        self.sw = smoothl1_weight

    def forward(self, z_student_proj, z_teacher):
        """
        Args:
            z_student_proj: (B,T,C) — projected student features (requires grad)
            z_teacher:      (B,T,C) — frozen teacher features (already detached)
        Returns:
            scalar loss, dict of components
        """
        # Cosine term: align direction, per (B,T) vector over channel dim
        cos = F.cosine_similarity(z_student_proj, z_teacher, dim=-1)   # (B,T)
        l_cos = (1.0 - cos).mean()

        # Smooth-L1 term: align magnitude, robust to outliers
        l_sl1 = F.smooth_l1_loss(z_student_proj, z_teacher)

        total = self.cw * l_cos + self.sw * l_sl1
        return total, {
            'l_distill': total.item(),
            'l_distill_cos': l_cos.item(),
            'l_distill_sl1': l_sl1.item(),
        }


if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    torch.manual_seed(0)
    B, T, C = 4, 64, 128

    proj = DistillProjection(C, C)
    dloss = FeatureDistillLoss(cosine_weight=1.0, smoothl1_weight=1.0)

    # Fake student feature (requires grad) + frozen teacher feature (detached)
    z_student = torch.randn(B, T, C, requires_grad=True)
    z_teacher = torch.randn(B, T, C).detach()

    z_proj = proj(z_student)
    loss, d = dloss(z_proj, z_teacher)
    print(f"shapes: z_student{tuple(z_student.shape)} proj{tuple(z_proj.shape)} "
          f"z_teacher{tuple(z_teacher.shape)}")
    print(f"loss={loss.item():.4f} | {d}")

    loss.backward()
    # gradient must reach student feature (through proj) ...
    assert z_student.grad is not None and z_student.grad.abs().sum() > 0, "no grad to student!"
    # ... and proj params
    assert all(p.grad is not None for p in proj.parameters()), "no grad to proj!"
    # ... and NEVER to teacher (it's a leaf with requires_grad=False)
    assert z_teacher.grad is None, "teacher got gradient!"
    print("grad -> student: OK | grad -> proj: OK | grad -> teacher: None (correct)")

    # perfect-alignment sanity: loss should be ~0 when proj(student)==teacher
    with torch.no_grad():
        l0, _ = dloss(z_teacher, z_teacher)
    print(f"loss(teacher,teacher) = {l0.item():.6f} (should be ~0)")
    print("OK")