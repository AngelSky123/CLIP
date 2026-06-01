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
# 3) Kinematic prior (bone-length consistency + bilateral symmetry)
# ----------------------------------------------------------------------
# MMFi / H36M-17 skeleton bone connectivity (parent, child). Matches the
# adjacency used by the model's SkeletonRefiner (models/pose_decoder.py).
H36M_BONES = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(9,10),
              (8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]

# Left/right symmetric bone index pairs (indices into H36M_BONES):
#   0:(0,1) L.hip      <-> 3:(0,4) R.hip
#   1:(1,2) L.thigh    <-> 4:(4,5) R.thigh
#   2:(2,3) L.shin     <-> 5:(5,6) R.shin
#  10:(8,11) R.clav    <-> 13:(8,14) L.clav
#  11:(11,12) R.uarm   <-> 14:(14,15) L.uarm
#  12:(12,13) R.farm   <-> 15:(15,16) L.farm
H36M_SYM_PAIRS = [(0,3),(1,4),(2,5),(10,13),(11,14),(12,15)]


class KinematicPriorLoss(nn.Module):
    """Bone-length consistency (vs GT) + bilateral symmetry prior.

    Targets PA-MPJPE specifically. PA error after Procrustes alignment is
    dominated by limb-structure errors (disproportionate bones, misplaced
    joints — exactly DT-Pose's "structural fidelity gap"). The model's GCN
    SkeletonRefiner learns topology implicitly; this loss gives it an explicit
    structural objective.

    Two terms:
      - bone : smooth-L1 between predicted and GT bone lengths. The primary
               signal; GT bone lengths are the correct target.
      - sym  : |left_bone - right_bone| on symmetric limb pairs (prediction
               only). A prior that limb pairs are equal length; helps regularize
               where GT is noisy. Lighter weight by default.

    Scale/translation invariant: bone lengths don't depend on global position,
    so this is MPJPE-neutral (won't disturb absolute localization) while
    directly improving structural fidelity → PA.
    """
    def __init__(self, bones=None, sym_pairs=None,
                 bone_weight=1.0, sym_weight=0.3, beta=0.02):
        super().__init__()
        self.bones = bones if bones is not None else H36M_BONES
        self.sym_pairs = sym_pairs if sym_pairs is not None else H36M_SYM_PAIRS
        self.bone_weight = bone_weight
        self.sym_weight = sym_weight
        self.beta = beta  # SmoothL1 transition, meters (0.02 = 2cm bone-len error)
        # Precompute parent/child index tensors for vectorized bone extraction
        parents = torch.tensor([b[0] for b in self.bones], dtype=torch.long)
        children = torch.tensor([b[1] for b in self.bones], dtype=torch.long)
        self.register_buffer('bone_parents', parents)
        self.register_buffer('bone_children', children)

    def _bone_lengths(self, pose):
        """pose: (B,T,J,3) -> bone lengths (B,T,num_bones)."""
        p = pose[..., self.bone_parents, :]    # (B,T,num_bones,3)
        c = pose[..., self.bone_children, :]
        return torch.linalg.norm(c - p, dim=-1)   # (B,T,num_bones)

    def forward(self, p_pred, p_gt):
        """
        Args:
            p_pred: (B,T,J,3) predicted pose (requires grad)
            p_gt:   (B,T,J,3) ground-truth pose
        """
        bl_pred = self._bone_lengths(p_pred)
        bl_gt   = self._bone_lengths(p_gt)
        l_bone = F.smooth_l1_loss(bl_pred, bl_gt, beta=self.beta)

        # Symmetry on predicted bone lengths
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

# ----------------------------------------------------------------------
# Kinematic prior sandbox (run: python distill_loss.py kine)
# ----------------------------------------------------------------------
def _sandbox_kinematic():
    import warnings; warnings.filterwarnings('ignore')
    torch.manual_seed(0)
    B, T, J = 2, 8, 17
    print("=" * 60); print("KinematicPriorLoss"); print("=" * 60)

    kine = KinematicPriorLoss(bone_weight=1.0, sym_weight=0.3, beta=0.02)

    # Build a GT pose with KNOWN symmetric bone lengths from a canonical skeleton
    gt = torch.randn(B, T, J, 3) * 0.4
    gt = gt.requires_grad_(False)

    # Test 1: pred == gt -> BONE term ~ 0 (sym term reflects GT's own asymmetry)
    pred = gt.clone().requires_grad_(True)
    l, d = kine(pred, gt)
    print(f"loss(pred==gt): bone={d['l_kine_bone']:.6f} sym={d['l_kine_sym']:.4f} "
          f"bone_mm={d['l_kine_bone_mm']:.3f}")
    assert d['l_kine_bone'] < 1e-6, "bone term should vanish when pred==gt"
    print("  bone term -> 0 when pred==gt: OK (sym>0 only because random GT isn't symmetric)")

    # Bone-only loss for the invariance tests (isolate bone term)
    kine_bone = KinematicPriorLoss(bone_weight=1.0, sym_weight=0.0, beta=0.02)

    # Test 2: grad flows to pred, not gt
    pred2 = (gt + torch.randn_like(gt) * 0.05).requires_grad_(True)
    l2, d2 = kine(pred2, gt)
    l2.backward()
    assert pred2.grad is not None and pred2.grad.abs().sum() > 0
    assert gt.grad is None
    print(f"loss(pred=gt+noise) = {l2.item():.4f}  bone_mm={d2['l_kine_bone_mm']:.1f}")
    print("grad -> pred: OK | grad -> gt: None")

    # Test 3: bone-length invariance to global translation (MPJPE-neutral check)
    shifted = (gt + torch.tensor([2.0, -3.0, 1.0])).requires_grad_(False)
    l3, _ = kine_bone(shifted, gt)
    print(f"bone-loss(pred=gt+translation) = {l3.item():.6f} (should be ~0: translation-invariant)")
    assert l3.item() < 1e-5, "bone lengths should be translation-invariant!"

    # Test 4: bone-length invariance to global rotation
    theta = 0.7
    Rz = torch.tensor([[torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta)), 0.],
                       [torch.sin(torch.tensor(theta)),  torch.cos(torch.tensor(theta)), 0.],
                       [0., 0., 1.]])
    rotated = (gt @ Rz.T).requires_grad_(False)
    l4, _ = kine_bone(rotated, gt)
    print(f"bone-loss(pred=gt rotated) = {l4.item():.6f} (should be ~0: rotation-invariant)")
    assert l4.item() < 1e-5, "bone lengths should be rotation-invariant!"

    # Test 5: symmetry term detects asymmetry
    # Make left thigh (bone 1: joints 1->2) much longer than right thigh (bone 4: 4->5)
    asym = gt.clone()
    asym[:, :, 2, :] = asym[:, :, 1, :] + torch.tensor([0., 1.0, 0.])   # L.knee far from L.hip
    asym[:, :, 5, :] = asym[:, :, 4, :] + torch.tensor([0., 0.2, 0.])   # R.knee near R.hip
    kine_sym_only = KinematicPriorLoss(bone_weight=0.0, sym_weight=1.0)
    l5, d5 = kine_sym_only(asym, gt)
    print(f"sym loss on L/R-asymmetric pose = {d5['l_kine_sym']:.4f} (should be > 0)")
    assert d5['l_kine_sym'] > 0.1

    # Test 6: bone connectivity sanity — 16 bones for 17 joints
    assert len(H36M_BONES) == 16
    assert len(H36M_SYM_PAIRS) == 6
    all_joints = set()
    for a,b in H36M_BONES: all_joints.add(a); all_joints.add(b)
    assert all_joints == set(range(17)), f"bones don't cover all 17 joints: {sorted(all_joints)}"
    print(f"skeleton: {len(H36M_BONES)} bones cover all 17 joints, {len(H36M_SYM_PAIRS)} sym pairs: OK")

    print("\n[KINEMATIC OK]")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "kine":
    import sys
    _sandbox_kinematic()