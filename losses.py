"""
Training objectives v5 — MotionGuidanceLoss for temporal collapse

Loss structure:
  L_pose = L_coord + λ1·L_bone + λ2·L_vel + λ3·L_motion
  L_total = L_pose(clean) + α·L_pose(masked) + β·L_cons + γ·(div losses) + δ·L_action

New in v5 — MotionGuidanceLoss (λ3):
  VelocitySmoothLoss fails to break temporal collapse because its gradient
  is proportional to GT acceleration (≈0 for smooth human motion).
  MotionGuidanceLoss uses .detach() to decouple consecutive frames, giving
  each frame an independent gradient in the GT displacement direction.

  Gradient analysis at static prediction (pred[t]=c for all t):
    VelocitySmoothLoss: ∂L/∂pred[t] ∝ gt_accel[t] → ≈0 (smooth motion)
    MotionGuidanceLoss: ∂L/∂pred[t] = -sign(gt_disp[t]) → always ±1 ✓
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pose_decoder import H36M_BONES


class CoordinateLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        dist = torch.norm(pred - gt, dim=-1)
        return dist.mean()


class HipPositionLoss(nn.Module):
    """专门惩罚 hip (joint 0) 的全局位置预测误差.

    CSI 信号原则上能编码人在房间里的位置 (多径效应), 但模型对此学得很慢.
    通过显式加权 hip 关节的回归损失, 引导模型更专注全局定位预测.

    使用方式: pose_loss = base_loss + lambda_hip * HipPositionLoss
    建议起始权重 1.0, 如果 PA-MPJPE 显著上升说明姿态质量被挤压, 降到 0.5.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        hip_pred = pred[:, :, 0, :]  # (B, T, 3)
        hip_gt = gt[:, :, 0, :]
        return torch.norm(hip_pred - hip_gt, dim=-1).mean()



class BoneConsistencyLoss(nn.Module):
    def __init__(self, bones=None):
        super().__init__()
        self.bones = bones or H36M_BONES

    def compute_bone_lengths(self, joints):
        bone_lengths = []
        for i, j in self.bones:
            length = torch.norm(joints[:, :, i] - joints[:, :, j], dim=-1)
            bone_lengths.append(length)
        return torch.stack(bone_lengths, dim=-1)

    def forward(self, pred, gt):
        return F.l1_loss(self.compute_bone_lengths(pred),
                         self.compute_bone_lengths(gt))


class VelocitySmoothLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        pred_vel = pred[:, 1:] - pred[:, :-1]
        gt_vel = gt[:, 1:] - gt[:, :-1]
        return torch.norm(pred_vel - gt_vel, dim=-1).mean()


class MotionGuidanceLoss(nn.Module):
    """Break temporal collapse via detach-based per-frame motion guidance.

    Problem with VelocitySmoothLoss:
      loss = ||pred_vel[t] - gt_vel[t]||
      When pred is static, ∂loss/∂pred[t] ∝ gt_acceleration (≈0 for smooth motion).
      The gradients from consecutive frames CANCEL because they enter as a difference.

    Solution: detach pred[t-1] to break the gradient coupling.
      target[t] = pred[t-1].detach() + (gt[t] - gt[t-1])
      loss = ||pred[t] - target[t]||

    When pred is static (pred[t] = c for all t):
      ∂loss/∂pred[t] = sign(c - (c + gt_disp)) = -sign(gt_disp[t])
      → pushes pred[t] in the GT displacement direction
      → independent per frame, no cancellation
      → magnitude is always 1 (L1), regardless of motion smoothness
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        gt_disp = gt[:, 1:] - gt[:, :-1]          # (B, T-1, 17, 3)
        target = pred[:, :-1].detach() + gt_disp   # where pred[t] should be
        return F.l1_loss(pred[:, 1:], target)


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_clean, pred_masked):
        return torch.norm(pred_clean.detach() - pred_masked, dim=-1).mean()


class DiversityLoss(nn.Module):
    """Penalize low variance in batch predictions.
    
    Uses hinge-style loss: max(0, margin - std).
    More stable than -log(var) which has unbounded gradients near zero.
    margin is set to ~30% of typical GT std (64mm → ~20mm = 0.02m).
    """

    def __init__(self, margin=0.02):
        super().__init__()
        self.margin = margin

    def forward(self, pred):
        B = pred.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=pred.device)
        mean_pose = pred.mean(dim=1)  # (B, 17, 3)
        std = mean_pose.std(dim=0).mean()  # scalar, in meters
        # Hinge: penalize when std < margin, no penalty when std >= margin
        return F.relu(self.margin - std)


class TemporalDiversityLoss(nn.Module):
    """Penalize temporal motion that is too static compared to GT."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        pred_motion = torch.norm(pred[:, 1:] - pred[:, :-1], dim=-1).mean(dim=(1, 2))
        gt_motion = torch.norm(gt[:, 1:] - gt[:, :-1], dim=-1).mean(dim=(1, 2))
        ratio = pred_motion / (gt_motion + 1e-6)
        return F.relu(1.0 - ratio).mean()


class InputSensitivityLoss(nn.Module):
    """Penalize predictions that don't vary when GT varies.
    Vectorized implementation (no O(B^2) loop).
    """

    def __init__(self, margin=0.05):
        super().__init__()
        self.margin = margin

    def forward(self, pred, gt):
        B = pred.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=pred.device)

        pred_flat = pred.mean(dim=1).reshape(B, -1)  # (B, 51)
        gt_flat = gt.mean(dim=1).reshape(B, -1)

        # Pairwise distances (vectorized)
        gt_dist = torch.cdist(gt_flat.unsqueeze(0), gt_flat.unsqueeze(0)).squeeze(0)  # (B, B)
        pred_dist = torch.cdist(pred_flat.unsqueeze(0), pred_flat.unsqueeze(0)).squeeze(0)

        # Upper triangle only (avoid self-pairs and double counting)
        mask = torch.triu(torch.ones(B, B, device=pred.device), diagonal=1).bool()
        gt_d = gt_dist[mask]
        pred_d = pred_dist[mask]

        # Only penalize pairs where GT is sufficiently different
        valid = gt_d > self.margin
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        ratio = pred_d[valid] / (gt_d[valid] + 1e-6)
        return F.relu(0.5 - ratio).mean()


class PoseLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=0.5, lambda3=2.0, lambda_hip=1.0):
        super().__init__()
        self.coord_loss = CoordinateLoss()
        self.bone_loss = BoneConsistencyLoss()
        self.vel_loss = VelocitySmoothLoss()
        self.motion_guidance = MotionGuidanceLoss()
        self.hip_loss = HipPositionLoss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda_hip = lambda_hip

    def forward(self, pred, gt):
        l_coord = self.coord_loss(pred, gt)
        l_bone = self.bone_loss(pred, gt)
        l_vel = self.vel_loss(pred, gt)
        l_motion = self.motion_guidance(pred, gt)
        l_hip = self.hip_loss(pred, gt)
        total = (l_coord
                 + self.lambda1 * l_bone
                 + self.lambda2 * l_vel
                 + self.lambda3 * l_motion
                 + self.lambda_hip * l_hip)
        return total, {
            'l_coord': l_coord.item(),
            'l_bone': l_bone.item(),
            'l_vel': l_vel.item(),
            'l_motion': l_motion.item(),
            'l_hip': l_hip.item(),
        }


class TotalLoss(nn.Module):
    """域泛化联合损失 (用于 train.py 的 forward_rsc 训练路径).

    L = L_pose(clean)                  ← backbone + decoder 梯度 (主损失)
      + α · L_pose(masked)             ← backbone + decoder 梯度 (RSC 正则化)
      + β · L_cons                     ← 一致性约束
      + γ · (L_div + L_temp + L_input) ← 反塌陷正则化
      + δ · L_action                   ← 动作分类辅助任务

    v7.1 梯度流说明:
      clean 路径和 masked 路径的梯度都会回传到 backbone (CSI Encoder +
      Transformer). masked 路径之所以也能更新 backbone, 是因为
      RSCGlobalChallenger 在 z_global_raw (保留计算图) 上做 mask,
      未被遮挡的元素保留了完整的反向传播路径.

    动作先验解耦:
      forward_rsc 中以 50% 概率将 action_emb 置零 (Action Dropout),
      迫使 decoder 不过度依赖动作标签, 提升跨域鲁棒性.
    """

    def __init__(self, lambda1=1.0, lambda2=0.5, lambda3=2.0, alpha=0.5, beta=2.0,
                 gamma=0.005, delta=0.02, lambda_hip=1.0):
        super().__init__()
        self.pose_loss = PoseLoss(lambda1, lambda2, lambda3, lambda_hip)
        self.cons_loss = ConsistencyLoss()
        self.div_loss = DiversityLoss()
        self.temp_div_loss = TemporalDiversityLoss()
        self.input_sens_loss = InputSensitivityLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def forward(self, outputs, gt, training=True, action_loss=None):
        loss_dict = {}

        if training and 'p_final_masked' in outputs:
            # Clean path: backbone + decoder receive gradients (主损失)
            l_pose_clean, cd = self.pose_loss(outputs['p_final_clean'], gt)
            # Masked path: backbone + decoder receive gradients (RSC 正则化)
            # v7.1: z_global_masked 保留了 backbone 的计算图,
            # 梯度经未遮挡元素回传到 Transformer 和 CSI Encoder
            l_pose_masked, md = self.pose_loss(outputs['p_final_masked'], gt)
            l_cons = self.cons_loss(outputs['p_final_clean'],
                                    outputs['p_final_masked'])

            # Anti-collapse (on clean predictions)
            pred_clean = outputs['p_final_clean']
            l_div = self.div_loss(pred_clean)
            l_temp_div = self.temp_div_loss(pred_clean, gt)
            l_input_sens = self.input_sens_loss(pred_clean, gt)

            total = (l_pose_clean
                     + self.alpha * l_pose_masked
                     + self.beta * l_cons
                     + self.gamma * (l_div + l_temp_div + l_input_sens))

            if action_loss is not None:
                total = total + self.delta * action_loss

            loss_dict.update({
                'l_total': total.item(),
                'l_pose_clean': l_pose_clean.item(),
                'l_pose_masked': l_pose_masked.item(),
                'l_cons': l_cons.item(),
                'l_div': l_div.item(),
                'l_temp_div': l_temp_div.item(),
                'l_input_sens': l_input_sens.item(),
                'l_action': action_loss.item() if action_loss is not None else 0,
                'l_coord_clean': cd['l_coord'],
                'l_motion_clean': cd['l_motion'],
                'l_coord_masked': md['l_coord'],
                'l_motion_masked': md['l_motion'],
                'l_bone_masked': md['l_bone'],
                'l_vel_masked': md['l_vel'],
            })
        else:
            pred = outputs.get('p_final', outputs.get('p_final_clean'))
            l_pose, details = self.pose_loss(pred, gt)
            total = l_pose
            loss_dict.update({
                'l_total': total.item(),
                'l_pose': l_pose.item(),
                **details,
            })

        return total, loss_dict