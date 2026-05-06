"""
Training objectives v3 — fixed gradient flow documentation

Loss structure:
  L_total = L_pose(clean) + α·L_pose(masked) + β·L_cons + γ·(div losses) + δ·L_action

  - L_pose(clean):  PRIMARY loss — gradients flow to BACKBONE + DECODER
  - L_pose(masked): RSC regularization — gradients flow to DECODER only
  - L_cons:         Consistency between clean and masked predictions
  - L_div/L_temp:   Anti-collapse regularization
  - L_action:       Action classification auxiliary task
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
    def __init__(self, lambda1=1.0, lambda2=0.5):
        super().__init__()
        self.coord_loss = CoordinateLoss()
        self.bone_loss = BoneConsistencyLoss()
        self.vel_loss = VelocitySmoothLoss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, pred, gt):
        l_coord = self.coord_loss(pred, gt)
        l_bone = self.bone_loss(pred, gt)
        l_vel = self.vel_loss(pred, gt)
        total = l_coord + self.lambda1 * l_bone + self.lambda2 * l_vel
        return total, {
            'l_coord': l_coord.item(),
            'l_bone': l_bone.item(),
            'l_vel': l_vel.item(),
        }


class TotalLoss(nn.Module):
    """
    L = L_pose(clean)                  ← backbone + decoder gradients
      + α · L_pose(masked)             ← decoder only (RSC regularization)
      + β · L_cons                     ← consistency
      + γ · (L_div + L_temp + L_input) ← anti-collapse
      + δ · L_action                   ← auxiliary task
    """

    def __init__(self, lambda1=1.0, lambda2=0.5, alpha=0.5, beta=2.0,
                 gamma=0.005, delta=0.02):
        super().__init__()
        self.pose_loss = PoseLoss(lambda1, lambda2)
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
            # Clean path: backbone + decoder receive gradients
            l_pose_clean, cd = self.pose_loss(outputs['p_final_clean'], gt)
            # Masked path: only decoder receives gradients
            l_pose_masked, md = self.pose_loss(outputs['p_final_masked'], gt)
            l_cons = self.cons_loss(outputs['p_final_clean'],
                                    outputs['p_final_masked'])

            # Anti-collapse (on clean predictions)
            pred_clean = outputs['p_final_clean']
            l_div = self.div_loss(pred_clean)
            l_temp_div = self.temp_div_loss(pred_clean, gt)
            l_input_sens = self.input_sens_loss(pred_clean, gt)

            # Clean is PRIMARY (backbone gradients), masked is secondary
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
                'l_coord_masked': md['l_coord'],
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