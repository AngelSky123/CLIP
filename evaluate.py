"""
Module 10: 评估指标
- MPJPE (mm), PA-MPJPE (mm), PCK@50 (%), PCK@20 (%)

Note: Input data is ALREADY root-relative (dataset subtracts Hip joint).
Do NOT apply root_relative() again here.
"""
import numpy as np
import torch


def mpjpe(pred, gt):
    """Mean Per Joint Position Error.
    Assumes inputs are already root-relative.
    """
    if pred.dim() == 4:
        pred = pred.reshape(-1, pred.shape[-2], 3)
        gt = gt.reshape(-1, gt.shape[-2], 3)
    dist = torch.norm(pred - gt, dim=-1)
    return dist.mean().item()


def pa_mpjpe(pred, gt):
    """Procrustes-Aligned MPJPE."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if pred.ndim == 4:
        pred = pred.reshape(-1, pred.shape[-2], 3)
        gt = gt.reshape(-1, gt.shape[-2], 3)
    N = pred.shape[0]
    total_error = 0.0
    for i in range(N):
        aligned = procrustes_align(pred[i], gt[i])
        error = np.linalg.norm(aligned - gt[i], axis=-1).mean()
        total_error += error
    return total_error / N


def procrustes_align(pred, gt):
    mu_pred = pred.mean(axis=0, keepdims=True)
    mu_gt = gt.mean(axis=0, keepdims=True)
    pred_centered = pred - mu_pred
    gt_centered = gt - mu_gt
    norm_pred = np.linalg.norm(pred_centered)
    norm_gt = np.linalg.norm(gt_centered)
    pred_scaled = pred_centered / (norm_pred + 1e-8)
    gt_scaled = gt_centered / (norm_gt + 1e-8)
    H = pred_scaled.T @ gt_scaled
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.eye(3)
    sign_matrix[2, 2] = np.sign(d)
    R = Vt.T @ sign_matrix @ U.T
    aligned = norm_gt * (pred_scaled @ R.T) + mu_gt
    return aligned


def pck(pred, gt, threshold=0.05):
    """Percentage of Correct Keypoints.
    Assumes inputs are already root-relative.
    """
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = pred
    if isinstance(gt, torch.Tensor):
        gt_np = gt.detach().cpu().numpy()
    else:
        gt_np = gt
    if pred_np.ndim == 4:
        pred_np = pred_np.reshape(-1, pred_np.shape[-2], 3)
        gt_np = gt_np.reshape(-1, gt_np.shape[-2], 3)
    dist = np.linalg.norm(pred_np - gt_np, axis=-1)
    correct = (dist < threshold).astype(np.float32)
    return correct.mean() * 100.0


class PoseEvaluator:
    def __init__(self, unit='meter'):
        self.unit = unit
        self.scale = 1000.0 if unit == 'meter' else 1.0

    def evaluate(self, pred, gt):
        mpjpe_val = mpjpe(pred, gt) * self.scale
        pa_mpjpe_val = pa_mpjpe(pred, gt) * self.scale
        threshold_50 = 0.05 if self.unit == 'meter' else 50.0
        threshold_20 = 0.02 if self.unit == 'meter' else 20.0
        pck_50 = pck(pred, gt, threshold=threshold_50)
        pck_20 = pck(pred, gt, threshold=threshold_20)
        return {
            'MPJPE (mm)': mpjpe_val,
            'PA-MPJPE (mm)': pa_mpjpe_val,
            'PCK@50 (%)': pck_50,
            'PCK@20 (%)': pck_20,
        }