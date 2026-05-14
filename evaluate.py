"""
评估指标 v3 — DT-Pose 严格对齐版本

不再假设输入是 root-relative. 模型现在学绝对坐标, 评测也用绝对坐标.

指标定义 (与 DT-Pose `utils.py` 对应):
  MPJPE (mm)              <-> calulate_error(align=False) 返回的 mpjpe
  MPJPE_aligned (mm)      <-> calulate_error(align=True), hip 对齐后逐关节误差
  PA-MPJPE (mm)           <-> compute_similarity_transform + 计算欧氏距离
  PCK@thr (normalized)    <-> compute_pck_pckh, 按 GT 右肩-左髋距离归一化

所有 mm 指标输入单位米, 内部 ×1000.
"""
import numpy as np
import torch


# ============================================================
# DT-Pose 风格 PCK: 按身长 (右肩->左髋) 归一化
# ============================================================
def pck_normalized(pred, gt, thr=0.5):
    """DT-Pose 风格 PCK.

    Args:
        pred, gt: (B, T, 17, 3) 或 (N, 17, 3) 张量 / numpy, 单位 m
        thr: 误差阈值占 (右肩-左髋) 距离的比例 (0.5 = PCK@50)

    Returns:
        float: 所有关节平均 PCK 百分比 (0-100)

    与 DT-Pose `compute_pck_pckh` 的对应:
      DT-Pose: scale = ||gt[:, :, 5] - gt[:, :, 12]||  # transpose 后 [:, dim, joint]
        Joint 5 = LShoulder, Joint 12 = LHip (H36M 索引)
        → "右肩到左髋"的说法实际上是 LShoulder 到 LHip
      pck = 100 * mean(dist / scale <= thr)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if pred.ndim == 4:
        pred = pred.reshape(-1, pred.shape[-2], 3)
        gt = gt.reshape(-1, gt.shape[-2], 3)

    # H36M 索引: 11 = LShoulder, 4 = LHip
    # DT-Pose 在 transpose 后用 [:, :, 5] 和 [:, :, 12], 对应原始 joint 5 (LShoulder?) 和 12 (LHip?)
    # MMFi 用 17 关节 H36M 格式: 0=Hip 1=RHip 2=RKnee 3=RAnkle 4=LHip 5=LKnee 6=LAnkle
    # 7=Spine 8=Thorax 9=Neck 10=Head 11=LShoulder 12=LElbow 13=LWrist 14=RShoulder 15=RElbow 16=RWrist
    # DT-Pose 注释写的"右肩-左髋"实际是 LShoulder(11) 和 LHip(4), 因为它用的 H36M 索引
    # 但 DT-Pose 代码里写的是 5 和 12. 我们对齐它的代码, 用 LShoulder(11) 和 LElbow(12) 是不对的
    # 重新核对: DT-Pose 训练时 collate_fn_padd 的 _output 是 (B, 17, 3), 没有 transpose
    # pck 函数里 dt_kpts.transpose(0,2,1) 后形状变成 (B, 3, 17) → [:, :, 5] = joint 5
    # 所以 DT-Pose 用的是 joint 5 (LKnee in H36M) 和 joint 12 (LElbow in H36M)... 但注释说是右肩-左髋
    # 这是 DT-Pose 代码的潜在 bug, 但为了严格对齐, 我们就用它的索引: 5 和 12
    scale = np.sqrt(np.sum((gt[:, 5, :] - gt[:, 12, :]) ** 2, axis=1))  # (N,)
    scale = np.maximum(scale, 1e-6)  # 防止除零

    dist = np.linalg.norm(pred - gt, axis=-1)  # (N, 17)
    dist_normalized = dist / scale[:, None]

    pck_per_joint = (dist_normalized <= thr).astype(np.float32).mean(axis=0)  # (17,)
    return float(pck_per_joint.mean() * 100.0)


# ============================================================
# DT-Pose 风格 Procrustes Alignment
# ============================================================
def compute_similarity_transform(X, Y, compute_optimal_scale=True):
    """DT-Pose utils.py 的 Procrustes (含 optimal scaling).

    Args:
        X: (J, 3) 目标 (GT)
        Y: (J, 3) 待对齐 (Pred)
    Returns:
        Z: 对齐后的 Y (J, 3)
    """
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    normX = np.sqrt((X0 ** 2).sum())
    normY = np.sqrt((Y0 ** 2).sum())
    X0 /= (normX + 1e-8)
    Y0 /= (normY + 1e-8)

    A = X0.T @ Y0
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = V @ U.T
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = V @ U.T
    traceTA = s.sum()

    if compute_optimal_scale:
        b = traceTA * normX / (normY + 1e-8)
        Z = normX * traceTA * (Y0 @ T) + muX
    else:
        Z = normY * (Y0 @ T) + muX
    return Z


# ============================================================
# MPJPE / MPJPE_aligned / PA-MPJPE
# ============================================================
def mpjpe(pred, gt, align_hip=False):
    """Mean Per Joint Position Error (绝对坐标).

    Args:
        pred, gt: 任意 shape, 末两维 (J, 3)
        align_hip: 若 True, 先把 pred 的 hip(0号关节) 平移到 gt hip 位置
                   对应 DT-Pose calulate_error(align=True)

    Returns:
        float: 平均误差 (单位与输入一致)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if pred.ndim == 4:
        pred = pred.reshape(-1, pred.shape[-2], 3)
        gt = gt.reshape(-1, gt.shape[-2], 3)

    if align_hip:
        offset = gt[:, 0:1, :] - pred[:, 0:1, :]
        pred = pred + offset

    return float(np.linalg.norm(pred - gt, axis=-1).mean())


def pa_mpjpe(pred, gt):
    """Procrustes-aligned MPJPE (DT-Pose 风格: 含 optimal scaling)."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if pred.ndim == 4:
        pred = pred.reshape(-1, pred.shape[-2], 3)
        gt = gt.reshape(-1, gt.shape[-2], 3)

    N = pred.shape[0]
    errs = []
    for i in range(N):
        Z = compute_similarity_transform(gt[i], pred[i], compute_optimal_scale=True)
        errs.append(np.linalg.norm(Z - gt[i], axis=-1).mean())
    return float(np.mean(errs))


# ============================================================
# 兼容旧 API 的 pck (绝对阈值版, 保留用于消融)
# ============================================================
def pck(pred, gt, threshold=0.05):
    """绝对阈值 PCK (单位 m). 不是 DT-Pose 对应指标, 仅供内部参考."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if pred.ndim == 4:
        pred = pred.reshape(-1, pred.shape[-2], 3)
        gt = gt.reshape(-1, gt.shape[-2], 3)
    dist = np.linalg.norm(pred - gt, axis=-1)
    return float((dist < threshold).astype(np.float32).mean() * 100.0)


class PoseEvaluator:
    """评估器: 输出与 DT-Pose 可直接对比的所有指标."""

    def __init__(self, unit='meter'):
        self.unit = unit
        self.scale = 1000.0 if unit == 'meter' else 1.0

    def evaluate(self, pred, gt):
        # MPJPE 两个版本: 绝对 + hip 对齐
        mpjpe_abs = mpjpe(pred, gt, align_hip=False) * self.scale
        mpjpe_aligned = mpjpe(pred, gt, align_hip=True) * self.scale
        pa_mpjpe_val = pa_mpjpe(pred, gt) * self.scale

        # PCK 两个版本: DT-Pose 风格 (按身长归一化) + 绝对阈值
        pck_norm_50 = pck_normalized(pred, gt, thr=0.5)
        pck_norm_20 = pck_normalized(pred, gt, thr=0.2)
        pck_abs_50 = pck(pred, gt, threshold=0.05 if self.unit == 'meter' else 50.0)
        pck_abs_20 = pck(pred, gt, threshold=0.02 if self.unit == 'meter' else 20.0)

        return {
            'MPJPE (mm)': mpjpe_abs,
            'MPJPE_aligned (mm)': mpjpe_aligned,
            'PA-MPJPE (mm)': pa_mpjpe_val,
            'PCK@50_norm (%)': pck_norm_50,
            'PCK@20_norm (%)': pck_norm_20,
            'PCK@50_abs (%)': pck_abs_50,
            'PCK@20_abs (%)': pck_abs_20,
        }
