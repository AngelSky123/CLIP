"""
单样本测试可视化脚本 v2 — 修复坐标轴映射

MMFi/H36M 坐标系: dim0=X(左右), dim1=Y(身高/垂直), dim2=Z(前后/深度)
Matplotlib 3D:    X轴=左右, Y轴=前后, Z轴=垂直

映射: plot_X=data_X, plot_Y=data_Z, plot_Z=data_Y

用法:
  python visualize.py --checkpoint checkpoints/best_model.pth \
                      --env E04 --subject S31 --action A01 \
                      --frame 50 --save_dir viz_output
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import glob
from scipy.io import loadmat

from config import get_config
from models.full_model import CSIRSCPoseDG
from dataset import CSIPreprocessor
from evaluate import mpjpe, pa_mpjpe, pck, pck_normalized


# ==================== H36M Skeleton ====================
JOINT_NAMES = [
    'Hip', 'RHip', 'RKnee', 'RAnkle',
    'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck', 'Head',
    'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist'
]

BONES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

BONE_COLORS_GT = {
    (0, 1): '#2ca02c', (1, 2): '#d62728', (2, 3): '#d62728',
    (0, 4): '#2ca02c', (4, 5): '#1f77b4', (5, 6): '#1f77b4',
    (0, 7): '#2ca02c', (7, 8): '#2ca02c', (8, 9): '#2ca02c', (9, 10): '#2ca02c',
    (8, 11): '#1f77b4', (11, 12): '#1f77b4', (12, 13): '#1f77b4',
    (8, 14): '#d62728', (14, 15): '#d62728', (15, 16): '#d62728',
}

BONE_COLORS_PRED = {
    (0, 1): '#98df8a', (1, 2): '#ff9896', (2, 3): '#ff9896',
    (0, 4): '#98df8a', (4, 5): '#aec7e8', (5, 6): '#aec7e8',
    (0, 7): '#98df8a', (7, 8): '#98df8a', (8, 9): '#98df8a', (9, 10): '#98df8a',
    (8, 11): '#aec7e8', (11, 12): '#aec7e8', (12, 13): '#aec7e8',
    (8, 14): '#ff9896', (14, 15): '#ff9896', (15, 16): '#ff9896',
}


def detect_vertical_axis(joints):
    """Auto-detect which axis is vertical and whether to flip.
    
    Args:
        joints: (T, 17, 3) or (17, 3)
    Returns:
        axis_order: tuple (plot_x_idx, plot_y_idx, plot_z_idx)
        flip_z: bool — True if vertical axis needs to be negated
    """
    if joints.ndim == 3:
        j = joints[0]
    else:
        j = joints
    
    hip_to_head = j[10] - j[0]
    vertical_axis = np.argmax(np.abs(hip_to_head))
    
    # If Hip->Head is negative along vertical axis, we need to flip
    flip_z = hip_to_head[vertical_axis] < 0
    
    axes = [0, 1, 2]
    axes.remove(vertical_axis)
    
    return (axes[0], axes[1], vertical_axis), flip_z


def remap_joints(joints, axis_order, flip_z=False):
    """Remap joint coordinates to plotting axes.
    
    Args:
        joints: (..., 3)
        axis_order: (x_idx, y_idx, z_idx)
        flip_z: if True, negate the vertical axis (person upside down fix)
    Returns:
        remapped: (..., 3)
    """
    x_idx, y_idx, z_idx = axis_order
    z_data = joints[..., z_idx]
    if flip_z:
        z_data = -z_data
    return np.stack([joints[..., x_idx], 
                     joints[..., y_idx], 
                     z_data], axis=-1)


def load_single_sample(data_root, env, subject, action, start_frame=0, seq_len=64):
    """Load a single CSI sample and its ground truth."""
    csi_dir = os.path.join(data_root, env, subject, action, 'wifi-csi')
    gt_path = os.path.join(data_root, env, subject, action, 'ground_truth.npy')

    assert os.path.exists(csi_dir), f"CSI dir not found: {csi_dir}"
    assert os.path.exists(gt_path), f"GT not found: {gt_path}"

    num_frames = len(glob.glob(os.path.join(csi_dir, 'frame*.mat')))
    actual_len = min(seq_len, num_frames - start_frame)
    print(f"Loading {env}/{subject}/{action}: frames {start_frame}~{start_frame+actual_len-1} "
          f"(total {num_frames})")

    preprocessor = CSIPreprocessor()
    amps, phases = [], []
    for i in range(start_frame, start_frame + actual_len):
        frame_path = os.path.join(csi_dir, f'frame{i + 1:03d}.mat')
        if os.path.exists(frame_path):
            mat = loadmat(frame_path)
            amp = np.nan_to_num(mat['CSIamp'].astype(np.float32))
            pha = np.nan_to_num(mat['CSIphase'].astype(np.float32))
        else:
            amp = np.zeros((3, 114, 10), dtype=np.float32)
            pha = np.zeros((3, 114, 10), dtype=np.float32)
        amps.append(amp)
        phases.append(pha)

    amp = np.stack(amps, axis=0)
    phase = np.stack(phases, axis=0)
    csi = preprocessor.preprocess(amp, phase)

    gt = np.load(gt_path).astype(np.float32)
    gt_clip = gt[start_frame:start_frame + actual_len]
    # 绝对坐标 (与 dataset.py 一致, 不再做 root 减法)

    if actual_len < seq_len:
        pad_len = seq_len - actual_len
        csi = np.pad(csi, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='edge')
        gt_clip = np.pad(gt_clip, ((0, pad_len), (0, 0), (0, 0)), mode='edge')

    csi_tensor = torch.from_numpy(csi).unsqueeze(0)
    gt_tensor = torch.from_numpy(gt_clip).unsqueeze(0)
    return csi_tensor, gt_tensor, actual_len


def draw_skeleton_3d(ax, joints, bones, colors, label, alpha=1.0, lw=2.5, ms=5):
    """Draw 3D skeleton. joints should already be remapped to (plot_x, plot_y, plot_z)."""
    for bone in bones:
        i, j = bone
        ax.plot([joints[i, 0], joints[j, 0]],
                [joints[i, 1], joints[j, 1]],
                [joints[i, 2], joints[j, 2]],
                color=colors[bone], alpha=alpha, linewidth=lw)

    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               c='black', s=ms**2, alpha=alpha, zorder=5)
    ax.plot([], [], [], color=colors[bones[0]], label=label, linewidth=lw)


def set_axes_equal(ax, joints_list):
    """Set equal aspect ratio for 3D plot."""
    all_joints = np.concatenate(joints_list, axis=0)
    mid = all_joints.mean(axis=0)
    max_range = (all_joints.max(axis=0) - all_joints.min(axis=0)).max() / 2
    max_range = max(max_range, 0.3)

    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_zlabel('Z / Height (m)', fontsize=9)


def visualize_single_frame(gt, pred, frame_idx, save_path, axis_order, flip_z=False, metrics_text=""):
    """GT vs Pred from two views."""
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f'Frame {frame_idx}  {metrics_text}', fontsize=13, fontweight='bold')

    gt_f = remap_joints(gt[frame_idx], axis_order, flip_z)
    pred_f = remap_joints(pred[frame_idx], axis_order, flip_z)

    for idx, (elev, azim, title) in enumerate([
        (20, -60, 'Front View'),
        (80, -90, 'Top View'),
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        draw_skeleton_3d(ax, gt_f, BONES, BONE_COLORS_GT, 'Ground Truth',
                         alpha=0.9, lw=3, ms=6)
        draw_skeleton_3d(ax, pred_f, BONES, BONE_COLORS_PRED, 'Prediction',
                         alpha=0.7, lw=2.5, ms=5)
        for j in range(17):
            ax.plot([gt_f[j, 0], pred_f[j, 0]],
                    [gt_f[j, 1], pred_f[j, 1]],
                    [gt_f[j, 2], pred_f[j, 2]],
                    'k--', alpha=0.25, linewidth=0.8)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_title(title, fontsize=11)
        ax.view_init(elev=elev, azim=azim)
        set_axes_equal(ax, [gt_f, pred_f])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_error_analysis(gt, pred, actual_len, save_path):
    """Per-joint error heatmap + temporal + bar chart."""
    gt_np = gt[:actual_len]
    pred_np = pred[:actual_len]
    per_joint_error = np.linalg.norm(gt_np - pred_np, axis=-1) * 1000

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(per_joint_error.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax1.set_yticks(range(17))
    ax1.set_yticklabels(JOINT_NAMES, fontsize=8)
    ax1.set_xlabel('Frame', fontsize=10)
    ax1.set_title('Per-Joint Error Over Time (mm)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax1, shrink=0.8).set_label('Error (mm)', fontsize=9)

    ax2 = fig.add_subplot(gs[1, 0])
    mean_err = per_joint_error.mean(axis=1)
    ax2.plot(mean_err, color='#d62728', linewidth=1.5)
    ax2.fill_between(range(len(mean_err)), per_joint_error.min(axis=1),
                     per_joint_error.max(axis=1), alpha=0.2, color='#d62728')
    ax2.set_xlabel('Frame', fontsize=10)
    ax2.set_ylabel('MPJPE (mm)', fontsize=10)
    ax2.set_title('Temporal Error', fontsize=11, fontweight='bold')
    ax2.axhline(y=mean_err.mean(), color='gray', linestyle='--', alpha=0.5,
                label=f'Avg: {mean_err.mean():.1f}mm')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    joint_err = per_joint_error.mean(axis=0)
    colors = ['#d62728' if e > np.median(joint_err) else '#2ca02c' for e in joint_err]
    ax3.barh(range(17), joint_err, color=colors, alpha=0.8)
    ax3.set_yticks(range(17))
    ax3.set_yticklabels(JOINT_NAMES, fontsize=8)
    ax3.set_xlabel('Mean Error (mm)', fontsize=10)
    ax3.set_title('Per-Joint Mean Error', fontsize=11, fontweight='bold')
    ax3.axvline(x=joint_err.mean(), color='gray', linestyle='--', alpha=0.5)
    for i, v in enumerate(joint_err):
        ax3.text(v + 1, i, f'{v:.0f}', va='center', fontsize=7)
    ax3.grid(True, alpha=0.3, axis='x')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_multi_frame(gt, pred, actual_len, save_path, axis_order, flip_z=False, num_frames=8):
    """Multiple frames in a grid."""
    step = max(1, actual_len // num_frames)
    frames = list(range(0, actual_len, step))[:num_frames]

    ncols = min(4, len(frames))
    nrows = (len(frames) + ncols - 1) // ncols
    fig = plt.figure(figsize=(4.5 * ncols, 4.5 * nrows))
    fig.suptitle('Multi-Frame Skeleton (GT=solid, Pred=light)', fontsize=13, fontweight='bold')

    for plot_idx, f_idx in enumerate(frames):
        ax = fig.add_subplot(nrows, ncols, plot_idx + 1, projection='3d')
        gt_f = remap_joints(gt[f_idx], axis_order, flip_z)
        pred_f = remap_joints(pred[f_idx], axis_order, flip_z)

        draw_skeleton_3d(ax, gt_f, BONES, BONE_COLORS_GT, 'GT', alpha=0.9, lw=2, ms=4)
        draw_skeleton_3d(ax, pred_f, BONES, BONE_COLORS_PRED, 'Pred', alpha=0.6, lw=1.5, ms=3)

        err = np.linalg.norm(gt[f_idx] - pred[f_idx], axis=-1).mean() * 1000
        ax.set_title(f'F{f_idx} ({err:.0f}mm)', fontsize=9)
        ax.view_init(elev=20, azim=-60)
        set_axes_equal(ax, [gt_f, pred_f])
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_csi_input(csi_np, actual_len, save_path):
    """CSI input amplitude + phase heatmaps."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 8))
    fig.suptitle('CSI Input Visualization', fontsize=13, fontweight='bold')
    clip = csi_np[:actual_len]

    for ant in range(3):
        amp_map = clip[:, ant, :, :].mean(axis=-1)
        ax = axes[ant, 0]
        im = ax.imshow(amp_map.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_ylabel(f'Ant{ant+1}\nSubcarrier', fontsize=9)
        if ant == 0: ax.set_title('Amplitude', fontsize=11, fontweight='bold')
        if ant == 2: ax.set_xlabel('Frame', fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.7)

        phase_map = clip[:, 3 + ant, :, :].mean(axis=-1)
        ax = axes[ant, 1]
        im = ax.imshow(phase_map.T, aspect='auto', cmap='twilight', interpolation='nearest')
        if ant == 0: ax.set_title('Phase (sin)', fontsize=11, fontweight='bold')
        if ant == 2: ax.set_xlabel('Frame', fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Single sample visualization')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--data_root', type=str, default='/home/a123456/PerceptAlign/MMFi')
    parser.add_argument('--env', type=str, default='E04')
    parser.add_argument('--subject', type=str, default='S31')
    parser.add_argument('--action', type=str, default='A01')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--frame', type=int, default=30)
    parser.add_argument('--save_dir', type=str, default='viz_output')
    args_viz = parser.parse_args()

    os.makedirs(args_viz.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_args = get_config(inference_only=True)
    print(f"\n{'='*60}")
    print(f"Loading model from: {args_viz.checkpoint}")
    model = CSIRSCPoseDG(model_args).to(device)

    if os.path.exists(args_viz.checkpoint):
        ckpt = torch.load(args_viz.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
        if 'metrics' in ckpt and ckpt['metrics']:
            print(f"Checkpoint metrics: {ckpt['metrics']}")
    else:
        print(f"WARNING: Checkpoint not found, using random weights!")

    model.eval()

    print(f"\n{'='*60}")
    csi_tensor, gt_tensor, actual_len = load_single_sample(
        args_viz.data_root, args_viz.env, args_viz.subject, args_viz.action,
        args_viz.start_frame, model_args.seq_len
    )

    # Get action index from args
    action_idx_val = int(args_viz.action[1:]) - 1
    action_idx_tensor = torch.tensor([action_idx_val], dtype=torch.long, device=device)

    print(f"\nRunning inference (action={args_viz.action}, idx={action_idx_val})...")
    with torch.no_grad():
        # With GT action (oracle)
        outputs = model(csi_tensor.to(device), action_idx=action_idx_tensor)
        pred_tensor = outputs['p_final'].cpu()

    gt_np = gt_tensor[0].numpy()
    pred_np = pred_tensor[0].numpy()
    csi_np = csi_tensor[0].numpy()

    # Auto-detect vertical axis
    axis_order, flip_z = detect_vertical_axis(gt_np)
    axis_names = ['X', 'Y', 'Z']
    print(f"\nAuto-detected axes: "
          f"plot_X=data_{axis_names[axis_order[0]]}, "
          f"plot_Y=data_{axis_names[axis_order[1]]}, "
          f"plot_Z(vertical)=data_{axis_names[axis_order[2]]}")

    # Verify: print Hip->Head in remapped coords
    gt_remapped = remap_joints(gt_np[0], axis_order, flip_z)
    hip_head = gt_remapped[10] - gt_remapped[0]
    print(f"Hip->Head (remapped): [{hip_head[0]:.3f}, {hip_head[1]:.3f}, {hip_head[2]:.3f}]")
    print(f"Vertical component (Z) should be the largest: {hip_head[2]:.3f}")

    # Metrics
    print(f"\n{'='*60}")
    print("Evaluation Metrics:")
    gt_eval = gt_tensor[:, :actual_len]
    pred_eval = pred_tensor[:, :actual_len]
    mpjpe_val = mpjpe(pred_eval, gt_eval) * 1000
    pa_mpjpe_val = pa_mpjpe(pred_eval, gt_eval) * 1000
    pck50_val = pck_normalized(pred_eval, gt_eval, thr=0.5)
    pck20_val = pck_normalized(pred_eval, gt_eval, thr=0.2)
    frame_idx = min(args_viz.frame, actual_len - 1)
    frame_err = np.linalg.norm(gt_np[frame_idx] - pred_np[frame_idx], axis=-1).mean() * 1000

    print(f"  [Frame {frame_idx}]")
    print(f"    MPJPE:    {frame_err:.2f} mm")
    print(f"  [Sequence avg]")
    print(f"    MPJPE:    {mpjpe_val:.2f} mm")
    print(f"    PA-MPJPE: {pa_mpjpe_val:.2f} mm")
    print(f"    PCK@50_n: {pck50_val:.1f}%")
    print(f"    PCK@20_n: {pck20_val:.1f}%")

    metrics_text = f'MPJPE={mpjpe_val:.1f}mm  PA-MPJPE={pa_mpjpe_val:.1f}mm'
    prefix = f"{args_viz.env}_{args_viz.subject}_{args_viz.action}"

    print(f"\n{'='*60}")
    print("Generating visualizations...")

    visualize_single_frame(
        gt_np, pred_np, frame_idx,
        os.path.join(args_viz.save_dir, f'{prefix}_frame{frame_idx}_skeleton.png'),
        axis_order, flip_z,
        metrics_text=f'Frame MPJPE={frame_err:.1f}mm  |  Seq {metrics_text}'
    )

    visualize_multi_frame(
        gt_np, pred_np, actual_len,
        os.path.join(args_viz.save_dir, f'{prefix}_multi_frame.png'),
        axis_order, flip_z, num_frames=8
    )

    visualize_error_analysis(
        gt_np, pred_np, actual_len,
        os.path.join(args_viz.save_dir, f'{prefix}_error_analysis.png')
    )

    visualize_csi_input(
        csi_np, actual_len,
        os.path.join(args_viz.save_dir, f'{prefix}_csi_input.png')
    )

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {args_viz.save_dir}/")
    for f in sorted(os.listdir(args_viz.save_dir)):
        if f.endswith('.png'):
            print(f"  {f}")


if __name__ == '__main__':
    main()