"""
eval_dtpose_faithful.py — 与 DT-Pose 逐帧评测协议严格对齐的最终评测脚本。

设计 (解读 A):
  * 模型仍然吃 seq_len=64 帧上下文 (保留 GlobalTemporalModeler/TCN 的时序建模,
    这是本方法的卖点, 不退化成单帧)。
  * 但评测的「打分粒度」与 DT-Pose 完全一致:
      - E04 (S31-S40) 全部 27 个动作、每个序列的【全部帧】都进入 MPJPE 池;
      - 每个全局帧【恰好被预测一次】(无重叠铺窗 + 尾窗只补未覆盖帧),
        不做重叠平均 (重叠平均是 DT-Pose 没有的平滑, 会引入不可比的偏差);
      - 不做 edge-padding (padding 帧会污染 MPJPE);
      - 严格 DG: action_idx=None, 不使用任何测试集 GT 标签作为输入。
  * 标准指标 (MPJPE / PA-MPJPE / PCK) 一律复用仓库 evaluate.PoseEvaluator,
    其公式已与 DT-Pose utils.calulate_error / compute_similarity_transform 核对一致,
    所有帧拼成一个大数组【一次性】计算 = DT-Pose 的「逐帧入池、等权平均」。

与 DT-Pose 评测唯一剩下的差异: 本模型每帧的预测使用了 64 帧时序上下文,
而 DT-Pose 是逐帧 (无时序上下文)。这是方法层面的差异, 应作为贡献写进论文,
不是评测口径不一致。

用法:
    python eval_dtpose_faithful.py \
        --data_root /home/a123456/PerceptAlign/MMFi \
        --ckpt ./checkpoints/distill_hipout/<run>/best_mpjpe_ema.pth \
        --test_env E04 --seq_len 64 \
        --variance        # 可选: 额外量化评测协议噪声底 (mean±std over strides)
"""
import os
import sys
import argparse
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import CSIRSCPoseDG
from evaluate import PoseEvaluator
# 复用 v2 里已经写好、且与 dataset.CSIPreprocessor 逐元素一致的整段读盘/预处理,
# 以及 hip_error / 多 stride 方差工具。
from evaluate_v2 import (
    iter_env_sequences, _get_pose, _np, hip_error,
    multi_stride_variance, format_variance,
)

# DT-Pose 论文 Table 1, Setting 3 (Cross-Environment) × Protocol 3 (all 27)
DTPOSE_MPJPE = 316.8
DTPOSE_PA = 104.2


# ----------------------------------------------------------------------
# 模型构建: 复刻 train_distill_pretrained.py 的架构默认值
# (该 run 的所有架构维度都用的是默认值; 如你改过, 用 --override 同步)
# ----------------------------------------------------------------------
def build_model_args(seq_len):
    return SimpleNamespace(
        amp_channels=3, phase_channels=6,
        encoder_hidden_dim=32, encoder_out_dim=64,
        local_hidden_dim=64, local_out_dim=64, num_res3d_blocks=2,
        global_dim=128, num_transformer_layers=3, num_heads=4,
        tcn_channels=[128, 128], tcn_kernel_size=3, transformer_dropout=0.3,
        coarse_hidden_dim=256, gcn_hidden_dim=128, num_gcn_layers=3,
        num_joints=17, num_actions=27,
        rsc2_time_drop_pct=0.5, rsc2_channel_drop_pct=0.5, rsc2_batch_pct=0.5,
        seq_len=seq_len, use_vision_backbone=False,
    )


def load_student(ckpt_path, seq_len, device):
    margs = build_model_args(seq_len)
    student = CSIRSCPoseDG(margs).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' not in ckpt:
        raise KeyError(f"ckpt 缺少 'model_state_dict'; got {list(ckpt.keys())}")
    miss, unexp = student.load_state_dict(ckpt['model_state_dict'], strict=True)
    # strict=True 下若有不匹配会直接抛错; 这里仅作信息打印兜底
    if miss or unexp:
        raise RuntimeError(f"state_dict 不匹配: missing={miss[:5]} unexpected={unexp[:5]}")
    student.eval()
    saved = ckpt.get('metrics', {})
    return student, ckpt.get('epoch', None), saved


# ----------------------------------------------------------------------
# 核心: 整段序列预测, 每个全局帧恰好预测一次 (无重叠铺窗 + 尾窗补未覆盖帧)
# ----------------------------------------------------------------------
@torch.no_grad()
def predict_full_sequence(student, csi_full, seq_len, device):
    """csi_full: (T, 9, 114, 10) -> preds: (T, 17, 3), 每帧恰好一个预测。

    铺窗策略 (T=297, seq_len=64 为例):
      非重叠窗 [0:64],[64:128],[128:192],[192:256], 再补尾窗 [233:297],
      但尾窗只写入 256-296 这些尚未覆盖的帧 (233-255 已由 [192:256] 写入)。
    => 全 297 帧覆盖、每帧一次、零 padding。
    末帧附近因此拥有完整左侧时序上下文。
    """
    T = csi_full.shape[0]
    csi_t = torch.from_numpy(csi_full)
    preds = np.zeros((T, 17, 3), dtype=np.float64)
    covered = np.zeros(T, dtype=bool)

    if T <= seq_len:
        # 短序列: 整段一次喂入, 不 padding。模型对变长 T 是兼容的。
        win = csi_t.unsqueeze(0).to(device)
        p = _np(_get_pose(student(win, action_idx=None)).squeeze(0))  # (T,17,3)
        preds[:] = p[:T]
        return preds

    starts = list(range(0, T - seq_len + 1, seq_len))   # 非重叠铺窗
    if starts[-1] + seq_len < T:
        starts.append(T - seq_len)                       # 结束于末帧的尾窗

    for st in starts:
        win = csi_t[st:st + seq_len].unsqueeze(0).to(device)
        p = _np(_get_pose(student(win, action_idx=None)).squeeze(0))  # (seq_len,17,3)
        for t in range(p.shape[0]):
            g = st + t
            if g < T and not covered[g]:
                preds[g] = p[t]
                covered[g] = True
        if device != 'cpu':
            torch.cuda.empty_cache()

    if not covered.all():
        raise RuntimeError(f"有 {int((~covered).sum())} 帧未被覆盖, 铺窗逻辑有误")
    return preds


@torch.no_grad()
def evaluate_dtpose_faithful(student, data_root, env, seq_len, device):
    """对 env 的全部 (subject, action) 序列做逐帧评测, 所有帧拼池一次性算指标。"""
    student.eval()
    all_p, all_g = [], []
    n_seq, n_frames = 0, 0
    for seq_id, csi_full, gt_full in iter_env_sequences(data_root, env, seq_len):
        n = min(csi_full.shape[0], gt_full.shape[0])   # csi/gt 帧数对齐兜底
        if n == 0:
            continue
        csi_full, gt_full = csi_full[:n], gt_full[:n]
        p = predict_full_sequence(student, csi_full, seq_len, device)[:n]
        all_p.append(p)
        all_g.append(gt_full.astype(np.float64))
        n_seq += 1
        n_frames += n

    if n_seq == 0:
        raise RuntimeError(f"{env} 下没有读到任何序列, 检查 data_root")

    preds = np.concatenate(all_p, 0)   # (N_total_frames, 17, 3)
    gts = np.concatenate(all_g, 0)

    ev = PoseEvaluator(unit='meter')   # 公式与 DT-Pose 一致, 米 -> mm 内部 ×1000
    m = ev.evaluate(preds, gts)
    m['hip_error (mm)'] = hip_error(preds, gts) * 1000.0
    return m, n_seq, n_frames


def main():
    ap = argparse.ArgumentParser(description='DT-Pose 逐帧协议对齐的最终评测 (解读A)')
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--ckpt', type=str, required=True,
                    help='推荐 best_mpjpe_ema.pth')
    ap.add_argument('--test_env', type=str, default='E04')
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--variance', action='store_true',
                    help='额外跑 multi_stride_variance, 量化评测协议噪声底 (mean±std)')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    student, epoch, saved = load_student(args.ckpt, args.seq_len, device)
    print('=' * 70)
    print(f'Loaded: {args.ckpt}  (epoch={epoch})')
    if isinstance(saved, dict) and 'e04' in saved:
        e04 = saved['e04']
        print(f'  训练时该 ckpt 在 E04 的【滑窗】监控值: '
              f"MPJPE={e04.get('MPJPE (mm)', float('nan')):.2f}  "
              f"PA={e04.get('PA-MPJPE (mm)', float('nan')):.2f}")
    print('=' * 70)

    print(f'[逐帧评测] env={args.test_env} seq_len={args.seq_len} '
          f'(每帧一次 / 全帧覆盖 / 无padding / action_idx=None)')
    m, n_seq, n_frames = evaluate_dtpose_faithful(
        student, args.data_root, args.test_env, args.seq_len, device)

    mpjpe = m['MPJPE (mm)']
    pa = m['PA-MPJPE (mm)']
    print(f'  序列数={n_seq}  总帧数={n_frames}')
    print(f'  MPJPE        : {mpjpe:.2f} mm')
    print(f'  MPJPE_aligned: {m["MPJPE_aligned (mm)"]:.2f} mm')
    print(f'  PA-MPJPE     : {pa:.2f} mm')
    print(f'  hip_error    : {m["hip_error (mm)"]:.2f} mm')
    print(f'  PCK@50_norm  : {m["PCK@50_norm (%)"]:.1f} %')
    print(f'  PCK@20_norm  : {m["PCK@20_norm (%)"]:.1f} %')
    print('-' * 70)
    print(f'  DT-Pose (S3,P3): MPJPE={DTPOSE_MPJPE}  PA={DTPOSE_PA}')
    print(f'  ΔMPJPE = {mpjpe - DTPOSE_MPJPE:+.2f} mm   '
          f'ΔPA = {pa - DTPOSE_PA:+.2f} mm')
    print('=' * 70)

    if args.variance:
        print('\n[噪声底] multi_stride_variance (各 stride 单独评一次, 报 mean±std)')
        print('  注: 此处用的是重叠平均聚合, 仅用于量化「评测口径方差」,')
        print('      不是上面的逐帧口径; 看 mpjpe 的 σ 判断领先是否显著。')
        res = multi_stride_variance(student, args.data_root, args.test_env,
                                    device, seq_len=args.seq_len)
        print(format_variance(res['summary'], prefix='    '))
        sigma = res['summary'].get('mpjpe', {}).get('std', float('nan'))
        lead = DTPOSE_MPJPE - mpjpe
        print('-' * 70)
        if not np.isnan(sigma):
            verdict = ('显著 (领先 > σ)' if lead > sigma else
                       '不显著 (领先 ≤ σ, 只能写"持平")')
            print(f'  领先量 {lead:+.2f} mm  vs  σ_mpjpe {sigma:.2f} mm  ->  {verdict}')


if __name__ == '__main__':
    main()