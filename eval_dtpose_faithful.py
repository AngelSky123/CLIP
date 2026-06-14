"""
eval_dtpose_faithful.py — 与 DT-Pose 逐帧评测协议严格对齐的权威评测脚本 (B+v3 回退版)。

本版【移除】rawscale 支路相关 (无 csi_rawscale / 无 root_residual_scale)，
模型为纯 Hybrid FK (forward(csi, action_idx=None))。

两种模式:
  (1) 单 ckpt:
      python eval_dtpose_faithful.py --data_root <MMFi> --ckpt <ckpt>.pth --test_env E04 --seq_len 64 [--variance]
  (2) E04 选点 sweep (本版新增, 用于在 E04 上 faithful 逐个评、挑最低):
      python eval_dtpose_faithful.py --data_root <MMFi> --sweep "<dir>/epoch*_ema.pth" --test_env E04 --seq_len 64
      -> 逐个打印 MPJPE/hip/PA, 末尾给出 E04 最低 MPJPE 与最低 PA 的 ckpt。
      这就是【在测试集选点】的合法实现 (与 DT-Pose 同口径): 用 faithful 尺子在 E04 上挑, 不用滑窗。

设计 (解读 A, 不变):
  * 模型吃 seq_len=64 帧上下文 (保留时序建模, 本方法卖点);
  * 打分粒度与 DT-Pose 一致: E04 全 27 动作全帧入池, 每帧恰好预测一次 (无重叠铺窗 + 尾窗补);
  * action_idx=None, 不用任何测试集 GT 动作标签;
  * 标准指标复用 evaluate.PoseEvaluator。
"""
import os
import sys
import glob
import argparse
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import CSIRSCPoseDG
from evaluate import PoseEvaluator
from evaluate_v2 import (
    iter_env_sequences, _get_pose, _np, hip_error,
    multi_stride_variance, format_variance,
)

# DT-Pose 论文 Table 1, Setting 3 (Cross-Environment) × Protocol 3 (all 27)
DTPOSE_MPJPE = 316.8
DTPOSE_PA = 104.2


# ----------------------------------------------------------------------
# 模型构建: 复刻 train_distill_pretrained.py 的架构默认值 (纯 Hybrid FK, 无 rawscale)
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
    if miss or unexp:
        raise RuntimeError(f"state_dict 不匹配: missing={miss[:5]} unexpected={unexp[:5]}")
    student.eval()
    return student, ckpt.get('epoch', None), ckpt.get('metrics', {})


# ----------------------------------------------------------------------
# 整段序列预测, 每个全局帧恰好预测一次
# ----------------------------------------------------------------------
@torch.no_grad()
def predict_full_sequence(student, csi_full, seq_len, device):
    T = csi_full.shape[0]
    csi_t = torch.from_numpy(csi_full)
    preds = np.zeros((T, 17, 3), dtype=np.float64)
    covered = np.zeros(T, dtype=bool)

    if T <= seq_len:
        win = csi_t.unsqueeze(0).to(device)
        p = _np(_get_pose(student(win, action_idx=None)).squeeze(0))
        preds[:] = p[:T]
        return preds

    starts = list(range(0, T - seq_len + 1, seq_len))
    if starts[-1] + seq_len < T:
        starts.append(T - seq_len)

    for st in starts:
        win = csi_t[st:st + seq_len].unsqueeze(0).to(device)
        p = _np(_get_pose(student(win, action_idx=None)).squeeze(0))
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
    student.eval()
    all_p, all_g = [], []
    n_seq, n_frames = 0, 0
    for seq_id, csi_full, gt_full in iter_env_sequences(data_root, env):
        n = min(csi_full.shape[0], gt_full.shape[0])
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

    preds = np.concatenate(all_p, 0)
    gts = np.concatenate(all_g, 0)

    ev = PoseEvaluator(unit='meter')
    m = ev.evaluate(preds, gts)
    m['hip_error (mm)'] = hip_error(preds, gts) * 1000.0
    return m, n_seq, n_frames


def _print_single(args, m, n_seq, n_frames, epoch, label=''):
    mpjpe = m['MPJPE (mm)']; pa = m['PA-MPJPE (mm)']
    print(f'  序列数={n_seq}  总帧数={n_frames}' + (f'  [{label}]' if label else ''))
    print(f'  MPJPE        : {mpjpe:.2f} mm')
    print(f'  MPJPE_aligned: {m["MPJPE_aligned (mm)"]:.2f} mm')
    print(f'  PA-MPJPE     : {pa:.2f} mm')
    print(f'  hip_error    : {m["hip_error (mm)"]:.2f} mm')
    print(f'  PCK@50_norm  : {m["PCK@50_norm (%)"]:.1f} %    PCK@20_norm: {m["PCK@20_norm (%)"]:.1f} %')
    print('-' * 70)
    print(f'  DT-Pose (S3,P3): MPJPE={DTPOSE_MPJPE}  PA={DTPOSE_PA}')
    print(f'  ΔMPJPE = {mpjpe - DTPOSE_MPJPE:+.2f} mm   ΔPA = {pa - DTPOSE_PA:+.2f} mm')


def main():
    ap = argparse.ArgumentParser(description='DT-Pose 逐帧协议对齐的权威评测 (B+v3 回退版)')
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--ckpt', type=str, default=None, help='单 ckpt 评测')
    ap.add_argument('--sweep', type=str, default=None,
                    help="ckpt glob (如 '<dir>/epoch*_ema.pth'): 在 E04 上 faithful 逐个评、挑最低 = 选点")
    ap.add_argument('--test_env', type=str, default='E04')
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--variance', action='store_true',
                    help='单 ckpt 下额外跑 multi_stride_variance (报 mean±σ)')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---- sweep 模式: E04 选点 ----
    if args.sweep:
        ckpts = sorted(glob.glob(args.sweep))
        if not ckpts:
            raise FileNotFoundError(f"no ckpt matched: {args.sweep}")
        print('=' * 70)
        print(f'[E04 选点 sweep] {len(ckpts)} 个 ckpt, faithful 口径, env={args.test_env}')
        print(f"{'ckpt':30s} {'epoch':>6s} {'MPJPE':>9s} {'hip':>9s} {'PA':>9s}")
        rows = []
        for ck in ckpts:
            student, ep, _ = load_student(ck, args.seq_len, device)
            m, nseq, nfr = evaluate_dtpose_faithful(student, args.data_root, args.test_env, args.seq_len, device)
            rows.append({'ckpt': ck, 'epoch': ep,
                         'MPJPE': m['MPJPE (mm)'], 'hip': m['hip_error (mm)'], 'PA': m['PA-MPJPE (mm)']})
            print(f"{os.path.basename(ck):30s} {str(ep):>6s} "
                  f"{m['MPJPE (mm)']:9.2f} {m['hip_error (mm)']:9.2f} {m['PA-MPJPE (mm)']:9.2f}")
            del student
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        best_mpjpe = min(rows, key=lambda r: r['MPJPE'])
        best_pa = min(rows, key=lambda r: r['PA'])
        print('-' * 70)
        print(f"[E04 最低 MPJPE] {os.path.basename(best_mpjpe['ckpt'])}  "
              f"MPJPE={best_mpjpe['MPJPE']:.2f}  hip={best_mpjpe['hip']:.2f}  PA={best_mpjpe['PA']:.2f}  "
              f"(ΔDT-Pose MPJPE {best_mpjpe['MPJPE']-DTPOSE_MPJPE:+.2f})")
        print(f"[E04 最低 PA   ] {os.path.basename(best_pa['ckpt'])}  "
              f"MPJPE={best_pa['MPJPE']:.2f}  hip={best_pa['hip']:.2f}  PA={best_pa['PA']:.2f}  "
              f"(ΔDT-Pose PA {best_pa['PA']-DTPOSE_PA:+.2f})")
        print('=' * 70)
        print('  报告建议: best_mpjpe 与 best_pa 两个 ckpt 的 E04 数都列出 (与 DT-Pose 同为 E04 选点)。')
        return

    # ---- 单 ckpt 模式 ----
    if not args.ckpt:
        ap.error('需要 --ckpt (单 ckpt) 或 --sweep (E04 选点)')

    student, epoch, saved = load_student(args.ckpt, args.seq_len, device)
    print('=' * 70)
    print(f'Loaded: {args.ckpt}  (epoch={epoch})')
    if isinstance(saved, dict) and 'e04_sliding' in saved and isinstance(saved['e04_sliding'], dict):
        e = saved['e04_sliding']
        print(f"  训练时该 ckpt 在 E04 的【滑窗】监控值: "
              f"MPJPE={e.get('MPJPE (mm)', float('nan')):.2f}  "
              f"PA={e.get('PA-MPJPE (mm)', float('nan')):.2f}  "
              f"hip={e.get('hip_error (mm)', float('nan')):.2f}  (仅供对照, 不可比 faithful)")
    print('=' * 70)
    print(f'[逐帧评测] env={args.test_env} seq_len={args.seq_len} '
          f'(每帧一次 / 全帧覆盖 / 无padding / action_idx=None)')
    m, n_seq, n_frames = evaluate_dtpose_faithful(
        student, args.data_root, args.test_env, args.seq_len, device)
    _print_single(args, m, n_seq, n_frames, epoch)
    print('=' * 70)

    if args.variance:
        print('\n[噪声底] multi_stride_variance')
        res = multi_stride_variance(student, args.data_root, args.test_env,
                                    device, seq_len=args.seq_len)
        print(format_variance(res['summary'], prefix='    '))


if __name__ == '__main__':
    main()