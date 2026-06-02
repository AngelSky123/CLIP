"""
evaluate_v2.py  —  CSI-RSC-PoseDG 改进版评测模块  (v2, 已按真实仓库核对)

针对日志分析的两个根因:
  ① 评测不稳: 逐 epoch ±100mm 抖动里很大一部分是评测协议方差, 不是泛化能力
     的真实变化。-> multi_stride_variance 把噪声底直接量化出来 (mean±std)。
  ② 全局定位才是病灶: MPJPE 与 PA 异常解耦, 姿态结构 (MPJPE_a) 已收敛, 乱跳
     几乎全来自 hip(root) 的全局平移。-> 新增 hip_error 单独监控这个最不稳的量。

与 v1 草稿的关键修正 (核对仓库源码后):
  - 模型输出键是 'p_final' (不是 'pose'); 严格 DG 评测必须 action_idx=None。
  - root/hip 关节索引在本仓库到处都是 0 (HipPositionLoss / OutputDistillLoss
    hip_joint_idx=0 / evaluate.py 的 align / visualize.py JOINT_NAMES[0]='Hip')。
  - GT 单位是米, 标准指标走仓库自带的 PoseEvaluator(unit='meter') (内部 ×1000),
    保证 PA 等数值与你已报告的 ~106mm 完全同尺度, 不引入实现差异。
  - 标准指标 (MPJPE / MPJPE_a / PA / PCK) 一律复用 evaluate.PoseEvaluator,
    本模块只「额外」补 hip_error 和方差/聚合工具, 不重造轮子。

三种用法:
  A) 把 evaluate_v2() 直接替换 train_distill_pretrained.py 里的 evaluate():
     零新增基础设施, 立刻多打印一列 hip_err (+ 可选 per-action 分解)。
  B) 量化评测噪声底 (推荐先做这步):
     multi_stride_variance(student, data_root, 'E04', device, seq_len=64, scale=1000)
     -> 各指标 mean±std。若 mpjpe 的 σ ~ 30-50mm, 则 310 vs 316.8 的"超越"不显著。
  C) 用 make_holdout_split 从 E01-E03 切固定 val 子集做选点 (别在 E04 上挑 best)。
"""

import os
import glob
import numpy as np
from collections import defaultdict

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


# =====================================================================
# 工具
# =====================================================================
def _np(x):
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _get_pose(out):
    """从模型输出里取姿态张量。本仓库 student(csi, action_idx=None) 返回
    dict 且键为 'p_final', 形状 (B,T,17,3)。保留 tuple/其他键的兜底。"""
    if isinstance(out, dict):
        if 'p_final' in out:
            return out['p_final']
        # 兜底: 取第一个 4 维张量
        for v in out.values():
            if _HAS_TORCH and isinstance(v, torch.Tensor) and v.dim() == 4:
                return v
        return next(iter(out.values()))
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


# =====================================================================
# 一、唯一新增的核心指标: 纯全局定位误差
#     (其余标准指标全部复用仓库 evaluate.PoseEvaluator)
# =====================================================================
def hip_error(pred, gt, root_idx=0):
    """pred/gt: (..., J, 3)。返回 root 关节预测与真值的平均欧氏距离。
    这是 MPJPE 逐 epoch 乱跳的直接来源, 单独监控。单位 = 输入单位。"""
    pred, gt = _np(pred), _np(gt)
    return float(np.linalg.norm(
        pred[..., root_idx, :] - gt[..., root_idx, :], axis=-1).mean())


def mpjpe_raw(pred, gt):
    """标准 MPJPE (含全局平移), 仅用于无 PoseEvaluator 时的自包含路径。"""
    pred, gt = _np(pred), _np(gt)
    return float(np.linalg.norm(pred - gt, axis=-1).mean())


def mpjpe_aligned(pred, gt, root_idx=0):
    """hip 对齐后的 MPJPE (= 日志里的 MPJPE_a), 纯姿态结构误差。"""
    pred, gt = _np(pred), _np(gt)
    pred_a = pred - pred[..., root_idx:root_idx + 1, :]
    gt_a = gt - gt[..., root_idx:root_idx + 1, :]
    return float(np.linalg.norm(pred_a - gt_a, axis=-1).mean())


# =====================================================================
# 二、用法 A: 严格 DG 的 drop-in evaluate(), 复用 PoseEvaluator + 补 hip_err
#     可直接替换 train_distill_pretrained.py / train.py 里的 evaluate()。
# =====================================================================
def evaluate_v2(student, test_loader, device, evaluator, logger,
                root_idx=0, per_action=False):
    """与 train_distill_pretrained.py 的 evaluate() 同签名, 行为完全一致,
    只是日志里多了 hip_err 一列, 并把 hip_error 写进返回的 metrics。

    严格 DG: 模型只吃 CSI, action_idx=None; GT 仅用于算指标。
    evaluator: 传入仓库的 PoseEvaluator(unit='meter') (标准指标由它算, 同尺度)。
    """
    if not _HAS_TORCH:
        raise RuntimeError("evaluate_v2 需要 torch")
    student.eval()
    all_preds, all_gts, all_actions = [], [], []
    action_correct, action_total = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            csi = batch['csi'].to(device)
            pose_3d = batch['pose_3d'].to(device)
            outputs = student(csi, action_idx=None)
            pred = _get_pose(outputs)
            all_preds.append(pred.cpu())
            all_gts.append(pose_3d.cpu())
            if per_action:
                all_actions.extend(batch['action'])

            if isinstance(outputs, dict) and 'action_logits' in outputs:
                labels = torch.tensor(
                    [int(a[1:]) - 1 for a in batch['action']],
                    dtype=torch.long, device=device)
                action_correct += (outputs['action_logits'].argmax(-1) == labels).sum().item()
                action_total += labels.shape[0]

            del outputs, csi, pose_3d
            if device != 'cpu':
                torch.cuda.empty_cache()

    preds = torch.cat(all_preds)
    gts = torch.cat(all_gts)

    # 标准指标: 复用仓库 evaluator (米 -> mm 由它处理, 与已报告数值同尺度)
    metrics = evaluator.evaluate(preds, gts)
    # 额外补全局定位误差 (preds/gts 是米, ×1000 -> mm 与其余指标一致)
    metrics['hip_error (mm)'] = hip_error(preds, gts, root_idx) * 1000.0
    metrics['pred_std'] = preds.mean(dim=1).std(dim=0).mean().item() * 1000
    metrics['action_acc'] = 100.0 * action_correct / max(action_total, 1)

    logger.info(
        f'[Eval] MPJPE: {metrics["MPJPE (mm)"]:.2f}mm | '
        f'MPJPE_a: {metrics["MPJPE_aligned (mm)"]:.2f}mm | '
        f'hip_err: {metrics["hip_error (mm)"]:.2f}mm | '       # 新增: 盯这个
        f'PA: {metrics["PA-MPJPE (mm)"]:.2f}mm | '
        f'P50n: {metrics["PCK@50_norm (%)"]:.1f}% | '
        f'PredStd: {metrics["pred_std"]:.1f}mm | '
        f'ActAcc: {metrics["action_acc"]:.1f}%')

    if per_action and all_actions:
        # 看 hip_err 在哪些动作上爆掉 (定位误差通常和动作幅度相关)
        by_act = defaultdict(list)
        p_np, g_np = _np(preds), _np(gts)            # (N,T,J,3)
        for i, a in enumerate(all_actions):
            by_act[a].append(i)
        logger.info('  [per-action hip_err]')
        for a in sorted(by_act):
            idx = by_act[a]
            he = hip_error(p_np[idx], g_np[idx], root_idx) * 1000.0
            mp = mpjpe_raw(p_np[idx], g_np[idx]) * 1000.0
            logger.info(f'    {a}: MPJPE={mp:7.1f}  hip_err={he:7.1f}  n={len(idx)}')

    return metrics


# =====================================================================
# 三、用法 B: 序列级多 stride 重叠聚合 -> 量化评测噪声底
#     现在 test 用 stride=seq_len 无重叠, 边界帧只被预测一次、缺时序上下文 ->
#     方差大。这里按 (subject, action) 整段序列读盘, 用多个 stride 重叠推理,
#     对同一帧的多次预测求平均, 并报告各指标在 stride 间的 mean±std。
#
#     CSI 预处理直接复用 dataset.CSIPreprocessor, 与训练/测试输入完全一致。
# =====================================================================

_ENV_SUBJECTS = {
    'E01': range(1, 11), 'E02': range(11, 21),
    'E03': range(21, 31), 'E04': range(31, 41),
}


# --- 纯 numpy CSI 预处理 (复刻 dataset.CSIPreprocessor, 但去趋势用闭式解) ---
def _detrend_linear_np(x):
    """沿最后一维线性去趋势, 纯 numpy 闭式最小二乘。
    与 scipy.signal.detrend(type='linear') 逐元素一致 (误差 ~1e-15), 不碰 LAPACK。"""
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[-1]
    t = np.arange(N, dtype=np.float64)
    tc = t - t.mean()
    denom = (tc * tc).sum()
    xm = x.mean(axis=-1, keepdims=True)
    slope = (tc * (x - xm)).sum(axis=-1, keepdims=True) / (denom + 1e-12)
    return (x - (slope * tc + xm))


def _normalize_amplitude_np(amp):
    amin = amp.min(axis=(-2, -1), keepdims=True)
    amax = amp.max(axis=(-2, -1), keepdims=True)
    denom = amax - amin
    denom = np.where(denom < 1e-8, 1.0, denom)
    return np.nan_to_num((amp - amin) / denom, nan=0.0, posinf=0.0, neginf=0.0)


def _process_phase_np(phase):
    pu = np.unwrap(phase, axis=-2)
    shape = pu.shape
    pf = pu.reshape(-1, shape[-2])              # 与 CSIPreprocessor 同样的 reshape
    pd = _detrend_linear_np(pf).reshape(shape)
    sin_p, cos_p = np.sin(pd), np.cos(pd)
    return np.nan_to_num(np.concatenate([sin_p, cos_p], axis=1), nan=0.0)


def _preprocess_csi(amp, phase):
    """amp/phase: (T,3,114,10) -> (T,9,114,10) float32。等价于 CSIPreprocessor.preprocess。"""
    amp_norm = _normalize_amplitude_np(np.nan_to_num(amp.astype(np.float32)))
    phase_enc = _process_phase_np(np.nan_to_num(phase.astype(np.float32)))
    return np.concatenate([amp_norm, phase_enc], axis=1).astype(np.float32)


def iter_env_sequences(data_root, env, seq_len=64, num_actions=27):
    """按 (subject, action) 逐段 yield (seq_id, csi_full, gt_full)。
    csi_full: (T_total, 9, 114, 10) float32 ; gt_full: (T_total, 17, 3) (米)。
    预处理与 dataset.CSIPreprocessor 逐元素一致, 但用纯 numpy 的去趋势
    (闭式最小二乘), 不走 scipy.signal.detrend -> LAPACK, 因此不受某些
    scipy/numpy/BLAS 构建上 lstsq 崩溃 ('illegal value ... internal None') 的影响。"""
    from scipy.io import loadmat

    for sid in _ENV_SUBJECTS.get(env, []):
        subj = f'S{sid:02d}'
        for aid in range(1, num_actions + 1):
            act = f'A{aid:02d}'
            base = os.path.join(data_root, env, subj, act)
            csi_dir = os.path.join(base, 'wifi-csi')
            gt_path = os.path.join(base, 'ground_truth.npy')
            if not os.path.isdir(csi_dir) or not os.path.exists(gt_path):
                continue
            n = len(glob.glob(os.path.join(csi_dir, 'frame*.mat')))
            if n == 0:
                continue
            amps, phases = [], []
            for i in range(n):
                fp = os.path.join(csi_dir, f'frame{i + 1:03d}.mat')
                if os.path.exists(fp):
                    m = loadmat(fp)
                    amps.append(np.nan_to_num(m['CSIamp'].astype(np.float32)))
                    phases.append(np.nan_to_num(m['CSIphase'].astype(np.float32)))
                else:
                    amps.append(np.zeros((3, 114, 10), np.float32))
                    phases.append(np.zeros((3, 114, 10), np.float32))
            # 纯 numpy 预处理整段序列 (与 CSIPreprocessor 逐元素一致, 不碰 scipy detrend)
            csi = _preprocess_csi(np.stack(amps), np.stack(phases))  # (n,9,114,10)
            gt = np.load(gt_path).astype(np.float32)[:n]             # (n,17,3)
            yield f'{env}/{subj}/{act}', csi, gt


class _FrameAggregator:
    """把多个重叠窗口对同一全局帧的预测累加求平均。key=(seq_id, frame)。"""
    def __init__(self):
        self._sum = {}
        self._cnt = defaultdict(int)
        self._gt = {}

    def add(self, seq_id, start, pred_win, gt_win):
        pred_win, gt_win = _np(pred_win), _np(gt_win)
        for t in range(pred_win.shape[0]):
            key = (seq_id, start + t)
            if key not in self._sum:
                self._sum[key] = np.zeros_like(pred_win[t])
            self._sum[key] += pred_win[t]
            self._cnt[key] += 1
            self._gt[key] = gt_win[t]

    def finalize(self):
        keys = sorted(self._sum.keys())
        preds = np.stack([self._sum[k] / self._cnt[k] for k in keys], 0)
        gts = np.stack([self._gt[k] for k in keys], 0)
        return preds, gts


def _strides_for(seq_len, n_levels=3):
    out, s = [], seq_len
    for _ in range(n_levels):
        out.append(max(1, int(s)))
        s = s / 2
    return sorted(set(out))


def _metrics_from_arrays(preds, gts, root_idx=0, scale=1000.0):
    """自包含指标 (米输入, scale=1000 -> mm)。优先复用仓库 PoseEvaluator
    的标准指标, 失败则退回内部实现; hip_error 始终由本模块补。"""
    out = {}
    try:
        from evaluate import PoseEvaluator
        ev = PoseEvaluator(unit='meter')
        std = ev.evaluate(preds, gts)
        out['mpjpe'] = std['MPJPE (mm)']
        out['mpjpe_aligned'] = std['MPJPE_aligned (mm)']
        out['pa_mpjpe'] = std['PA-MPJPE (mm)']
        out['pck50_norm'] = std['PCK@50_norm (%)']
    except Exception:
        out['mpjpe'] = mpjpe_raw(preds, gts) * scale
        out['mpjpe_aligned'] = mpjpe_aligned(preds, gts, root_idx) * scale
    out['hip_error'] = hip_error(preds, gts, root_idx) * scale
    return out


def evaluate_sequences(student, data_root, env, device, seq_len=64,
                       strides=None, root_idx=0, scale=1000.0):
    """多 stride 重叠窗口评测 (聚合所有帧后算一次指标)。返回 dict。"""
    if not _HAS_TORCH:
        raise RuntimeError("evaluate_sequences 需要 torch")
    student.eval()
    if strides is None:
        strides = _strides_for(seq_len)

    agg = _FrameAggregator()
    with torch.no_grad():
        for seq_id, csi_full, gt_full in iter_env_sequences(data_root, env, seq_len):
            T_total = csi_full.shape[0]
            csi_t = torch.from_numpy(csi_full)
            for stride in strides:
                if T_total < seq_len:
                    starts = [0]
                else:
                    starts = list(range(0, T_total - seq_len + 1, stride))
                    if starts[-1] != T_total - seq_len:
                        starts.append(T_total - seq_len)
                for st in starts:
                    win = csi_t[st:st + seq_len].unsqueeze(0).to(device)
                    pred = _get_pose(student(win, action_idx=None)).squeeze(0)
                    pred = _np(pred)
                    actual_T = pred.shape[0]
                    agg.add(seq_id, st, pred, gt_full[st:st + actual_T])
    preds, gts = agg.finalize()
    return _metrics_from_arrays(preds, gts, root_idx, scale)


def multi_stride_variance(student, data_root, env, device, seq_len=64,
                          stride_list=None, root_idx=0, scale=1000.0):
    """对每个 stride 各跑一次 (单 stride, 非聚合), 报告各指标 mean±std。
    σ 就是评测协议的噪声底: 若 mpjpe 的 σ ~ 30-50mm, 那么 310 vs 316.8
    的"超越"在噪声之内、不显著 —— 这正是先量化噪声、再谈改进的依据。"""
    if stride_list is None:
        stride_list = _strides_for(seq_len)
    per = []
    for stride in stride_list:
        m = evaluate_sequences(student, data_root, env, device, seq_len,
                               strides=[stride], root_idx=root_idx, scale=scale)
        m['_stride'] = stride
        per.append(m)
    keys = [k for k in per[0] if not k.startswith('_')]
    summary = {}
    for k in keys:
        v = np.array([r[k] for r in per], dtype=np.float64)
        summary[k] = {'mean': float(v.mean()), 'std': float(v.std(ddof=0)),
                      'min': float(v.min()), 'max': float(v.max())}
    return {'per_stride': per, 'summary': summary}


def aggregate_runs(metric_dicts):
    """把多次评测 (多 epoch / 多 seed / 多 stride) 聚合成 mean±std。"""
    keys = [k for k in metric_dicts[0] if not k.startswith('_')]
    return {k: {'mean': float(np.mean([d[k] for d in metric_dicts])),
                'std': float(np.std([d[k] for d in metric_dicts], ddof=0))}
            for k in keys}


# =====================================================================
# 四、用法 C: 从 E01-E03 切 held-out val (按 subject), 别在 E04 上挑 best
# =====================================================================
def make_holdout_split(dataset, group_key_fn=None, val_ratio=0.2, seed=0):
    """按分组键 (默认 subject) 把 dataset 索引划成 (train_idx, val_idx), 组间不重叠。
    本仓库 dataset.samples[i]['subject'] 可用 -> 默认 group_key_fn 取它。"""
    if group_key_fn is None:
        def group_key_fn(ds, i):
            s = ds.samples[i]
            return s['subject'] if isinstance(s, dict) else s[1]
    groups = defaultdict(list)
    for i in range(len(dataset)):
        groups[group_key_fn(dataset, i)].append(i)
    keys = sorted(groups.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_ratio)))
    val_keys = set(keys[:n_val])
    train_idx, val_idx = [], []
    for k, idxs in groups.items():
        (val_idx if k in val_keys else train_idx).extend(idxs)
    return sorted(train_idx), sorted(val_idx)


# =====================================================================
# 日志格式化
# =====================================================================
def format_variance(summary, prefix=''):
    return '\n'.join(
        f"{prefix}{k:14s}: {v['mean']:7.2f} ± {v['std']:5.2f} "
        f"[{v['min']:.2f}, {v['max']:.2f}]"
        for k, v in summary.items())


# =====================================================================
# 自检 (不依赖仓库/GPU)
# =====================================================================
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    gt = rng.randn(64, 17, 3) * 0.3                  # 米
    noise = rng.randn(64, 17, 3) * 0.04
    trans = rng.randn(64, 1, 3) * 0.2                # 注入全局平移误差
    pred = gt + noise + trans

    he = hip_error(pred, gt) * 1000
    ma = mpjpe_aligned(pred, gt) * 1000
    mp = mpjpe_raw(pred, gt) * 1000
    print(f"[self-test] MPJPE={mp:.1f}  MPJPE_a={ma:.1f}  hip_err={he:.1f} (mm)")
    print(f"[self-test] 注入了全局平移 -> hip_err({he:.1f}) 应 >> 结构误差 MPJPE_a({ma:.1f})")
    assert he > ma, "hip_error 未能隔离全局定位误差"

    # _get_pose: 模拟本仓库 dict 输出
    class _Stub:
        def eval(self): pass
        def __call__(self, win, action_idx=None):
            T = win.shape[1] if hasattr(win, 'shape') else 64
            return {'p_final': np.zeros((1, T, 17, 3)), 'action_logits': None}
    p = _get_pose(_Stub()(type('W', (), {'shape': (1, 8)})()))
    assert np.asarray(p).shape == (1, 8, 17, 3)
    print("[self-test] _get_pose 取 'p_final' 正确, 形状 (1,8,17,3)")
    print("[ALL OK]")