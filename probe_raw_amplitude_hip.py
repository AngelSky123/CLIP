"""
probe_raw_amplitude_hip.py  (v2: 数值稳健版)
============================================
决定性探针：原始绝对幅度里到底有没有 E04 hip 的绝对定位信息？

v2 修复 (v1 因原始 CSIamp 含 inf 导致 raw_abs 溢出成 NaN, 误报"天花板"):
  - 原始幅度先做 posinf/neginf/nan 清理 + 稳健分位裁剪, 杜绝溢出。
  - 标准化输出再 clip 一道, 残余极值不污染回归。
  - 新增 raw_log 一路: log1p(裁剪后绝对幅度) —— 物理上 log 功率对距离更线性、
    且天然压缩动态范围, 抗尺度爆炸。这一路往往比线性 raw_abs 更能体现真实信息量。

四种输入特征, 闭式岭回归在 E01-E03 训练、E04 测试 hip(joint0) 绝对 xyz:
  (A) raw_abs  : 原始绝对幅度 (3,114,10) 沿 packet 均值 -> (3,114)=342 维, 保留绝对尺度
  (B) raw_log  : log1p(原始绝对幅度) 同样处理 (log 功率)
  (C) norm_abs : dataset.py 那套逐帧 min-max 后同样处理 (你现在的做法)
  (D) mean_base: 永远预测训练集 hip 均值 (零信息标尺)

判读看 E04 hip MPJPE(mm): raw_abs/raw_log 若显著低于 norm_abs 和 mean_base
-> 预处理删了定位信号, 冲 316 有杠杆; 若都贴着 mean_base -> 真·跨域天花板。

用法:
    python probe_raw_amplitude_hip.py \
        --data_root /home/a123456/PerceptAlign/MMFi \
        --train_envs E01 E02 E03 --test_env E04 \
        --max_frames_per_seq 60 --ridge 1e-2
"""
import os, sys, glob, argparse
import numpy as np
from scipy.io import loadmat

ENV_SUBJECTS = {'E01': range(1, 11), 'E02': range(11, 21),
                'E03': range(21, 31), 'E04': range(31, 41)}
HIP_JOINT = 0
AMP_CAP = 1e4   # 原始幅度稳健上限, 超出视为饱和/损坏


def iter_sequences(data_root, envs):
    for env in envs:
        for sid in ENV_SUBJECTS[env]:
            subj = f'S{sid:02d}'
            for aid in range(1, 28):
                act = f'A{aid:02d}'
                csi_dir = os.path.join(data_root, env, subj, act, 'wifi-csi')
                gt_path = os.path.join(data_root, env, subj, act, 'ground_truth.npy')
                if not os.path.isdir(csi_dir) or not os.path.exists(gt_path):
                    continue
                n = len(glob.glob(os.path.join(csi_dir, 'frame*.mat')))
                if n:
                    yield env, subj, act, csi_dir, gt_path, n


def clean_amp(amp):
    """inf/nan -> 0, 再裁剪到 [0, AMP_CAP], 全 float64。"""
    amp = np.asarray(amp, np.float64)
    amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(amp, 0.0, AMP_CAP, out=amp)
    return amp


def norm_amp_per_frame(amp):
    amin = amp.min(axis=(-2, -1), keepdims=True)
    amax = amp.max(axis=(-2, -1), keepdims=True)
    denom = np.where((amax - amin) < 1e-8, 1.0, amax - amin)
    return (amp - amin) / denom


def build_features(data_root, envs, max_frames_per_seq):
    Xr, Xl, Xn, Y = [], [], [], []
    n_seq, n_bad = 0, 0
    for env, subj, act, csi_dir, gt_path, n in iter_sequences(data_root, envs):
        gt = np.load(gt_path).astype(np.float64)
        F = min(n, gt.shape[0])
        idxs = (np.linspace(0, F - 1, max_frames_per_seq).astype(int)
                if max_frames_per_seq and F > max_frames_per_seq else np.arange(F))
        for i in idxs:
            fp = os.path.join(csi_dir, f'frame{i + 1:03d}.mat')
            if not os.path.exists(fp):
                continue
            try:
                raw = loadmat(fp)['CSIamp']
            except Exception:
                n_bad += 1; continue
            if np.asarray(raw).shape != (3, 114, 10):
                n_bad += 1; continue
            amp = clean_amp(raw)                       # (3,114,10), 干净
            Xr.append(amp.mean(-1).reshape(-1))        # 线性绝对幅度
            Xl.append(np.log1p(amp).mean(-1).reshape(-1))  # log 功率
            Xn.append(norm_amp_per_frame(amp).mean(-1).reshape(-1))  # 逐帧归一化
            Y.append(gt[i, HIP_JOINT, :])
        n_seq += 1
    print(f"  [{'+'.join(envs)}] sequences={n_seq}  frames={len(Y)}  bad={n_bad}")
    return (np.asarray(Xr), np.asarray(Xl), np.asarray(Xn), np.asarray(Y))


def standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    f = lambda X: np.clip((X - mu) / sd, -10.0, 10.0)
    return f(Xtr), f(Xte)


def ridge_fit(X, Y, lam):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], 1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ Y)


def predict(X, W):
    return np.concatenate([X, np.ones((X.shape[0], 1))], 1) @ W


def hip_mm(pred, gt):
    return float(np.sqrt(((pred - gt) ** 2).sum(1)).mean() * 1000.0)


def run(name, Xtr, Xte, Ytr, Yte, lam):
    a, b = standardize(Xtr, Xte)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        print(f"  {name:10s}  特征仍含非有限值, 跳过"); return None
    W = ridge_fit(a, Ytr, lam)
    ein, eout = hip_mm(predict(a, W), Ytr), hip_mm(predict(b, W), Yte)
    print(f"  {name:10s}  held-in(E01-03)={ein:7.1f} mm   E04(test)={eout:7.1f} mm")
    return eout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--train_envs', nargs='+', default=['E01', 'E02', 'E03'])
    ap.add_argument('--test_env', default='E04')
    ap.add_argument('--max_frames_per_seq', type=int, default=60)
    ap.add_argument('--ridge', type=float, default=1e-2)
    a = ap.parse_args()

    print("=" * 66)
    print("  原始绝对幅度 vs 归一化幅度 —— E04 hip 绝对定位信息探针 (v2)")
    print("=" * 66)
    print("Loading TRAIN ...")
    Xr, Xl, Xn, Y = build_features(a.data_root, a.train_envs, a.max_frames_per_seq)
    print("Loading TEST  ...")
    Xr_t, Xl_t, Xn_t, Y_t = build_features(a.data_root, [a.test_env], a.max_frames_per_seq)
    if len(Y) == 0 or len(Y_t) == 0:
        print("!! 没读到数据"); sys.exit(1)

    mh = Y.mean(0, keepdims=True)
    bi, bo = hip_mm(np.repeat(mh, len(Y), 0), Y), hip_mm(np.repeat(mh, len(Y_t), 0), Y_t)
    print("\n--- E04 hip MPJPE (越低=信息越多) ---")
    print(f"  {'mean_base':10s}  held-in(E01-03)={bi:7.1f} mm   E04(test)={bo:7.1f} mm   <- 零信息标尺")
    e_raw = run("raw_abs", Xr, Xr_t, Y, Y_t, a.ridge)
    e_log = run("raw_log", Xl, Xl_t, Y, Y_t, a.ridge)
    e_norm = run("norm_abs", Xn, Xn_t, Y, Y_t, a.ridge)

    print("\n--- 判读 ---")
    best = min([x for x in (e_raw, e_log) if x is not None], default=None)
    if best is None:
        print("  raw 两路都没算出有效数, 检查数据。")
    elif best < bo - 15 and (e_norm is None or best < e_norm - 10):
        print(f"  raw 最好 {best:.1f}mm, 比 mean_base 低 {bo-best:.1f}mm、比 norm_abs 低 {(e_norm-best) if e_norm else float('nan'):.1f}mm")
        print("  => 预处理删了可用 E04 定位信号, 冲 316 有真实杠杆。")
        print("     下一步: 给网络补一路*保留绝对尺度*的幅度输入(全局统一缩放, 不逐帧 min-max), 重训蒸馏。")
    elif best < bo - 15:
        print(f"  raw 比 mean_base 低 {bo-best:.1f}mm, 但和 norm_abs 差不多 -> 信息有限, 期望不高。")
    else:
        print(f"  raw 最好仅比 mean_base 低 {bo-best:.1f}mm (≈贴零线) -> 真·跨域天花板, 冲 316 到此为止。")
    print("=" * 66)


if __name__ == "__main__":
    main()