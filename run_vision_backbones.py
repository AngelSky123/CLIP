"""
Run several vision backbones (resnet18 / resnet50 / swin / vit) through
train_vision.py and summarize into a single comparison table + CSV.

Mirrors run_all_experiments.py: subprocess orchestration, then collect best
metrics from each run's best_model.pth and print a paper-style table.

Usage:
  python run_vision_backbones.py                       # all 4 backbones
  python run_vision_backbones.py --backbones resnet18 swin_tiny
  python run_vision_backbones.py --freeze              # frozen-backbone DG baseline
  python run_vision_backbones.py --collect_only        # re-summarize existing runs
  python run_vision_backbones.py --epochs 50 --test_env E04

Each backbone trains on --train_envs (default E01 E02 E03), evaluated strictly
on --test_env (default E04) with action_idx=None. Effective batch is kept ~16
per backbone (heavier transformers use smaller batch + more accumulation).
"""
import os
import re
import sys
import csv
import argparse
import subprocess
from datetime import datetime


# name -> launch config. img_size/batch_size/accum tuned so transformers @224
# don't OOM while keeping effective batch (batch*accum) ~= 16.
BACKBONES = {
    'resnet18':  dict(arch='resnet18',                       img_size=112, batch_size=4, accum=4),
    'resnet50':  dict(arch='resnet50',                       img_size=112, batch_size=4, accum=4),
    'swin_tiny': dict(arch='swin_tiny_patch4_window7_224',   img_size=224, batch_size=2, accum=8),
    'vit_small': dict(arch='vit_small_patch16_224',          img_size=224, batch_size=2, accum=8),
}
ORDER = ['resnet18', 'resnet50', 'swin_tiny', 'vit_small']

# Reference row: original dual-branch encoder, Plan A+B baseline (from README).
# Shown for context only; not retrained here.
BASELINE_REF = {
    'name': 'orig-encoder (Plan A+B)',
    'params': '1.62M',
    'MPJPE (mm)': 345.35,
    'PA-MPJPE (mm)': 104.68,
    'MPJPE_aligned (mm)': 125.83,
    'PCK@50_norm (%)': 52.7,
    'PCK@20_norm (%)': float('nan'),
    'action_acc': float('nan'),
}

METRIC_KEYS = ['MPJPE (mm)', 'PA-MPJPE (mm)', 'MPJPE_aligned (mm)',
               'PCK@50_norm (%)', 'PCK@20_norm (%)', 'action_acc']


# ----------------------------------------------------------------------
# Collection helpers
# ----------------------------------------------------------------------
def find_latest_run(save_dir):
    """get_config() appends run_<timestamp>; return the newest one's path."""
    if not os.path.isdir(save_dir):
        return None
    runs = [d for d in os.listdir(save_dir)
            if d.startswith('run_') and os.path.isdir(os.path.join(save_dir, d))]
    if not runs:
        # maybe save_dir already IS a run dir
        if os.path.exists(os.path.join(save_dir, 'best_model.pth')) or \
           os.path.exists(os.path.join(save_dir, 'train.log')):
            return save_dir
        return None
    return os.path.join(save_dir, sorted(runs)[-1])


def parse_log(run_dir):
    """Pull param count and total time from train.log (best-effort)."""
    log = os.path.join(run_dir, 'train.log')
    params, total_time = None, None
    if os.path.exists(log):
        with open(log, errors='ignore') as f:
            text = f.read()
        m = re.findall(r'Model parameters:\s*([\d,]+)', text)
        if m:
            n = int(m[-1].replace(',', ''))
            params = f'{n/1e6:.2f}M'
        t = re.findall(r'Total time:\s*([0-9:]+)', text)
        if t:
            total_time = t[-1]
    return params, total_time


def read_metrics(run_dir):
    """Best metrics from best_model.pth['metrics']; fall back to last eval line."""
    ckpt = os.path.join(run_dir, 'best_model.pth')
    if os.path.exists(ckpt):
        try:
            import torch
            m = torch.load(ckpt, map_location='cpu').get('metrics', {})
            if m:
                return m
        except Exception as e:
            print(f'  (warn) could not read {ckpt}: {e}')
    # Fallback: parse the most recent [Eval] line in the log
    log = os.path.join(run_dir, 'train.log')
    if os.path.exists(log):
        with open(log, errors='ignore') as f:
            evals = [ln for ln in f if '[Eval]' in ln]
        if evals:
            ln = evals[-1]
            def grab(pat):
                mm = re.search(pat, ln)
                return float(mm.group(1)) if mm else float('nan')
            return {
                'MPJPE (mm)':        grab(r'MPJPE:\s*([\d.]+)'),
                'MPJPE_aligned (mm)':grab(r'MPJPE_a:\s*([\d.]+)'),
                'PA-MPJPE (mm)':     grab(r'PA:\s*([\d.]+)'),
                'PCK@50_norm (%)':   grab(r'P50n:\s*([\d.]+)'),
                'PCK@20_norm (%)':   grab(r'P20n:\s*([\d.]+)'),
                'action_acc':        grab(r'ActAcc:\s*([\d.]+)'),
            }
    return None


def collect(save_root, names):
    rows = []
    for name in names:
        run_dir = find_latest_run(os.path.join(save_root, name))
        if run_dir is None:
            rows.append({'name': name, 'params': '—', 'time': '—', 'metrics': None})
            continue
        params, total_time = parse_log(run_dir)
        rows.append({'name': name, 'params': params or '—',
                     'time': total_time or '—', 'metrics': read_metrics(run_dir),
                     'run_dir': run_dir})
    return rows


# ----------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------
def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and v != v):  # None or NaN
        return '—'
    return f'{v:.{nd}f}'


def print_table(rows, save_root):
    hdr = (f'{"Backbone":<24}{"Params":>9}{"MPJPE↓":>10}{"PA↓":>9}'
           f'{"MPJPE_a↓":>10}{"PCK@50n↑":>10}{"PCK@20n↑":>10}{"ActAcc↑":>9}{"Time":>10}')
    print('\n' + '=' * len(hdr))
    print('VISION BACKBONE COMPARISON  (cross-env DG, strict: test action_idx=None)')
    print('=' * len(hdr))
    print(hdr)
    print('-' * len(hdr))

    # reference row
    b = BASELINE_REF
    print(f'{b["name"]:<24}{b["params"]:>9}'
          f'{_fmt(b["MPJPE (mm)"]):>10}{_fmt(b["PA-MPJPE (mm)"]):>9}'
          f'{_fmt(b["MPJPE_aligned (mm)"]):>10}{_fmt(b["PCK@50_norm (%)"],1):>10}'
          f'{_fmt(b["PCK@20_norm (%)"],1):>10}{_fmt(b["action_acc"],1):>9}{"—":>10}')
    print('-' * len(hdr))

    best_mpjpe, best_name = float('inf'), None
    for r in rows:
        m = r['metrics']
        if not m:
            print(f'{r["name"]:<24}{r["params"]:>9}{"—":>10}{"—":>9}{"—":>10}'
                  f'{"—":>10}{"—":>10}{"—":>9}{r["time"]:>10}')
            continue
        mp = m.get('MPJPE (mm)', float('nan'))
        if mp == mp and mp < best_mpjpe:
            best_mpjpe, best_name = mp, r['name']
        print(f'{r["name"]:<24}{r["params"]:>9}'
              f'{_fmt(mp):>10}{_fmt(m.get("PA-MPJPE (mm)")):>9}'
              f'{_fmt(m.get("MPJPE_aligned (mm)")):>10}{_fmt(m.get("PCK@50_norm (%)"),1):>10}'
              f'{_fmt(m.get("PCK@20_norm (%)"),1):>10}{_fmt(m.get("action_acc"),1):>9}'
              f'{r["time"]:>10}')
    print('=' * len(hdr))
    if best_name:
        delta = best_mpjpe - BASELINE_REF['MPJPE (mm)']
        sign = '+' if delta >= 0 else ''
        print(f'Best vision backbone: {best_name} @ {best_mpjpe:.2f}mm MPJPE '
              f'({sign}{delta:.2f}mm vs orig-encoder baseline)')
    print('NB: judge generalization by PA-MPJPE + per-env spread, not MPJPE alone.\n')

    # CSV
    csv_path = os.path.join(save_root, 'vision_backbone_summary.csv')
    os.makedirs(save_root, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['backbone', 'params', 'time'] + METRIC_KEYS)
        w.writerow([BASELINE_REF['name'], BASELINE_REF['params'], ''] +
                   [BASELINE_REF.get(k, '') for k in METRIC_KEYS])
        for r in rows:
            m = r['metrics'] or {}
            w.writerow([r['name'], r['params'], r['time']] +
                       [m.get(k, '') for k in METRIC_KEYS])
    print(f'CSV saved: {csv_path}')


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
def run_one(name, cfg, args):
    save_dir = os.path.join(args.save_root, name)
    bs = args.batch_size if args.batch_size else cfg['batch_size']
    accum = args.accum if args.accum else cfg['accum']
    cmd = [
        sys.executable, 'train_vision.py',
        '--data_root', args.data_root,
        '--train_envs', *args.train_envs,
        '--test_env', args.test_env,
        '--vision_arch', cfg['arch'],
        '--vision_img_size', str(cfg['img_size']),
        '--epochs', str(args.epochs),
        '--batch_size', str(bs),
        '--accumulate_grad', str(accum),
        '--lr_backbone', str(args.lr_backbone),
        '--lr_head', str(args.lr_head),
        '--patience', str(args.patience),
        '--seed', str(args.seed),
        '--save_dir', save_dir,
    ]
    if args.freeze:
        cmd.append('--vision_freeze')
    print(f'\n{"#"*70}\n# {name}  ({cfg["arch"]} @ {cfg["img_size"]}, '
          f'bs={bs} accum={accum})\n{"#"*70}')
    print('  ' + ' '.join(cmd))
    rc = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__))).returncode
    if rc != 0:
        print(f'WARNING: {name} exited with code {rc}')
    return rc == 0


def main():
    ap = argparse.ArgumentParser(description='Run + compare vision backbones')
    ap.add_argument('--backbones', nargs='+', default=ORDER,
                    choices=list(BACKBONES.keys()),
                    help='subset of backbones to run')
    ap.add_argument('--collect_only', action='store_true',
                    help='only summarize existing runs, do not train')
    ap.add_argument('--freeze', action='store_true',
                    help='freeze backbones (heads-only training)')
    ap.add_argument('--data_root', type=str,
                    default='/home/a123456/PerceptAlign/MMFi')
    ap.add_argument('--train_envs', nargs='+', default=['E01', 'E02', 'E03'])
    ap.add_argument('--test_env', type=str, default='E04')
    ap.add_argument('--save_root', type=str, default='./checkpoints/vision_compare')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=None,
                    help='override per-backbone default batch size')
    ap.add_argument('--accum', type=int, default=None,
                    help='override per-backbone default grad accumulation')
    ap.add_argument('--lr_backbone', type=float, default=1e-4)
    ap.add_argument('--lr_head', type=float, default=5e-4)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    names = [n for n in ORDER if n in args.backbones]  # keep stable order

    if not args.collect_only:
        try:
            import timm  # noqa: F401
        except ImportError:
            print('ERROR: timm not installed. Run: pip install timm')
            sys.exit(1)
        print(f'Running {len(names)} backbones: {names}')
        print(f'  train_envs={args.train_envs}  test_env={args.test_env}  '
              f'epochs={args.epochs}  freeze={args.freeze}')
        for name in names:
            run_one(name, BACKBONES[name], args)

    rows = collect(args.save_root, names)
    print_table(rows, args.save_root)


if __name__ == '__main__':
    main()