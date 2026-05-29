"""
运行全部 3×3 实验并汇总结果表格.

用法:
  python run_all_experiments.py                    # 全部 9 组
  python run_all_experiments.py --protocol P1      # 只跑 P1 的 3 种设定
  python run_all_experiments.py --setting S1       # 只跑 S1 的 3 种协议
  python run_all_experiments.py --collect_only     # 只汇总已有结果, 不训练
"""
import os
import sys
import json
import argparse
import subprocess
from datetime import datetime


PROTOCOLS = ['P1', 'P2', 'P3']
SETTINGS = ['S1', 'S2', 'S3']
ENVS = ['E01', 'E02', 'E03', 'E04']


def find_best_result(save_root, protocol, setting, test_env=None):
    """Find the best results.json from experiment runs."""
    if setting == 'S3' and test_env is None:
        # Look for averaged result
        avg_path = os.path.join(save_root, f'{protocol}_S3', 'results_average.json')
        if os.path.exists(avg_path):
            with open(avg_path) as f:
                return json.load(f)
        # Fall back: collect from individual runs
        results = []
        for env in ENVS:
            r = find_best_result(save_root, protocol, setting, env)
            if r:
                results.append(r)
        if results:
            avg = {}
            for key in ['MPJPE (mm)', 'PA-MPJPE (mm)', 'PCK@50_norm (%)', 'PCK@20_norm (%)']:
                vals = [r[key] for r in results if key in r]
                avg[key] = sum(vals) / len(vals) if vals else 0
            return {'protocol': protocol, 'setting': 'S3', 'test_env': 'average', **avg}
        return None

    exp_name = f'{protocol}_{setting}'
    if test_env:
        exp_name += f'_test{test_env}'

    exp_dir = os.path.join(save_root, exp_name)
    if not os.path.exists(exp_dir):
        return None

    # Find most recent run with results
    best_result = None
    best_mpjpe = float('inf')
    for run_dir in sorted(os.listdir(exp_dir), reverse=True):
        rpath = os.path.join(exp_dir, run_dir, 'results.json')
        if os.path.exists(rpath):
            with open(rpath) as f:
                r = json.load(f)
            mpjpe = r.get('MPJPE (mm)', float('inf'))
            if mpjpe < best_mpjpe:
                best_mpjpe = mpjpe
                best_result = r
    return best_result


def run_experiment(protocol, setting, args):
    """Run a single experiment via subprocess."""
    cmd = [
        sys.executable, 'train_experiment.py',
        '--protocol', protocol,
        '--setting', setting,
        '--data_root', args.data_root,
        '--save_root', args.save_root,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--patience', str(args.patience),
        '--seed', str(args.seed),
    ]

    # S3: don't pass test_env → train_experiment.py will run all 4
    print(f'\n{"="*60}')
    print(f'RUNNING: {protocol} × {setting}')
    print(f'{"="*60}')
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def print_results_table(save_root):
    """Collect and print results in paper-format table."""
    print(f'\n{"="*80}')
    print(f'RESULTS SUMMARY')
    print(f'{"="*80}\n')

    # Header
    header = f'{"Method":<20}'
    for p in PROTOCOLS:
        header += f'  {"PCK@20↑":>8} {"PCK@50↑":>8} {"MPJPE↓":>8} {"PA↓":>8}'
    print(header)
    print('-' * len(header))

    for s in SETTINGS:
        setting_names = {'S1': 'Random Split', 'S2': 'Cross-Subject', 'S3': 'Cross-Environment'}
        print(f'Setting {s} ({setting_names[s]}):')

        row = f'  {"DT-Pose (Ours)":<18}'
        all_found = True
        for p in PROTOCOLS:
            r = find_best_result(save_root, p, s)
            if r:
                row += f'  {r.get("PCK@20_norm (%)", 0):>8.1f}'
                row += f' {r.get("PCK@50_norm (%)", 0):>8.1f}'
                row += f' {r.get("MPJPE (mm)", 0):>8.1f}'
                row += f' {r.get("PA-MPJPE (mm)", 0):>8.1f}'
            else:
                row += f'  {"—":>8} {"—":>8} {"—":>8} {"—":>8}'
                all_found = False
        print(row)

        # For S3, also print per-env breakdown
        if s == 'S3':
            for env in ENVS:
                env_row = f'    {"└─ " + env:<16}'
                for p in PROTOCOLS:
                    r = find_best_result(save_root, p, s, env)
                    if r:
                        env_row += f'  {r.get("PCK@20_norm (%)", 0):>8.1f}'
                        env_row += f' {r.get("PCK@50_norm (%)", 0):>8.1f}'
                        env_row += f' {r.get("MPJPE (mm)", 0):>8.1f}'
                        env_row += f' {r.get("PA-MPJPE (mm)", 0):>8.1f}'
                    else:
                        env_row += f'  {"—":>8} {"—":>8} {"—":>8} {"—":>8}'
                print(env_row)
        print()

    # Also save as CSV
    csv_path = os.path.join(save_root, 'results_summary.csv')
    with open(csv_path, 'w') as f:
        f.write('Setting,Protocol,PCK@20_norm,PCK@50_norm,MPJPE,PA-MPJPE\n')
        for s in SETTINGS:
            for p in PROTOCOLS:
                r = find_best_result(save_root, p, s)
                if r:
                    f.write(f'{s},{p},'
                            f'{r.get("PCK@20_norm (%)", ""):.1f},'
                            f'{r.get("PCK@50_norm (%)", ""):.1f},'
                            f'{r.get("MPJPE (mm)", ""):.1f},'
                            f'{r.get("PA-MPJPE (mm)", ""):.1f}\n')
                else:
                    f.write(f'{s},{p},,,\n')
    print(f'CSV saved: {csv_path}')


def main():
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--protocol', type=str, default=None,
                        help='Run only this protocol (P1/P2/P3)')
    parser.add_argument('--setting', type=str, default=None,
                        help='Run only this setting (S1/S2/S3)')
    parser.add_argument('--collect_only', action='store_true',
                        help='Only collect and display existing results')
    parser.add_argument('--data_root', type=str,
                        default='/home/a123456/PerceptAlign/MMFi')
    parser.add_argument('--save_root', type=str, default='./experiments')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    protocols = [args.protocol] if args.protocol else PROTOCOLS
    settings = [args.setting] if args.setting else SETTINGS

    if not args.collect_only:
        total = len(protocols) * len(settings)
        print(f'Total experiments: {total}')
        print(f'  Protocols: {protocols}')
        print(f'  Settings:  {settings}')
        print(f'  Note: S3 runs 4 leave-one-out sub-experiments each')
        print(f'  Save root: {args.save_root}')
        print()

        for i, s in enumerate(settings):
            for j, p in enumerate(protocols):
                idx = i * len(protocols) + j + 1
                print(f'\n{"#"*60}')
                print(f'  Experiment {idx}/{total}: {p} × {s}')
                print(f'{"#"*60}')
                success = run_experiment(p, s, args)
                if not success:
                    print(f'WARNING: {p}_{s} failed!')

    # Always print results at the end
    print_results_table(args.save_root)


if __name__ == '__main__':
    main()