"""
CSI-RSC-PoseDG: 统一实验脚本

支持 3 种协议 × 3 种划分设定的任意组合.

协议:
  P1: A01-A14 (14 类日常动作)
  P2: A15-A27 (13 类康复动作)
  P3: A01-A27 (全部 27 类动作)

划分设定:
  S1: 随机划分 (训练:测试 = 3:1)
  S2: 跨受试者 (32 训 8 测, 每环境 8:2)
  S3: 跨环境 (3 训 1 测, leave-one-out)

用法:
  python train_experiment.py --protocol P1 --setting S1
  python train_experiment.py --protocol P3 --setting S3 --test_env E04
"""
import os
import sys
import argparse
import glob
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.io import loadmat
from scipy.signal import detrend
from datetime import datetime

from models.full_model import CSIRSCPoseDG
from losses import PoseLoss
from evaluate import PoseEvaluator
from utils import set_seed, setup_logger, count_parameters, save_checkpoint, AverageMeter, Timer


# ==============================================================
# Protocol definitions
# ==============================================================
PROTOCOLS = {
    'P1': list(range(1, 15)),   # A01-A14: daily actions
    'P2': list(range(15, 28)),  # A15-A27: rehabilitation actions
    'P3': list(range(1, 28)),   # A01-A27: all actions
}

ENV_SUBJECTS = {
    'E01': list(range(1, 11)),
    'E02': list(range(11, 21)),
    'E03': list(range(21, 31)),
    'E04': list(range(31, 41)),
}

ALL_ENVS = ['E01', 'E02', 'E03', 'E04']


# ==============================================================
# CSI Preprocessor (same as existing)
# ==============================================================
class CSIPreprocessor:
    @staticmethod
    def normalize_amplitude(amp):
        amin = amp.min(axis=(-2, -1), keepdims=True)
        amax = amp.max(axis=(-2, -1), keepdims=True)
        denom = amax - amin
        denom = np.where(denom < 1e-8, 1.0, denom)
        result = (amp - amin) / denom
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def process_phase(phase):
        phase_unwrap = np.unwrap(phase, axis=-2)
        shape = phase_unwrap.shape
        phase_flat = phase_unwrap.reshape(-1, shape[-2])
        phase_detrend = detrend(phase_flat, axis=-1)
        phase_detrend = phase_detrend.reshape(shape)
        sin_p = np.sin(phase_detrend)
        cos_p = np.cos(phase_detrend)
        return np.nan_to_num(np.concatenate([sin_p, cos_p], axis=1), nan=0.0)

    @staticmethod
    def preprocess(amp, phase):
        amp_norm = CSIPreprocessor.normalize_amplitude(amp)
        phase_enc = CSIPreprocessor.process_phase(phase)
        return np.concatenate([amp_norm, phase_enc], axis=1).astype(np.float32)


# ==============================================================
# Unified Dataset
# ==============================================================
class MMFiExperimentDataset(Dataset):
    """Flexible dataset supporting protocol and setting combinations."""

    def __init__(self, data_root, envs, subject_ids, action_ids,
                 seq_len=64, stride=32):
        self.data_root = data_root
        self.seq_len = seq_len
        self.preprocessor = CSIPreprocessor()
        self.samples = []

        for env in envs:
            env_subs = ENV_SUBJECTS.get(env, [])
            for subj_id in env_subs:
                if subj_id not in subject_ids:
                    continue
                subj_str = f'S{subj_id:02d}'
                for act_id in action_ids:
                    act_str = f'A{act_id:02d}'
                    csi_dir = os.path.join(data_root, env, subj_str, act_str, 'wifi-csi')
                    gt_path = os.path.join(data_root, env, subj_str, act_str, 'ground_truth.npy')
                    if not os.path.exists(csi_dir) or not os.path.exists(gt_path):
                        continue
                    num_frames = len(glob.glob(os.path.join(csi_dir, 'frame*.mat')))
                    if num_frames == 0:
                        continue
                    if num_frames < seq_len:
                        self.samples.append({
                            'env': env, 'subject': subj_str, 'action': act_str,
                            'start_frame': 0, 'num_frames': num_frames,
                            'csi_dir': csi_dir, 'gt_path': gt_path,
                        })
                    else:
                        starts = list(range(0, num_frames - seq_len + 1, stride))
                        if starts and starts[-1] + seq_len < num_frames:
                            starts.append(num_frames - seq_len)
                        for start in starts:
                            self.samples.append({
                                'env': env, 'subject': subj_str, 'action': act_str,
                                'start_frame': start, 'num_frames': num_frames,
                                'csi_dir': csi_dir, 'gt_path': gt_path,
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        start = sample['start_frame']
        actual_len = min(self.seq_len, sample['num_frames'] - start)

        amps, phases = [], []
        for i in range(start, start + actual_len):
            fp = os.path.join(sample['csi_dir'], f'frame{i+1:03d}.mat')
            if os.path.exists(fp):
                mat = loadmat(fp)
                amps.append(np.nan_to_num(mat['CSIamp'].astype(np.float32)))
                phases.append(np.nan_to_num(mat['CSIphase'].astype(np.float32)))
            else:
                amps.append(np.zeros((3, 114, 10), dtype=np.float32))
                phases.append(np.zeros((3, 114, 10), dtype=np.float32))

        csi = self.preprocessor.preprocess(np.stack(amps), np.stack(phases))
        gt = np.load(sample['gt_path']).astype(np.float32)
        gt_clip = gt[start:start + actual_len]
        # 绝对坐标 (DT-Pose 对齐)

        if actual_len < self.seq_len:
            pad = self.seq_len - actual_len
            csi = np.pad(csi, ((0, pad), (0, 0), (0, 0), (0, 0)), mode='edge')
            gt_clip = np.pad(gt_clip, ((0, pad), (0, 0), (0, 0)), mode='edge')

        return {
            'csi': torch.from_numpy(csi),
            'pose_3d': torch.from_numpy(gt_clip),
            'env': sample['env'],
            'subject': sample['subject'],
            'action': sample['action'],
        }


# ==============================================================
# Data splitting logic
# ==============================================================
def build_splits(setting, protocol, seed=42, test_env=None):
    """Return (train_envs, train_subjects, test_envs, test_subjects).

    All subjects are returned as sets of integer IDs.
    """
    action_ids = PROTOCOLS[protocol]
    all_subjects = set(range(1, 41))
    rng = np.random.RandomState(seed)

    if setting == 'S1':
        # Random 3:1 split — ALL subjects in both train and test
        # Splitting happens at sequence level in build_dataloaders
        return ALL_ENVS, all_subjects, ALL_ENVS, all_subjects

    elif setting == 'S2':
        # Cross-subject: 32 train, 8 test (2 per env)
        train_subs, test_subs = set(), set()
        for env, subs in ENV_SUBJECTS.items():
            shuffled = rng.permutation(subs).tolist()
            n_test = 2
            test_subs.update(shuffled[:n_test])
            train_subs.update(shuffled[n_test:])
        return ALL_ENVS, train_subs, ALL_ENVS, test_subs

    elif setting == 'S3':
        # Cross-environment: leave-one-out
        assert test_env is not None, "S3 requires --test_env"
        train_envs = [e for e in ALL_ENVS if e != test_env]
        test_envs = [test_env]
        train_subs = set()
        for e in train_envs:
            train_subs.update(ENV_SUBJECTS[e])
        test_subs = set(ENV_SUBJECTS[test_env])
        return train_envs, train_subs, test_envs, test_subs

    else:
        raise ValueError(f"Unknown setting: {setting}")


def build_dataloaders(data_root, setting, protocol, batch_size=2,
                      num_workers=4, seed=42, test_env=None, seq_len=64):
    action_ids = PROTOCOLS[protocol]
    train_envs, train_subs, test_envs, test_subs = build_splits(
        setting, protocol, seed, test_env
    )

    train_dataset = MMFiExperimentDataset(
        data_root, train_envs, train_subs, action_ids, seq_len=seq_len
    )
    test_dataset = MMFiExperimentDataset(
        data_root, test_envs, test_subs, action_ids, seq_len=seq_len, stride=seq_len
    )

    # For S1, do sequence-level random split (same subjects in both)
    if setting == 'S1':
        full_dataset = train_dataset  # both have same subjects
        n_total = len(full_dataset)
        indices = np.random.RandomState(seed).permutation(n_total).tolist()
        n_train = int(n_total * 0.75)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        train_dataset = Subset(full_dataset, train_idx)
        test_dataset = Subset(full_dataset, test_idx)
        train_subs = set(range(1, 41))  # all subjects in both

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader, train_subs, test_subs


# ==============================================================
# Model config (same architecture for all experiments)
# ==============================================================
def get_model_args(num_actions=27):
    class Args:
        pass
    a = Args()
    a.amp_channels = 3
    a.phase_channels = 6
    a.encoder_hidden_dim = 32
    a.encoder_out_dim = 64
    a.local_hidden_dim = 64
    a.local_out_dim = 64
    a.num_res3d_blocks = 2
    a.global_dim = 128
    a.num_transformer_layers = 3
    a.num_heads = 4
    a.tcn_channels = [128, 128]
    a.tcn_kernel_size = 3
    a.transformer_dropout = 0.1
    a.coarse_hidden_dim = 256
    a.gcn_hidden_dim = 128
    a.num_gcn_layers = 3
    a.num_joints = 17
    a.num_actions = num_actions
    a.seq_len = 64
    # RSC params
    a.rsc2_time_drop_pct = 0.5
    a.rsc2_batch_pct = 0.5
    return a


# ==============================================================
# Training & Evaluation
# ==============================================================
def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip=1.0, accum=4):
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        csi = batch['csi'].to(device)
        pose = batch['pose_3d'].to(device)

        action_labels = torch.tensor(
            [int(a[1:]) - 1 for a in batch['action']],
            dtype=torch.long, device=device
        )
        out = model(csi, action_idx=action_labels)
        loss, _ = loss_fn(out['p_final'], pose)
        # Action classification loss
        act_loss = nn.CrossEntropyLoss()(out['action_logits'], action_labels)
        loss = loss + 0.5 * act_loss
        loss = loss / accum
        loss.backward()

        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        loss_meter.update(loss.item() * accum, csi.shape[0])
        del out, loss, csi, pose
        torch.cuda.empty_cache()

    return loss_meter.avg


@torch.no_grad()
def evaluate(model, loader, device, evaluator):
    """严格评估: action_idx=None, 不使用测试集 GT 动作标签."""
    model.eval()
    all_preds, all_gts = [], []
    action_correct, action_total = 0, 0

    for batch in loader:
        csi = batch['csi'].to(device)
        pose = batch['pose_3d'].to(device)

        # ★ 严格 DG: 不透露测试集动作标签
        out = model(csi, action_idx=None)
        all_preds.append(out['p_final'].cpu())
        all_gts.append(pose.cpu())

        # 动作准确率 (仅指标, 不作为模型输入)
        action_labels = torch.tensor(
            [int(a[1:]) - 1 for a in batch['action']],
            dtype=torch.long, device=device
        )
        action_pred = out['action_logits'].argmax(dim=-1)
        action_correct += (action_pred == action_labels).sum().item()
        action_total += action_labels.shape[0]

        del out, csi, pose
        torch.cuda.empty_cache()

    preds = torch.cat(all_preds)
    gts = torch.cat(all_gts)
    metrics = evaluator.evaluate(preds, gts)
    metrics['action_acc'] = 100.0 * action_correct / max(action_total, 1)
    return metrics


def run_single_experiment(args):
    """Run a single experiment (one protocol + one setting + optional test_env)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # Setup save dir
    exp_name = f'{args.protocol}_{args.setting}'
    if args.setting == 'S3' and args.test_env:
        exp_name += f'_test{args.test_env}'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_root, exp_name, f'run_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(exp_name, os.path.join(save_dir, 'train.log'))
    logger.info(f'Experiment: {exp_name}')
    logger.info(f'Protocol: {args.protocol} → actions {PROTOCOLS[args.protocol]}')
    logger.info(f'Setting: {args.setting}')
    if args.test_env:
        logger.info(f'Test env: {args.test_env}')

    # Data
    # IMPORTANT: Always use 27 (full action space) for the embedding table.
    # Protocol only controls which actions appear in the data, not the index range.
    # P1: data has A01-A14 (idx 0-13), P2: A15-A27 (idx 14-26), P3: all.
    # Using len(PROTOCOLS[protocol]) would create a 13-slot table for P2,
    # but action indices are still 14-26 → out-of-bounds crash.
    num_actions = 27
    train_loader, test_loader, train_subs, test_subs = build_dataloaders(
        args.data_root, args.setting, args.protocol,
        batch_size=args.batch_size, num_workers=args.num_workers,
        seed=args.seed, test_env=args.test_env, seq_len=64
    )
    logger.info(f'Train: {len(train_loader.dataset)} samples ({len(train_subs)} subjects)')
    logger.info(f'Test:  {len(test_loader.dataset)} samples ({len(test_subs)} subjects)')
    logger.info(f'Train batches: {len(train_loader)}, Test batches: {len(test_loader)}')

    if len(train_loader) == 0 or len(test_loader) == 0:
        logger.error('Empty dataloader! Skipping.')
        return None

    # Model
    model_args = get_model_args(num_actions)
    model = CSIRSCPoseDG(model_args).to(device)
    logger.info(f'Model parameters: {count_parameters(model):,}')

    # Training setup
    loss_fn = PoseLoss(lambda1=1.0, lambda2=0.5, lambda3=2.0)
    evaluator = PoseEvaluator(unit='meter')
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Train
    timer = Timer()
    timer.start()
    best_mpjpe = float('inf')
    patience_counter = 0
    best_metrics = {}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            grad_clip=1.0, accum=args.accum
        )
        scheduler.step()

        if epoch % 3 == 0 or epoch == args.epochs:
            metrics = evaluate(model, test_loader, device, evaluator)
            cur = metrics['MPJPE (mm)']
            logger.info(
                f'[Epoch {epoch:3d}] Loss: {train_loss:.4f} | '
                f'MPJPE: {cur:.2f} PA: {metrics["PA-MPJPE (mm)"]:.2f} '
                f'P50: {metrics["PCK@50_norm (%)"]:.1f} P20: {metrics["PCK@20_norm (%)"]:.1f}'
            )
            if cur < best_mpjpe:
                best_mpjpe = cur
                best_metrics = metrics.copy()
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch, metrics,
                                os.path.join(save_dir, 'best_model.pth'))
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break

    logger.info(f'Best MPJPE: {best_mpjpe:.2f}mm | Time: {timer.elapsed_str()}')

    # Save results
    result = {
        'protocol': args.protocol,
        'setting': args.setting,
        'test_env': args.test_env,
        'n_train': len(train_loader.dataset),
        'n_test': len(test_loader.dataset),
        'best_epoch': best_metrics.get('epoch', '?'),
        **best_metrics,
    }
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


# ==============================================================
# Main
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description='CSI Pose Estimation Experiment')
    parser.add_argument('--protocol', type=str, required=True, choices=['P1', 'P2', 'P3'])
    parser.add_argument('--setting', type=str, required=True, choices=['S1', 'S2', 'S3'])
    parser.add_argument('--test_env', type=str, default=None,
                        help='Test environment for S3 (e.g., E04)')
    parser.add_argument('--data_root', type=str,
                        default='/home/a123456/PerceptAlign/MMFi')
    parser.add_argument('--save_root', type=str, default='./experiments')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    if args.setting == 'S3' and args.test_env is None:
        # Run all 4 leave-one-out experiments
        print(f'\n{"="*60}')
        print(f'S3: Running 4 leave-one-out experiments for {args.protocol}')
        print(f'{"="*60}\n')

        results = []
        for test_env in ALL_ENVS:
            print(f'\n>>> Test env: {test_env} <<<\n')
            args.test_env = test_env
            r = run_single_experiment(args)
            if r:
                results.append(r)

        # Compute average
        if results:
            avg = {}
            for key in ['MPJPE (mm)', 'PA-MPJPE (mm)', 'PCK@50_norm (%)', 'PCK@20_norm (%)']:
                vals = [r[key] for r in results if key in r]
                avg[key] = np.mean(vals) if vals else 0
            print(f'\n{"="*60}')
            print(f'{args.protocol} S3 Average (4 envs):')
            for k, v in avg.items():
                print(f'  {k}: {v:.2f}')

            # Save averaged result
            avg_result = {
                'protocol': args.protocol, 'setting': 'S3',
                'test_env': 'average', 'sub_results': results, **avg,
            }
            avg_dir = os.path.join(args.save_root, f'{args.protocol}_S3')
            os.makedirs(avg_dir, exist_ok=True)
            with open(os.path.join(avg_dir, 'results_average.json'), 'w') as f:
                json.dump(avg_result, f, indent=2)
    else:
        run_single_experiment(args)


if __name__ == '__main__':
    main()