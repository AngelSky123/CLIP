"""
MMFi Dataset — 标准 8:2 划分模式

划分策略: 按被试(subject)划分, 保证同一被试的数据不会同时出现在训练和测试集中.
  - 每个环境10个被试, 4个环境共40人
  - 8:2 → 32人训练, 8人测试 (每个环境随机选2人作为测试)

这比按序列随机划分更合理, 避免数据泄露.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy.signal import detrend


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


class MMFiStandardDataset(Dataset):
    """MMFi with subject-based 8:2 split across ALL environments."""

    def __init__(self, data_root, envs, subject_ids, seq_len=64, stride=32,
                 augment=False):
        """
        Args:
            data_root: path to MMFi/
            envs: list of all environments ['E01','E02','E03','E04']
            subject_ids: list of subject IDs to include (e.g., [1,2,...,32] for train)
            seq_len: temporal window
            stride: sliding window stride
            augment: enable data augmentation
        """
        self.data_root = data_root
        self.envs = envs
        self.subject_ids = set(subject_ids)
        self.seq_len = seq_len
        self.stride = stride
        self.preprocessor = CSIPreprocessor()

        self.augmentor = None
        if augment:
            try:
                from augmentation import CSIAugmentor
                self.augmentor = CSIAugmentor(p=0.5)
            except ImportError:
                pass

        self.samples = []
        self._build_index()

    def _build_index(self):
        for env in self.envs:
            env_subject_map = {
                'E01': list(range(1, 11)),
                'E02': list(range(11, 21)),
                'E03': list(range(21, 31)),
                'E04': list(range(31, 41)),
            }
            subjects = env_subject_map.get(env, [])

            for subj_id in subjects:
                if subj_id not in self.subject_ids:
                    continue

                subj_str = f'S{subj_id:02d}'
                for act_id in range(1, 28):
                    act_str = f'A{act_id:02d}'
                    csi_dir = os.path.join(self.data_root, env, subj_str, act_str, 'wifi-csi')
                    gt_path = os.path.join(self.data_root, env, subj_str, act_str, 'ground_truth.npy')

                    if not os.path.exists(csi_dir) or not os.path.exists(gt_path):
                        continue

                    num_frames = len(glob.glob(os.path.join(csi_dir, 'frame*.mat')))
                    if num_frames == 0:
                        continue

                    if num_frames < self.seq_len:
                        self.samples.append({
                            'env': env, 'subject': subj_str, 'action': act_str,
                            'start_frame': 0, 'num_frames': num_frames,
                            'csi_dir': csi_dir, 'gt_path': gt_path,
                        })
                    else:
                        starts = list(range(0, num_frames - self.seq_len + 1, self.stride))
                        if starts and starts[-1] + self.seq_len < num_frames:
                            starts.append(num_frames - self.seq_len)
                        for start in starts:
                            self.samples.append({
                                'env': env, 'subject': subj_str, 'action': act_str,
                                'start_frame': start, 'num_frames': num_frames,
                                'csi_dir': csi_dir, 'gt_path': gt_path,
                            })

        print(f"[MMFiStandardDataset] {len(self.samples)} samples, "
              f"{len(self.subject_ids)} subjects from {self.envs}")

    def _load_csi_sequence(self, csi_dir, start_frame, length):
        amps, phases = [], []
        for i in range(start_frame, start_frame + length):
            frame_path = os.path.join(csi_dir, f'frame{i + 1:03d}.mat')
            if os.path.exists(frame_path):
                mat = loadmat(frame_path)
                amp = np.nan_to_num(mat['CSIamp'].astype(np.float32))
                pha = np.nan_to_num(mat['CSIphase'].astype(np.float32))
                amps.append(amp)
                phases.append(pha)
            else:
                amps.append(np.zeros((3, 114, 10), dtype=np.float32))
                phases.append(np.zeros((3, 114, 10), dtype=np.float32))
        return np.stack(amps, axis=0), np.stack(phases, axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        start = sample['start_frame']
        num_frames = sample['num_frames']
        actual_len = min(self.seq_len, num_frames - start)

        amp, phase = self._load_csi_sequence(sample['csi_dir'], start, actual_len)
        csi = self.preprocessor.preprocess(amp, phase)

        gt = np.load(sample['gt_path']).astype(np.float32)
        gt_clip = gt[start:start + actual_len]

        # 绝对坐标 (DT-Pose 对齐)
        if actual_len < self.seq_len:
            pad_len = self.seq_len - actual_len
            csi = np.pad(csi, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='edge')
            gt_clip = np.pad(gt_clip, ((0, pad_len), (0, 0), (0, 0)), mode='edge')

        csi_tensor = torch.from_numpy(csi)
        gt_tensor = torch.from_numpy(gt_clip)

        if self.augmentor is not None:
            csi_tensor = self.augmentor(csi_tensor)

        return {
            'csi': csi_tensor,
            'pose_3d': gt_tensor,
            'env': sample['env'],
            'subject': sample['subject'],
            'action': sample['action'],
        }


def get_subject_split(seed=42, train_ratio=0.8):
    """Subject-based 8:2 split: stratified by environment.

    Each env has 10 subjects. With train_ratio=0.8:
      - 8 subjects per env for training (32 total)
      - 2 subjects per env for testing (8 total)

    This ensures every environment is represented in both splits.

    Returns:
        train_ids: list of subject IDs for training
        test_ids: list of subject IDs for testing
    """
    rng = np.random.RandomState(seed)

    env_subjects = {
        'E01': list(range(1, 11)),
        'E02': list(range(11, 21)),
        'E03': list(range(21, 31)),
        'E04': list(range(31, 41)),
    }

    num_test_per_env = max(1, round(10 * (1 - train_ratio)))  # 2

    train_ids = []
    test_ids = []

    for env, subjects in env_subjects.items():
        shuffled = rng.permutation(subjects).tolist()
        test_ids.extend(shuffled[:num_test_per_env])
        train_ids.extend(shuffled[num_test_per_env:])

    print(f"[Split] Train: {len(train_ids)} subjects, Test: {len(test_ids)} subjects")
    print(f"  Train IDs: {sorted(train_ids)}")
    print(f"  Test IDs:  {sorted(test_ids)}")

    return sorted(train_ids), sorted(test_ids)


def build_dataloaders(args):
    """Build train/test dataloaders with subject-based 8:2 split."""
    train_ids, test_ids = get_subject_split(seed=args.seed, train_ratio=args.train_ratio)

    train_dataset = MMFiStandardDataset(
        args.data_root, args.all_envs, train_ids,
        seq_len=args.seq_len, augment=True,
    )
    test_dataset = MMFiStandardDataset(
        args.data_root, args.all_envs, test_ids,
        seq_len=args.seq_len, stride=args.seq_len, augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, test_loader