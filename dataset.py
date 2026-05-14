"""
MMFi Dataset loader v3: with CSI augmentation.
"""
import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy.signal import detrend
from augmentation import CSIAugmentor


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
        phase_encoded = np.concatenate([sin_p, cos_p], axis=1)
        return np.nan_to_num(phase_encoded, nan=0.0)

    @staticmethod
    def preprocess(amp, phase):
        amp_norm = CSIPreprocessor.normalize_amplitude(amp)
        phase_enc = CSIPreprocessor.process_phase(phase)
        csi = np.concatenate([amp_norm, phase_enc], axis=1)
        return csi.astype(np.float32)


class MMFiDataset(Dataset):
    def __init__(self, data_root, envs, seq_len=64, stride=32,
                 augment=False):
        self.data_root = data_root
        self.envs = envs
        self.seq_len = seq_len
        self.stride = stride
        self.preprocessor = CSIPreprocessor()

        # CSI augmentation (only for training)
        self.augmentor = CSIAugmentor(p=0.8) if augment else None

        self.samples = []
        self._build_index()

    def _get_subject_range(self, env):
        env_subject_map = {
            'E01': list(range(1, 11)),
            'E02': list(range(11, 21)),
            'E03': list(range(21, 31)),
            'E04': list(range(31, 41)),
        }
        return env_subject_map.get(env, [])

    def _build_index(self):
        for env in self.envs:
            subjects = self._get_subject_range(env)
            for subj_id in subjects:
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
                        for start in range(0, num_frames - self.seq_len + 1, self.stride):
                            self.samples.append({
                                'env': env, 'subject': subj_str, 'action': act_str,
                                'start_frame': start, 'num_frames': num_frames,
                                'csi_dir': csi_dir, 'gt_path': gt_path,
                            })
        print(f"[MMFiDataset] Built {len(self.samples)} samples from envs {self.envs}"
              f" (augment={'ON' if self.augmentor else 'OFF'})")

    def _load_csi_sequence(self, csi_dir, start_frame, length):
        amps, phases = [], []
        for i in range(start_frame, start_frame + length):
            frame_path = os.path.join(csi_dir, f'frame{i + 1:03d}.mat')
            if os.path.exists(frame_path):
                mat = loadmat(frame_path)
                amp = np.nan_to_num(mat['CSIamp'].astype(np.float32), nan=0.0)
                pha = np.nan_to_num(mat['CSIphase'].astype(np.float32), nan=0.0)
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

        # 绝对坐标 (DT-Pose 对齐, 不减 root)
        # 注: 原 Root-relative 代码已删除
        if actual_len < self.seq_len:
            pad_len = self.seq_len - actual_len
            csi = np.pad(csi, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='edge')
            gt_clip = np.pad(gt_clip, ((0, pad_len), (0, 0), (0, 0)), mode='edge')

        csi_tensor = torch.from_numpy(csi)
        gt_tensor = torch.from_numpy(gt_clip)

        # Apply CSI augmentation
        if self.augmentor is not None:
            csi_tensor = self.augmentor(csi_tensor)

        return {
            'csi': csi_tensor,
            'pose_3d': gt_tensor,
            'env': sample['env'],
            'subject': sample['subject'],
            'action': sample['action'],
        }


class MMFiSyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=64, num_envs=3):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_envs = num_envs
        self.envs = [f'E{i+1:02d}' for i in range(num_envs)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'csi': torch.randn(self.seq_len, 9, 114, 10),
            'pose_3d': torch.randn(self.seq_len, 17, 3) * 0.3,
            'env': self.envs[idx % self.num_envs],
            'subject': f'S{idx % 10 + 1:02d}',
            'action': f'A{idx % 27 + 1:02d}',
        }


def build_dataloaders(args, synthetic=False):
    if synthetic:
        train_dataset = MMFiSyntheticDataset(200, args.seq_len, len(args.train_envs))
        test_dataset = MMFiSyntheticDataset(50, args.seq_len, 1)
    else:
        train_dataset = MMFiDataset(
            args.data_root, args.train_envs, args.seq_len,
            augment=True,  # Enable augmentation for training
        )
        test_dataset = MMFiDataset(
            args.data_root, [args.test_env], args.seq_len,
            stride=args.seq_len,  # 无重叠滑窗: 每帧评测一次 (DT-Pose 对齐)
            augment=False,
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