"""
MMFi image dataset for the image three-stage pipeline.

Difference vs dataset.py:
  * Output is a rendered image sequence (T, 3, H, W) from csi_image.py,
    instead of the (T, 9, 114, 10) raw-channel tensor.
  * Augmentation runs on the RAW amp/phase (the physical domain) BEFORE
    rendering — you cannot meaningfully run the old CSIAugmentor on an image.
    A small raw-domain augmentor (amp scale, phase noise, subcarrier drop) keeps
    the cross-environment augmentation spirit.

Returns dicts with the same keys as dataset.py ('csi','pose_3d','env',...), so
the training loops are unchanged — only the shape of 'csi' differs.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

from csi_image import CSIImagePreprocessor


ENV_SUBJECTS = {
    'E01': list(range(1, 11)),
    'E02': list(range(11, 21)),
    'E03': list(range(21, 31)),
    'E04': list(range(31, 41)),
}


class RawCSIAugmentor:
    """Lightweight raw-domain augmentation on (T,3,114,10) amp & phase, pre-render."""
    def __init__(self, amp_scale=(0.7, 1.3), phase_noise_std=0.15,
                 subcarrier_drop_pct=0.1, p=0.8):
        self.amp_scale = amp_scale
        self.phase_noise_std = phase_noise_std
        self.subcarrier_drop_pct = subcarrier_drop_pct
        self.p = p

    def __call__(self, amp, phase):
        if np.random.rand() > self.p:
            return amp, phase
        amp = amp.copy(); phase = phase.copy()
        H = amp.shape[-2]
        # NOTE: per-antenna amplitude scaling was removed. The renderer applies a
        # per-image min-max normalization, which cancels any global amplitude
        # scale, so the augmentation had no effect on the rendered image while
        # risking float32 overflow on large/inf raw CSIamp values.
        if np.random.rand() < 0.7:                              # phase noise
            phase = phase + np.random.randn(*phase.shape).astype(np.float32) * self.phase_noise_std
        if np.random.rand() < 0.5:                              # subcarrier dropout
            k = max(1, int(self.subcarrier_drop_pct * H))
            idx = np.random.choice(H, k, replace=False)
            amp[:, :, idx, :] = 0.0
        return amp, phase


class MMFiImageDataset(Dataset):
    def __init__(self, data_root, envs, seq_len=64, stride=32, augment=False,
                 img_size=112, channels=("phase", "amp", "doppler"),
                 antenna_quotient=True):
        self.data_root = data_root
        self.envs = envs
        self.seq_len = seq_len
        self.stride = stride
        self.renderer = CSIImagePreprocessor(
            img_size=img_size, channels=channels, antenna_quotient=antenna_quotient)
        self.aug = RawCSIAugmentor() if augment else None
        self.samples = []
        self._build_index()

    def _build_index(self):
        for env in self.envs:
            for subj_id in ENV_SUBJECTS.get(env, []):
                subj = f'S{subj_id:02d}'
                for act_id in range(1, 28):
                    act = f'A{act_id:02d}'
                    csi_dir = os.path.join(self.data_root, env, subj, act, 'wifi-csi')
                    gt_path = os.path.join(self.data_root, env, subj, act, 'ground_truth.npy')
                    if not os.path.exists(csi_dir) or not os.path.exists(gt_path):
                        continue
                    n = len(glob.glob(os.path.join(csi_dir, 'frame*.mat')))
                    if n == 0:
                        continue
                    if n < self.seq_len:
                        self.samples.append(dict(env=env, subject=subj, action=act,
                                                 start_frame=0, num_frames=n,
                                                 csi_dir=csi_dir, gt_path=gt_path))
                    else:
                        for s in range(0, n - self.seq_len + 1, self.stride):
                            self.samples.append(dict(env=env, subject=subj, action=act,
                                                     start_frame=s, num_frames=n,
                                                     csi_dir=csi_dir, gt_path=gt_path))
        print(f"[MMFiImageDataset] {len(self.samples)} samples from {self.envs} "
              f"(augment={'ON' if self.aug else 'OFF'}, img={self.renderer.img_size})")

    def _load_raw(self, csi_dir, start, length):
        amps, phases = [], []
        for i in range(start, start + length):
            fp = os.path.join(csi_dir, f'frame{i + 1:03d}.mat')
            if os.path.exists(fp):
                m = loadmat(fp)
                amps.append(np.nan_to_num(m['CSIamp'].astype(np.float32),
                                          nan=0.0, posinf=0.0, neginf=0.0))
                phases.append(np.nan_to_num(m['CSIphase'].astype(np.float32),
                                            nan=0.0, posinf=0.0, neginf=0.0))
            else:
                amps.append(np.zeros((3, 114, 10), np.float32))
                phases.append(np.zeros((3, 114, 10), np.float32))
        return np.stack(amps), np.stack(phases)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        start, n = s['start_frame'], s['num_frames']
        L = min(self.seq_len, n - start)

        amp, phase = self._load_raw(s['csi_dir'], start, L)      # (L,3,114,10)
        if self.aug is not None:
            amp, phase = self.aug(amp, phase)
        img = self.renderer.preprocess(amp, phase)               # (L,3,H,W)

        gt = np.load(s['gt_path']).astype(np.float32)[start:start + L]

        if L < self.seq_len:
            pad = self.seq_len - L
            img = np.pad(img, ((0, pad), (0, 0), (0, 0), (0, 0)), mode='edge')
            gt = np.pad(gt, ((0, pad), (0, 0), (0, 0)), mode='edge')

        return {
            'csi': torch.from_numpy(img),          # (T,3,H,W)
            'pose_3d': torch.from_numpy(gt),
            'env': s['env'], 'subject': s['subject'], 'action': s['action'],
        }


class MMFiImageSyntheticDataset(Dataset):
    """Offline smoke-test dataset (no .mat files needed)."""
    def __init__(self, num_samples=40, seq_len=64, img_size=112, num_envs=3):
        self.n = num_samples; self.seq_len = seq_len; self.img = img_size
        self.envs = [f'E{i+1:02d}' for i in range(num_envs)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            'csi': torch.rand(self.seq_len, 3, self.img, self.img),
            'pose_3d': torch.randn(self.seq_len, 17, 3) * 0.3,
            'env': self.envs[idx % len(self.envs)],
            'subject': f'S{idx % 10 + 1:02d}',
            'action': f'A{idx % 27 + 1:02d}',
        }


def build_image_dataloaders(args, synthetic=False):
    img_size = getattr(args, 'vision_img_size', 112)
    if synthetic:
        train_ds = MMFiImageSyntheticDataset(80, args.seq_len, img_size, len(args.train_envs))
        test_ds = MMFiImageSyntheticDataset(20, args.seq_len, img_size, 1)
    else:
        train_ds = MMFiImageDataset(args.data_root, args.train_envs, args.seq_len,
                                    augment=True, img_size=img_size)
        test_ds = MMFiImageDataset(args.data_root, [args.test_env], args.seq_len,
                                   stride=args.seq_len, augment=False, img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    return train_loader, test_loader