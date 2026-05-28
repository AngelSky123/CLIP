"""
MMFi distillation dataset: loads aligned depth frames (+ optional CSI) + GT pose.

Depth handling (critical):
  * 16-bit millimetre PNGs (480x640). Normalized with a FIXED physical scale
    (clip to [0, depth_clip] mm, divide by depth_clip), NOT per-image min-max.
    Per-image normalization would erase absolute distance — exactly the cue we
    want the teacher to keep for global hip localization.
  * Resized to (img_size, img_size); invalid pixels (0) stay 0.

with_csi / with_depth flags:
  * Step A (teacher):       with_depth=True,  with_csi=False
  * Step B (distillation):  with_depth=True,  with_csi=True   (train)
                            depth not needed at test (student is CSI-only)
"""
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ENV_SUBJECTS = {
    'E01': list(range(1, 11)), 'E02': list(range(11, 21)),
    'E03': list(range(21, 31)), 'E04': list(range(31, 41)),
}


def load_depth_seq(depth_dir, start, length, img_size, depth_clip):
    """Return (length, 1, img_size, img_size) float in [0,1], fixed-scale."""
    frames = []
    for i in range(start, start + length):
        fp = os.path.join(depth_dir, f'frame{i + 1:03d}.png')
        if os.path.exists(fp):
            d = np.array(Image.open(fp)).astype(np.float32)        # (480,640) uint16->f32 mm
        else:
            d = np.zeros((480, 640), np.float32)
        frames.append(d)
    d = torch.from_numpy(np.stack(frames)).unsqueeze(1)            # (L,1,480,640)
    d = torch.clamp(d, 0.0, depth_clip) / depth_clip               # fixed metric scale -> [0,1]
    d = F.interpolate(d, size=(img_size, img_size), mode='bilinear', align_corners=False)
    return d                                                       # (L,1,img,img)


class MMFiDistillDataset(Dataset):
    def __init__(self, data_root, envs, seq_len=64, stride=32,
                 with_depth=True, with_csi=False,
                 depth_img=112, depth_clip=5000.0, csi_augment=False):
        self.data_root = data_root
        self.envs = envs
        self.seq_len = seq_len
        self.stride = stride
        self.with_depth = with_depth
        self.with_csi = with_csi
        self.depth_img = depth_img
        self.depth_clip = float(depth_clip)
        self._csi_pre = None
        if with_csi:
            from dataset import CSIPreprocessor          # reuse the exact CSI pipeline
            self._csi_pre = CSIPreprocessor()
            self._csi_aug = None
            if csi_augment:
                try:
                    from augmentation import CSIAugmentor
                    self._csi_aug = CSIAugmentor(p=0.8)
                except Exception:
                    self._csi_aug = None
        self.samples = []
        self._build_index()

    def _build_index(self):
        for env in self.envs:
            for sid in ENV_SUBJECTS.get(env, []):
                subj = f'S{sid:02d}'
                for aid in range(1, 28):
                    act = f'A{aid:02d}'
                    base = os.path.join(self.data_root, env, subj, act)
                    gt = os.path.join(base, 'ground_truth.npy')
                    depth_dir = os.path.join(base, 'depth')
                    csi_dir = os.path.join(base, 'wifi-csi')
                    if not os.path.exists(gt):
                        continue
                    if self.with_depth and not os.path.isdir(depth_dir):
                        continue
                    if self.with_csi and not os.path.isdir(csi_dir):
                        continue
                    ref = depth_dir if self.with_depth else csi_dir
                    pat = 'frame*.png' if self.with_depth else 'frame*.mat'
                    n = len(glob.glob(os.path.join(ref, pat)))
                    if n == 0:
                        continue
                    if n < self.seq_len:
                        self.samples.append(dict(env=env, subject=subj, action=act,
                                                 start=0, n=n, base=base))
                    else:
                        for s in range(0, n - self.seq_len + 1, self.stride):
                            self.samples.append(dict(env=env, subject=subj, action=act,
                                                     start=s, n=n, base=base))
        print(f"[MMFiDistillDataset] {len(self.samples)} samples from {self.envs} "
              f"(depth={self.with_depth}, csi={self.with_csi})")

    def _load_csi(self, base, start, L):
        from scipy.io import loadmat
        csi_dir = os.path.join(base, 'wifi-csi')
        amps, phases = [], []
        for i in range(start, start + L):
            fp = os.path.join(csi_dir, f'frame{i + 1:03d}.mat')
            if os.path.exists(fp):
                m = loadmat(fp)
                amps.append(np.nan_to_num(m['CSIamp'].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
                phases.append(np.nan_to_num(m['CSIphase'].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
            else:
                amps.append(np.zeros((3, 114, 10), np.float32))
                phases.append(np.zeros((3, 114, 10), np.float32))
        csi = self._csi_pre.preprocess(np.stack(amps), np.stack(phases))   # (L,9,114,10)
        return csi

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        start, n = s['start'], s['n']
        L = min(self.seq_len, n - start)
        gt = np.load(os.path.join(s['base'], 'ground_truth.npy')).astype(np.float32)[start:start + L]

        out = {'pose_3d': None, 'env': s['env'], 'subject': s['subject'], 'action': s['action']}

        if self.with_depth:
            depth = load_depth_seq(os.path.join(s['base'], 'depth'), start, L,
                                   self.depth_img, self.depth_clip)        # (L,1,H,W) tensor
            if L < self.seq_len:
                depth = torch.cat([depth, depth[-1:].repeat(self.seq_len - L, 1, 1, 1)], 0)
            out['depth'] = depth

        if self.with_csi:
            csi = self._load_csi(s['base'], start, L)                      # (L,9,114,10)
            if L < self.seq_len:
                csi = np.pad(csi, ((0, self.seq_len - L), (0, 0), (0, 0), (0, 0)), mode='edge')
            csi_t = torch.from_numpy(csi)
            if self._csi_aug is not None:
                csi_t = self._csi_aug(csi_t)
            out['csi'] = csi_t

        if L < self.seq_len:
            gt = np.pad(gt, ((0, self.seq_len - L), (0, 0), (0, 0)), mode='edge')
        out['pose_3d'] = torch.from_numpy(gt)
        return out


class MMFiDistillSyntheticDataset(Dataset):
    def __init__(self, num_samples=40, seq_len=64, depth_img=112,
                 with_depth=True, with_csi=False, num_envs=3):
        self.n = num_samples; self.seq_len = seq_len; self.img = depth_img
        self.with_depth = with_depth; self.with_csi = with_csi
        self.envs = [f'E{i+1:02d}' for i in range(num_envs)]
    def __len__(self): return self.n
    def __getitem__(self, idx):
        out = {'pose_3d': torch.randn(self.seq_len, 17, 3) * 0.3,
               'env': self.envs[idx % len(self.envs)],
               'subject': f'S{idx%10+1:02d}', 'action': f'A{idx%27+1:02d}'}
        if self.with_depth:
            out['depth'] = torch.rand(self.seq_len, 1, self.img, self.img)
        if self.with_csi:
            out['csi'] = torch.randn(self.seq_len, 9, 114, 10)
        return out


def build_teacher_dataloaders(args, synthetic=False):
    """Step A: depth-only loaders (E01-E03 train, E04 eval)."""
    if synthetic:
        tr = MMFiDistillSyntheticDataset(80, args.seq_len, args.depth_img, True, False, len(args.train_envs))
        te = MMFiDistillSyntheticDataset(20, args.seq_len, args.depth_img, True, False, 1)
    else:
        tr = MMFiDistillDataset(args.data_root, args.train_envs, args.seq_len,
                                with_depth=True, with_csi=False,
                                depth_img=args.depth_img, depth_clip=args.depth_clip)
        te = MMFiDistillDataset(args.data_root, [args.test_env], args.seq_len, stride=args.seq_len,
                                with_depth=True, with_csi=False,
                                depth_img=args.depth_img, depth_clip=args.depth_clip)
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
    vl = DataLoader(te, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    return tl, vl