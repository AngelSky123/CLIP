"""
MMFi distillation dataset: aligned teacher frames (depth and/or RGB) + optional CSI + GT pose.

本版改动:
  * 新增 with_rgb / load_rgb_seq: 逐帧 PNG (640x480 RGB, 与 depth 同名编号),
    读为 float[0,1] (L,3,img,img)。ImageNet 归一化在 RGB 教师模型内部做, 这里只出 [0,1]。
  * 新增 rgb_root: RGB 可放在独立磁盘 (如机械盘), GT/CSI/depth 仍从 data_root 读。
    自动探测两种布局: <rgb_root>/E/S/A/rgb/frameNNN.png 或 <rgb_root>/E/S/A/frameNNN.png。
    rgb_root=None 时退回 data_root (原行为)。
  * 移除 csi_rawscale (rawscale 实验已证否并回退, 主线无人消费)。
  * depth 路径一字不动。

Depth handling (critical, 不变):
  * 16-bit millimetre PNGs (480x640). Normalized with a FIXED physical scale
    (clip to [0, depth_clip] mm, divide by depth_clip), NOT per-image min-max.

flags:
  * 教师训练 (depth): with_depth=True,  with_csi=False
  * 教师训练 (rgb):   with_rgb=True,   with_csi=False
  * 蒸馏 Step B:      with_csi=True + (with_depth 或 with_rgb, 按 --teacher_modality)
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


def load_rgb_seq(rgb_dir, start, length, img_size):
    """Return (length, 3, img_size, img_size) float in [0,1] (ImageNet 归一化在模型内做)。"""
    frames = []
    for i in range(start, start + length):
        fp = os.path.join(rgb_dir, f'frame{i + 1:03d}.png')
        if os.path.exists(fp):
            im = np.array(Image.open(fp).convert('RGB'), dtype=np.float32)  # (480,640,3)
        else:
            im = np.zeros((480, 640, 3), np.float32)
        frames.append(im)
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2) / 255.0      # (L,3,480,640)
    x = F.interpolate(x, size=(img_size, img_size), mode='bilinear', align_corners=False)
    return x                                                                # (L,3,img,img)


class MMFiDistillDataset(Dataset):
    def __init__(self, data_root, envs, seq_len=64, stride=32,
                 with_depth=True, with_rgb=False, with_csi=False,
                 depth_img=112, depth_clip=5000.0, rgb_img=112,
                 rgb_root=None, csi_augment=False):
        self.data_root = data_root
        self.rgb_root = rgb_root if rgb_root else data_root
        self.envs = envs
        self.seq_len = seq_len
        self.stride = stride
        self.with_depth = with_depth
        self.with_rgb = with_rgb
        self.with_csi = with_csi
        self.depth_img = depth_img
        self.depth_clip = float(depth_clip)
        self.rgb_img = rgb_img
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

    def _resolve_rgb_dir(self, env, subj, act):
        """自动探测 RGB 布局: <rgb_root>/E/S/A/rgb 或 <rgb_root>/E/S/A (帧直接放动作目录)。"""
        c1 = os.path.join(self.rgb_root, env, subj, act, 'rgb')
        c2 = os.path.join(self.rgb_root, env, subj, act)
        if os.path.isdir(c1):
            return c1
        if os.path.isdir(c2) and glob.glob(os.path.join(c2, 'frame*.png')):
            return c2
        return None

    def _build_index(self):
        for env in self.envs:
            for sid in ENV_SUBJECTS.get(env, []):
                subj = f'S{sid:02d}'
                for aid in range(1, 28):
                    act = f'A{aid:02d}'
                    base = os.path.join(self.data_root, env, subj, act)
                    gt = os.path.join(base, 'ground_truth.npy')
                    depth_dir = os.path.join(base, 'depth')
                    rgb_dir = self._resolve_rgb_dir(env, subj, act) if self.with_rgb else None
                    csi_dir = os.path.join(base, 'wifi-csi')
                    if not os.path.exists(gt):
                        continue
                    if self.with_depth and not os.path.isdir(depth_dir):
                        continue
                    if self.with_rgb and rgb_dir is None:
                        continue
                    if self.with_csi and not os.path.isdir(csi_dir):
                        continue
                    # 帧数参照: depth > rgb > csi (有哪个用哪个)
                    if self.with_depth:
                        ref, pat = depth_dir, 'frame*.png'
                    elif self.with_rgb:
                        ref, pat = rgb_dir, 'frame*.png'   # rgb_dir 已是解析后的绝对路径
                    else:
                        ref, pat = csi_dir, 'frame*.mat'
                    n = len(glob.glob(os.path.join(ref, pat)))
                    if n == 0:
                        continue
                    if n < self.seq_len:
                        self.samples.append(dict(env=env, subject=subj, action=act,
                                                 start=0, n=n, base=base, rgb_dir=rgb_dir))
                    else:
                        for s in range(0, n - self.seq_len + 1, self.stride):
                            self.samples.append(dict(env=env, subject=subj, action=act,
                                                     start=s, n=n, base=base, rgb_dir=rgb_dir))
        print(f"[MMFiDistillDataset] {len(self.samples)} samples from {self.envs} "
              f"(depth={self.with_depth}, rgb={self.with_rgb}, csi={self.with_csi}, "
              f"rgb_root={self.rgb_root if self.with_rgb else '-'})")
        if self.with_rgb and len(self.samples) == 0:
            e, s = self.envs[0], f'S{ENV_SUBJECTS.get(self.envs[0], [1])[0]:02d}'
            print(f"[MMFiDistillDataset][HINT] 0 样本。请确认以下两点:\n"
                  f"  1) GT 在 data_root 下:    {os.path.join(self.data_root, e, s, 'A01', 'ground_truth.npy')}\n"
                  f"  2) RGB 在 rgb_root 下:    {os.path.join(self.rgb_root, e, s, 'A01', 'rgb')}  或\n"
                  f"                            {os.path.join(self.rgb_root, e, s, 'A01')}/frame*.png\n"
                  f"  GT/CSI 用 --data_root, RGB 盘用 --rgb_root, 两者可不同。")

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
        return self._csi_pre.preprocess(np.stack(amps), np.stack(phases))   # (L,9,114,10)

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
                                   self.depth_img, self.depth_clip)        # (L,1,H,W)
            if L < self.seq_len:
                depth = torch.cat([depth, depth[-1:].repeat(self.seq_len - L, 1, 1, 1)], 0)
            out['depth'] = depth

        if self.with_rgb:
            rgb = load_rgb_seq(s['rgb_dir'], start, L, self.rgb_img)                    # (L,3,H,W)
            if L < self.seq_len:
                rgb = torch.cat([rgb, rgb[-1:].repeat(self.seq_len - L, 1, 1, 1)], 0)
            out['rgb'] = rgb

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
    def __init__(self, num_samples=40, seq_len=64, depth_img=112, rgb_img=112,
                 with_depth=True, with_rgb=False, with_csi=False, num_envs=3):
        self.n = num_samples; self.seq_len = seq_len
        self.dimg = depth_img; self.rimg = rgb_img
        self.with_depth = with_depth; self.with_rgb = with_rgb; self.with_csi = with_csi
        self.envs = [f'E{i+1:02d}' for i in range(num_envs)]
    def __len__(self): return self.n
    def __getitem__(self, idx):
        out = {'pose_3d': torch.randn(self.seq_len, 17, 3) * 0.3,
               'env': self.envs[idx % len(self.envs)],
               'subject': f'S{idx%10+1:02d}', 'action': f'A{idx%27+1:02d}'}
        if self.with_depth:
            out['depth'] = torch.rand(self.seq_len, 1, self.dimg, self.dimg)
        if self.with_rgb:
            out['rgb'] = torch.rand(self.seq_len, 3, self.rimg, self.rimg)
        if self.with_csi:
            out['csi'] = torch.randn(self.seq_len, 9, 114, 10)
        return out


def build_teacher_dataloaders(args, synthetic=False, modality='depth'):
    """Step A: 教师 loaders (E01-E03 train, E04 eval)。modality: 'depth' | 'rgb'。"""
    assert modality in ('depth', 'rgb'), modality
    wd, wr = (modality == 'depth'), (modality == 'rgb')
    rgb_img = getattr(args, 'rgb_img', 112)
    rgb_root = getattr(args, 'rgb_root', None)
    if synthetic:
        tr = MMFiDistillSyntheticDataset(80, args.seq_len, args.depth_img, rgb_img, wd, wr, False, len(args.train_envs))
        te = MMFiDistillSyntheticDataset(20, args.seq_len, args.depth_img, rgb_img, wd, wr, False, 1)
    else:
        tr = MMFiDistillDataset(args.data_root, args.train_envs, args.seq_len,
                                with_depth=wd, with_rgb=wr, with_csi=False,
                                depth_img=args.depth_img, depth_clip=args.depth_clip,
                                rgb_img=rgb_img, rgb_root=rgb_root)
        te = MMFiDistillDataset(args.data_root, [args.test_env], args.seq_len, stride=args.seq_len,
                                with_depth=wd, with_rgb=wr, with_csi=False,
                                depth_img=args.depth_img, depth_clip=args.depth_clip,
                                rgb_img=rgb_img, rgb_root=rgb_root)
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
    vl = DataLoader(te, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    return tl, vl