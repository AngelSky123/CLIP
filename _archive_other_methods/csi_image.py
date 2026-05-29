"""
CSI -> image renderer (PerceptAlign-style), adapted for MMFi.

Goal: keep the three-stage framework (MAE -> Action -> Pose) unchanged, and only
change the DATA representation: render each CSI frame into a float 3-channel
"image", exactly in the spirit of PerceptAlign/tools/preprocess.py:

  PerceptAlign per (frame, receiver):
      q   = antenna_quotient(ant0 / ant1)           # denoise phase
      ch0 = normalize(resize(angle(q)))             # phase  image
      ch1 = normalize(resize(abs(q)))               # amp    image
      ch2 = normalize(resize(|STFT(q)|.mean))       # DFS / Doppler image
      img = cat([ch0, ch1, ch2])  -> (3, 224, 224)  # fed to ResNet

MMFi adaptation
---------------
MMFi raw per frame = (3 antenna, 114 subcarrier, 10 packet). The packet axis is
only length 10, so PerceptAlign's STFT-based Doppler is not reproducible
per-frame. We substitute a *coarse per-frame micro-Doppler*: the magnitude of a
10-point FFT along the packet axis. Same 3-channel spirit (phase / amp /
doppler), produced per frame so the downstream T-frame token structure is kept.

Output: (T, 3, H, W) float32 in [0, 1] — a drop-in replacement for the
(T, 9, 114, 10) tensor returned by the original CSIPreprocessor.preprocess().

IMPORTANT: any raw-CSI augmentation (CSIAugmentor) must run on the raw amp/phase
BEFORE rendering — it operates on (T,9,114,10), not on images. Render last.
"""
from typing import List

import numpy as np
import torch
import torch.fft
import torch.nn.functional as F


def _resize_norm(planes: torch.Tensor, size) -> torch.Tensor:
    """planes: (T, C, H0, W0) real -> (T, C, size, size), per-image min-max to [0,1].

    Mirrors PerceptAlign.compute_feature: bilinear interpolate then normalize.
    """
    planes = F.interpolate(planes, size=size, mode="bilinear", align_corners=False)
    T, C, H, W = planes.shape
    flat = planes.reshape(T, C, H * W)
    mn = flat.amin(dim=-1, keepdim=True)
    mx = flat.amax(dim=-1, keepdim=True)
    rng = (mx - mn).clamp_min(1e-12)
    flat = (flat - mn) / rng                      # constant images -> ~0
    return flat.reshape(T, C, H, W)


class CSIImagePreprocessor:
    """Render MMFi CSI sequences into PerceptAlign-style 3-channel images.

    Args:
        img_size:     output square size (PerceptAlign uses 224; 112 is faster).
        channels:     ordered subset of {'phase','amp','doppler'} (default all 3).
        antenna_quotient: if True, denoise via ant0/ant1 complex quotient
                          (PerceptAlign-style); else use per-antenna mean.
    """

    def __init__(self,
                 img_size: int = 224,
                 channels: List[str] = ("phase", "amp", "doppler"),
                 antenna_quotient: bool = True,
                 eps: float = 1e-9):
        self.img_size = int(img_size)
        self.channels = list(channels)
        self.antenna_quotient = bool(antenna_quotient)
        self.eps = float(eps)

    def _complex_planes(self, amp: np.ndarray, phase: np.ndarray):
        """amp/phase: (T, 3, 114, 10) -> dict of real planes (T, 114, 10)."""
        amp = torch.from_numpy(np.nan_to_num(amp.astype(np.float32)))
        phase = torch.from_numpy(np.nan_to_num(phase.astype(np.float32)))
        c = amp * torch.exp(1j * phase.to(torch.float32))     # (T,3,114,10) complex

        if self.antenna_quotient and c.shape[1] >= 2:
            q = c[:, 0] / (c[:, 1] + self.eps)                 # (T,114,10) complex
        else:
            q = c.mean(dim=1)                                  # (T,114,10) complex

        out = {}
        if "phase" in self.channels:
            out["phase"] = torch.angle(q)
        if "amp" in self.channels:
            out["amp"] = torch.abs(q)
        if "doppler" in self.channels:
            # coarse per-frame micro-Doppler: |FFT along the 10-packet axis|
            dop = torch.fft.fft(q, dim=-1)                     # (T,114,10) complex
            out["doppler"] = torch.abs(dop)
        return out

    def preprocess(self, amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """amp, phase: (T, 3, 114, 10) -> (T, len(channels), img_size, img_size) float32."""
        planes = self._complex_planes(amp, phase)
        stack = torch.stack([planes[k] for k in self.channels], dim=1)  # (T,C,114,10)
        img = _resize_norm(stack, (self.img_size, self.img_size))       # (T,C,H,W)
        return np.nan_to_num(img.numpy().astype(np.float32))


if __name__ == "__main__":
    # Shape sanity check on synthetic MMFi frames.
    T = 8
    amp = np.random.rand(T, 3, 114, 10).astype(np.float32)
    phase = (np.random.rand(T, 3, 114, 10).astype(np.float32) - 0.5) * 6.28

    for size in (112, 224):
        pre = CSIImagePreprocessor(img_size=size)
        out = pre.preprocess(amp, phase)
        assert out.shape == (T, 3, size, size), out.shape
        print(f"img_size={size}: (T,3,114,10) -> {out.shape} | "
              f"range [{out.min():.3f}, {out.max():.3f}] | "
              f"per-channel mean {out.mean((0,2,3)).round(3)}")
    # single-channel / no-quotient variants
    pre2 = CSIImagePreprocessor(img_size=112, channels=["amp"], antenna_quotient=False)
    print("amp-only, antenna-mean:", pre2.preprocess(amp, phase).shape)
    print("OK")