"""
MAE Self-Supervised Pretraining for CSI — Stage 1A.

Adapted to the actual CSI-RSC-PoseDG codebase:
  - seq_len = 64 (not 30)
  - 4 separate backbone modules: csi_encoder, local_encoder,
    feature_pooling, global_modeler
  - Input CSI shape: (B, T=64, 9, 114, 10)
  - Patch defaults chosen so that 64 % patch_t == 0, 114 % patch_s == 0,
    10 % patch_a == 0:
        patch_t=4  -> 16 chunks
        patch_s=19 -> 6  chunks
        patch_a=5  -> 2  chunks
        => 192 patches per sample
        => patch_dim = 4 * 19 * 5 * 9 = 3420
"""
import math
import torch
import torch.nn as nn


class CSIPatchify(nn.Module):
    def __init__(self, in_channels: int = 9,
                 patch_t: int = 4, patch_s: int = 19, patch_a: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.patch_t = patch_t
        self.patch_s = patch_s
        self.patch_a = patch_a
        self.patch_dim = in_channels * patch_t * patch_s * patch_a

    def forward(self, x: torch.Tensor):
        B, T, C, S, A = x.shape
        Pt, Ps, Pa = self.patch_t, self.patch_s, self.patch_a
        assert T % Pt == 0 and S % Ps == 0 and A % Pa == 0, (
            f"shape ({T},{S},{A}) not divisible by patch ({Pt},{Ps},{Pa})"
        )
        nt, ns, na = T // Pt, S // Ps, A // Pa
        x = x.reshape(B, nt, Pt, C, ns, Ps, na, Pa)
        x = x.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
        return x.reshape(B, nt * ns * na, self.patch_dim), (nt, ns, na)

    def unpatchify(self, patches: torch.Tensor, grid):
        B, N, _ = patches.shape
        nt, ns, na = grid
        Pt, Ps, Pa, C = self.patch_t, self.patch_s, self.patch_a, self.in_channels
        x = patches.reshape(B, nt, ns, na, Pt, Ps, Pa, C)
        x = x.permute(0, 1, 4, 7, 2, 5, 3, 6).contiguous()
        T, S, A = nt * Pt, ns * Ps, na * Pa
        return x.reshape(B, T, C, S, A)


def random_masking(patches: torch.Tensor, mask_ratio: float):
    B, N, _ = patches.shape
    n_mask = int(N * mask_ratio)
    noise = torch.rand(B, N, device=patches.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_mask = ids_shuffle[:, :n_mask]
    mask = torch.zeros(B, N, device=patches.device)
    mask.scatter_(1, ids_mask, 1.0)
    masked_patches = patches * (1.0 - mask).unsqueeze(-1)
    return masked_patches, mask


class CSIMaeModel(nn.Module):
    """Wraps the 4 existing backbone modules and adds a lightweight decoder."""

    def __init__(self,
                 csi_encoder: nn.Module,
                 local_encoder: nn.Module,
                 feature_pooling: nn.Module,
                 global_modeler: nn.Module,
                 in_channels: int = 9,
                 patch_t: int = 4,
                 patch_s: int = 19,
                 patch_a: int = 5,
                 global_dim: int = 128,
                 mask_ratio: float = 0.75,
                 decoder_hidden: int = 256):
        super().__init__()
        self.csi_encoder = csi_encoder
        self.local_encoder = local_encoder
        self.feature_pooling = feature_pooling
        self.global_modeler = global_modeler

        self.patchifier = CSIPatchify(in_channels, patch_t, patch_s, patch_a)
        self.patch_dim = self.patchifier.patch_dim
        self.mask_ratio = mask_ratio
        self.global_dim = global_dim

        self.pos_embed_dim = 32
        self.decoder = nn.Sequential(
            nn.Linear(global_dim + self.pos_embed_dim, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, self.patch_dim),
        )

        self.register_buffer('_pos_buf', torch.zeros(1, self.pos_embed_dim),
                             persistent=False)
        self._pos_n = 0

    def _sinusoidal_pos(self, n: int, device: torch.device):
        if self._pos_n == n and self._pos_buf.device == device:
            return self._pos_buf
        pe = torch.zeros(n, self.pos_embed_dim, device=device)
        position = torch.arange(n, device=device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.pos_embed_dim, 2, device=device).float()
            * -(math.log(10000.0) / self.pos_embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self._pos_buf = pe
        self._pos_n = n
        return pe

    def forward_backbone(self, csi: torch.Tensor) -> torch.Tensor:
        feat = self.csi_encoder(csi)
        z_local = self.local_encoder(feat)
        z_pooled = self.feature_pooling(z_local)
        z_global = self.global_modeler(z_pooled)
        return z_global

    def forward(self, csi: torch.Tensor):
        B, T, C, S, A = csi.shape

        target_patches, (nt, ns, na) = self.patchifier(csi)
        masked_patches, mask = random_masking(target_patches, self.mask_ratio)
        masked_csi = self.patchifier.unpatchify(masked_patches, (nt, ns, na))

        z_global = self.forward_backbone(masked_csi)

        Pt = self.patchifier.patch_t
        z_chunked = z_global.reshape(B, nt, Pt, self.global_dim).mean(dim=2)
        z_per_patch = z_chunked.unsqueeze(2).unsqueeze(3).expand(
            B, nt, ns, na, self.global_dim
        ).reshape(B, nt * ns * na, self.global_dim)

        n_spatial = ns * na
        pos = self._sinusoidal_pos(n_spatial, csi.device)
        pos = pos.unsqueeze(0).unsqueeze(0).expand(B, nt, n_spatial, self.pos_embed_dim)
        pos = pos.reshape(B, nt * n_spatial, self.pos_embed_dim)

        decoder_input = torch.cat([z_per_patch, pos], dim=-1)
        pred_patches = self.decoder(decoder_input)

        mean = target_patches.mean(dim=-1, keepdim=True)
        var = target_patches.var(dim=-1, keepdim=True)
        target_norm = (target_patches - mean) / (var + 1e-6).sqrt()

        loss_per_patch = ((pred_patches - target_norm) ** 2).mean(dim=-1)
        mask_sum = mask.sum() + 1e-6
        loss = (loss_per_patch * mask).sum() / mask_sum

        return loss, {
            'mask': mask,
            'mask_ratio_actual': mask.mean().item(),
            'pred': pred_patches.detach(),
            'z_global': z_global,
        }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hmsf_pooling import HierarchicalLocalFeaturePooling

    class StubCsiEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(9, 64, 1)
        def forward(self, x):
            x = x.permute(0, 2, 1, 3, 4)
            x = self.conv(x)
            return x.permute(0, 2, 1, 3, 4).contiguous()

    class StubLocalEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(64, 64, 3, padding=1)
        def forward(self, x):
            x = x.permute(0, 2, 1, 3, 4)
            x = self.conv(x)
            return x.permute(0, 2, 1, 3, 4).contiguous()

    class StubGlobal(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(128, 128)
        def forward(self, x):
            return self.proj(x)

    mae = CSIMaeModel(
        csi_encoder=StubCsiEnc(),
        local_encoder=StubLocalEnc(),
        feature_pooling=HierarchicalLocalFeaturePooling(64, 128),
        global_modeler=StubGlobal(),
    )
    x = torch.randn(2, 64, 9, 114, 10)
    loss, info = mae(x)
    print(f"input  : {x.shape}")
    print(f"loss   : {loss.item():.4f}")
    print(f"mask ratio : {info['mask_ratio_actual']:.3f}")
    print(f"n_patches  : {info['mask'].shape[1]}")
    print(f"patch_dim  : {mae.patch_dim}")
    loss.backward()
    print("backward OK")
    for name, mod in [('csi_encoder', mae.csi_encoder),
                      ('local_encoder', mae.local_encoder),
                      ('feature_pooling', mae.feature_pooling),
                      ('global_modeler', mae.global_modeler)]:
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in mod.parameters())
        print(f"  {name}: grad={'OK' if has_grad else 'NONE'}")