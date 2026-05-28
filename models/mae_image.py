"""
Image-version MAE for the three-stage framework (Stage 1A).

Mirrors models/mae_pretrain.py exactly, but for the IMAGE representation:
  * Input is a rendered CSI image sequence (B, T, 3, H, W) from csi_image.py.
  * Backbone = VisionBackboneEncoder (per-frame) + GlobalTemporalModeler (temporal),
    i.e. the same two modules the image three-stage pipeline shares.
  * Patchify is standard 2D image patchify (original MAE), not the custom CSI
    patchify, because the input is now a 2D image.

Reconstruction design (identical in spirit to CSIMaeModel):
  masked image -> backbone -> z_global (B,T,global_dim) [one vector per frame]
  decoder reconstructs each masked patch from (per-frame global feature + patch
  position embedding). Loss = normalized MSE on masked patches only.

This pretrains BOTH vision_backbone and global_modeler, which Stage 1B/2 then load.
"""
import torch
import torch.nn as nn


def random_masking(patches: torch.Tensor, mask_ratio: float):
    """patches: (N, num_patches, patch_dim). Returns masked patches + mask (1=masked)."""
    N, P, _ = patches.shape
    n_mask = int(P * mask_ratio)
    noise = torch.rand(N, P, device=patches.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_mask = ids_shuffle[:, :n_mask]
    mask = torch.zeros(N, P, device=patches.device)
    mask.scatter_(1, ids_mask, 1.0)
    masked = patches * (1.0 - mask).unsqueeze(-1)
    return masked, mask


class ImagePatchify(nn.Module):
    """Standard non-overlapping 2D patchify, per frame.

    (N, C, H, W) <-> (N, num_patches, C*patch*patch)
    """
    def __init__(self, in_channels: int = 3, img_size: int = 112, patch_size: int = 16):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size {img_size} not divisible by patch_size {patch_size}"
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.patch_dim = in_channels * patch_size * patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        P, g = self.patch_size, self.grid
        x = x.reshape(N, C, g, P, g, P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()      # (N, g, g, C, P, P)
        return x.reshape(N, g * g, C * P * P)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        N = patches.shape[0]
        C, P, g = self.in_channels, self.patch_size, self.grid
        x = patches.reshape(N, g, g, C, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()      # (N, C, g, P, g, P)
        return x.reshape(N, C, g * P, g * P)


class ImageMaeModel(nn.Module):
    """Wraps vision_backbone + global_modeler and adds a lightweight patch decoder."""

    def __init__(self,
                 vision_backbone: nn.Module,
                 global_modeler: nn.Module,
                 in_channels: int = 3,
                 img_size: int = 112,
                 patch_size: int = 16,
                 global_dim: int = 128,
                 mask_ratio: float = 0.75,
                 pos_embed_dim: int = 32,
                 decoder_hidden: int = 256):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.global_modeler = global_modeler

        # vision_backbone resizes internally to its own img_size; keep them equal
        # so the patch mask isn't smeared by interpolation.
        assert getattr(vision_backbone, 'img_size', img_size) == img_size, (
            f"vision_backbone.img_size ({getattr(vision_backbone,'img_size',None)}) "
            f"must equal MAE img_size ({img_size})")

        self.patchifier = ImagePatchify(in_channels, img_size, patch_size)
        self.patch_dim = self.patchifier.patch_dim
        self.num_patches = self.patchifier.num_patches
        self.mask_ratio = mask_ratio
        self.global_dim = global_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, pos_embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.pos_embed_dim = pos_embed_dim

        self.decoder = nn.Sequential(
            nn.Linear(global_dim + pos_embed_dim, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, self.patch_dim),
        )

    def forward_backbone(self, img: torch.Tensor) -> torch.Tensor:
        """img (B,T,C,H,W) -> z_global (B,T,global_dim)"""
        z_seq = self.vision_backbone(img)        # (B,T,global_dim)
        return self.global_modeler(z_seq)        # (B,T,global_dim)

    def forward(self, img: torch.Tensor):
        B, T, C, H, W = img.shape
        x = img.reshape(B * T, C, H, W)

        target = self.patchifier(x)                                  # (B*T, np, pd)
        masked, mask = random_masking(target, self.mask_ratio)
        masked_img = self.patchifier.unpatchify(masked).reshape(B, T, C, H, W)

        z_global = self.forward_backbone(masked_img)                 # (B,T,gd)
        z = z_global.reshape(B * T, self.global_dim).unsqueeze(1)
        z = z.expand(B * T, self.num_patches, self.global_dim)
        pos = self.pos_embed.expand(B * T, self.num_patches, self.pos_embed_dim)
        pred = self.decoder(torch.cat([z, pos], dim=-1))             # (B*T, np, pd)

        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target_norm = (target - mean) / (var + 1e-6).sqrt()

        loss_per_patch = ((pred - target_norm) ** 2).mean(dim=-1)    # (B*T, np)
        loss = (loss_per_patch * mask).sum() / (mask.sum() + 1e-6)
        return loss, {
            'mask_ratio_actual': mask.mean().item(),
            'num_patches': self.num_patches,
            'pred': pred.detach(),
        }


if __name__ == "__main__":
    import os, sys, warnings
    warnings.filterwarnings('ignore')
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.vision_backbone import VisionBackboneEncoder
    from models.global_encoder import GlobalTemporalModeler

    IMG = 112
    vb = VisionBackboneEncoder(in_channels=3, out_dim=128, arch='resnet18',
                               pretrained=False, img_size=IMG)
    gm = GlobalTemporalModeler(in_dim=128, global_dim=128, num_transformer_layers=3,
                               num_heads=4, tcn_channels=[128, 128], tcn_kernel_size=3,
                               dropout=0.1, max_seq_len=128)
    mae = ImageMaeModel(vb, gm, in_channels=3, img_size=IMG, patch_size=16,
                        global_dim=128, mask_ratio=0.75)

    x = torch.randn(2, 8, 3, IMG, IMG)
    loss, info = mae(x)
    print(f"input {tuple(x.shape)} | loss {loss.item():.4f} | "
          f"patches {info['num_patches']} | mask {info['mask_ratio_actual']:.2f} | "
          f"patch_dim {mae.patch_dim}")
    loss.backward()
    for name, mod in [('vision_backbone', mae.vision_backbone),
                      ('global_modeler', mae.global_modeler),
                      ('decoder', mae.decoder)]:
        g = any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())
        print(f"  {name} grad: {'OK' if g else 'NONE'}")
    print("OK")