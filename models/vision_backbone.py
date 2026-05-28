"""
Vision-backbone CSI encoder (ImageNet-pretrained) — Plan: "use a vision backbone".

Drop-in replacement for the front-end trio of CSIRSCPoseDG:
    csi_encoder -> local_encoder -> feature_pooling
i.e. the part that maps each CSI frame (9, 114, 10) to a per-frame feature.
Everything downstream (GlobalTemporalModeler, RSC, PoseDecoder, ActionClassifier)
is left UNCHANGED because the output shape is identical: (B, T, out_dim).

Pipeline per frame:
    (9, 114, 10)
      -> InstanceNorm2d  (per-sample env normalization; a DG helper the
                          ImageNet stem otherwise lacks)
      -> bilinear resize to (img_size, img_size)
      -> timm backbone (in_chans=9, num_classes=0, global_pool='avg')
      -> Linear -> LayerNorm -> GELU  (project backbone dim -> out_dim=global_dim)

Why this exists
---------------
It is an ALTERNATIVE to your Stage-1A MAE pretraining: instead of self-supervised
pretraining on source CSI, we borrow ImageNet-pretrained spatial features. So the
fair comparison is:  (single-stage train.py with this backbone)  vs
                     (MAE -> Stage 2 with the original backbone).

Honest caveats (please read before trusting the numbers)
--------------------------------------------------------
1. ImageNet statistics (edges/textures of natural scenes) are far from CSI
   subcarrier-antenna heatmaps. Transfer is NOT guaranteed to help and may hurt.
2. The (114, 10) plane lacks the spatial-physical correspondence image backbones
   assume (this repo already documents the HMSF negative result for the same
   reason). Upsampling the 10-wide packet axis to img_size is heavy interpolation.
3. ImageNet backbones bring BatchNorm running stats that absorb domain shift, and
   they drop your InstanceNorm-gating + MixStyle. We re-add InstanceNorm on the
   input and keep MixStyleTemporal in GlobalTemporalModeler, but DG robustness may
   still be weaker than the original encoder. Watch PA-MPJPE and per-env spread.
4. The backbone is pretrained; the rest is fresh. Use a LOWER LR on the backbone
   (see train_vision.py differential LR), or a flat lr ~1e-3 will wreck the
   ImageNet features in the first few steps.

Dependency: pip install timm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    _HAS_TIMM = True
except ImportError:  # keep `import models` working for non-vision users
    _HAS_TIMM = False


# Backbones that need an explicit img_size at construction (patch/window based)
_TRANSFORMER_KEYS = ('vit', 'swin', 'deit', 'beit', 'pvt', 'twins', 'cait', 'xcit')


class VisionBackboneEncoder(nn.Module):
    """ImageNet-pretrained per-frame CSI encoder.

    Args:
        in_channels:  CSI channels (3 amp + 6 phase = 9).
        out_dim:      output feature dim per frame. MUST equal args.global_dim
                      (128) so GlobalTemporalModeler's in_dim matches.
        arch:         any timm model name, e.g. 'resnet18' / 'resnet50' /
                      'vit_small_patch16_224' / 'swin_tiny_patch4_window7_224'.
        pretrained:   load ImageNet weights (downloads on first use).
        img_size:     each frame is bilinearly resized to (img_size, img_size).
        instance_norm: per-sample InstanceNorm on the raw CSI channels.
        freeze_backbone: freeze backbone params (train only adapter+proj+head).
        chunk_size:   process T in chunks of this many frames to bound memory.
    """

    def __init__(self,
                 in_channels: int = 9,
                 out_dim: int = 128,
                 arch: str = 'resnet18',
                 pretrained: bool = True,
                 img_size: int = 112,
                 instance_norm: bool = True,
                 freeze_backbone: bool = False,
                 weights_path: str = None,
                 chunk_size: int = 16):
        """
        weights_path: path to a LOCAL timm-format checkpoint (.safetensors/.pth).
            Use this on offline machines that cannot reach huggingface.co. The
            file is loaded through timm's own pipeline so the pretrained 3-channel
            stem is correctly adapted to in_channels (do NOT load it by hand —
            a manual load_state_dict silently drops the first conv on the channel
            mismatch and leaves the stem random). Overrides `pretrained`.
        """
        super().__init__()
        if not _HAS_TIMM:
            raise ImportError(
                "VisionBackboneEncoder requires `timm`. Install: pip install timm")

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.img_size = img_size
        self.chunk_size = chunk_size
        self.arch = arch

        # Per-sample env normalization (mirrors EnvironmentNormalization in the
        # original csi_encoder). Helps cancel per-environment global statistics.
        self.env_norm = (nn.InstanceNorm2d(in_channels, affine=True)
                         if instance_norm else nn.Identity())

        # timm adapts the pretrained stem to arbitrary in_chans automatically
        # (sums/repeats the RGB conv weights), and num_classes=0 + global_pool='avg'
        # returns a pooled feature vector.
        is_transformer = any(k in arch for k in _TRANSFORMER_KEYS)
        kwargs = dict(num_classes=0, global_pool='avg', in_chans=in_channels)
        if is_transformer:
            # transformers fix sequence length at build time -> tell them img_size
            kwargs['img_size'] = img_size

        if weights_path:
            # Offline: load a local checkpoint through timm's pipeline so the
            # 3->in_channels stem adaptation still happens.
            kwargs['pretrained'] = True
            kwargs['pretrained_cfg_overlay'] = dict(file=weights_path)
        else:
            kwargs['pretrained'] = pretrained

        try:
            self.backbone = timm.create_model(arch, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to build timm backbone '{arch}'.\n"
                f"If this is a download/connection error (e.g. huggingface.co "
                f"timed out), the machine cannot fetch ImageNet weights. Options:\n"
                f"  1) Use a mirror:   export HF_ENDPOINT=https://hf-mirror.com\n"
                f"  2) Offline cache:  download on a connected box, copy "
                f"~/.cache/huggingface, then export HF_HUB_OFFLINE=1\n"
                f"  3) Local file:     pass --vision_weights /path/to/{arch}.safetensors\n"
                f"  4) No pretrain:    pass --vision_scratch (defeats the purpose)\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e
        feat_dim = self.backbone.num_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Project backbone feature dim -> out_dim (global_dim). Matches the
        # Linear+LayerNorm+GELU projection style of the original LocalFeaturePooling.
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, H, W) -> (N, out_dim)"""
        x = self.env_norm(x)
        x = F.interpolate(x, size=(self.img_size, self.img_size),
                          mode='bilinear', align_corners=False)
        feat = self.backbone(x)         # (N, feat_dim)
        return self.proj(feat)          # (N, out_dim)

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        """csi: (B, T, C, H, W) -> (B, T, out_dim)"""
        B, T, C, H, W = csi.shape
        outs = []
        for t0 in range(0, T, self.chunk_size):
            t1 = min(t0 + self.chunk_size, T)
            n = t1 - t0
            chunk = csi[:, t0:t1].reshape(B * n, C, H, W)
            feat = self._encode_frames(chunk)          # (B*n, out_dim)
            outs.append(feat.reshape(B, n, self.out_dim))
        return torch.cat(outs, dim=1)                  # (B, T, out_dim)


if __name__ == "__main__":
    # Shape sanity check. pretrained=False here so it runs offline; on your
    # machine set pretrained=True (the default in the model).
    for arch in ['resnet18', 'resnet50']:
        enc = VisionBackboneEncoder(in_channels=9, out_dim=128, arch=arch,
                                    pretrained=False, img_size=112)
        x = torch.randn(2, 8, 9, 114, 10)   # (B=2, T=8, 9, 114, 10)
        y = enc(x)
        n_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        print(f"{arch:>10}: in {tuple(x.shape)} -> out {tuple(y.shape)} | "
              f"params {n_train/1e6:.2f}M | backbone_feat {enc.backbone.num_features}")
    assert y.shape == (2, 8, 128)
    print("OK")