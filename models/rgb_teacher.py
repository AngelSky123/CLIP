"""
RGB -> 3D pose teacher (Step A variant: RGB 教师).

与 DepthPoseTeacher 接口【逐位一致】:
    forward(rgb (B,T,3,H,W)) -> {'p_final': (B,T,17,3), 'z_global': (B,T,128)}
蒸馏管线 (feature/output distill) 因此一行不用改, 只换教师。

与 depth 教师的差异:
  * 逐帧编码器: ImageNet 预训练 ResNet18 截断 (--backbone resnet18, 默认);
    无 torchvision / 无法下载权重时退 --backbone scratch (与 DepthEncoder 同构的 3 通道 CNN)。
  * 输入: RGB float [0,1], (B,T,3,H,W)。ImageNet mean/std 归一化在模型内部做
    (注册为 buffer), dataloader 只负责 [0,1] —— 与 depth 管线的"dataset 出定标值"约定一致。
  * 时序建模 / pose head 与 depth 教师完全相同 (共享 GlobalTemporalModeler 几何)。

预期定位 (与 README §9 一致): RGB 教师的价值在【相对结构/PA】(ImageNet 先验);
单目 RGB 的绝对 root 只会比深度教师更差, 不解决 hip。蒸馏时建议把
--out_distill_hip_weight 从 4.0 降到 1.0 (教师 hip 更不可信, 别放大噪声)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .global_encoder import GlobalTemporalModeler

# ImageNet 统计 (RGB, [0,1] 输入)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class _ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
    def forward(self, x):
        return F.gelu(self.bn(self.conv(x)))


class _ScratchRGBEncoder(nn.Module):
    """从零训的 3 通道 CNN, 与 DepthEncoder 同构 (兜底, 无 torchvision 依赖)。"""
    def __init__(self, out_dim=128, widths=(32, 64, 128, 128)):
        super().__init__()
        chans = [3] + list(widths)
        self.stem = nn.Sequential(
            nn.Conv2d(3, chans[1], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(chans[1]), nn.GELU(),
        )
        blocks = []
        for i in range(1, len(widths)):
            blocks.append(_ConvBlock(chans[i], chans[i + 1], stride=2))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(widths[-1], out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x):                       # (N,3,H,W) -> (N,out_dim)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class _ResNet18Encoder(nn.Module):
    """ImageNet 预训练 ResNet18 截断 (conv1..layer4 + avgpool) -> proj 到 out_dim。"""
    def __init__(self, out_dim=128, pretrained=True):
        super().__init__()
        try:
            import torchvision
            from torchvision.models import resnet18
        except ImportError as e:
            raise ImportError(
                "backbone=resnet18 需要 torchvision (torch 1.13 配 torchvision==0.14.0); "
                "装不了就用 --backbone scratch") from e
        try:    # torchvision >= 0.13 新 API
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet18(weights=weights)
        except ImportError:  # 老 API
            net = resnet18(pretrained=pretrained)
        self.features = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4, net.avgpool)
        self.proj = nn.Sequential(
            nn.Linear(512, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x):                       # (N,3,H,W) -> (N,out_dim)
        h = self.features(x).flatten(1)         # (N,512)
        return self.proj(h)


class RGBEncoder(nn.Module):
    """逐帧 RGB -> 特征向量, 带 ImageNet 归一化与时间维 chunk。
    (B,T,3,H,W) float[0,1] -> (B,T,out_dim)。"""
    def __init__(self, out_dim=128, backbone='resnet18', pretrained=True, chunk_size=16):
        super().__init__()
        assert backbone in ('resnet18', 'scratch'), backbone
        self.backbone_name = backbone
        self.chunk_size = chunk_size
        self.register_buffer('px_mean', torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('px_std', torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))
        if backbone == 'resnet18':
            self.enc = _ResNet18Encoder(out_dim=out_dim, pretrained=pretrained)
        else:
            self.enc = _ScratchRGBEncoder(out_dim=out_dim)

    def _encode(self, x):                       # (N,3,H,W) in [0,1]
        x = (x - self.px_mean) / self.px_std
        return self.enc(x)

    def forward(self, rgb):                     # (B,T,3,H,W) -> (B,T,out_dim)
        B, T, C, H, W = rgb.shape
        outs = []
        for t0 in range(0, T, self.chunk_size):
            t1 = min(t0 + self.chunk_size, T)
            n = t1 - t0
            chunk = rgb[:, t0:t1].reshape(B * n, C, H, W)
            outs.append(self._encode(chunk).reshape(B, n, -1))
        return torch.cat(outs, dim=1)


class RGBPoseTeacher(nn.Module):
    """RGB -> z_global -> 3D pose。接口与 DepthPoseTeacher 逐位一致。"""
    def __init__(self, global_dim=128, num_joints=17, seq_len=64,
                 num_transformer_layers=3, num_heads=4,
                 tcn_channels=(128, 128), tcn_kernel_size=3, dropout=0.1,
                 backbone='resnet18', pretrained=True):
        super().__init__()
        self.num_joints = num_joints
        self.backbone_name = backbone
        self.encoder = RGBEncoder(out_dim=global_dim, backbone=backbone, pretrained=pretrained)
        self.global_modeler = GlobalTemporalModeler(
            in_dim=global_dim, global_dim=global_dim,
            num_transformer_layers=num_transformer_layers, num_heads=num_heads,
            tcn_channels=list(tcn_channels), tcn_kernel_size=tcn_kernel_size,
            dropout=dropout, max_seq_len=seq_len + 50)
        self.pose_head = nn.Sequential(
            nn.Linear(global_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, num_joints * 3))

    def forward(self, rgb):                     # (B,T,3,H,W) float[0,1]
        feat = self.encoder(rgb)                # (B,T,gd)
        z_global = self.global_modeler(feat)    # (B,T,gd)
        B, T, _ = z_global.shape
        pose = self.pose_head(z_global).reshape(B, T, self.num_joints, 3)
        return {'p_final': pose, 'z_global': z_global}


if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')

    class _StubGM(nn.Module):
        def __init__(self, **kw): super().__init__(); self.l = nn.Linear(128, 128)
        def forward(self, x): return self.l(x)

    # scratch 模式 (无 torchvision 依赖) 完整前向/反传
    import sys as _sys
    m = RGBPoseTeacher(global_dim=128, seq_len=64, backbone='scratch')
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    x = torch.rand(2, 8, 3, 112, 112)
    out = m(x)
    print(f"[scratch] rgb {tuple(x.shape)} -> pose {tuple(out['p_final'].shape)} "
          f"z_global {tuple(out['z_global'].shape)} | params {n/1e6:.2f}M")
    assert out['p_final'].shape == (2, 8, 17, 3) and out['z_global'].shape == (2, 8, 128)
    out['p_final'].sum().backward()
    print("[scratch] backward OK")

    # resnet18 结构 (weights=None, 不下载) — 仅在 torchvision 可用时测
    try:
        m2 = RGBPoseTeacher(global_dim=128, seq_len=64, backbone='resnet18', pretrained=False)
        out2 = m2(torch.rand(1, 4, 3, 112, 112))
        n2 = sum(p.numel() for p in m2.parameters() if p.requires_grad)
        assert out2['p_final'].shape == (1, 4, 17, 3)
        print(f"[resnet18] OK | params {n2/1e6:.2f}M")
    except ImportError as e:
        print(f"[resnet18] torchvision 不可用, 跳过 ({e})")