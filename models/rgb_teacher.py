"""
RGB -> 3D pose teacher (DG 强化版)。drop-in 替换 models/rgb_teacher.py。

相对原版的改动 (只为跨环境泛化 + 稳住 MPJPE, 不改对外接口):
  forward(rgb (B,T,3,H,W)) -> {'p_final': (B,T,17,3), 'z_global': (B,T,128)}  ← 一字不变

  1. MixStyle 插进 ResNet 骨干 (layer1/2/3 之后): 在特征层混合不同样本的实例统计量,
     生成"虚拟域", 直接打散源域房间风格捷径。MixStyle2D 无可学习参数 ->
     【不改变 state_dict】, eval 时自动直通。
  2. freeze_stages: 冻结 ResNet 浅层 (stem + 前 N 个 stage)。浅层是通用低级纹理,
     源域过拟合主要发生在深层语义; 冻结浅层 = 更少自由度去记房间。
  3. backbone_dropout: avgpool 后、proj 前加一道 dropout, 抑制 z_global 过拟合。
  scratch 兜底骨干同样插 MixStyle。

为什么这同时帮 MPJPE: 现日志里 E04 MPJPE 从 ep3 的 293 一路涨到 ep48 的 369,
纯粹是过拟合导致收敛后变差。把过拟合按住后, 收敛模型的 E04 MPJPE 不再塌,
结构项 (MPJPE_aligned) 也跟着稳 -> 可在源内 val 选到一个真·收敛且不烂的教师。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .global_encoder import GlobalTemporalModeler
from .mixstyle import MixStyle2D

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
    """从零训的 3 通道 CNN, 块间插 MixStyle (兜底, 无 torchvision 依赖)。"""
    def __init__(self, out_dim=128, widths=(32, 64, 128, 128),
                 dg_mixstyle=True, mixstyle_p=0.5, mixstyle_alpha=0.3):
        super().__init__()
        chans = [3] + list(widths)
        self.stem = nn.Sequential(
            nn.Conv2d(3, chans[1], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(chans[1]), nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [_ConvBlock(chans[i], chans[i + 1], stride=2) for i in range(1, len(widths))])
        self.mix = MixStyle2D(p=mixstyle_p, alpha=mixstyle_alpha) if dg_mixstyle else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(widths[-1], out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x):                       # (N,3,H,W) -> (N,out_dim)
        x = self.stem(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.mix is not None and i < len(self.blocks) - 1:   # 末块后不混
                x = self.mix(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class _ResNet18Encoder(nn.Module):
    """ImageNet ResNet18 截断, layer1/2/3 之后插 MixStyle, 支持冻结浅层。"""
    def __init__(self, out_dim=128, pretrained=True,
                 dg_mixstyle=True, mixstyle_p=0.5, mixstyle_alpha=0.3,
                 freeze_stages=0):
        super().__init__()
        try:
            import torchvision
            from torchvision.models import resnet18
        except ImportError as e:
            raise ImportError(
                "backbone=resnet18 需要 torchvision; 装不了用 --backbone scratch") from e
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet18(weights=weights)
        except ImportError:
            net = resnet18(pretrained=pretrained)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.avgpool = net.avgpool
        self.mix = MixStyle2D(p=mixstyle_p, alpha=mixstyle_alpha) if dg_mixstyle else None
        self.proj = nn.Sequential(
            nn.Linear(512, out_dim), nn.LayerNorm(out_dim), nn.GELU())

        # 冻结浅层: 1=stem, 2=+layer1, 3=+layer2, 4=+layer3 (layer4 永远训)
        self._frozen = []
        order = [self.stem, self.layer1, self.layer2, self.layer3]
        for i in range(min(freeze_stages, len(order))):
            for p in order[i].parameters():
                p.requires_grad = False
            order[i].eval()
            self._frozen.append(order[i])

    def train(self, mode=True):
        super().train(mode)
        for m in self._frozen:          # 冻结段保持 eval (BN 统计不更新)
            m.eval()
        return self

    def forward(self, x):               # (N,3,H,W) -> (N,out_dim)
        x = self.stem(x)
        x = self.layer1(x); x = self.mix(x) if self.mix is not None else x
        x = self.layer2(x); x = self.mix(x) if self.mix is not None else x
        x = self.layer3(x); x = self.mix(x) if self.mix is not None else x
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.proj(x)


class RGBEncoder(nn.Module):
    """逐帧 RGB -> 特征向量, 带 ImageNet 归一化、时间维 chunk、可选 DG。"""
    def __init__(self, out_dim=128, backbone='resnet18', pretrained=True, chunk_size=16,
                 dg_mixstyle=True, mixstyle_p=0.5, mixstyle_alpha=0.3, freeze_stages=0):
        super().__init__()
        assert backbone in ('resnet18', 'scratch'), backbone
        self.backbone_name = backbone
        self.chunk_size = chunk_size
        self.register_buffer('px_mean', torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('px_std', torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))
        if backbone == 'resnet18':
            self.enc = _ResNet18Encoder(out_dim, pretrained, dg_mixstyle,
                                        mixstyle_p, mixstyle_alpha, freeze_stages)
        else:
            self.enc = _ScratchRGBEncoder(out_dim, dg_mixstyle=dg_mixstyle,
                                          mixstyle_p=mixstyle_p, mixstyle_alpha=mixstyle_alpha)

    def _encode(self, x):
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
                 backbone='resnet18', pretrained=True,
                 dg_mixstyle=True, mixstyle_p=0.5, mixstyle_alpha=0.3,
                 freeze_stages=0, backbone_dropout=0.1):
        super().__init__()
        self.num_joints = num_joints
        self.backbone_name = backbone
        self.encoder = RGBEncoder(out_dim=global_dim, backbone=backbone, pretrained=pretrained,
                                  dg_mixstyle=dg_mixstyle, mixstyle_p=mixstyle_p,
                                  mixstyle_alpha=mixstyle_alpha, freeze_stages=freeze_stages)
        self.backbone_drop = nn.Dropout(backbone_dropout)
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
        feat = self.backbone_drop(feat)
        z_global = self.global_modeler(feat)    # (B,T,gd)
        B, T, _ = z_global.shape
        pose = self.pose_head(z_global).reshape(B, T, self.num_joints, 3)
        return {'p_final': pose, 'z_global': z_global}


if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    m = RGBPoseTeacher(global_dim=128, seq_len=64, backbone='scratch',
                       dg_mixstyle=True).train()
    x = torch.rand(2, 8, 3, 112, 112)
    out = m(x)
    assert out['p_final'].shape == (2, 8, 17, 3) and out['z_global'].shape == (2, 8, 128)
    out['p_final'].sum().backward()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[scratch+DG] forward/backward OK | trainable params {n/1e6:.2f}M")

    # MixStyle 不引入参数: 关/开 DG 的 scratch 参数量应一致
    m_off = RGBPoseTeacher(backbone='scratch', dg_mixstyle=False)
    a = sum(p.numel() for p in m.parameters())
    b = sum(p.numel() for p in m_off.parameters())
    print(f"[param check] DG-on={a/1e6:.4f}M  DG-off={b/1e6:.4f}M  diff={a-b} (应=0)")
    assert a == b, "MixStyle 不应改变参数量"
    print("[OK] MixStyle 无参数, state_dict 兼容")