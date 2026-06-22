"""
dtpose_feature_teacher.py — 乙′: 用 A 训出的 DT-Pose 式预训练编码器作为【冻结特征教师】,
把它学到的【可跨域的表征】通过 feature 蒸馏注入你原系统的 z_global。

动机 (已被实验证据支持):
  - A 的 pretrain_dtpose_400.pt 经 400 轮自监督 (MAE+相邻帧对比+uniformity) 后,
    阶段二能把 E04 MPJPE 推到 323、hip 外推到 301 (< 零信息基线 315) ——
    说明这个编码器的表征里【含有你原 backbone 没提取到的跨域定位信息】。
  - 你原系统 PA 已 103 (形状强), 短板在 MPJPE 的 root。本模块不替换你的 backbone,
    只把 DT-Pose 表征作为一路 feature 蒸馏目标, 让你的 z_global 对齐它 ——
    架构/DG模块/PA 全保留, 只借它的跨域表征补 root 短板。

接口:
  teacher = DTPoseFeatureTeacher(ckpt_path, device)
  z_teacher = teacher(csi)      # csi: (B,T,9,114,10) 你的标准输入
                                # 返回 (B,T,emb_dim) 逐帧 cls 特征 (detach, 冻结)
  emb_dim 由 ckpt 自动探测 (A 默认 256)。

蒸馏侧: 用你已有的 DistillProjection(128->emb_dim) + FeatureDistillLoss
  让 student z_global 投到 emb_dim 后对齐 z_teacher (cosine + smoothL1)。

口径对齐 (关键):
  DT-Pose 预训练吃的是【全局 min-max 归一化的纯幅度单帧 (3,114,10)】。
  你 dataset.py 的幅度是【逐帧 min-max】(前3通道)。二者不同。
  本模块对每帧按 DT-Pose 训练时的方式重做归一化 (per-frame min-max over (3,114,10),
  与 pretrain 的 _read_amp 逐帧等价), 保证喂给教师的分布与它训练时一致。
"""
import sys
import torch
import torch.nn as nn


def _inject_mae_classes():
    """torch.load 整对象 MAE_ViT 需要其类在 __main__ 可见。"""
    import importlib.util, os
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'pretrain_dtpose_style.py')
    spec = importlib.util.spec_from_file_location('pretrain_dtpose_style', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    main_mod = sys.modules['__main__']
    # 注入 pretrain_dtpose_style.py 里【实际存在的全部类名】(精确匹配, 含下划线),
    # 否则 torch.load 整对象时在 __main__ 找不到对应类会报 AttributeError。
    for name in ('MAE_ViT', 'MAE_Encoder', 'MAE_Decoder', 'Block',
                 'PatchShuffle', 'FramePairDataset', '_Attention', '_Mlp'):
        if hasattr(mod, name):
            setattr(main_mod, name, getattr(mod, name))
    return mod


class DTPoseFeatureTeacher(nn.Module):
    """冻结的 DT-Pose 式预训练编码器, 逐帧输出 cls 表征作为蒸馏目标。"""
    def __init__(self, ckpt_path, device='cuda', amp_channels=3):
        super().__init__()
        _inject_mae_classes()
        try:
            obj = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            obj = torch.load(ckpt_path, map_location=device)
        # ckpt 可能是整个 MAE_ViT, 也可能是 dict
        if hasattr(obj, 'encoder'):
            encoder = obj.encoder
        elif isinstance(obj, dict) and 'model' in obj and hasattr(obj['model'], 'encoder'):
            encoder = obj['model'].encoder
        else:
            raise ValueError(f"无法从 {ckpt_path} 取出 .encoder; got {type(obj)}")
        self.encoder = encoder.to(device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.amp_channels = amp_channels
        # 探测 emb_dim
        self.emb_dim = self.encoder.cls_token.shape[-1]

    @staticmethod
    def _amp_to_dtpose_input(csi):
        """csi: (B,T,9,114,10) -> 取幅度3通道, 逐帧 DT-Pose 式 min-max -> (B*T,3,114,10)。
        DT-Pose _read_amp: 全局(整帧) min-max。这里逐帧做, 与其单帧训练等价。"""
        B, T, C, H, W = csi.shape
        amp = csi[:, :, :3, :, :]                       # (B,T,3,114,10) 幅度通道
        x = amp.reshape(B * T, 3, H, W)
        # 逐帧 (整张3x114x10) min-max, 与 pretrain 的 _read_amp 一致
        flat = x.reshape(B * T, -1)
        mn = flat.min(dim=1, keepdim=True)[0]
        mx = flat.max(dim=1, keepdim=True)[0]
        x = (flat - mn) / (mx - mn + 1e-8)
        return x.reshape(B * T, 3, H, W)

    @torch.no_grad()
    def forward(self, csi):
        """csi: (B,T,9,114,10) -> z_teacher: (B,T,emb_dim) detached。"""
        B, T = csi.shape[0], csi.shape[1]
        x = self._amp_to_dtpose_input(csi)              # (B*T,3,114,10)
        feat = self.encoder.feature_extract(x)          # (B*T, emb_dim) cls 特征
        return feat.reshape(B, T, self.emb_dim).detach()


if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='pretrain_dtpose_400.pt')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(a.ckpt):
        # 沙箱无 ckpt: 直接在内存构造 encoder 测维度逻辑 (不走 pickle 存盘)
        mod = _inject_mae_classes()
        m = mod.MAE_ViT(image_size=(114,10), patch_size=(2,2),
                        emb_dim=256, encoder_layer=4, decoder_layer=2,
                        input_dim=3, mask_ratio=0.8)
        teacher = DTPoseFeatureTeacher.__new__(DTPoseFeatureTeacher)
        nn.Module.__init__(teacher)
        teacher.encoder = m.encoder.to(dev).eval()
        for p in teacher.encoder.parameters():
            p.requires_grad = False
        teacher.amp_channels = 3
        teacher.emb_dim = teacher.encoder.cls_token.shape[-1]
        print(f"[sandbox] 内存构造 encoder 自测 (无 ckpt)")
    else:
        teacher = DTPoseFeatureTeacher(a.ckpt, device=dev)
    print(f"emb_dim 探测 = {teacher.emb_dim}")
    n_train = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in teacher.parameters())
    print(f"教师参数: 总 {n_total/1e6:.2f}M, 可训练 {n_train} (应=0, 已冻结)")
    assert n_train == 0, "教师未冻结!"

    csi = torch.randn(2, 64, 9, 114, 10, device=dev)
    z = teacher(csi)
    print(f"输入 {tuple(csi.shape)} -> z_teacher {tuple(z.shape)} (应 (2,64,{teacher.emb_dim}))")
    assert z.shape == (2, 64, teacher.emb_dim)
    assert not z.requires_grad, "z_teacher 应 detached"

    # 验证逐帧归一化输出范围 [0,1]
    x = teacher._amp_to_dtpose_input(csi)
    print(f"归一化后输入范围: [{x.min().item():.3f}, {x.max().item():.3f}] (应 [0,1])")
    assert x.min() >= -1e-4 and x.max() <= 1 + 1e-4

    # 验证可作为蒸馏目标: student z_global(128) 投到 emb_dim 后能算 feature loss
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from distill_loss import DistillProjection, FeatureDistillLoss
        proj = DistillProjection(128, teacher.emb_dim).to(dev)
        floss = FeatureDistillLoss().to(dev)
        z_student = torch.randn(2, 64, 128, device=dev, requires_grad=True)
        l, d = floss(proj(z_student), z)
        l.backward()
        print(f"蒸馏对齐自测: loss={l.item():.4f}, grad->student OK")
    except ImportError:
        print("[sandbox] distill_loss 不在路径, 跳过蒸馏对齐自测 (真实环境会有)")
    print("[ALL OK]")