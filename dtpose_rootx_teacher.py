"""
dtpose_rootx_teacher.py — 层次B (B1): 把 DT-Pose 复现线的 x 轴【轨迹形状】蒸进你的
axis_split FK 支的 x 轴。

== 动机 (缝合诊断证据) ==
  * axis_split 单模型: x 轴 std 已涨到 47~82 (敢动了), 但【无偏】—— ema 平均后
    hip 回到 ~330 (≈零信息线), 因为你 backbone 没 DT 那 400 轮预训练, x 敢动但
    不知往哪动。
  * 缝合 S5: 用 DT 的 x -> 源域 hip 一致降 14~40mm, E04 hip 290。DT 的 x 里有
    "x 该往哪动" 的方向信号 (来自其自监督预训练表征)。
  * => B1: 训练时让你 FK 支的 x 轨迹对齐 DT 的 x 轨迹, 把 DT 的 x 方向信号注进单模型,
          让 x 从 "无偏乱抖" 变 "有偏向对的方向"。

== 关键口径 (两个坑, 已处理) ==
  1. 输入口径: DT 吃单帧纯幅度全局 min-max (3,114,10)。你 batch 是 (B,T,9,114,10)
     逐帧 min-max(含相位)。复用乙' dtpose_feature_teacher 的 _amp_to_dtpose_input
     做口径转换, 保证喂 DT 的分布与其训练时一致。
  2. 坐标系: 两线 hip 互差 99mm、x 互偏 38mm 且随域变 (debias 证明)。直接对齐
     x 绝对值会把【域相关的不可观测偏移】学进来。=> 只蒸【x 去均值轨迹】(每段序列
     各自减自己的 x 时间均值), 蒸 "x 怎么动" 不蒸 "x 在哪"。这与 axis_split 的
     x=cumsum(vx) 轨迹本质自洽。

== 接口 ==
  teacher = DTPoseRootXTeacher(dt_ckpt, dt_decoder_ckpt, device)
  x_traj_teacher = teacher(csi)     # csi: (B,T,9,114,10) -> (B,T) DT 的 hip.x 去均值轨迹 (detach)
  # 训练侧: 取你 student 的 p_final hip.x, 去均值, 与之对齐 (smooth_l1)。
"""
import sys
import os
import torch
import torch.nn as nn

# 复用乙' 的 MAE 类注入 + 口径转换
from dtpose_feature_teacher import _inject_mae_classes, DTPoseFeatureTeacher
from train_pose_dtpose_style import ViT_Pose_Decoder

HIP = 0


class DTPoseRootXTeacher(nn.Module):
    """冻结的 DT-Pose 完整解码器, 逐帧出 17x3 pose, 取 hip 的 x 轴去均值轨迹作蒸馏目标。

    dt_ckpt:         预训练整对象 (含 encoder), 或 train_pose_dtpose_style 存的解码器 state_dict。
    dt_decoder_ckpt: 若 dt_ckpt 只是 encoder, 解码器权重 (prompt/GCN/Transformer/fc) 单独给。
    """
    def __init__(self, dt_ckpt, dt_decoder_ckpt=None, device='cuda'):
        super().__init__()
        _inject_mae_classes()
        try:
            obj = torch.load(dt_ckpt, map_location=device, weights_only=False)
        except TypeError:
            obj = torch.load(dt_ckpt, map_location=device)
        if hasattr(obj, 'encoder'):
            encoder = obj.encoder
        elif isinstance(obj, dict) and 'model' in obj and hasattr(obj['model'], 'encoder'):
            encoder = obj['model'].encoder
        else:
            raise ValueError(f"无法从 {dt_ckpt} 取 encoder; got {type(obj)}")

        self.dt = ViT_Pose_Decoder(encoder, keypoints=17, coor_num=3).to(device)
        if dt_decoder_ckpt:
            sd = torch.load(dt_decoder_ckpt, map_location=device)
            sd = sd.get('model_state_dict', sd)
            miss, unexp = self.dt.load_state_dict(sd, strict=False)
            if len(miss) > 10:
                print(f"[DTPoseRootXTeacher][警告] decoder missing 较多 {miss[:5]}, "
                      f"确认 dt_decoder_ckpt 与 encoder 配套")
        self.dt.eval()
        for p in self.dt.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, csi):
        """csi: (B,T,9,114,10) -> (B,T) DT 的 hip.x 去均值轨迹 (detach)。"""
        B, T = csi.shape[0], csi.shape[1]
        # 复用乙' 的口径转换: 取幅度3通道, 逐帧 DT 式 min-max -> (B*T,3,114,10)
        x = DTPoseFeatureTeacher._amp_to_dtpose_input(csi)      # (B*T,3,114,10)
        # DT 完整解码器逐帧出 pose (分块防显存)
        outs = []
        bs = 512
        for s in range(0, x.shape[0], bs):
            outs.append(self.dt(x[s:s + bs]))                   # (b,17,3)
        pose = torch.cat(outs, 0).reshape(B, T, 17, 3)
        x_hip = pose[:, :, HIP, 0]                              # (B,T) DT 的 hip x
        x_traj = x_hip - x_hip.mean(dim=1, keepdim=True)        # 去均值轨迹 (只蒸怎么动)
        return x_traj.detach()


if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dt_ckpt', default='pretrain_dtpose_400.pt')
    ap.add_argument('--dt_decoder_ckpt', default='pose_dtpose.pt')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(a.dt_ckpt):
        print(f"[sandbox] 无 {a.dt_ckpt}, 跳过真实加载自测")
        sys.exit(0)
    t = DTPoseRootXTeacher(a.dt_ckpt, a.dt_decoder_ckpt, device=dev)
    n_train = sum(p.numel() for p in t.parameters() if p.requires_grad)
    print(f"DT-x 教师可训练参数 = {n_train} (应=0, 已冻结)")
    assert n_train == 0
    csi = torch.randn(2, 64, 9, 114, 10, device=dev)
    xt = t(csi)
    print(f"输入 {tuple(csi.shape)} -> x_traj {tuple(xt.shape)} (应 (2,64))")
    assert xt.shape == (2, 64)
    print(f"x_traj 去均值检查: 每段均值 ~ {xt.mean(1).abs().max().item():.2e} (应~0)")
    assert xt.mean(1).abs().max().item() < 1e-4
    assert not xt.requires_grad
    print("[OK]")