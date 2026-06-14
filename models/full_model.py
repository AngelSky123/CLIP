"""
CSI-RSC-PoseDG v9 (B+v3 回退版) — 纯 Hybrid FK 解码器
  pose_decoder = HybridFKPoseDecoder(...)  (结构支 + FK 支 + α 融合)
  root anchor 以【损失项】形式 (L_anchor) 在 trainer 里施加 (--w_root_anchor)。

本版【移除】了后续 rawscale 实验的两处附加 (已证否, 见 README §9e):
  - raw_scale_encoder 支路 (整条删除)
  - PriorRootDecoder 包装 (root 退回 FK 直接预测 + L_anchor 软正则)
其余 (RSC / action dropout / 蒸馏 / vision backbone 分支) 一行不动。

核心机制 (沿用):
1. RSCGlobalChallenger 真正启用, Mask 保留 backbone 梯度。
2. Action Dropout: 训练时 50% 概率阻断 action_emb。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .csi_encoder import DualBranchCSIEncoder
from .local_encoder import LocalSpatioTemporalEncoder, LocalFeaturePooling
from .global_encoder import GlobalTemporalModeler
from .pose_decoder import PoseDecoder, ActionClassifier
from .rsc import RSCGlobalChallenger

# 结构支 (FK Hybrid) 在仓库根目录, 靠各入口 sys.path.insert 可见
from fk_decoder import HybridFKPoseDecoder


class CSIRSCPoseDG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._debug_printed = False

        action_embed_dim = 32
        self.use_vision_backbone = getattr(args, 'use_vision_backbone', False)

        if self.use_vision_backbone:
            from .vision_backbone import VisionBackboneEncoder
            self.vision_backbone = VisionBackboneEncoder(
                in_channels=getattr(args, 'vision_in_channels',
                                    args.amp_channels + args.phase_channels),
                out_dim=args.global_dim,
                arch=getattr(args, 'vision_arch', 'resnet18'),
                pretrained=not getattr(args, 'vision_scratch', False),
                img_size=getattr(args, 'vision_img_size', 112),
                instance_norm=not getattr(args, 'vision_no_instance_norm', False),
                freeze_backbone=getattr(args, 'vision_freeze', False),
                weights_path=getattr(args, 'vision_weights', None),
            )
        else:
            self.csi_encoder = DualBranchCSIEncoder(
                amp_channels=args.amp_channels,
                phase_channels=args.phase_channels,
                hidden_dim=args.encoder_hidden_dim,
                out_dim=args.encoder_out_dim,
            )
            self.local_encoder = LocalSpatioTemporalEncoder(
                in_channels=args.encoder_out_dim,
                hidden_dim=args.local_hidden_dim,
                out_dim=args.local_out_dim,
                num_blocks=args.num_res3d_blocks,
            )
            self.feature_pooling = LocalFeaturePooling(
                in_channels=args.local_out_dim,
                out_channels=args.global_dim,
            )

        self.global_modeler = GlobalTemporalModeler(
            in_dim=args.global_dim,
            global_dim=args.global_dim,
            num_transformer_layers=args.num_transformer_layers,
            num_heads=args.num_heads,
            tcn_channels=args.tcn_channels,
            tcn_kernel_size=args.tcn_kernel_size,
            dropout=args.transformer_dropout,
            max_seq_len=args.seq_len + 50,
        )

        self.rsc_global = RSCGlobalChallenger(
            time_drop_pct=getattr(args, 'rsc2_time_drop_pct', 0.5),
            channel_drop_pct=getattr(args, 'rsc2_channel_drop_pct', 0.5),
            batch_pct=getattr(args, 'rsc2_batch_pct', 0.5)
        )

        # ------ Decoder & Classifier (纯 Hybrid FK, 无 prior-root 包装) ------
        self.pose_decoder = HybridFKPoseDecoder(
            in_dim=args.global_dim, hidden_dim=args.coarse_hidden_dim,
            gcn_hidden=args.gcn_hidden_dim, num_gcn_layers=args.num_gcn_layers,
            num_joints=args.num_joints, action_embed_dim=action_embed_dim,
        )

        self.action_classifier = ActionClassifier(
            in_dim=args.global_dim,
            num_actions=args.num_actions,
            embed_dim=action_embed_dim,
        )

    def forward_backbone(self, csi):
        if self.use_vision_backbone:
            z_seq = self.vision_backbone(csi)
            z_global = self.global_modeler(z_seq)
            return z_seq, z_global
        feat = self.csi_encoder(csi)
        z_local = self.local_encoder(feat)
        z_pooled = self.feature_pooling(z_local)
        z_global = self.global_modeler(z_pooled)
        return z_local, z_global

    def forward_decoder(self, z_global, action_emb):
        # HybridFKPoseDecoder: forward(z_global, action_emb) -> (p_coarse, p_final)
        return self.pose_decoder(z_global, action_emb)

    def forward(self, csi, action_idx=None):
        z_local, z_global = self.forward_backbone(csi)
        action_logits = self.action_classifier(z_global)
        action_probs = F.softmax(action_logits, dim=-1)

        if action_idx is not None:
            action_emb = self.action_classifier.get_action_embedding(action_idx=action_idx)
        else:
            action_emb = self.action_classifier.get_action_embedding(action_probs=action_probs)

        p_coarse, p_final = self.forward_decoder(z_global, action_emb)

        return {
            'p_coarse': p_coarse,
            'p_final': p_final,
            'z_local': z_local,
            'z_global': z_global,
            'action_logits': action_logits,
        }

    def forward_rsc(self, csi, pose_3d, loss_fn, action_idx=None):
        """RSC 训练模式: 携带梯度修复与动作先验解耦。"""
        z_local, z_global_raw = self.forward_backbone(csi)

        action_logits = self.action_classifier(z_global_raw)
        action_probs = F.softmax(action_logits, dim=-1)
        if action_idx is not None:
            action_emb = self.action_classifier.get_action_embedding(action_idx)
        else:
            action_emb = self.action_classifier.get_action_embedding(action_probs=action_probs)

        # Action Dropout: 训练时 50% 概率阻断 action_emb (相对骨架的动作条件)
        if self.training and torch.rand(1).item() < 0.5:
            action_emb_for_decoder = torch.zeros_like(action_emb)
        else:
            action_emb_for_decoder = action_emb

        # Step 2A: 干净路径 (主图)
        p_coarse_clean, p_final_clean = self.forward_decoder(z_global_raw, action_emb_for_decoder)

        # Step 3: RSC 梯度 (分离图上找主导特征)
        z_global_detached = z_global_raw.detach().clone().requires_grad_(True)
        _, p_final_for_grad = self.forward_decoder(z_global_detached, action_emb_for_decoder.detach())
        loss_for_grad = loss_fn(p_final_for_grad, pose_3d)
        grad_global = torch.autograd.grad(
            loss_for_grad, z_global_detached,
            create_graph=False, retain_graph=False,
        )[0]

        # Step 4: RSC Masking (保留 backbone 梯度)
        z_global_masked = self.rsc_global(z_global_raw, grad_global.detach())

        if not self._debug_printed:
            with torch.no_grad():
                diff = (z_global_raw.detach() - z_global_masked.detach()).abs()
                pct = 100.0 * (diff > 1e-8).float().sum().item() / diff.numel()
            print(f"[RSC DEBUG] z_global: {z_global_raw.shape}, "
                  f"masked {pct:.1f}%, grad_norm={grad_global.abs().mean():.6f}")
            self._debug_printed = True

        # Step 5: 被 Mask 后的解码
        p_coarse_masked, p_final_masked = self.forward_decoder(
            z_global_masked, action_emb_for_decoder.detach())

        return {
            'p_coarse_clean': p_coarse_clean,
            'p_final_clean': p_final_clean,
            'p_coarse_masked': p_coarse_masked,
            'p_final_masked': p_final_masked,
            'z_local': z_local,
            'z_global': z_global_raw,
            'z_global_masked': z_global_masked,
            'action_logits': action_logits,
        }