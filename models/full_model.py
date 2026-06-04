"""
CSI-RSC-PoseDG v7.1 — Action-Conditioned Pose Decoder (Fixed)
  + v8.x: optional ImageNet vision backbone front-end (use_vision_backbone).

核心修正:
1. 修复计算图断裂 (Detach Bug): 真正启用 RSCGlobalChallenger，确保 Mask 操作保留 Backbone 梯度。
2. 动作特征随机失活 (Action Dropout): 训练时 50% 概率阻断 Action 先验，彻底解决跨域时的级联失效。

NEW — Vision backbone option:
  If args.use_vision_backbone is True, the (csi_encoder + local_encoder +
  feature_pooling) trio is replaced by a single VisionBackboneEncoder that maps
  each CSI frame (9,114,10) through an ImageNet-pretrained backbone and outputs a
  per-frame feature sequence (B, T, global_dim). GlobalTemporalModeler, RSC,
  PoseDecoder, ActionClassifier and BOTH forward paths are unchanged.
  Default is False -> identical behaviour to the original model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .csi_encoder import DualBranchCSIEncoder
from .local_encoder import LocalSpatioTemporalEncoder, LocalFeaturePooling
from .global_encoder import GlobalTemporalModeler
from .pose_decoder import PoseDecoder, ActionClassifier
from root_decoupled_decoder import RootDecoupledPoseDecoder

# 核心修正：引入你已经写好但之前被闲置的 RSC 模块
from .rsc import RSCGlobalChallenger


class CSIRSCPoseDG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._debug_printed = False

        action_embed_dim = 32

        # ------ NEW: choose front-end (original CSI trio vs vision backbone) ------
        self.use_vision_backbone = getattr(args, 'use_vision_backbone', False)

        if self.use_vision_backbone:
            # Lazy import so non-vision users don't need timm installed.
            from .vision_backbone import VisionBackboneEncoder
            self.vision_backbone = VisionBackboneEncoder(
                in_channels=getattr(args, 'vision_in_channels',
                                    args.amp_channels + args.phase_channels),
                out_dim=args.global_dim,                 # MUST equal global_modeler in_dim
                arch=getattr(args, 'vision_arch', 'resnet18'),
                pretrained=not getattr(args, 'vision_scratch', False),
                img_size=getattr(args, 'vision_img_size', 112),
                instance_norm=not getattr(args, 'vision_no_instance_norm', False),
                freeze_backbone=getattr(args, 'vision_freeze', False),
                weights_path=getattr(args, 'vision_weights', None),
            )
        else:
            # ------ 原始 Backbone ------
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

        # ------ 全局时序建模器 (always present, unchanged) ------
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

        # ------ 核心修正：实例化多维特征自挑战模块 ------
        self.rsc_global = RSCGlobalChallenger(
            time_drop_pct=getattr(args, 'rsc2_time_drop_pct', 0.5),
            channel_drop_pct=getattr(args, 'rsc2_channel_drop_pct', 0.5),
            batch_pct=getattr(args, 'rsc2_batch_pct', 0.5)
        )

        # ------ 初始化 Decoder & Classifier ------
        self.pose_decoder = PoseDecoder(
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
            # Vision path: per-frame ImageNet features -> temporal model.
            z_seq = self.vision_backbone(csi)          # (B, T, global_dim)
            z_global = self.global_modeler(z_seq)
            # z_local is unused by losses/RSC downstream; return z_seq as a
            # harmless placeholder to keep the (z_local, z_global) contract.
            return z_seq, z_global

        feat = self.csi_encoder(csi)
        z_local = self.local_encoder(feat)
        z_pooled = self.feature_pooling(z_local)
        z_global = self.global_modeler(z_pooled)
        return z_local, z_global

    def forward_decoder(self, z_global, action_emb):
        return self.pose_decoder(z_global, action_emb)

    def forward(self, csi, action_idx=None):
        """Standard forward pass (推理模式)."""
        z_local, z_global = self.forward_backbone(csi)
        action_logits = self.action_classifier(z_global)

        if action_idx is not None:
            action_emb = self.action_classifier.get_action_embedding(
                action_idx=action_idx
            )
        else:
            action_probs = F.softmax(action_logits, dim=-1)
            action_emb = self.action_classifier.get_action_embedding(
                action_probs=action_probs
            )

        p_coarse, p_final = self.forward_decoder(z_global, action_emb)

        return {
            'p_coarse': p_coarse,
            'p_final': p_final,
            'z_local': z_local,
            'z_global': z_global,
            'action_logits': action_logits,
        }

    def forward_rsc(self, csi, pose_3d, loss_fn, action_idx=None):
        """RSC 训练模式：携带梯度修复与动作先验解耦"""
        # Step 1: Backbone 前向传播
        z_local, z_global_raw = self.forward_backbone(csi)

        # 动作分类与 Embedding
        action_logits = self.action_classifier(z_global_raw)
        if action_idx is not None:
            action_emb = self.action_classifier.get_action_embedding(action_idx)
        else:
            action_probs = F.softmax(action_logits, dim=-1)
            action_emb = self.action_classifier.get_action_embedding(action_probs=action_probs)

        # === 修复 2：Action Dropout (动作特征解耦) ===
        if self.training and torch.rand(1).item() < 0.5:
            action_emb_for_decoder = torch.zeros_like(action_emb)
        else:
            action_emb_for_decoder = action_emb

        # Step 2A: 干净路径 (主图，负责传递绝大部分基础梯度)
        p_coarse_clean, p_final_clean = self.forward_decoder(
            z_global_raw, action_emb_for_decoder
        )

        # Step 3: RSC 梯度计算 (在分离的图上寻找主导特征)
        z_global_detached = z_global_raw.detach().clone().requires_grad_(True)
        _, p_final_for_grad = self.forward_decoder(
            z_global_detached, action_emb_for_decoder.detach()
        )

        loss_for_grad = loss_fn(p_final_for_grad, pose_3d)
        grad_global = torch.autograd.grad(
            loss_for_grad, z_global_detached,
            create_graph=False, retain_graph=False,
        )[0]

        # Step 4: RSC Masking (特征自挑战应用) — 保留 Backbone 梯度
        z_global_masked = self.rsc_global(
            z_global_raw, grad_global.detach()
        )

        # Debug 打印监控
        if not self._debug_printed:
            with torch.no_grad():
                diff = (z_global_raw.detach() - z_global_masked.detach()).abs()
                pct = 100.0 * (diff > 1e-8).float().sum().item() / diff.numel()
            print(f"[RSC DEBUG] z_global: {z_global_raw.shape}, "
                  f"masked {pct:.1f}%, "
                  f"grad_norm={grad_global.abs().mean():.6f}")
            self._debug_printed = True

        # Step 5: 被 Mask 后的解码 (迫使网络发掘次优特征)
        p_coarse_masked, p_final_masked = self.forward_decoder(
            z_global_masked, action_emb_for_decoder.detach()
        )

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