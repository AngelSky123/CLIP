"""
CSI-RSC-PoseDG v7.1 — Action-Conditioned Pose Decoder (Fixed)

核心修正:
1. 修复计算图断裂 (Detach Bug): 真正启用 RSCGlobalChallenger，确保 Mask 操作保留 Backbone 梯度。
2. 动作特征随机失活 (Action Dropout): 训练时 50% 概率阻断 Action 先验，彻底解决跨域时的级联失效。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .csi_encoder import DualBranchCSIEncoder
from .local_encoder import LocalSpatioTemporalEncoder, LocalFeaturePooling
from .global_encoder import GlobalTemporalModeler
from .pose_decoder import PoseDecoder, ActionClassifier

# 核心修正：引入你已经写好但之前被闲置的 RSC 模块
from .rsc import RSCGlobalChallenger 


class CSIRSCPoseDG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._debug_printed = False

        action_embed_dim = 32

        # ------ 初始化 Backbone ------
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

        # ------ 核心修正：实例化多维特征自挑战模块 ------
        self.rsc_global = RSCGlobalChallenger(
            time_drop_pct=getattr(args, 'rsc2_time_drop_pct', 0.5),
            channel_drop_pct=getattr(args, 'rsc2_channel_drop_pct', 0.5),
            batch_pct=getattr(args, 'rsc2_batch_pct', 0.5)
        )

        # ------ 初始化 Decoder & Classifier ------
        self.pose_decoder = PoseDecoder(
            in_dim=args.global_dim,
            hidden_dim=args.coarse_hidden_dim,
            gcn_hidden=args.gcn_hidden_dim,
            num_gcn_layers=args.num_gcn_layers,
            num_joints=args.num_joints,
            action_embed_dim=action_embed_dim,
        )
        self.action_classifier = ActionClassifier(
            in_dim=args.global_dim,
            num_actions=args.num_actions,
            embed_dim=action_embed_dim,
        )

    def forward_backbone(self, csi):
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
            # Training / Oracle: use GT action
            action_emb = self.action_classifier.get_action_embedding(
                action_idx=action_idx
            )
        else:
            # Inference: use predicted action (soft)
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
        # 训练时以 50% 概率将动作先验置零。
        # 这迫使 Decoder 必须学会仅通过 CSI 骨干特征来推演 3D 骨架，
        # 防止其过度依赖 Action 导致在未知域彻底崩盘。
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

        # Step 4: RSC Masking (特征自挑战应用)
        # === 修复 1：保留 Backbone 梯度 ===
        # 传入带有 requires_grad=True 的 z_global_raw，
        # Mask 后的张量将会把惩罚梯度一路反传回 Transformer 和 CSI Encoder。
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