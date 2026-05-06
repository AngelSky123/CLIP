"""
CSI-RSC-PoseDG v7 — Action-Conditioned Pose Decoder

核心思想: 将动作信息作为显式条件输入解码器.
  - 训练时: 使用 GT 动作标签 → one-hot → embedding → decoder
  - 推理时: 使用预测的动作概率 → softmax → weighted embedding → decoder

这从架构上保证不同动作必须产生不同预测.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .csi_encoder import DualBranchCSIEncoder
from .local_encoder import LocalSpatioTemporalEncoder, LocalFeaturePooling
from .global_encoder import GlobalTemporalModeler
from .pose_decoder import PoseDecoder, ActionClassifier


class CSIRSCPoseDG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._debug_printed = False
        self.rsc_drop_pct = args.rsc2_time_drop_pct
        self.rsc_batch_pct = args.rsc2_batch_pct

        action_embed_dim = 32

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
        """Standard forward pass.
        
        Args:
            csi: (B, T, 9, 114, 10)
            action_idx: (B,) int64 action labels. 
                        If None → use predicted action (inference mode).
        """
        z_local, z_global = self.forward_backbone(csi)
        action_logits = self.action_classifier(z_global)

        if action_idx is not None:
            # Training: use GT action
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

    def _apply_rsc_mask(self, z, gradient):
        B, T, C = z.shape
        num_apply = max(1, int(B * self.rsc_batch_pct))
        perm = torch.randperm(B, device=z.device)
        apply_indices = perm[:num_apply]
        z_masked = z.clone()
        for idx in apply_indices:
            g = gradient[idx].abs()
            g_flat = g.reshape(-1)
            num_to_drop = max(1, int(self.rsc_drop_pct * g_flat.numel()))
            num_to_keep = max(1, g_flat.numel() - num_to_drop)
            threshold, _ = g_flat.kthvalue(num_to_keep)
            mask = (g < threshold).float()
            z_masked[idx] = z[idx] * mask
        return z_masked

    def forward_rsc(self, csi, pose_3d, loss_fn, action_idx=None):
        """RSC training with action conditioning.
        
        Args:
            csi: (B, T, 9, 114, 10)
            pose_3d: (B, T, 17, 3) GT poses
            loss_fn: callable(pred, gt) → scalar loss
            action_idx: (B,) GT action labels
        """
        # Step 1: Backbone
        z_local, z_global_raw = self.forward_backbone(csi)

        # Action prediction + embedding
        action_logits = self.action_classifier(z_global_raw)
        if action_idx is not None:
            action_emb = self.action_classifier.get_action_embedding(
                action_idx=action_idx
            )
        else:
            action_probs = F.softmax(action_logits, dim=-1)
            action_emb = self.action_classifier.get_action_embedding(
                action_probs=action_probs
            )

        # Step 2A: Clean path (backbone receives gradients)
        p_coarse_clean, p_final_clean = self.forward_decoder(
            z_global_raw, action_emb
        )

        # Step 3: RSC gradient computation (detached)
        z_global_detached = z_global_raw.detach().clone().requires_grad_(True)
        _, p_final_for_grad = self.forward_decoder(
            z_global_detached, action_emb.detach()
        )

        loss_for_grad = loss_fn(p_final_for_grad, pose_3d)
        grad_global = torch.autograd.grad(
            loss_for_grad, z_global_detached,
            create_graph=False, retain_graph=False,
        )[0]

        # Step 4: RSC masking
        with torch.no_grad():
            z_global_masked = self._apply_rsc_mask(
                z_global_raw.detach(), grad_global.detach()
            )
        z_global_masked = z_global_masked.requires_grad_(True)

        # Debug
        if not self._debug_printed:
            with torch.no_grad():
                diff = (z_global_raw.detach() - z_global_masked.detach()).abs()
                pct = 100.0 * (diff > 1e-8).float().sum().item() / diff.numel()
            print(f"[RSC DEBUG] z_global: {z_global_raw.shape}, "
                  f"masked {pct:.1f}%, "
                  f"grad_norm={grad_global.abs().mean():.6f}")
            self._debug_printed = True

        # Step 5: Masked decode
        p_coarse_masked, p_final_masked = self.forward_decoder(
            z_global_masked, action_emb.detach()
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