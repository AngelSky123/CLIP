"""
Stage 2: Pose Fine-tuning from Pretrained Backbone.

This script is a thin extension of your existing train.py:
  * Loads pretrained backbone weights (Stage 1A MAE or Stage 1B Action) into CSIRSCPoseDG.
  * Uses differential LR: backbone (pretrained) gets lower LR than fresh heads.
  * Everything else (TotalLoss, RSC, Action Dropout, evaluate()) reuses your code.

REQUIRED: HMSF must be applied to full_model.py BEFORE running this script.
See README.md / apply_hmsf_patch.py for the one-line edit.

Usage:
    python train_stage2.py \
        --data_root /home/a123456/PerceptAlign/MMFi \
        --train_envs E01 E02 E03 \
        --test_env E04 \
        --pretrain_ckpt ./checkpoints/stage1b_action/action_best.pt \
        --epochs 50 \
        --batch_size 2 \
        --accumulate_grad 4 \
        --lr_backbone 1e-4 \
        --lr_head 5e-4 \
        --lambda_hip 0.3 \
        --save_dir ./checkpoints/stage2_pose
"""
import os
import sys
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config
from dataset import build_dataloaders
from models import CSIRSCPoseDG
from losses import TotalLoss, PoseLoss
from evaluate import PoseEvaluator
from utils import (set_seed, setup_logger, count_parameters,
                   save_checkpoint, AverageMeter, Timer)

# Reuse training loop and evaluate from existing train.py
from train import train_one_epoch, evaluate, action_to_index


def parse_stage2_extras():
    """Parse only the Stage-2-specific args; rest come from config.get_config()."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pretrain_ckpt', type=str, required=True,
                        help='Path to Stage 1A or 1B checkpoint.')
    parser.add_argument('--lr_backbone', type=float, default=1e-4,
                        help='LR for pretrained backbone (slow).')
    parser.add_argument('--lr_head', type=float, default=5e-4,
                        help='LR for fresh heads (fast).')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0,
                        help='If > 0, freeze backbone for the first N epochs.')
    known, _ = parser.parse_known_args()
    return known


def load_pretrained_backbone(model, ckpt_path, logger, device):
    """Load Stage 1A or 1B checkpoint into CSIRSCPoseDG's backbone submodules.

    FIX 🔴 #3: now warns on key mismatch and aborts if essentially random init.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Pretrain ckpt not found: {ckpt_path}')

    sd = torch.load(ckpt_path, map_location=device)
    keys = list(sd.keys())
    logger.info(f'Pretrain ckpt keys: {keys}')

    loaded = []
    failed = []
    for name in ['csi_encoder', 'local_encoder', 'feature_pooling', 'global_modeler']:
        if name in sd:
            m, u = getattr(model, name).load_state_dict(sd[name], strict=False)
            logger.info(f'  {name}: missing={len(m)} unexpected={len(u)}')

            if len(m) > 0 or len(u) > 0:
                logger.warning(
                    f'  ⚠ {name} KEY MISMATCH! '
                    f'missing[:5]={m[:5]}, unexpected[:5]={u[:5]}'
                )
                expected_keys = len(getattr(model, name).state_dict())
                load_ratio = (expected_keys - len(m)) / max(expected_keys, 1)
                if load_ratio < 0.5:
                    logger.error(
                        f'  ❌ {name} loaded only {load_ratio*100:.1f}% of expected params! '
                        f'This is essentially random init.'
                    )
                    failed.append(name)
            loaded.append(name)

    # Stage 1B 的 action_classifier（如果 ckpt 里有的话）
    if 'action_classifier' in sd:
        try:
            m, u = model.action_classifier.load_state_dict(
                sd['action_classifier'], strict=False)
            logger.info(f'  action_classifier (from Stage 1B): '
                        f'missing={len(m)} unexpected={len(u)}')
            if len(m) > 0 or len(u) > 0:
                logger.warning(
                    f'  ⚠ action_classifier mismatch! '
                    f'missing[:5]={m[:5]}, unexpected[:5]={u[:5]}'
                )
            loaded.append('action_classifier')
        except Exception as e:
            logger.warning(f'action_classifier load failed: {e}')

    if failed:
        logger.error(f'❌ ABORTED: modules failed to load: {failed}')
        logger.error(f'   Stage 2 would essentially train from random weights for these modules.')
        raise RuntimeError(f'Backbone loading failed for: {failed}')

    logger.info(f'Loaded pretrained components: {loaded}')


def make_optimizer(model, lr_backbone, lr_head, weight_decay):
    """Two param groups: pretrained backbone vs fresh heads."""
    backbone_modules = ['csi_encoder', 'local_encoder',
                        'feature_pooling', 'global_modeler']

    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(m + '.') for m in backbone_modules):
            backbone_params.append(param)
        else:
            head_params.append(param)

    return AdamW([
        {'params': backbone_params, 'lr': lr_backbone, 'name': 'backbone'},
        {'params': head_params, 'lr': lr_head, 'name': 'head'},
    ], weight_decay=weight_decay, betas=(0.9, 0.999))


def main():
    # Get base config (same as train.py)
    args = get_config()
    extras = parse_stage2_extras()
    args.pretrain_ckpt = extras.pretrain_ckpt
    args.lr_backbone = extras.lr_backbone
    args.lr_head = extras.lr_head
    args.freeze_backbone_epochs = extras.freeze_backbone_epochs

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    logger = setup_logger('Stage2-Pose',
                          log_file=os.path.join(args.save_dir, 'train.log'))
    logger.info(f'Stage 2: pose fine-tuning')
    logger.info(f'Pretrain ckpt: {args.pretrain_ckpt}')
    logger.info(f'lr_backbone={args.lr_backbone}, lr_head={args.lr_head}')
    logger.info(f'Config: {vars(args)}')

    # Data (source for training, target for eval only)
    train_loader, test_loader = build_dataloaders(args, synthetic=False)
    logger.info(f'Train: {len(train_loader)} batches | Test (E04): {len(test_loader)} batches')

    # Model — note this assumes full_model.py has been patched to use HMSF
    model = CSIRSCPoseDG(args).to(device)
    logger.info(f'Model parameters: {count_parameters(model):,}')

    # Load Stage 1A/1B weights
    load_pretrained_backbone(model, args.pretrain_ckpt, logger, device)

    # Loss
    loss_fn = TotalLoss(
        lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
        alpha=args.alpha, beta=args.beta,
        gamma=args.gamma, delta=args.delta,
        lambda_hip=getattr(args, 'lambda_hip', 0.3),
    )
    pose_loss_fn = PoseLoss(
        lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
        lambda_hip=getattr(args, 'lambda_hip', 0.3),
    )
    evaluator = PoseEvaluator(unit='meter')

    # Differential LR optimizer
    optimizer = make_optimizer(
        model, args.lr_backbone, args.lr_head, args.weight_decay
    )

    # Cosine schedule (same as train.py)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    timer = Timer(); timer.start()
    best_mpjpe = float('inf')
    patience_counter = 0
    patience = getattr(args, 'patience', 15)

    for epoch in range(1, args.epochs + 1):
        # Optional: freeze backbone for first N epochs
        if args.freeze_backbone_epochs > 0 and epoch <= args.freeze_backbone_epochs:
            for name in ['csi_encoder', 'local_encoder', 'feature_pooling', 'global_modeler']:
                for p in getattr(model, name).parameters():
                    p.requires_grad = False
            if epoch == 1:
                logger.info(f'Backbone FROZEN for first {args.freeze_backbone_epochs} epochs')
        elif args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            for name in ['csi_encoder', 'local_encoder', 'feature_pooling', 'global_modeler']:
                for p in getattr(model, name).parameters():
                    p.requires_grad = True
            logger.info(f'Backbone UNFROZEN at epoch {epoch}')

        logger.info(f'\n{"=" * 60}')
        cur_lrs = [g['lr'] for g in optimizer.param_groups]
        logger.info(f'Epoch {epoch}/{args.epochs} | LR: '
                    f'backbone={cur_lrs[0]:.2e} head={cur_lrs[1]:.2e}')

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, pose_loss_fn,
            device, epoch, logger, args,
        )
        logger.info(
            f'[Train] Epoch {epoch} | Loss: {train_metrics["loss"]:.4f} '
            f'Act: {train_metrics["l_action"]:.4f} | Time: {timer.elapsed_str()}'
        )
        scheduler.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            eval_metrics = evaluate(model, test_loader, device, evaluator, logger)
            cur = eval_metrics['MPJPE (mm)']
            if cur < best_mpjpe:
                best_mpjpe = cur
                patience_counter = 0
                save_checkpoint(
                    model, optimizer, epoch, eval_metrics,
                    os.path.join(args.save_dir, 'best_model.pth'),
                )
                logger.info(f'*** New best MPJPE: {best_mpjpe:.2f}mm ***')
            else:
                patience_counter += 1
                logger.info(f'No improvement. Patience: {patience_counter}/{patience}')
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break

        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, {},
                os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'),
            )

    logger.info(f'\nStage 2 done. Best MPJPE: {best_mpjpe:.2f}mm')
    logger.info(f'Total time: {timer.elapsed_str()}')


if __name__ == '__main__':
    main()