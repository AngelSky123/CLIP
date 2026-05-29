"""
Vision-backbone training (cross-environment DG) — analogous to train_stage2.py.

Trains CSIRSCPoseDG with an ImageNet-pretrained vision backbone front-end,
on source envs (E01-E03), evaluated strictly on the target env (E04) with
action_idx=None. Reuses train.py's train_one_epoch / evaluate verbatim, so the
RSC + Action-Dropout + TotalLoss logic is identical to your Plan A+B baseline.

The ONLY differences vs train.py:
  * args.use_vision_backbone = True (+ vision_* options)
  * differential LR: the pretrained timm backbone gets a low LR; every fresh
    module (input InstanceNorm, projection, global_modeler, decoder, classifier)
    gets a high LR. A flat lr=1e-3 would destroy the ImageNet features instantly.

This is the fair counterpart to "MAE Stage 1A -> Stage 2": ImageNet pretraining
replaces MAE pretraining as the source of good spatial features.

Usage:
    python train_vision.py \
        --data_root /home/a123456/PerceptAlign/MMFi \
        --train_envs E01 E02 E03 --test_env E04 \
        --vision_arch resnet18 \
        --vision_img_size 112 \
        --epochs 50 --batch_size 4 --accumulate_grad 4 \
        --lr_backbone 1e-4 --lr_head 5e-4 \
        --save_dir ./checkpoints/vision_resnet18

Try also: --vision_arch resnet50
          --vision_arch swin_tiny_patch4_window7_224 --vision_img_size 224
          --vision_arch vit_small_patch16_224       --vision_img_size 224
Add --vision_freeze to train only the heads (fast sanity check / strong DG baseline).
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import get_config
from dataset import build_dataloaders
from models import CSIRSCPoseDG
from losses import TotalLoss, PoseLoss
from evaluate import PoseEvaluator
from utils import (set_seed, setup_logger, count_parameters,
                   save_checkpoint, Timer)

# Reuse the EXACT training loop + eval from train.py (RSC, Action Dropout, etc.)
from train import train_one_epoch, evaluate, action_to_index


def parse_vision_extras():
    """Vision-specific args; the rest come from config.get_config()."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--vision_arch', type=str, default='resnet18',
                   help="timm model name: resnet18/resnet50/"
                        "vit_small_patch16_224/swin_tiny_patch4_window7_224 ...")
    p.add_argument('--vision_img_size', type=int, default=112,
                   help='each CSI frame is resized to this square size. '
                        'Use 224 for vit_*_224 / swin_*_224.')
    p.add_argument('--vision_scratch', action='store_true', default=False,
                   help='train backbone from random init (disable ImageNet weights)')
    p.add_argument('--vision_freeze', action='store_true', default=False,
                   help='freeze the backbone; train only adapter+proj+heads')
    p.add_argument('--vision_no_instance_norm', action='store_true', default=False,
                   help='disable per-sample InstanceNorm on the CSI input')
    p.add_argument('--vision_weights', type=str, default=None,
                   help='local timm-format checkpoint (.safetensors/.pth) for '
                        'offline machines that cannot reach huggingface.co')
    p.add_argument('--lr_backbone', type=float, default=1e-4,
                   help='LR for the pretrained backbone (slow)')
    p.add_argument('--lr_head', type=float, default=5e-4,
                   help='LR for fresh modules (fast)')
    known, _ = p.parse_known_args()
    return known


def make_optimizer(model, lr_backbone, lr_head, weight_decay):
    """Two param groups:
        slow  = the pretrained timm backbone (vision_backbone.backbone.*)
        fast  = everything fresh (input norm, projection, global_modeler,
                pose_decoder, action_classifier)
    """
    slow_prefix = 'vision_backbone.backbone.'
    slow, fast = [], []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        (slow if name.startswith(slow_prefix) else fast).append(prm)

    groups = []
    if slow:
        groups.append({'params': slow, 'lr': lr_backbone, 'name': 'backbone'})
    groups.append({'params': fast, 'lr': lr_head, 'name': 'head'})
    return AdamW(groups, weight_decay=weight_decay, betas=(0.9, 0.999))


def main():
    # Base config (cross-env DG) + vision extras
    args = get_config()
    extras = parse_vision_extras()
    args.use_vision_backbone = True
    args.vision_arch = extras.vision_arch
    args.vision_img_size = extras.vision_img_size
    args.vision_scratch = extras.vision_scratch
    args.vision_freeze = extras.vision_freeze
    args.vision_no_instance_norm = extras.vision_no_instance_norm
    args.vision_weights = extras.vision_weights
    args.lr_backbone = extras.lr_backbone
    args.lr_head = extras.lr_head

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    logger = setup_logger('Vision-DG',
                          log_file=os.path.join(args.save_dir, 'train.log'))
    logger.info('Vision-backbone cross-environment DG training')
    logger.info(f'arch={args.vision_arch} img_size={args.vision_img_size} '
                f'pretrained={not args.vision_scratch} freeze={args.vision_freeze}')
    logger.info(f'lr_backbone={args.lr_backbone} lr_head={args.lr_head}')
    logger.info(f'Strict DG: test-time action_idx=None (no GT labels)')
    logger.info(f'Config: {vars(args)}')

    data_exists = os.path.exists(args.data_root)
    train_loader, test_loader = build_dataloaders(args, synthetic=not data_exists)
    logger.info(f'Train batches: {len(train_loader)}, Test batches: {len(test_loader)}')

    model = CSIRSCPoseDG(args).to(device)
    logger.info(f'Model parameters: {count_parameters(model):,}')

    loss_fn = TotalLoss(
        lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta,
        lambda_hip=getattr(args, 'lambda_hip', 1.0),
    )
    pose_loss_fn = PoseLoss(
        lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
        lambda_hip=getattr(args, 'lambda_hip', 1.0),
    )
    evaluator = PoseEvaluator(unit='meter')

    optimizer = make_optimizer(model, args.lr_backbone, args.lr_head, args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    timer = Timer(); timer.start()
    best_mpjpe = float('inf')
    patience_counter = 0
    patience = getattr(args, 'patience', 15)

    for epoch in range(1, args.epochs + 1):
        cur_lrs = [g['lr'] for g in optimizer.param_groups]
        logger.info(f'\n{"="*60}')
        logger.info(f'Epoch {epoch}/{args.epochs} | LR: ' +
                    ' '.join(f'{g["name"]}={lr:.2e}'
                             for g, lr in zip(optimizer.param_groups, cur_lrs)))

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
                save_checkpoint(model, optimizer, epoch, eval_metrics,
                                os.path.join(args.save_dir, 'best_model.pth'))
                logger.info(f'*** New best MPJPE: {best_mpjpe:.2f}mm ***')
            else:
                patience_counter += 1
                logger.info(f'No improvement. Patience: {patience_counter}/{patience}')
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break

        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, {},
                            os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'))

    logger.info(f'\nVision-backbone training complete. Best MPJPE: {best_mpjpe:.2f}mm')
    logger.info(f'Total time: {timer.elapsed_str()}')


if __name__ == '__main__':
    main()