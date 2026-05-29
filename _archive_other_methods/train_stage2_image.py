"""
Stage 2 (image): pose fine-tuning from a pretrained image backbone.

Loads vision_backbone + global_modeler (from Stage 1A MAE or Stage 1B action) into
CSIRSCPoseDG(use_vision_backbone=True), then fine-tunes the full pose decoder with
RSC + Action Dropout. Reuses train.py's train_one_epoch / evaluate verbatim, so the
DG training logic is identical to the raw-CSI pipeline — only the input is images.

Differential LR: pretrained backbone (vision_backbone + global_modeler) slow,
fresh heads (pose_decoder, action_classifier) fast.

Usage:
    python train_stage2_image.py \
        --data_root /home/a123456/PerceptAlign/MMFi \
        --train_envs E01 E02 E03 --test_env E04 \
        --vision_arch resnet18 --vision_img_size 112 \
        --pretrain_ckpt ./checkpoints/img_stage1a_mae/mae_latest.pt \
        --epochs 50 --batch_size 8 --accumulate_grad 2 \
        --lr_backbone 1e-4 --lr_head 5e-4 --lambda_hip 0.3 \
        --save_dir ./checkpoints/img_stage2_pose
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import get_config
from dataset_image import build_image_dataloaders
from models import CSIRSCPoseDG
from losses import TotalLoss, PoseLoss
from evaluate import PoseEvaluator
from utils import set_seed, setup_logger, count_parameters, save_checkpoint, Timer

# Reuse the exact training loop + eval from train.py
from train import train_one_epoch, evaluate


def parse_extras():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--pretrain_ckpt', type=str, required=True,
                   help='Stage 1A (mae_*.pt) or 1B (action_*.pt) checkpoint.')
    p.add_argument('--vision_arch', type=str, default='resnet18')
    p.add_argument('--vision_img_size', type=int, default=112)
    p.add_argument('--vision_scratch', action='store_true', default=False)
    p.add_argument('--vision_weights', type=str, default=None)
    p.add_argument('--lr_backbone', type=float, default=1e-4)
    p.add_argument('--lr_head', type=float, default=5e-4)
    known, _ = p.parse_known_args()
    return known


def load_pretrained(model, ckpt_path, logger, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Pretrain ckpt not found: {ckpt_path}')
    sd = torch.load(ckpt_path, map_location=device)
    loaded = []
    if 'vision_backbone' in sd:
        m, _ = model.vision_backbone.load_state_dict(sd['vision_backbone'], strict=False)
        logger.info(f'  vision_backbone: missing={len(m)}'); loaded.append('vision_backbone')
    if 'global_modeler' in sd:
        m, _ = model.global_modeler.load_state_dict(sd['global_modeler'], strict=False)
        logger.info(f'  global_modeler: missing={len(m)}'); loaded.append('global_modeler')
    if 'action_classifier' in sd:                       # present only from Stage 1B
        try:
            m, _ = model.action_classifier.load_state_dict(sd['action_classifier'], strict=False)
            logger.info(f'  action_classifier: missing={len(m)}'); loaded.append('action_classifier')
        except Exception as e:
            logger.warning(f'action_classifier load failed: {e}')
    logger.info(f'Loaded pretrained: {loaded}')


def make_optimizer(model, lr_backbone, lr_head, weight_decay):
    slow_prefixes = ('vision_backbone.', 'global_modeler.')
    slow, fast = [], []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        (slow if name.startswith(slow_prefixes) else fast).append(prm)
    groups = []
    if slow:
        groups.append({'params': slow, 'lr': lr_backbone, 'name': 'backbone'})
    groups.append({'params': fast, 'lr': lr_head, 'name': 'head'})
    return AdamW(groups, weight_decay=weight_decay, betas=(0.9, 0.999))


def main():
    args = get_config()
    ex = parse_extras()
    args.use_vision_backbone = True
    args.vision_in_channels = 3                 # rendered images are 3-channel
    args.vision_arch = ex.vision_arch
    args.vision_img_size = ex.vision_img_size
    args.vision_scratch = ex.vision_scratch
    args.vision_weights = ex.vision_weights
    args.vision_no_instance_norm = False
    args.vision_freeze = False
    args.pretrain_ckpt = ex.pretrain_ckpt
    args.lr_backbone = ex.lr_backbone
    args.lr_head = ex.lr_head

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('Img-Stage2', os.path.join(args.save_dir, 'train.log'))
    logger.info('Stage 2 (image) pose fine-tuning')
    logger.info(f'arch={args.vision_arch} img={args.vision_img_size} '
                f'pretrain={args.pretrain_ckpt}')
    logger.info(f'lr_backbone={args.lr_backbone} lr_head={args.lr_head}')

    data_exists = os.path.exists(args.data_root)
    train_loader, test_loader = build_image_dataloaders(args, synthetic=not data_exists)
    logger.info(f'Train batches: {len(train_loader)}, Test batches: {len(test_loader)}')

    model = CSIRSCPoseDG(args).to(device)
    logger.info(f'Model parameters: {count_parameters(model):,}')
    load_pretrained(model, args.pretrain_ckpt, logger, device)

    loss_fn = TotalLoss(lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
                        alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta,
                        lambda_hip=getattr(args, 'lambda_hip', 0.3))
    pose_loss_fn = PoseLoss(lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
                            lambda_hip=getattr(args, 'lambda_hip', 0.3))
    evaluator = PoseEvaluator(unit='meter')

    optimizer = make_optimizer(model, args.lr_backbone, args.lr_head, args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    timer = Timer(); timer.start()
    best_mpjpe = float('inf'); patience_counter = 0
    patience = getattr(args, 'patience', 15)

    for epoch in range(1, args.epochs + 1):
        cur_lrs = [g['lr'] for g in optimizer.param_groups]
        logger.info(f'\n{"="*60}')
        logger.info(f'Epoch {epoch}/{args.epochs} | LR: ' +
                    ' '.join(f'{g["name"]}={lr:.2e}' for g, lr in zip(optimizer.param_groups, cur_lrs)))
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn,
                                         pose_loss_fn, device, epoch, logger, args)
        logger.info(f'[Train] Epoch {epoch} | Loss: {train_metrics["loss"]:.4f} '
                    f'Act: {train_metrics["l_action"]:.4f} | Time: {timer.elapsed_str()}')
        scheduler.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            eval_metrics = evaluate(model, test_loader, device, evaluator, logger)
            cur = eval_metrics['MPJPE (mm)']
            if cur < best_mpjpe:
                best_mpjpe = cur; patience_counter = 0
                save_checkpoint(model, optimizer, epoch, eval_metrics,
                                os.path.join(args.save_dir, 'best_model.pth'))
                logger.info(f'*** New best MPJPE: {best_mpjpe:.2f}mm ***')
            else:
                patience_counter += 1
                logger.info(f'No improvement. Patience: {patience_counter}/{patience}')
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch}'); break

    logger.info(f'\nStage 2 (image) done. Best MPJPE: {best_mpjpe:.2f}mm')
    logger.info(f'Total time: {timer.elapsed_str()}')


if __name__ == '__main__':
    main()