"""
Stage 1A (image): MAE self-supervised pretraining on rendered CSI images.

Source-only (target env never appears). Pretrains vision_backbone + global_modeler,
saving both module state_dicts for Stage 1B / Stage 2 to load.

Usage:
    python train_mae_image.py \
        --data_root /home/a123456/PerceptAlign/MMFi \
        --train_envs E01 E02 E03 \
        --vision_arch resnet18 --vision_img_size 112 \
        --patch_size 16 --mask_ratio 0.75 \
        --epochs 300 --batch_size 8 --accumulate_grad 2 \
        --lr 1.5e-4 \
        --save_dir ./checkpoints/img_stage1a_mae
"""
import os
import sys
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_image import MMFiImageDataset, MMFiImageSyntheticDataset
from models.vision_backbone import VisionBackboneEncoder
from models.global_encoder import GlobalTemporalModeler
from models.mae_image import ImageMaeModel
from utils import set_seed, setup_logger, count_parameters, AverageMeter, Timer


def get_args():
    p = argparse.ArgumentParser(description='Stage 1A image MAE')
    p.add_argument('--data_root', type=str, default='/home/a123456/PerceptAlign/MMFi')
    p.add_argument('--train_envs', nargs='+', default=['E01', 'E02', 'E03'])
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--stride', type=int, default=32)
    # image / vision
    p.add_argument('--vision_arch', type=str, default='resnet18')
    p.add_argument('--vision_img_size', type=int, default=112)
    p.add_argument('--vision_scratch', action='store_true', default=False)
    p.add_argument('--vision_weights', type=str, default=None)
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--mask_ratio', type=float, default=0.75)
    p.add_argument('--decoder_hidden', type=int, default=256)
    p.add_argument('--global_dim', type=int, default=128)
    p.add_argument('--num_transformer_layers', type=int, default=3)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--tcn_channels', type=int, nargs='+', default=[128, 128])
    p.add_argument('--tcn_kernel_size', type=int, default=3)
    p.add_argument('--transformer_dropout', type=float, default=0.1)
    # optim
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--accumulate_grad', '--accum', type=int, default=2)
    p.add_argument('--lr', type=float, default=1.5e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int, default=20)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--save_dir', type=str, default='./checkpoints/img_stage1a_mae')
    p.add_argument('--save_every', type=int, default=20)
    p.add_argument('--log_interval', type=int, default=20)
    p.add_argument('--resume', type=str, default='')
    return p.parse_args()


def build_backbone(args):
    vb = VisionBackboneEncoder(
        in_channels=3, out_dim=args.global_dim, arch=args.vision_arch,
        pretrained=not args.vision_scratch, img_size=args.vision_img_size,
        weights_path=args.vision_weights,
    )
    gm = GlobalTemporalModeler(
        in_dim=args.global_dim, global_dim=args.global_dim,
        num_transformer_layers=args.num_transformer_layers, num_heads=args.num_heads,
        tcn_channels=list(args.tcn_channels), tcn_kernel_size=args.tcn_kernel_size,
        dropout=args.transformer_dropout, max_seq_len=args.seq_len + 50,
    )
    return vb, gm


def main():
    args = get_args()
    set_seed(args.seed)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('Img-Stage1A', os.path.join(args.save_dir, 'train.log'))
    logger.info(f'Stage 1A (image MAE) on {args.train_envs}')
    logger.info(f'arch={args.vision_arch} img={args.vision_img_size} '
                f'patch={args.patch_size} mask={args.mask_ratio}')

    data_exists = os.path.exists(args.data_root)
    if data_exists:
        train_set = MMFiImageDataset(args.data_root, args.train_envs, args.seq_len,
                                     args.stride, augment=True,
                                     img_size=args.vision_img_size)
    else:
        logger.warning('data_root missing -> synthetic smoke data')
        train_set = MMFiImageSyntheticDataset(80, args.seq_len, args.vision_img_size,
                                              len(args.train_envs))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    logger.info(f'Train: {len(train_set)} samples, {len(train_loader)} batches')

    vb, gm = build_backbone(args)
    model = ImageMaeModel(vb, gm, in_channels=3, img_size=args.vision_img_size,
                          patch_size=args.patch_size, global_dim=args.global_dim,
                          mask_ratio=args.mask_ratio,
                          decoder_hidden=args.decoder_hidden).to(device)
    logger.info(f'Model params: {count_parameters(model):,}')

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                      weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // args.accumulate_grad)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch, global_step = 0, 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        scheduler.load_state_dict(ck['scheduler_state_dict'])
        start_epoch = ck['epoch'] + 1
        global_step = ck.get('global_step', 0)
        logger.info(f'Resumed from {args.resume}, epoch={start_epoch}')

    timer = Timer(); timer.start()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_meter, mask_meter = AverageMeter(), AverageMeter()
        optimizer.zero_grad()
        for it, batch in enumerate(train_loader):
            csi = batch['csi'].to(device, non_blocking=True)
            loss, info = model(csi)
            (loss / args.accumulate_grad).backward()
            loss_meter.update(loss.item(), csi.size(0))
            mask_meter.update(info['mask_ratio_actual'], csi.size(0))
            if (it + 1) % args.accumulate_grad == 0 or (it + 1) == len(train_loader):
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                global_step += 1
            if (it + 1) % args.log_interval == 0:
                logger.info(f'[Ep {epoch:3d} It {it+1:4d}/{len(train_loader)}] '
                            f'loss={loss_meter.avg:.4f} mask={mask_meter.avg:.3f} '
                            f'lr={optimizer.param_groups[0]["lr"]:.2e}')
            del loss, info
        torch.cuda.empty_cache()
        logger.info(f'[Epoch {epoch:3d}] avg_loss={loss_meter.avg:.4f} time={timer.elapsed_str()}')

        # checkpoint contract: save the two backbone modules for downstream stages
        ckpt = {
            'epoch': epoch, 'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'vision_backbone': model.vision_backbone.state_dict(),
            'global_modeler': model.global_modeler.state_dict(),
            'loss': loss_meter.avg,
        }
        torch.save(ckpt, os.path.join(args.save_dir, 'mae_latest.pt'))
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            torch.save({k: ckpt[k] for k in ('epoch', 'vision_backbone', 'global_modeler', 'loss')},
                       os.path.join(args.save_dir, f'mae_ep{epoch+1}.pt'))
            logger.info(f'Saved mae_ep{epoch+1}.pt')

    logger.info(f'Stage 1A done. Total time: {timer.elapsed_str()}')


if __name__ == '__main__':
    main()