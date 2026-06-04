"""
Stage 1A: MAE Self-Supervised Pretraining on Source Domains.

Place this file in your project ROOT (same directory as train.py).

Usage:
    python train_mae.py \
        --data_root /home/a123456/PerceptAlign/MMFi \
        --train_envs E01 E02 E03 \
        --epochs 300 \
        --batch_size 4 \
        --accumulate_grad 4 \
        --lr 1.5e-4 \
        --save_dir ./checkpoints/stage1a_mae

Source-only training: target env (E04) NEVER appears.
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
from dataset import MMFiDataset
from models.csi_encoder import DualBranchCSIEncoder
from models.local_encoder import LocalSpatioTemporalEncoder
from models.global_encoder import GlobalTemporalModeler
from models.local_encoder import LocalFeaturePooling
from models.mae_pretrain import CSIMaeModel
from utils import set_seed, setup_logger, count_parameters, AverageMeter, Timer
from domain_balanced_sampler import DomainBalancedBatchSampler


def get_args():
    p = argparse.ArgumentParser(description='MAE Stage 1A pretraining')
    p.add_argument('--data_root', type=str,
                   default='/home/a123456/PerceptAlign/MMFi')
    p.add_argument('--train_envs', nargs='+', default=['E01', 'E02', 'E03'])
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--stride', type=int, default=32)

    p.add_argument('--mae_mask_ratio', type=float, default=0.75)
    p.add_argument('--mae_patch_t', type=int, default=4)
    p.add_argument('--mae_patch_s', type=int, default=19)
    p.add_argument('--mae_patch_a', type=int, default=5)
    p.add_argument('--mae_decoder_hidden', type=int, default=256)

    p.add_argument('--amp_channels', type=int, default=3)
    p.add_argument('--phase_channels', type=int, default=6)
    p.add_argument('--encoder_hidden_dim', type=int, default=32)
    p.add_argument('--encoder_out_dim', type=int, default=64)
    p.add_argument('--local_hidden_dim', type=int, default=64)
    p.add_argument('--local_out_dim', type=int, default=64)
    p.add_argument('--num_res3d_blocks', type=int, default=2)
    p.add_argument('--global_dim', type=int, default=128)
    p.add_argument('--num_transformer_layers', type=int, default=3)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--tcn_channels', type=int, nargs='+', default=[128, 128])
    p.add_argument('--tcn_kernel_size', type=int, default=3)
    p.add_argument('--transformer_dropout', type=float, default=0.1)

    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--accumulate_grad', '--accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=1.5e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int, default=20)
    p.add_argument('--grad_clip', type=float, default=1.0)

    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--save_dir', type=str, default='./checkpoints/stage1a_mae')
    p.add_argument('--save_every', type=int, default=20)
    p.add_argument('--log_interval', type=int, default=20)
    p.add_argument('--resume', type=str, default='')

    p.add_argument('--lambda_unif', type=float, default=0.04)
    p.add_argument('--lambda_tcl',  type=float, default=0.1)
    p.add_argument('--lambda_dinv', type=float, default=0.2)
    return p.parse_args()


def build_backbone(args):
    csi_encoder = DualBranchCSIEncoder(
        amp_channels=args.amp_channels,
        phase_channels=args.phase_channels,
        hidden_dim=args.encoder_hidden_dim,
        out_dim=args.encoder_out_dim,
    )
    local_encoder = LocalSpatioTemporalEncoder(
        in_channels=args.encoder_out_dim,
        hidden_dim=args.local_hidden_dim,
        out_dim=args.local_out_dim,
        num_blocks=args.num_res3d_blocks,
    )
    feature_pooling = LocalFeaturePooling(
        in_channels=args.local_out_dim,
        out_channels=args.global_dim,
    )
    global_modeler = GlobalTemporalModeler(
        in_dim=args.global_dim,
        global_dim=args.global_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        tcn_channels=list(args.tcn_channels),
        tcn_kernel_size=args.tcn_kernel_size,
        dropout=args.transformer_dropout,
        max_seq_len=args.seq_len + 50,
    )
    return csi_encoder, local_encoder, feature_pooling, global_modeler


def get_lr_lambda(warmup_steps, total_steps):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn


def main():
    args = get_args()
    set_seed(args.seed)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    logger = setup_logger('Stage1A-MAE',
                          log_file=os.path.join(args.save_dir, 'train.log'))
    logger.info(f'Stage 1A: MAE pretraining on {args.train_envs}')
    logger.info(f'Args: {vars(args)}')

    train_set = MMFiDataset(
        data_root=args.data_root,
        envs=args.train_envs,
        seq_len=args.seq_len,
        stride=args.stride,
        augment=True,
    )
    # train_loader = DataLoader(
    #     train_set, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.num_workers, pin_memory=True, drop_last=True,
    # )
    batch_sampler = DomainBalancedBatchSampler(
        train_set, batch_size=args.batch_size, group_size=2, seed=args.seed)
    train_loader = DataLoader(
        train_set, batch_sampler=batch_sampler,
        num_workers=args.num_workers, pin_memory=True,
    )
    logger.info(f'Train: {len(train_set)} samples, {len(train_loader)} batches')

    csi_enc, local_enc, pool, global_mod = build_backbone(args)
    model = CSIMaeModel(
        csi_encoder=csi_enc,
        local_encoder=local_enc,
        feature_pooling=pool,
        global_modeler=global_mod,
        in_channels=args.amp_channels + args.phase_channels,
        patch_t=args.mae_patch_t,
        patch_s=args.mae_patch_s,
        patch_a=args.mae_patch_a,
        global_dim=args.global_dim,
        mask_ratio=args.mae_mask_ratio,
        decoder_hidden=args.mae_decoder_hidden,
    ).to(device)
    logger.info(f'Model params: {count_parameters(model):,}')

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      betas=(0.9, 0.95), weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // args.accumulate_grad)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, get_lr_lambda(warmup_steps, total_steps)
    )

    start_epoch = 0
    global_step = 0
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
        batch_sampler.set_epoch(epoch)
        model.train()
        loss_meter = AverageMeter()
        mask_meter = AverageMeter()
        optimizer.zero_grad()

        for it, batch in enumerate(train_loader):
            # csi = batch['csi'].to(device, non_blocking=True)
            # loss, info = model(csi)
            # (loss / args.accumulate_grad).backward()
            csi = batch['csi'].to(device, non_blocking=True)
            loss, info = model(csi)
            # === DT-Pose 式域一致表征 (翻译适配窗口架构) ===
            from mae_dcl_losses import (uniformity_loss,
                                        temporal_contrastive_loss,
                                        domain_invariant_loss)
            zg = info['z_global']
            env_ids = torch.tensor([{'E01':0,'E02':1,'E03':2}[e] for e in batch['env']],
                                   device=device)
            act_ids = torch.tensor([int(a[1:])-1 for a in batch['action']],
                                   device=device)
            loss = (loss
                    + args.lambda_unif * uniformity_loss(zg)
                    + args.lambda_tcl  * temporal_contrastive_loss(zg)
                    + args.lambda_dinv * domain_invariant_loss(zg, env_ids, act_ids))
            (loss / args.accumulate_grad).backward()

            loss_meter.update(loss.item(), csi.size(0))
            mask_meter.update(info['mask_ratio_actual'], csi.size(0))

            if (it + 1) % args.accumulate_grad == 0 or (it + 1) == len(train_loader):
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (it + 1) % args.log_interval == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f'[Ep {epoch:3d} It {it+1:4d}/{len(train_loader)}] '
                    f'loss={loss_meter.avg:.4f} mask={mask_meter.avg:.3f} '
                    f'lr={cur_lr:.2e}'
                )

            del loss, info
        torch.cuda.empty_cache()

        logger.info(
            f'[Epoch {epoch:3d}] avg_loss={loss_meter.avg:.4f} '
            f'time={timer.elapsed_str()}'
        )

        save_latest = os.path.join(args.save_dir, 'mae_latest.pt')
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'csi_encoder': model.csi_encoder.state_dict(),
            'local_encoder': model.local_encoder.state_dict(),
            'feature_pooling': model.feature_pooling.state_dict(),
            'global_modeler': model.global_modeler.state_dict(),
            'loss': loss_meter.avg,
        }, save_latest)

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            save_path = os.path.join(args.save_dir, f'mae_ep{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'csi_encoder': model.csi_encoder.state_dict(),
                'local_encoder': model.local_encoder.state_dict(),
                'feature_pooling': model.feature_pooling.state_dict(),
                'global_modeler': model.global_modeler.state_dict(),
                'loss': loss_meter.avg,
            }, save_path)
            logger.info(f'Saved {save_path}')

    logger.info(f'Stage 1A done. Total time: {timer.elapsed_str()}')


if __name__ == '__main__':
    main()