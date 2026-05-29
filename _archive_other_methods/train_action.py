"""
Stage 1B: Action Pretraining on Source Domains.

Loads MAE-pretrained backbone (from Stage 1A) and trains action classifier
on 27-class action labels across source envs only.

Usage:
    python train_action.py \
        --data_root /home/a123456/PerceptAlign/MMFi \
        --train_envs E01 E02 E03 \
        --mae_ckpt ./checkpoints/stage1a_mae/mae_latest.pt \
        --epochs 50 \
        --batch_size 8 \
        --accumulate_grad 2 \
        --lr_backbone 1e-4 \
        --lr_head 5e-4 \
        --save_dir ./checkpoints/stage1b_action

batch['action'] is a list of strings ["A01",...]. Converted via int(s[1:])-1.
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
from models.pose_decoder import ActionClassifier
from utils import set_seed, setup_logger, count_parameters, AverageMeter, Timer


def action_to_index(action_str: str) -> int:
    return int(action_str[1:]) - 1


class ActionPretrainModel(nn.Module):
    def __init__(self, csi_encoder, local_encoder, feature_pooling, global_modeler,
                 global_dim=128, num_actions=27, action_embed_dim=32):
        super().__init__()
        self.csi_encoder = csi_encoder
        self.local_encoder = local_encoder
        self.feature_pooling = feature_pooling
        self.global_modeler = global_modeler
        self.action_classifier = ActionClassifier(
            in_dim=global_dim, num_actions=num_actions, embed_dim=action_embed_dim,
        )

    def forward(self, csi):
        feat = self.csi_encoder(csi)
        z_local = self.local_encoder(feat)
        z_pooled = self.feature_pooling(z_local)
        z_global = self.global_modeler(z_pooled)
        logits = self.action_classifier(z_global)
        return logits, z_global


def get_args():
    p = argparse.ArgumentParser(description='Stage 1B action pretraining')
    p.add_argument('--data_root', type=str,
                   default='/home/a123456/PerceptAlign/MMFi')
    p.add_argument('--train_envs', nargs='+', default=['E01', 'E02', 'E03'])
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--stride', type=int, default=32)

    p.add_argument('--num_actions', type=int, default=27)
    p.add_argument('--action_embed_dim', type=int, default=32)
    p.add_argument('--label_smoothing', type=float, default=0.1)

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

    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--accumulate_grad', '--accum', type=int, default=2)
    p.add_argument('--lr_backbone', type=float, default=1e-4)
    p.add_argument('--lr_head', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--warmup_epochs', type=int, default=3)
    p.add_argument('--grad_clip', type=float, default=1.0)

    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--mae_ckpt', type=str, required=True)
    p.add_argument('--save_dir', type=str, default='./checkpoints/stage1b_action')
    p.add_argument('--log_interval', type=int, default=20)
    return p.parse_args()


def build_backbone(args):
    csi_encoder = DualBranchCSIEncoder(
        amp_channels=args.amp_channels, phase_channels=args.phase_channels,
        hidden_dim=args.encoder_hidden_dim, out_dim=args.encoder_out_dim,
    )
    local_encoder = LocalSpatioTemporalEncoder(
        in_channels=args.encoder_out_dim, hidden_dim=args.local_hidden_dim,
        out_dim=args.local_out_dim, num_blocks=args.num_res3d_blocks,
    )
    feature_pooling = LocalFeaturePooling(
        in_channels=args.local_out_dim, out_channels=args.global_dim,
    )
    global_modeler = GlobalTemporalModeler(
        in_dim=args.global_dim, global_dim=args.global_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads, tcn_channels=list(args.tcn_channels),
        tcn_kernel_size=args.tcn_kernel_size,
        dropout=args.transformer_dropout, max_seq_len=args.seq_len + 50,
    )
    return csi_encoder, local_encoder, feature_pooling, global_modeler


def main():
    args = get_args()
    set_seed(args.seed)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    logger = setup_logger('Stage1B-Action',
                          log_file=os.path.join(args.save_dir, 'train.log'))
    logger.info(f'Stage 1B: action pretraining on {args.train_envs}')
    logger.info(f'Args: {vars(args)}')

    train_set = MMFiDataset(
        data_root=args.data_root, envs=args.train_envs,
        seq_len=args.seq_len, stride=args.stride, augment=True,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    logger.info(f'Train: {len(train_set)} samples, {len(train_loader)} batches')

    csi_enc, local_enc, pool, global_mod = build_backbone(args)
    model = ActionPretrainModel(
        csi_encoder=csi_enc, local_encoder=local_enc,
        feature_pooling=pool, global_modeler=global_mod,
        global_dim=args.global_dim, num_actions=args.num_actions,
        action_embed_dim=args.action_embed_dim,
    ).to(device)
    logger.info(f'Model params: {count_parameters(model):,}')

    # Load Stage 1A backbone weights
    if not os.path.exists(args.mae_ckpt):
        raise FileNotFoundError(f'MAE ckpt not found: {args.mae_ckpt}')
    sd = torch.load(args.mae_ckpt, map_location=device)
    m1, _ = model.csi_encoder.load_state_dict(sd['csi_encoder'], strict=False)
    m2, _ = model.local_encoder.load_state_dict(sd['local_encoder'], strict=False)
    m3, _ = model.feature_pooling.load_state_dict(sd['feature_pooling'], strict=False)
    m4, _ = model.global_modeler.load_state_dict(sd['global_modeler'], strict=False)
    logger.info(f'MAE backbone loaded: missing csi={len(m1)} local={len(m2)} '
                f'pool={len(m3)} global={len(m4)}')

    # Differential LR
    backbone_params = (list(model.csi_encoder.parameters())
                       + list(model.local_encoder.parameters())
                       + list(model.feature_pooling.parameters())
                       + list(model.global_modeler.parameters()))
    head_params = list(model.action_classifier.parameters())
    optimizer = AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone, 'name': 'backbone'},
        {'params': head_params, 'lr': args.lr_head, 'name': 'head'},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.999))

    steps_per_epoch = max(1, len(train_loader) // args.accumulate_grad)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    timer = Timer(); timer.start()
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        optimizer.zero_grad()

        for it, batch in enumerate(train_loader):
            csi = batch['csi'].to(device, non_blocking=True)
            labels = torch.tensor(
                [action_to_index(a) for a in batch['action']],
                dtype=torch.long, device=device,
            )

            logits, _ = model(csi)
            loss = criterion(logits, labels)
            (loss / args.accumulate_grad).backward()

            with torch.no_grad():
                acc = (logits.argmax(-1) == labels).float().mean().item()
            loss_meter.update(loss.item(), csi.size(0))
            acc_meter.update(acc, csi.size(0))

            if (it + 1) % args.accumulate_grad == 0 or (it + 1) == len(train_loader):
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (it + 1) % args.log_interval == 0:
                lr_bb = optimizer.param_groups[0]['lr']
                lr_hd = optimizer.param_groups[1]['lr']
                logger.info(
                    f'[Ep {epoch:2d} It {it+1:4d}/{len(train_loader)}] '
                    f'loss={loss_meter.avg:.4f} acc={acc_meter.avg*100:.2f}% '
                    f'lr_bb={lr_bb:.2e} lr_hd={lr_hd:.2e}'
                )

        logger.info(
            f'[Epoch {epoch:2d}] avg_loss={loss_meter.avg:.4f} '
            f'avg_acc={acc_meter.avg*100:.2f}% time={timer.elapsed_str()}'
        )

        torch.save({
            'epoch': epoch,
            'csi_encoder': model.csi_encoder.state_dict(),
            'local_encoder': model.local_encoder.state_dict(),
            'feature_pooling': model.feature_pooling.state_dict(),
            'global_modeler': model.global_modeler.state_dict(),
            'action_classifier': model.action_classifier.state_dict(),
            'train_acc': acc_meter.avg,
        }, os.path.join(args.save_dir, 'action_latest.pt'))

        if acc_meter.avg > best_acc:
            best_acc = acc_meter.avg
            torch.save({
                'epoch': epoch,
                'csi_encoder': model.csi_encoder.state_dict(),
                'local_encoder': model.local_encoder.state_dict(),
                'feature_pooling': model.feature_pooling.state_dict(),
                'global_modeler': model.global_modeler.state_dict(),
                'action_classifier': model.action_classifier.state_dict(),
                'train_acc': acc_meter.avg,
            }, os.path.join(args.save_dir, 'action_best.pt'))
            logger.info(f'Saved best (train_acc={best_acc*100:.2f}%)')

        torch.cuda.empty_cache()

    logger.info(f'Stage 1B done. best train acc={best_acc*100:.2f}% '
                f'time={timer.elapsed_str()}')


if __name__ == '__main__':
    main()