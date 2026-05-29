"""
Stage 1B (image, OPTIONAL): action pretraining on rendered CSI images.

Loads the Stage-1A backbone (vision_backbone + global_modeler) and trains a
27-class action classifier on source envs only, with differential LR.

Usage:
    python train_action_image.py \
        --data_root /home/a123456/PerceptAlign/MMFi --train_envs E01 E02 E03 \
        --vision_arch resnet18 --vision_img_size 112 \
        --mae_ckpt ./checkpoints/img_stage1a_mae/mae_latest.pt \
        --epochs 50 --batch_size 8 --accumulate_grad 2 \
        --lr_backbone 1e-4 --lr_head 5e-4 \
        --save_dir ./checkpoints/img_stage1b_action
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
from models.pose_decoder import ActionClassifier
from utils import set_seed, setup_logger, count_parameters, AverageMeter, Timer


def action_to_index(a): return int(a[1:]) - 1


class ImageActionModel(nn.Module):
    def __init__(self, vision_backbone, global_modeler, global_dim=128,
                 num_actions=27, action_embed_dim=32):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.global_modeler = global_modeler
        self.action_classifier = ActionClassifier(global_dim, num_actions, action_embed_dim)

    def forward(self, csi):
        z = self.global_modeler(self.vision_backbone(csi))
        return self.action_classifier(z)


def get_args():
    p = argparse.ArgumentParser(description='Stage 1B image action')
    p.add_argument('--data_root', type=str, default='/home/a123456/PerceptAlign/MMFi')
    p.add_argument('--train_envs', nargs='+', default=['E01', 'E02', 'E03'])
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--stride', type=int, default=32)
    p.add_argument('--vision_arch', type=str, default='resnet18')
    p.add_argument('--vision_img_size', type=int, default=112)
    p.add_argument('--vision_scratch', action='store_true', default=False)
    p.add_argument('--global_dim', type=int, default=128)
    p.add_argument('--num_transformer_layers', type=int, default=3)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--tcn_channels', type=int, nargs='+', default=[128, 128])
    p.add_argument('--tcn_kernel_size', type=int, default=3)
    p.add_argument('--transformer_dropout', type=float, default=0.1)
    p.add_argument('--num_actions', type=int, default=27)
    p.add_argument('--action_embed_dim', type=int, default=32)
    p.add_argument('--label_smoothing', type=float, default=0.1)
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
    p.add_argument('--save_dir', type=str, default='./checkpoints/img_stage1b_action')
    p.add_argument('--log_interval', type=int, default=20)
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('Img-Stage1B', os.path.join(args.save_dir, 'train.log'))
    logger.info(f'Stage 1B (image action) on {args.train_envs}')

    data_exists = os.path.exists(args.data_root)
    if data_exists:
        train_set = MMFiImageDataset(args.data_root, args.train_envs, args.seq_len,
                                     args.stride, augment=True, img_size=args.vision_img_size)
    else:
        logger.warning('data_root missing -> synthetic smoke data')
        train_set = MMFiImageSyntheticDataset(80, args.seq_len, args.vision_img_size,
                                              len(args.train_envs))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    logger.info(f'Train: {len(train_set)} samples, {len(train_loader)} batches')

    vb = VisionBackboneEncoder(in_channels=3, out_dim=args.global_dim, arch=args.vision_arch,
                               pretrained=not args.vision_scratch, img_size=args.vision_img_size)
    gm = GlobalTemporalModeler(in_dim=args.global_dim, global_dim=args.global_dim,
                               num_transformer_layers=args.num_transformer_layers,
                               num_heads=args.num_heads, tcn_channels=list(args.tcn_channels),
                               tcn_kernel_size=args.tcn_kernel_size,
                               dropout=args.transformer_dropout, max_seq_len=args.seq_len + 50)
    model = ImageActionModel(vb, gm, args.global_dim, args.num_actions, args.action_embed_dim).to(device)
    logger.info(f'Model params: {count_parameters(model):,}')

    if not os.path.exists(args.mae_ckpt):
        raise FileNotFoundError(f'MAE ckpt not found: {args.mae_ckpt}')
    sd = torch.load(args.mae_ckpt, map_location=device)
    m1, _ = model.vision_backbone.load_state_dict(sd['vision_backbone'], strict=False)
    m2, _ = model.global_modeler.load_state_dict(sd['global_modeler'], strict=False)
    logger.info(f'MAE backbone loaded: missing vision={len(m1)} global={len(m2)}')

    backbone_params = list(model.vision_backbone.parameters()) + list(model.global_modeler.parameters())
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
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    timer = Timer(); timer.start(); best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        loss_meter, acc_meter = AverageMeter(), AverageMeter()
        optimizer.zero_grad()
        for it, batch in enumerate(train_loader):
            csi = batch['csi'].to(device, non_blocking=True)
            labels = torch.tensor([action_to_index(a) for a in batch['action']],
                                  dtype=torch.long, device=device)
            logits = model(csi)
            loss = criterion(logits, labels)
            (loss / args.accumulate_grad).backward()
            with torch.no_grad():
                acc = (logits.argmax(-1) == labels).float().mean().item()
            loss_meter.update(loss.item(), csi.size(0)); acc_meter.update(acc, csi.size(0))
            if (it + 1) % args.accumulate_grad == 0 or (it + 1) == len(train_loader):
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            if (it + 1) % args.log_interval == 0:
                logger.info(f'[Ep {epoch:2d} It {it+1:4d}/{len(train_loader)}] '
                            f'loss={loss_meter.avg:.4f} acc={acc_meter.avg*100:.2f}%')
        logger.info(f'[Epoch {epoch:2d}] avg_loss={loss_meter.avg:.4f} '
                    f'avg_acc={acc_meter.avg*100:.2f}% time={timer.elapsed_str()}')

        ckpt = {'epoch': epoch,
                'vision_backbone': model.vision_backbone.state_dict(),
                'global_modeler': model.global_modeler.state_dict(),
                'action_classifier': model.action_classifier.state_dict(),
                'train_acc': acc_meter.avg}
        torch.save(ckpt, os.path.join(args.save_dir, 'action_latest.pt'))
        if acc_meter.avg > best_acc:
            best_acc = acc_meter.avg
            torch.save(ckpt, os.path.join(args.save_dir, 'action_best.pt'))
            logger.info(f'Saved best (train_acc={best_acc*100:.2f}%)')
        torch.cuda.empty_cache()

    logger.info(f'Stage 1B done. best train acc={best_acc*100:.2f}% time={timer.elapsed_str()}')


if __name__ == '__main__':
    main()