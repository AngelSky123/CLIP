"""
Step A (RGB 变体): train the RGB -> 3D pose teacher (source envs E01-E03).

与 train_depth_teacher.py 同构, 仅输入模态换 RGB:
  * 模型 RGBPoseTeacher (ImageNet 预训练 ResNet18 截断, 兜底 --backbone scratch);
  * 分组学习率: 预训练 backbone 用 --lr_backbone (默认 1e-4), 其余 (proj/时序/头) 用 --lr (5e-4);
  * E04 eval 仍是教师强度 sanity check, 不是 DG 结果。

预期判读 (与 README §9 一致):
  * 看 MPJPE_aligned / PA —— RGB 教师的价值在相对结构 (能否优于深度教师的结构);
  * E04 hip 预计【差于】深度教师的 ~236mm (单目 RGB 无米制深度), 这不是失败,
    是已知物理事实; 蒸馏时据此把 --out_distill_hip_weight 降到 1.0。

Usage:
    python train_rgb_teacher.py \
        --data_root /home/a123456/PerceptAlign/MMFi \
        --train_envs E01 E02 E03 --test_env E04 \
        --rgb_img 112 --backbone resnet18 \
        --epochs 50 --batch_size 4 --accumulate_grad 4 \
        --save_dir ./checkpoints/rgb_teacher
依赖: torchvision==0.14.0 (配 torch 1.13)。装不上 -> --backbone scratch。
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_distill import build_teacher_dataloaders
from models.rgb_teacher import RGBPoseTeacher
from losses import PoseLoss
from evaluate import PoseEvaluator
from utils import (set_seed, setup_logger, count_parameters, save_checkpoint,
                   AverageMeter, Timer, save_run_config)


def get_args():
    p = argparse.ArgumentParser(description='Step A (RGB): rgb pose teacher')
    p.add_argument('--data_root', type=str, default='/home/a123456/PerceptAlign/MMFi')
    p.add_argument('--train_envs', nargs='+', default=['E01', 'E02', 'E03'])
    p.add_argument('--test_env', type=str, default='E04')
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--rgb_img', type=int, default=112)
    p.add_argument('--rgb_root', type=str, default=None,
                   help='RGB 数据独立根目录 (如机械盘); GT/CSI/depth 仍从 --data_root 读。None=与 data_root 相同')
    # depth_img/depth_clip 仅为 build_teacher_dataloaders 接口兼容, RGB 模态下不使用
    p.add_argument('--depth_img', type=int, default=112)
    p.add_argument('--depth_clip', type=float, default=5000.0)
    p.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'scratch'])
    p.add_argument('--no_pretrained', action='store_true', default=False,
                   help='resnet18 不加载 ImageNet 权重 (调试用)')
    p.add_argument('--global_dim', type=int, default=128)
    p.add_argument('--num_joints', type=int, default=17)
    p.add_argument('--lambda1', type=float, default=1.0)
    p.add_argument('--lambda2', type=float, default=0.5)
    p.add_argument('--lambda3', type=float, default=2.0)
    p.add_argument('--lambda_hip', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=4,
                   help='RGB ResNet18 比深度 CNN 重, 16G 上建议 4 (配 accum 4 等效 16)')
    p.add_argument('--accumulate_grad', '--accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-4, help='proj/时序/pose head 学习率')
    p.add_argument('--lr_backbone', type=float, default=1e-4, help='预训练 ResNet 主干学习率')
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--eval_interval', type=int, default=3)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--save_dir', type=str, default='./checkpoints/rgb_teacher')
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, evaluator, logger):
    model.eval()
    preds, gts = [], []
    for batch in loader:
        rgb = batch['rgb'].to(device)
        out = model(rgb)
        preds.append(out['p_final'].cpu()); gts.append(batch['pose_3d'])
        del out, rgb
    preds = torch.cat(preds); gts = torch.cat(gts)
    m = evaluator.evaluate(preds, gts)
    # hip 单独打出来: RGB 教师的 hip 决定蒸馏的 out_distill_hip_weight 取值
    hip = torch.norm(preds[..., 0, :] - gts[..., 0, :], dim=-1).mean().item() * 1000
    pred_std = preds.mean(dim=1).std(dim=0).mean().item() * 1000
    logger.info(f'[RGB Teacher Eval] MPJPE: {m["MPJPE (mm)"]:.2f}mm | MPJPE_a: {m["MPJPE_aligned (mm)"]:.2f}mm | '
                f'PA: {m["PA-MPJPE (mm)"]:.2f}mm | hip: {hip:.1f}mm | '
                f'P50n: {m["PCK@50_norm (%)"]:.1f}% | PredStd: {pred_std:.1f}mm')
    m['hip_error (mm)'] = hip
    return m


def build_param_groups(model, lr, lr_backbone):
    """ResNet 主干低 lr, 新增层 (proj/时序/头) 高 lr。scratch 模式下全部走 lr。"""
    bb, rest = [], []
    for name, p in model.named_parameters():
        if name.startswith('encoder.enc.features'):
            bb.append(p)
        else:
            rest.append(p)
    groups = [{'params': rest, 'lr': lr, 'name': 'heads'}]
    if bb:
        groups.append({'params': bb, 'lr': lr_backbone, 'name': 'resnet_backbone'})
    return groups


def main():
    args = get_args()
    set_seed(args.seed)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_run_config(args, args.save_dir, extra={"script": "train_rgb_teacher"})
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('RGBTeacher', os.path.join(args.save_dir, 'train.log'))
    logger.info(f'Step A (RGB): rgb pose teacher, train={args.train_envs} eval={args.test_env}')
    logger.info(f'rgb_img={args.rgb_img} backbone={args.backbone} '
                f'pretrained={not args.no_pretrained} lr={args.lr} lr_backbone={args.lr_backbone}')

    data_exists = os.path.exists(args.data_root)
    if not data_exists:
        logger.warning('data_root missing -> synthetic smoke data')
    train_loader, test_loader = build_teacher_dataloaders(args, synthetic=not data_exists,
                                                          modality='rgb')
    logger.info(f'Train batches: {len(train_loader)}, Test batches: {len(test_loader)}')

    model = RGBPoseTeacher(global_dim=args.global_dim, num_joints=args.num_joints,
                           seq_len=args.seq_len, backbone=args.backbone,
                           pretrained=not args.no_pretrained).to(device)
    logger.info(f'Teacher params: {count_parameters(model):,}')

    pose_loss_fn = PoseLoss(args.lambda1, args.lambda2, args.lambda3, args.lambda_hip)
    evaluator = PoseEvaluator(unit='meter')
    optimizer = AdamW(build_param_groups(model, args.lr, args.lr_backbone),
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    timer = Timer(); timer.start()
    best_mpjpe = float('inf'); patience = 0
    accum = args.accumulate_grad

    for epoch in range(1, args.epochs + 1):
        model.train()
        meter = AverageMeter(); optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            rgb = batch['rgb'].to(device); pose = batch['pose_3d'].to(device)
            out = model(rgb)
            loss, _ = pose_loss_fn(out['p_final'], pose)
            (loss / accum).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step(); optimizer.zero_grad()
            meter.update(loss.item(), rgb.size(0))
            if (i + 1) % args.log_interval == 0:
                logger.info(f'Epoch [{epoch}] Batch [{i+1}/{len(train_loader)}] Loss: {meter.avg:.4f}')
            del out, loss, rgb, pose
        logger.info(f'[Train] Epoch {epoch} | Loss: {meter.avg:.4f} | Time: {timer.elapsed_str()}')
        scheduler.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            m = evaluate(model, test_loader, device, evaluator, logger)
            cur = m['MPJPE (mm)']
            if cur < best_mpjpe:
                best_mpjpe = cur; patience = 0
                torch.save({'epoch': epoch, 'metrics': m,
                            'backbone': args.backbone,           # FrozenTeacher 重建模型要用
                            'model_state_dict': model.state_dict(),
                            'encoder': model.encoder.state_dict(),
                            'global_modeler': model.global_modeler.state_dict()},
                           os.path.join(args.save_dir, 'teacher_best.pt'))
                logger.info(f'*** New best RGB teacher MPJPE: {best_mpjpe:.2f}mm ***')
            else:
                patience += 1
                logger.info(f'No improvement. Patience: {patience}/{args.patience}')
            if patience >= args.patience:
                logger.info(f'Early stopping at epoch {epoch}'); break

    logger.info(f'\nStep A (RGB) done. Best RGB teacher MPJPE: {best_mpjpe:.2f}mm | Time: {timer.elapsed_str()}')
    logger.info('判读: 看 MPJPE_aligned/PA 是否优于深度教师 (结构价值); hip 预计差于深度教师 ~236mm,')
    logger.info('      蒸馏时把 --out_distill_hip_weight 降到 1.0 (教师 hip 不可信, 别放大噪声)。')


if __name__ == '__main__':
    main()