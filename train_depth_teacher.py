"""
Step A: train the depth -> 3D pose teacher (source envs E01-E03).

Eval on E04 is a SANITY CHECK on teacher strength ("if depth were available at
test, how good could pose be") — it is NOT the DG result. The DG student (Step B)
uses CSI only at test. If the teacher MPJPE here is strong (clearly below the CSI
baseline's 345mm, ideally well under 200mm), the depth feature carries the global
geometry we want to distill, and Step B is worth building.

Saves teacher_best.pt with full state plus the two modules used for distillation
(encoder, global_modeler).

Usage:
    python train_depth_teacher.py \
        --data_root /home/a123456/PerceptAlign/MMFi \
        --train_envs E01 E02 E03 --test_env E04 \
        --depth_img 112 --depth_clip 5000 \
        --epochs 50 --batch_size 8 --accumulate_grad 2 --lr 5e-4 \
        --save_dir ./checkpoints/depth_teacher
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
from models.depth_teacher import DepthPoseTeacher
from losses import PoseLoss
from evaluate import PoseEvaluator
from utils import set_seed, setup_logger, count_parameters, save_checkpoint, AverageMeter, Timer
from utils import set_seed, setup_logger, count_parameters, save_checkpoint, AverageMeter, Timer, save_run_config


def get_args():
    p = argparse.ArgumentParser(description='Step A: depth pose teacher')
    p.add_argument('--data_root', type=str, default='/home/a123456/PerceptAlign/MMFi')
    p.add_argument('--train_envs', nargs='+', default=['E01', 'E02', 'E03'])
    p.add_argument('--test_env', type=str, default='E04')
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--depth_img', type=int, default=112)
    p.add_argument('--depth_clip', type=float, default=5000.0)
    p.add_argument('--global_dim', type=int, default=128)
    p.add_argument('--num_joints', type=int, default=17)
    p.add_argument('--lambda1', type=float, default=1.0)
    p.add_argument('--lambda2', type=float, default=0.5)
    p.add_argument('--lambda3', type=float, default=2.0)
    p.add_argument('--lambda_hip', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--accumulate_grad', '--accum', type=int, default=2)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--eval_interval', type=int, default=3)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--save_dir', type=str, default='./checkpoints/depth_teacher')
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, evaluator, logger):
    model.eval()
    preds, gts = [], []
    for batch in loader:
        depth = batch['depth'].to(device)
        out = model(depth)
        preds.append(out['p_final'].cpu()); gts.append(batch['pose_3d'])
        del out, depth
    preds = torch.cat(preds); gts = torch.cat(gts)
    m = evaluator.evaluate(preds, gts)
    pred_std = preds.mean(dim=1).std(dim=0).mean().item() * 1000
    logger.info(f'[Teacher Eval] MPJPE: {m["MPJPE (mm)"]:.2f}mm | MPJPE_a: {m["MPJPE_aligned (mm)"]:.2f}mm | '
                f'PA: {m["PA-MPJPE (mm)"]:.2f}mm | P50n: {m["PCK@50_norm (%)"]:.1f}% | PredStd: {pred_std:.1f}mm')
    return m


def main():
    args = get_args()
    set_seed(args.seed)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    # ... main() 里 Path(args.save_dir).mkdir(...) 之后:
    save_run_config(args, args.save_dir, extra={"script": "train_depth_teacher"})
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('DepthTeacher', os.path.join(args.save_dir, 'train.log'))
    logger.info(f'Step A: depth pose teacher, train={args.train_envs} eval={args.test_env}')
    logger.info(f'depth_img={args.depth_img} depth_clip={args.depth_clip}mm')

    data_exists = os.path.exists(args.data_root)
    if not data_exists:
        logger.warning('data_root missing -> synthetic smoke data')
    train_loader, test_loader = build_teacher_dataloaders(args, synthetic=not data_exists)
    logger.info(f'Train batches: {len(train_loader)}, Test batches: {len(test_loader)}')

    model = DepthPoseTeacher(global_dim=args.global_dim, num_joints=args.num_joints,
                             seq_len=args.seq_len).to(device)
    logger.info(f'Teacher params: {count_parameters(model):,}')

    pose_loss_fn = PoseLoss(args.lambda1, args.lambda2, args.lambda3, args.lambda_hip)
    evaluator = PoseEvaluator(unit='meter')
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    timer = Timer(); timer.start()
    best_mpjpe = float('inf'); patience = 0
    accum = args.accumulate_grad

    for epoch in range(1, args.epochs + 1):
        model.train()
        meter = AverageMeter(); optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            depth = batch['depth'].to(device); pose = batch['pose_3d'].to(device)
            out = model(depth)
            loss, _ = pose_loss_fn(out['p_final'], pose)
            (loss / accum).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step(); optimizer.zero_grad()
            meter.update(loss.item(), depth.size(0))
            if (i + 1) % args.log_interval == 0:
                logger.info(f'Epoch [{epoch}] Batch [{i+1}/{len(train_loader)}] Loss: {meter.avg:.4f}')
            del out, loss, depth, pose
        logger.info(f'[Train] Epoch {epoch} | Loss: {meter.avg:.4f} | Time: {timer.elapsed_str()}')
        scheduler.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            m = evaluate(model, test_loader, device, evaluator, logger)
            cur = m['MPJPE (mm)']
            if cur < best_mpjpe:
                best_mpjpe = cur; patience = 0
                torch.save({'epoch': epoch, 'metrics': m,
                            'model_state_dict': model.state_dict(),
                            'encoder': model.encoder.state_dict(),
                            'global_modeler': model.global_modeler.state_dict()},
                           os.path.join(args.save_dir, 'teacher_best.pt'))
                logger.info(f'*** New best teacher MPJPE: {best_mpjpe:.2f}mm ***')
            else:
                patience += 1
                logger.info(f'No improvement. Patience: {patience}/{args.patience}')
            if patience >= args.patience:
                logger.info(f'Early stopping at epoch {epoch}'); break

    logger.info(f'\nStep A done. Best teacher MPJPE: {best_mpjpe:.2f}mm | Time: {timer.elapsed_str()}')
    logger.info('If teacher MPJPE << 345mm CSI baseline, depth carries useful geometry -> build Step B.')


if __name__ == '__main__':
    main()