"""
Step B: depth -> CSI cross-modal distillation (DIAGNOSTIC, single-stage).

The CSI student (CSIRSCPoseDG, native CSI front-end) is trained on E01-E03 with
its usual TotalLoss, PLUS a feature-level distillation term that aligns the
student's z_global to a FROZEN depth teacher's z_global (per frame, B,T,128).

  L = TotalLoss(student)  +  lambda_distill * FeatureDistillLoss(proj(z_s), z_t.detach())

Strict DG: test is CSI-only (action_idx=None). Depth is used ONLY during training
to produce the teacher target. The target env's depth is NEVER used.

This is INDEPENDENT of train.py / losses.py — it imports them read-only and adds
the distillation path on top, so the other (archived) methods are untouched.

Usage:
    python train_distill.py \
        --data_root /home/a123456/PerceptAlign/MMFi \
        --train_envs E01 E02 E03 --test_env E04 \
        --teacher_ckpt ./checkpoints/depth_teacher_full/teacher_best.pt \
        --depth_img 112 --depth_clip 5000 \
        --lambda_distill 0.1 \
        --epochs 50 --batch_size 8 --accumulate_grad 2 \
        --lr 1e-3 --num_workers 12 \
        --save_dir ./checkpoints/distill_lambda0.1
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import CSIRSCPoseDG
from models.global_encoder import GlobalTemporalModeler
from models.depth_teacher import DepthEncoder
from losses import TotalLoss, PoseLoss
from evaluate import PoseEvaluator
from dataset_distill import MMFiDistillDataset
from distill_loss import DistillProjection, FeatureDistillLoss
from utils import (set_seed, setup_logger, count_parameters,
                   save_checkpoint, AverageMeter, Timer, save_run_config)


def action_to_index(a):
    return int(a[1:]) - 1


# ----------------------------------------------------------------------
# Frozen depth teacher: reproduces z_global from depth, no gradient.
# ----------------------------------------------------------------------
class FrozenDepthTeacher(nn.Module):
    """Loads encoder + global_modeler from teacher_best.pt, frozen, eval-only.

    forward(depth) -> z_global (B,T,128), detached.
    """
    def __init__(self, ckpt_path, global_dim=128, seq_len=64,
                 num_transformer_layers=3, num_heads=4,
                 tcn_channels=(128, 128), tcn_kernel_size=3, device='cuda'):
        super().__init__()
        self.encoder = DepthEncoder(out_dim=global_dim)
        self.global_modeler = GlobalTemporalModeler(
            in_dim=global_dim, global_dim=global_dim,
            num_transformer_layers=num_transformer_layers, num_heads=num_heads,
            tcn_channels=list(tcn_channels), tcn_kernel_size=tcn_kernel_size,
            dropout=0.1, max_seq_len=seq_len + 50)

        ckpt = torch.load(ckpt_path, map_location=device)
        if 'encoder' not in ckpt or 'global_modeler' not in ckpt:
            raise KeyError(
                f"teacher ckpt missing encoder/global_modeler keys; got {list(ckpt.keys())}")
        me, ue = self.encoder.load_state_dict(ckpt['encoder'], strict=False)
        mg, ug = self.global_modeler.load_state_dict(ckpt['global_modeler'], strict=False)
        if me or ue or mg or ug:
            raise RuntimeError(
                f"teacher load mismatch: enc(missing={len(me)},unexpected={len(ue)}) "
                f"gm(missing={len(mg)},unexpected={len(ug)})")

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, depth):
        return self.global_modeler(self.encoder(depth)).detach()


def build_distill_loaders(args):
    """Train: E01-E03 with BOTH depth+csi (aligned). Test: E04 CSI-only (no depth)."""
    train_ds = MMFiDistillDataset(
        args.data_root, args.train_envs, args.seq_len, stride=32,
        with_depth=True, with_csi=True,
        depth_img=args.depth_img, depth_clip=args.depth_clip,
        csi_augment=True)
    # Test: student is CSI-only. with_depth=False so no target-env depth is touched.
    test_ds = MMFiDistillDataset(
        args.data_root, [args.test_env], args.seq_len, stride=args.seq_len,
        with_depth=False, with_csi=True,
        depth_img=args.depth_img, depth_clip=args.depth_clip,
        csi_augment=False)
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
    vl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    return tl, vl


def train_one_epoch(student, proj, teacher, loader, optimizer,
                    total_loss_fn, pose_loss_fn, distill_loss_fn,
                    device, epoch, logger, args):
    student.train(); proj.train()
    meters = {k: AverageMeter() for k in
              ['loss', 'l_pose_clean', 'l_cons', 'l_action', 'l_distill']}
    accum = getattr(args, 'accumulate_grad', 1)
    action_loss_fn = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        csi = batch['csi'].to(device)
        depth = batch['depth'].to(device)
        pose_3d = batch['pose_3d'].to(device)
        action_labels = torch.tensor(
            [action_to_index(a) for a in batch['action']],
            dtype=torch.long, device=device)

        # Student forward (RSC path) — identical to train.py's usage
        outputs = student.forward_rsc(
            csi, pose_3d,
            loss_fn=lambda p, g: pose_loss_fn(p, g)[0],
            action_idx=action_labels)

        action_loss = action_loss_fn(outputs['action_logits'], action_labels)
        base_loss, loss_dict = total_loss_fn(
            outputs, pose_3d, training=True, action_loss=action_loss)

        # Distillation: align student z_global to frozen teacher z_global
        z_student = outputs['z_global']                 # (B,T,128), grad on
        z_teacher = teacher(depth)                      # (B,T,128), detached
        z_proj = proj(z_student)
        l_distill, dd = distill_loss_fn(z_proj, z_teacher)

        total = base_loss + args.lambda_distill * l_distill
        (total / accum).backward()

        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(proj.parameters()),
                    args.grad_clip)
            optimizer.step(); optimizer.zero_grad()

        B = csi.shape[0]
        meters['loss'].update(total.item(), B)
        meters['l_pose_clean'].update(loss_dict.get('l_pose_clean', 0), B)
        meters['l_cons'].update(loss_dict.get('l_cons', 0), B)
        meters['l_action'].update(loss_dict.get('l_action', 0), B)
        meters['l_distill'].update(dd['l_distill'], B)

        if (i + 1) % args.log_interval == 0:
            logger.info(
                f'Epoch [{epoch}] Batch [{i+1}/{len(loader)}] '
                f'Loss: {meters["loss"].avg:.4f} '
                f'Pose(C): {meters["l_pose_clean"].avg:.4f} '
                f'Cons: {meters["l_cons"].avg:.4f} '
                f'Act: {meters["l_action"].avg:.4f} '
                f'Distill: {meters["l_distill"].avg:.4f}')

        del outputs, total, csi, depth, pose_3d, z_student, z_teacher, z_proj
        torch.cuda.empty_cache()

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def evaluate(student, loader, device, evaluator, logger):
    """Strict DG eval: CSI-only, action_idx=None. (Same as train.py.)"""
    student.eval()
    all_preds, all_gts = [], []
    action_correct, action_total = 0, 0
    for batch in loader:
        csi = batch['csi'].to(device)
        pose_3d = batch['pose_3d'].to(device)
        outputs = student(csi, action_idx=None)
        all_preds.append(outputs['p_final'].cpu())
        all_gts.append(pose_3d.cpu())
        action_labels = torch.tensor(
            [action_to_index(a) for a in batch['action']],
            dtype=torch.long, device=device)
        action_correct += (outputs['action_logits'].argmax(-1) == action_labels).sum().item()
        action_total += action_labels.shape[0]
        del outputs, csi, pose_3d
        torch.cuda.empty_cache()

    preds = torch.cat(all_preds); gts = torch.cat(all_gts)
    metrics = evaluator.evaluate(preds, gts)
    pred_std = preds.mean(dim=1).std(dim=0).mean().item() * 1000
    action_acc = 100.0 * action_correct / max(action_total, 1)
    logger.info(
        f'[Eval] MPJPE: {metrics["MPJPE (mm)"]:.2f}mm | '
        f'MPJPE_a: {metrics["MPJPE_aligned (mm)"]:.2f}mm | '
        f'PA: {metrics["PA-MPJPE (mm)"]:.2f}mm | '
        f'P50n: {metrics["PCK@50_norm (%)"]:.1f}% | '
        f'P20n: {metrics["PCK@20_norm (%)"]:.1f}% | '
        f'PredStd: {pred_std:.1f}mm | ActAcc: {action_acc:.1f}%')
    metrics['pred_std'] = pred_std
    metrics['action_acc'] = action_acc
    return metrics


def get_args():
    p = argparse.ArgumentParser(description='Step B: depth->CSI distillation')
    # data / envs
    p.add_argument('--data_root', type=str, default='/home/a123456/PerceptAlign/MMFi')
    p.add_argument('--train_envs', nargs='+', default=['E01', 'E02', 'E03'])
    p.add_argument('--test_env', type=str, default='E04')
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--depth_img', type=int, default=112)
    p.add_argument('--depth_clip', type=float, default=5000.0)
    # teacher
    p.add_argument('--teacher_ckpt', type=str, required=True)
    # distillation
    p.add_argument('--lambda_distill', type=float, default=0.1)
    p.add_argument('--distill_cos_w', type=float, default=1.0)
    p.add_argument('--distill_sl1_w', type=float, default=1.0)
    # student model dims (match CSIRSCPoseDG defaults / config.py)
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
    p.add_argument('--transformer_dropout', type=float, default=0.3)
    p.add_argument('--coarse_hidden_dim', type=int, default=256)
    p.add_argument('--gcn_hidden_dim', type=int, default=128)
    p.add_argument('--num_gcn_layers', type=int, default=3)
    p.add_argument('--num_joints', type=int, default=17)
    p.add_argument('--num_actions', type=int, default=27)
    p.add_argument('--rsc2_time_drop_pct', type=float, default=0.5)
    p.add_argument('--rsc2_channel_drop_pct', type=float, default=0.5)
    p.add_argument('--rsc2_batch_pct', type=float, default=0.5)
    # loss weights (match config.py)
    p.add_argument('--lambda1', type=float, default=1.0)
    p.add_argument('--lambda2', type=float, default=0.5)
    p.add_argument('--lambda3', type=float, default=2.0)
    p.add_argument('--lambda_hip', type=float, default=1.0)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--beta', type=float, default=2.0)
    p.add_argument('--gamma', type=float, default=0.0)
    p.add_argument('--delta', type=float, default=0.5)
    # training
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--accumulate_grad', '--accum', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--eval_interval', type=int, default=3)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--save_dir', type=str, default='./checkpoints/distill')
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_run_config(args, args.save_dir, extra={"script": "train_distill", "step": "B"})

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('Distill', os.path.join(args.save_dir, 'train.log'))
    logger.info(f'Step B distillation: train={args.train_envs} test={args.test_env}')
    logger.info(f'teacher={args.teacher_ckpt} lambda_distill={args.lambda_distill}')
    logger.info(f'Strict DG: test-time CSI-only, action_idx=None, no target-env depth')

    # Data
    train_loader, test_loader = build_distill_loaders(args)
    logger.info(f'Train batches: {len(train_loader)}, Test batches: {len(test_loader)}')

    # Student (native CSI front-end; NOT vision)
    args.use_vision_backbone = False
    student = CSIRSCPoseDG(args).to(device)
    proj = DistillProjection(args.global_dim, args.global_dim).to(device)
    logger.info(f'Student params: {count_parameters(student):,} | '
                f'Proj params: {count_parameters(proj):,}')

    # Frozen teacher
    teacher = FrozenDepthTeacher(
        args.teacher_ckpt, global_dim=args.global_dim, seq_len=args.seq_len,
        num_transformer_layers=args.num_transformer_layers, num_heads=args.num_heads,
        tcn_channels=tuple(args.tcn_channels), tcn_kernel_size=args.tcn_kernel_size,
        device=device).to(device)
    logger.info(f'Teacher loaded (frozen): {count_parameters(teacher):,} trainable '
                f'(should be 0)')

    # Losses
    total_loss_fn = TotalLoss(
        lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta,
        lambda_hip=args.lambda_hip)
    pose_loss_fn = PoseLoss(
        lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
        lambda_hip=args.lambda_hip)
    distill_loss_fn = FeatureDistillLoss(args.distill_cos_w, args.distill_sl1_w)
    evaluator = PoseEvaluator(unit='meter')

    # Optimizer: student + projection head (teacher excluded — it's frozen)
    optimizer = AdamW(
        list(student.parameters()) + list(proj.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    timer = Timer(); timer.start()
    best_mpjpe = float('inf'); best_pa = float('inf'); patience = 0

    for epoch in range(1, args.epochs + 1):
        logger.info(f'\n{"="*60}\nEpoch {epoch}/{args.epochs} | '
                    f'LR: {scheduler.get_last_lr()[0]:.6f}')
        tm = train_one_epoch(student, proj, teacher, train_loader, optimizer,
                             total_loss_fn, pose_loss_fn, distill_loss_fn,
                             device, epoch, logger, args)
        logger.info(f'[Train] Epoch {epoch} | Loss: {tm["loss"]:.4f} '
                    f'Distill: {tm["l_distill"]:.4f} Act: {tm["l_action"]:.4f} | '
                    f'Time: {timer.elapsed_str()}')
        scheduler.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            m = evaluate(student, test_loader, device, evaluator, logger)
            cur = m['MPJPE (mm)']
            # Track best by MPJPE (primary) but always log PA (the real target).
            if cur < best_mpjpe:
                best_mpjpe = cur; best_pa = m['PA-MPJPE (mm)']; patience = 0
                save_checkpoint(student, optimizer, epoch, m,
                                os.path.join(args.save_dir, 'best_model.pth'))
                # also persist projection head for analysis/reproducibility
                torch.save({'epoch': epoch, 'proj_state_dict': proj.state_dict()},
                           os.path.join(args.save_dir, 'proj_best.pth'))
                logger.info(f'*** New best MPJPE: {best_mpjpe:.2f}mm '
                            f'(PA: {best_pa:.2f}mm) ***')
            else:
                patience += 1
                logger.info(f'No improvement. Patience: {patience}/{args.patience}')
            if patience >= args.patience:
                logger.info(f'Early stopping at epoch {epoch}'); break

        if epoch % 10 == 0:
            save_checkpoint(student, optimizer, epoch, {},
                            os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'))

    logger.info(f'\nStep B done. Best MPJPE: {best_mpjpe:.2f}mm | PA: {best_pa:.2f}mm | '
                f'Time: {timer.elapsed_str()}')
    logger.info('Distillation target was PA-MPJPE: compare best PA vs CSI baseline ~104mm.')


if __name__ == '__main__':
    main()