"""
CSI-PoseDG: 标准训练脚本 v3 — 严格评估 (无 Oracle)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config_standard import get_config
from dataset_standard import build_dataloaders
from models import CSIRSCPoseDG
from losses import PoseLoss
from evaluate import PoseEvaluator
from utils import (
    set_seed, setup_logger, count_parameters,
    save_checkpoint, AverageMeter, Timer
)


def action_to_index(action_str):
    return int(action_str[1:]) - 1


def train_one_epoch(model, train_loader, optimizer, pose_loss_fn,
                    device, epoch, logger, args):
    model.train()
    meters = {'loss': AverageMeter(), 'l_action': AverageMeter()}
    accum_steps = getattr(args, 'accumulate_grad', 1)
    action_loss_fn = nn.CrossEntropyLoss()
    delta = getattr(args, 'delta', 0.5)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        csi = batch['csi'].to(device)
        pose_3d = batch['pose_3d'].to(device)
        action_labels = torch.tensor(
            [action_to_index(a) for a in batch['action']],
            dtype=torch.long, device=device
        )

        # 训练时用 GT 动作 (源域/训练集, 合规)
        outputs = model(csi, action_idx=action_labels)
        pose_loss, _ = pose_loss_fn(outputs['p_final'], pose_3d)
        action_loss = action_loss_fn(outputs['action_logits'], action_labels)
        total_loss = pose_loss + delta * action_loss
        total_loss = total_loss / accum_steps
        total_loss.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        B = csi.shape[0]
        meters['loss'].update(total_loss.item() * accum_steps, B)
        meters['l_action'].update(action_loss.item(), B)

        if (batch_idx + 1) % args.log_interval == 0:
            logger.info(
                f'Epoch [{epoch}] Batch [{batch_idx+1}/{len(train_loader)}] '
                f'Loss: {meters["loss"].avg:.4f} '
                f'Act: {meters["l_action"].avg:.4f}'
            )

        del outputs, total_loss, csi, pose_3d
        torch.cuda.empty_cache()

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def evaluate(model, test_loader, device, evaluator, logger):
    """严格评估: action_idx=None, 不使用测试集 GT 动作标签."""
    model.eval()
    all_preds, all_gts, all_envs = [], [], []
    action_correct, action_total = 0, 0

    for batch in test_loader:
        csi = batch['csi'].to(device)
        pose_3d = batch['pose_3d'].to(device)

        # ★ 不透露测试集动作标签
        out = model(csi, action_idx=None)
        all_preds.append(out['p_final'].cpu())
        all_gts.append(pose_3d.cpu())
        all_envs.extend(batch['env'])

        # 动作准确率 (仅指标, 不作为输入)
        action_labels = torch.tensor(
            [action_to_index(a) for a in batch['action']],
            dtype=torch.long, device=device
        )
        action_pred = out['action_logits'].argmax(dim=-1)
        action_correct += (action_pred == action_labels).sum().item()
        action_total += action_labels.shape[0]

        del out, csi, pose_3d
        torch.cuda.empty_cache()

    preds = torch.cat(all_preds)
    gts = torch.cat(all_gts)

    metrics = evaluator.evaluate(preds, gts)
    pred_std = preds.mean(dim=1).std(dim=0).mean().item() * 1000
    action_acc = 100.0 * action_correct / max(action_total, 1)

    logger.info(
        f'[Eval] MPJPE: {metrics["MPJPE (mm)"]:.2f}mm | '
        f'PA: {metrics["PA-MPJPE (mm)"]:.2f}mm | '
        f'P50: {metrics["PCK@50_norm (%)"]:.1f}% | '
        f'P20: {metrics["PCK@20_norm (%)"]:.1f}% | '
        f'PredStd: {pred_std:.1f}mm | '
        f'ActAcc: {action_acc:.1f}%'
    )

    # Per-env breakdown
    env_set = sorted(set(all_envs))
    if len(env_set) > 1:
        for env in env_set:
            mask = torch.tensor([i for i, e in enumerate(all_envs) if e == env])
            em = evaluator.evaluate(preds[mask], gts[mask])
            logger.info(f'  [{env}] MPJPE: {em["MPJPE (mm)"]:.2f}mm | n={len(mask)}')

    metrics['pred_std'] = pred_std
    metrics['action_acc'] = action_acc
    return metrics


def main():
    args = get_config()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('CSI-Standard',
                          log_file=os.path.join(args.save_dir, 'train.log'))

    logger.info(f'STANDARD 8:2 SPLIT — Strict eval (no Oracle)')
    logger.info(f'Configuration: {vars(args)}')

    train_loader, test_loader = build_dataloaders(args)
    logger.info(f'Train: {len(train_loader)}, Test: {len(test_loader)}')

    model = CSIRSCPoseDG(args).to(device)
    logger.info(f'Parameters: {count_parameters(model):,}')

    pose_loss_fn = PoseLoss(lambda1=args.lambda1, lambda2=args.lambda2, lambda3=getattr(args, "lambda3", 2.0))
    evaluator = PoseEvaluator(unit='meter')
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    timer = Timer()
    timer.start()
    best_mpjpe = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        logger.info(f'\n{"="*60}')
        logger.info(f'Epoch {epoch}/{args.epochs} | LR: {scheduler.get_last_lr()[0]:.6f}')

        tm = train_one_epoch(model, train_loader, optimizer, pose_loss_fn,
                             device, epoch, logger, args)
        logger.info(f'[Train] Epoch {epoch} | Loss: {tm["loss"]:.4f} '
                    f'Act: {tm["l_action"]:.4f} | Time: {timer.elapsed_str()}')
        scheduler.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            em = evaluate(model, test_loader, device, evaluator, logger)
            cur = em['MPJPE (mm)']
            if cur < best_mpjpe:
                best_mpjpe = cur
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch, em,
                                os.path.join(args.save_dir, 'best_model.pth'))
                logger.info(f'*** New best MPJPE: {best_mpjpe:.2f}mm ***')
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break

    logger.info(f'\nTraining complete! Best MPJPE: {best_mpjpe:.2f}mm')
    logger.info(f'Total time: {timer.elapsed_str()}')


if __name__ == '__main__':
    main()