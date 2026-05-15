"""
CSI-RSC-PoseDG: 训练脚本 v5 — 严格域泛化
  - 评估时完全不使用测试集 GT 动作标签
  - 模型必须自行从 CSI 预测动作 (action_idx=None)
  - Best model 按 Predicted 模式 MPJPE 选择
"""
import os
import sys
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
from utils import (
    set_seed, setup_logger, count_parameters,
    save_checkpoint, AverageMeter, Timer
)


def action_to_index(action_str):
    return int(action_str[1:]) - 1


def train_one_epoch(model, train_loader, optimizer, loss_fn, pose_loss_fn,
                    device, epoch, logger, args):
    model.train()
    meters = {
        'loss': AverageMeter(),
        'l_pose_clean': AverageMeter(),
        'l_pose_masked': AverageMeter(),
        'l_cons': AverageMeter(),
        'l_action': AverageMeter(),
    }
    accum_steps = getattr(args, 'accumulate_grad', 1)
    action_loss_fn = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        csi = batch['csi'].to(device)
        pose_3d = batch['pose_3d'].to(device)
        # 训练时使用 GT 动作标签 (仅源域数据)
        action_labels = torch.tensor(
            [action_to_index(a) for a in batch['action']],
            dtype=torch.long, device=device
        )

        outputs = model.forward_rsc(
            csi, pose_3d,
            loss_fn=lambda pred, gt: pose_loss_fn(pred, gt)[0],
            action_idx=action_labels,
        )

        action_loss = action_loss_fn(outputs['action_logits'], action_labels)
        total_loss, loss_dict = loss_fn(
            outputs, pose_3d, training=True, action_loss=action_loss
        )
        total_loss = total_loss / accum_steps
        total_loss.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        B = csi.shape[0]
        meters['loss'].update(loss_dict['l_total'], B)
        meters['l_pose_clean'].update(loss_dict.get('l_pose_clean', 0), B)
        meters['l_pose_masked'].update(loss_dict.get('l_pose_masked', 0), B)
        meters['l_cons'].update(loss_dict.get('l_cons', 0), B)
        meters['l_action'].update(loss_dict.get('l_action', 0), B)

        if (batch_idx + 1) % args.log_interval == 0:
            logger.info(
                f'Epoch [{epoch}] Batch [{batch_idx+1}/{len(train_loader)}] '
                f'Loss: {meters["loss"].avg:.4f} '
                f'Pose(C): {meters["l_pose_clean"].avg:.4f} '
                f'Pose(M): {meters["l_pose_masked"].avg:.4f} '
                f'Cons: {meters["l_cons"].avg:.4f} '
                f'Act: {meters["l_action"].avg:.4f}'
            )

        del outputs, total_loss, csi, pose_3d
        torch.cuda.empty_cache()

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def evaluate(model, test_loader, device, evaluator, logger):
    """严格域泛化评估: 不使用测试集任何 GT 标签作为模型输入.

    模型只接收 CSI 输入, 自行预测动作和姿态.
    GT 仅用于计算评估指标 (MPJPE 等), 不参与推理.
    """
    model.eval()
    all_preds, all_gts = [], []
    action_correct, action_total = 0, 0

    for batch in test_loader:
        csi = batch['csi'].to(device)
        pose_3d = batch['pose_3d'].to(device)

        # ★ 严格 DG: action_idx=None, 模型自行预测动作
        outputs = model(csi, action_idx=None)
        all_preds.append(outputs['p_final'].cpu())
        all_gts.append(pose_3d.cpu())

        # 动作分类准确率 (GT 仅用于计算指标, 不作为模型输入)
        action_labels = torch.tensor(
            [action_to_index(a) for a in batch['action']],
            dtype=torch.long, device=device
        )
        action_pred = outputs['action_logits'].argmax(dim=-1)
        action_correct += (action_pred == action_labels).sum().item()
        action_total += action_labels.shape[0]

        del outputs, csi, pose_3d
        torch.cuda.empty_cache()

    preds = torch.cat(all_preds)
    gts = torch.cat(all_gts)

    metrics = evaluator.evaluate(preds, gts)
    pred_std = preds.mean(dim=1).std(dim=0).mean().item() * 1000
    action_acc = 100.0 * action_correct / max(action_total, 1)

    logger.info(
    f'[Eval] '
    f'MPJPE: {metrics["MPJPE (mm)"]:.2f}mm | '
    f'MPJPE_a: {metrics["MPJPE_aligned (mm)"]:.2f}mm | '
    f'PA: {metrics["PA-MPJPE (mm)"]:.2f}mm | '
    f'P50n: {metrics["PCK@50_norm (%)"]:.1f}% | '
    f'P20n: {metrics["PCK@20_norm (%)"]:.1f}% | '
    f'PredStd: {pred_std:.1f}mm | '
    f'ActAcc: {action_acc:.1f}%'
)

    metrics['pred_std'] = pred_std
    metrics['action_acc'] = action_acc
    return metrics


def main():
    args = get_config()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger(
        'CSI-RSC-PoseDG',
        log_file=os.path.join(args.save_dir, 'train.log')
    )

    logger.info(f'Configuration: {vars(args)}')
    logger.info(f'Device: {device}')
    logger.info(f'Strict DG: test-time action_idx=None (no GT labels)')

    data_exists = os.path.exists(args.data_root)
    train_loader, test_loader = build_dataloaders(args, synthetic=not data_exists)
    logger.info(f'Train batches: {len(train_loader)}, Test batches: {len(test_loader)}')

    model = CSIRSCPoseDG(args).to(device)
    logger.info(f'Model parameters: {count_parameters(model):,}')

    loss_fn = TotalLoss(
        lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
        alpha=args.alpha, beta=args.beta,
        gamma=args.gamma, delta=args.delta,
        lambda_hip=getattr(args, 'lambda_hip', 1.0),
    )
    pose_loss_fn = PoseLoss(lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
                             lambda_hip=getattr(args, 'lambda_hip', 1.0))
    evaluator = PoseEvaluator(unit='meter')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    timer = Timer()
    timer.start()
    best_mpjpe = float('inf')
    patience_counter = 0
    patience = getattr(args, 'patience', 15)
    start_epoch = 1

    # ============================================================
    # Resume from checkpoint if specified
    # ============================================================
    if getattr(args, 'resume', ''):
        if not os.path.exists(args.resume):
            logger.error(f'Resume checkpoint not found: {args.resume}')
            raise FileNotFoundError(args.resume)
        logger.info(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception as e:
            logger.warning(f'Could not restore optimizer state: {e}')
        start_epoch = ckpt.get('epoch', 0) + 1
        if 'metrics' in ckpt and ckpt['metrics']:
            best_mpjpe = ckpt['metrics'].get('MPJPE (mm)', float('inf'))
            logger.info(f'  Loaded best_mpjpe = {best_mpjpe:.2f}mm from checkpoint metrics')
        # Step scheduler forward to current epoch
        for _ in range(start_epoch - 1):
            scheduler.step()
        logger.info(f'  Resuming at epoch {start_epoch}, LR = {scheduler.get_last_lr()[0]:.6f}')

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f'\n{"="*60}')
        logger.info(f'Epoch {epoch}/{args.epochs} | LR: {scheduler.get_last_lr()[0]:.6f}')

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, pose_loss_fn,
            device, epoch, logger, args
        )
        logger.info(
            f'[Train] Epoch {epoch} | Loss: {train_metrics["loss"]:.4f} '
            f'Act: {train_metrics["l_action"]:.4f} | '
            f'Time: {timer.elapsed_str()}'
        )

        scheduler.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            eval_metrics = evaluate(
                model, test_loader, device, evaluator, logger
            )
            current_mpjpe = eval_metrics['MPJPE (mm)']
            if current_mpjpe < best_mpjpe:
                best_mpjpe = current_mpjpe
                patience_counter = 0
                save_checkpoint(
                    model, optimizer, epoch, eval_metrics,
                    os.path.join(args.save_dir, 'best_model.pth')
                )
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

    logger.info(f'\n{"="*60}')
    logger.info(f'Training complete! Best MPJPE: {best_mpjpe:.2f}mm')
    logger.info(f'Total time: {timer.elapsed_str()}')


if __name__ == '__main__':
    main()