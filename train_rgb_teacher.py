"""
Step A (RGB, DG 强化版): train the RGB -> 3D pose teacher (源 E01-E03, 在 E04 选点)。
drop-in 替换原 train_rgb_teacher.py。

相对原版的改动 (目标: 跨环境 DG + 把 MPJPE 顶上去/稳住), 接口与 ckpt 字段保持兼容:
  [DG]
  1. RGB 序列光度增强 (rgb_augment.RGBSeqAugmentor): 训练循环里对 batch['rgb'] 施加,
     打散源域房间外观捷径; eval 不增强。 --rgb_aug/--no_rgb_aug --aug_p
  2. 骨干换 DG 版 (models.rgb_teacher 已插 MixStyle + 可冻结浅层 + dropout):
     --dg_mixstyle/--no_mixstyle --mixstyle_p --freeze_stages --backbone_dropout
  [稳 MPJPE / 提结构]
  3. 结构正则 (structural_losses.structural_loss): 骨长/对称/时序/root-relative,
     压 MPJPE_aligned 与 PA, 平移不变, 不伤绝对 hip。 --w_bone/--w_sym/--w_temp/--w_rel
  4. EMA: 平滑权重, 抑制评测抖动。 --use_ema/--no_ema --ema_decay
  [选点]
  5. 仍按【E04 测试集】选 teacher_best (主流做法, 与你其余流程口径一致)。
     选点指标默认 MPJPE, 可 --select_metric 切 PA-MPJPE / MPJPE_aligned。

依赖: torchvision (配 torch)。装不上 -> --backbone scratch。
"""
import os
import sys
import copy
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
from structural_losses import structural_loss
from evaluate import PoseEvaluator
from rgb_augment import RGBSeqAugmentor
from utils import (set_seed, setup_logger, count_parameters,
                   AverageMeter, Timer, save_run_config)


# ----------------------------------------------------------------------
# EMA (与 train_distill_pretrained.py 同款)
# ----------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.999, warmup=True):
        self.decay = decay; self.warmup = warmup; self.step = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = (min(self.decay, (1 + self.step) / (10 + self.step)) if self.warmup else self.decay)
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if torch.is_floating_point(v):
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return self.shadow


def get_args():
    p = argparse.ArgumentParser(description='Step A (RGB, DG): rgb pose teacher, E04 选点')
    p.add_argument('--data_root', type=str, default='/home/a123456/PerceptAlign/MMFi')
    p.add_argument('--train_envs', nargs='+', default=['E01', 'E02', 'E03'])
    p.add_argument('--test_env', type=str, default='E04')
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--rgb_img', type=int, default=112)
    p.add_argument('--rgb_root', type=str, default=None,
                   help='RGB 数据独立根目录; GT/CSI 仍从 --data_root 读。None=与 data_root 相同')
    # depth_img/clip 仅为 build_teacher_dataloaders 接口兼容, RGB 模态不使用
    p.add_argument('--depth_img', type=int, default=112)
    p.add_argument('--depth_clip', type=float, default=5000.0)
    p.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'scratch'])
    p.add_argument('--no_pretrained', action='store_true', default=False,
                   help='resnet18 不加载 ImageNet 权重 (调试用)')
    p.add_argument('--global_dim', type=int, default=128)
    p.add_argument('--num_joints', type=int, default=17)
    # ---- pose loss ----
    p.add_argument('--lambda1', type=float, default=1.0)
    p.add_argument('--lambda2', type=float, default=0.5)
    p.add_argument('--lambda3', type=float, default=2.0)
    p.add_argument('--lambda_hip', type=float, default=1.0,
                   help='教师绝对 hip 监督。想压 MPJPE 保留 1.0; 只要结构可设 0')
    # ---- DG: 增强 ----
    p.add_argument('--rgb_aug', dest='rgb_aug', action='store_true', default=True)
    p.add_argument('--no_rgb_aug', dest='rgb_aug', action='store_false')
    p.add_argument('--aug_p', type=float, default=0.9)
    # ---- DG: 骨干 ----
    p.add_argument('--dg_mixstyle', dest='dg_mixstyle', action='store_true', default=True)
    p.add_argument('--no_mixstyle', dest='dg_mixstyle', action='store_false')
    p.add_argument('--mixstyle_p', type=float, default=0.5)
    p.add_argument('--mixstyle_alpha', type=float, default=0.3)
    p.add_argument('--freeze_stages', type=int, default=2,
                   help='冻结 ResNet 浅层: 0=不冻, 1=stem, 2=+layer1, 3=+layer2, 4=+layer3')
    p.add_argument('--backbone_dropout', type=float, default=0.1)
    # ---- 结构正则 (压 MPJPE_aligned / PA) ----
    p.add_argument('--w_bone', type=float, default=1.0)
    p.add_argument('--w_sym',  type=float, default=0.1)
    p.add_argument('--w_temp', type=float, default=0.1)
    p.add_argument('--w_rel',  type=float, default=2.0)
    # ---- EMA ----
    p.add_argument('--use_ema', dest='use_ema', action='store_true', default=True)
    p.add_argument('--no_ema', dest='use_ema', action='store_false')
    p.add_argument('--ema_decay', type=float, default=0.999)
    # ---- 选点 (在 E04) ----
    p.add_argument('--select_metric', type=str, default='MPJPE',
                   choices=['MPJPE', 'PA-MPJPE', 'MPJPE_aligned'])
    # ---- 优化 ----
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--resume', type=str, default='',
                   help='从 teacher_last.pt 恢复 (恢复 model/optimizer/scheduler/ema/epoch/best/patience)')
    p.add_argument('--batch_size', type=int, default=4,
                   help='RGB ResNet18 比深度 CNN 重, 16G 上建议 4 (配 accum 4 等效 16)')
    p.add_argument('--accumulate_grad', '--accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-4, help='proj/时序/pose head 学习率')
    p.add_argument('--lr_backbone', type=float, default=1e-4, help='预训练 ResNet 主干学习率')
    p.add_argument('--weight_decay', type=float, default=5e-4,
                   help='骨干过拟合 -> 比原 1e-4 略大')
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--eval_interval', type=int, default=3)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--save_dir', type=str, default='./checkpoints/rgb_teacher_dg')
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, evaluator, logger, tag='E04'):
    model.eval()
    preds, gts = [], []
    for batch in loader:
        rgb = batch['rgb'].to(device)
        out = model(rgb)
        preds.append(out['p_final'].cpu()); gts.append(batch['pose_3d'])
        del out, rgb
    preds = torch.cat(preds); gts = torch.cat(gts)
    m = evaluator.evaluate(preds, gts)
    hip = torch.norm(preds[..., 0, :] - gts[..., 0, :], dim=-1).mean().item() * 1000
    pred_std = preds.mean(dim=1).std(dim=0).mean().item() * 1000
    m['hip_error (mm)'] = hip
    logger.info(f'[{tag} Teacher Eval] MPJPE: {m["MPJPE (mm)"]:.2f}mm | MPJPE_a: {m["MPJPE_aligned (mm)"]:.2f}mm | '
                f'PA: {m["PA-MPJPE (mm)"]:.2f}mm | hip: {hip:.1f}mm | '
                f'P50n: {m["PCK@50_norm (%)"]:.1f}% | PredStd: {pred_std:.1f}mm')
    return m


def build_param_groups(model, lr, lr_backbone):
    """ResNet 主干 (encoder.enc, 非 proj) 走低 lr; 其余走高 lr。已冻结参数自动排除。"""
    bb, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('encoder.enc') and 'proj' not in name:
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
    save_run_config(args, args.save_dir, extra={"script": "train_rgb_teacher_dg"})
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('RGBTeacherDG', os.path.join(args.save_dir, 'train.log'))
    logger.info('=' * 70)
    logger.info(f'Step A (RGB, DG): train={args.train_envs} eval/select={args.test_env}')
    logger.info(f'  backbone={args.backbone} pretrained={not args.no_pretrained} '
                f'mixstyle={args.dg_mixstyle}(p={args.mixstyle_p}) freeze_stages={args.freeze_stages} '
                f'bb_drop={args.backbone_dropout}')
    logger.info(f'  rgb_aug={args.rgb_aug}(p={args.aug_p}) wd={args.weight_decay} '
                f'struct[bone={args.w_bone} sym={args.w_sym} temp={args.w_temp} rel={args.w_rel}] '
                f'lambda_hip={args.lambda_hip}')
    logger.info(f'  EMA={args.use_ema}(decay={args.ema_decay}) select=E04 by {args.select_metric}')

    data_exists = os.path.exists(args.data_root)
    if not data_exists:
        logger.warning('data_root missing -> synthetic smoke data')
    train_loader, test_loader = build_teacher_dataloaders(args, synthetic=not data_exists,
                                                          modality='rgb')
    logger.info(f'Train batches: {len(train_loader)}, Test batches: {len(test_loader)}')

    model = RGBPoseTeacher(global_dim=args.global_dim, num_joints=args.num_joints,
                           seq_len=args.seq_len, backbone=args.backbone,
                           pretrained=not args.no_pretrained,
                           dg_mixstyle=args.dg_mixstyle, mixstyle_p=args.mixstyle_p,
                           mixstyle_alpha=args.mixstyle_alpha, freeze_stages=args.freeze_stages,
                           backbone_dropout=args.backbone_dropout).to(device)
    logger.info(f'Teacher params: {count_parameters(model):,} (trainable)')

    augmentor = RGBSeqAugmentor(p=args.aug_p).to(device).train() if args.rgb_aug else None
    pose_loss_fn = PoseLoss(args.lambda1, args.lambda2, args.lambda3, args.lambda_hip)
    evaluator = PoseEvaluator(unit='meter')
    optimizer = AdamW(build_param_groups(model, args.lr, args.lr_backbone),
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None

    timer = Timer(); timer.start()
    best_metric = float('inf'); patience = 0
    accum = args.accumulate_grad
    mkey = f'{args.select_metric} (mm)'
    start_epoch = 1

    # ---- 中断恢复 ----
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model_state_dict'], strict=True)   # 在线权重(非EMA)
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        if ema is not None and ck.get('ema_shadow') is not None:
            ema.shadow = {k: v.to(device) for k, v in ck['ema_shadow'].items()}
            ema.step = ck.get('ema_step', 0)
        best_metric = ck.get('best_metric', float('inf'))
        patience = ck.get('patience', 0)
        start_epoch = ck.get('epoch', 0) + 1
        logger.info(f'[resume] 从 {args.resume} 恢复 -> 续训 epoch {start_epoch}, '
                    f'best={best_metric:.2f}mm, patience={patience}')

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        meter = AverageMeter(); meter_s = AverageMeter(); optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            rgb = batch['rgb'].to(device)
            if augmentor is not None:
                rgb = augmentor(rgb)                       # DG: 序列一致光度增强
            pose = batch['pose_3d'].to(device)
            out = model(rgb)
            l_pose, _ = pose_loss_fn(out['p_final'], pose)
            l_struct, _ = structural_loss(out['p_final'], pose,
                                          w_bone=args.w_bone, w_sym=args.w_sym,
                                          w_temp=args.w_temp, w_rel=args.w_rel)
            loss = l_pose + l_struct
            (loss / accum).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step(); optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)
            meter.update(loss.item(), rgb.size(0))
            meter_s.update(float(l_struct.detach()), rgb.size(0))
            if (i + 1) % args.log_interval == 0:
                logger.info(f'Epoch [{epoch}] Batch [{i+1}/{len(train_loader)}] '
                            f'Loss: {meter.avg:.4f} (struct {meter_s.avg:.4f})')
            del out, loss, rgb, pose
        logger.info(f'[Train] Epoch {epoch} | Loss: {meter.avg:.4f} | Time: {timer.elapsed_str()}')
        scheduler.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            # 用 EMA 权重在 E04 评测/选点 (若开 EMA)
            if ema is not None:
                backup = copy.deepcopy(model.state_dict())
                ema.copy_to(model)
                m = evaluate(model, test_loader, device, evaluator, logger, tag='E04(ema)')
            else:
                m = evaluate(model, test_loader, device, evaluator, logger, tag='E04')

            cur = m[mkey]
            if cur < best_metric:
                best_metric = cur; patience = 0
                torch.save({'epoch': epoch, 'metrics': m,
                            'backbone': args.backbone,           # FrozenTeacher 重建要用
                            'model_state_dict': (ema.state_dict() if ema is not None else model.state_dict()),
                            'encoder': model.encoder.state_dict(),
                            'global_modeler': model.global_modeler.state_dict()},
                           os.path.join(args.save_dir, 'teacher_best.pt'))
                logger.info(f'*** New best RGB teacher ({args.select_metric}={best_metric:.2f}mm @E04) '
                            f'-> teacher_best.pt ***')
            else:
                patience += 1
                logger.info(f'No improvement. Patience: {patience}/{args.patience}')

            if ema is not None:
                model.load_state_dict(backup, strict=True)   # 还原在线权重继续训练

        # ---- 每个 epoch 存完整训练状态 (best/patience 已是最新), 任何中断都可 --resume ----
        torch.save({'epoch': epoch, 'best_metric': best_metric, 'patience': patience,
                    'backbone': args.backbone,
                    'model_state_dict': model.state_dict(),         # 在线权重(非EMA)
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'ema_shadow': (ema.state_dict() if ema is not None else None),
                    'ema_step': (ema.step if ema is not None else 0)},
                   os.path.join(args.save_dir, 'teacher_last.pt'))

        if patience >= args.patience:
            logger.info(f'Early stopping at epoch {epoch}'); break

    logger.info(f'\nStep A (RGB, DG) done. Best {args.select_metric}@E04: {best_metric:.2f}mm | '
                f'Time: {timer.elapsed_str()}')
    logger.info('判读: E04 的 MPJPE 应不再随训练单调上涨(过拟合被按住), 收敛点可低于原 epoch3 的 293mm。')
    logger.info('蒸馏 (Step B) 仍用 train_distill_pretrained.py --teacher_modality rgb 加载本 teacher_best.pt。')


if __name__ == '__main__':
    main()