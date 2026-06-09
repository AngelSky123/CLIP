"""
Step B+ v4 : depth -> CSI distillation, 从 Stage1B 预训练 backbone 出发。

v4 相对 v3 的改动 (目标: 在不动 PA-MPJPE 的前提下降 MPJPE):
  - pose_decoder 现在是 PriorRootDecoder(HybridFKPoseDecoder(...)):
      结构支保持不变, 全局 root 换成 [按动作源域先验 + tanh 限幅小残差]。
      改 root 在数学上不改 PA / MPJPE_aligned (逐帧去平移), 只改 raw MPJPE 的
      hip_error 项, 且在 E04 被先验+8cm 上界约束, 不再像 v3 那样后期乱漂。
  - 训练初始化: 用源域 canonical hip 初始化 action_prior (在建 EMA 之前)。
  - 去掉对 root 不可迁移的压力:
      lambda_hip 0.3 -> 0.0 (HipPositionLoss 只拉向源域 GT hip, 不可迁移, 对 PA 无关)
      out_distill_hip_weight 4.0 -> 1.0 (别再重压不可迁移的教师 hip)
  - 旧的 root_anchor / 路1 action-prior 分支已移除 (先验现在进了架构)。
  - 新增残差 L2 惩罚 (--w_root_res), 防止残差饱和。
  - epochs 默认 25 (E04 最优在 ~e9, 50 epoch 只过拟合源域)。
  - 新增周期性 checkpoint (每个 eval_interval 存 raw+ema), 便于对 val 选不出来的
    早期 epoch 单独跑 eval_dtpose_faithful。

选点仍在 E01-E03 held-out val; E04 仅监控、不参与选点。
"""
import os
import sys
import copy
import argparse
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import CSIRSCPoseDG
from models.depth_teacher import DepthPoseTeacher
from losses import TotalLoss, PoseLoss
from evaluate import PoseEvaluator
from dataset_distill import MMFiDistillDataset
from distill_loss import DistillProjection, FeatureDistillLoss, OutputDistillLoss
from utils import (set_seed, setup_logger, count_parameters,
                   save_checkpoint, AverageMeter, Timer, save_run_config)

from structural_losses import structural_loss, build_action_canonical


try:
    from evaluate_v2 import evaluate_v2 as _evaluate_v2
    _HAS_EVAL_V2 = True
except Exception:
    _HAS_EVAL_V2 = False

BACKBONE_MODULES = ('csi_encoder', 'local_encoder', 'feature_pooling', 'global_modeler')
HEAD_MODULES     = ('pose_decoder', 'action_classifier')


def action_to_index(a):
    return int(a[1:]) - 1


# ----------------------------------------------------------------------
# EMA
# ----------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.999, warmup=True):
        self.decay = decay; self.warmup = warmup; self.step = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = (min(self.decay, (1 + self.step) / (10 + self.step))
             if self.warmup else self.decay)
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


# ----------------------------------------------------------------------
# Frozen teacher
# ----------------------------------------------------------------------
class FrozenDepthTeacher(nn.Module):
    def __init__(self, ckpt_path, global_dim=128, num_joints=17, seq_len=64,
                 num_transformer_layers=3, num_heads=4,
                 tcn_channels=(128, 128), tcn_kernel_size=3, device='cuda'):
        super().__init__()
        self.teacher = DepthPoseTeacher(
            global_dim=global_dim, num_joints=num_joints, seq_len=seq_len,
            num_transformer_layers=num_transformer_layers, num_heads=num_heads,
            tcn_channels=tcn_channels, tcn_kernel_size=tcn_kernel_size)
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'model_state_dict' not in ckpt:
            raise KeyError(f"teacher ckpt missing 'model_state_dict'; got {list(ckpt.keys())}")
        miss, unex = self.teacher.load_state_dict(ckpt['model_state_dict'], strict=False)
        if miss or unex:
            raise RuntimeError(f"teacher load mismatch: missing={len(miss)} unexpected={len(unex)}")
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, depth):
        out = self.teacher(depth)
        return {'p_final': out['p_final'].detach(), 'z_global': out['z_global'].detach()}


def load_pretrained_backbone(student, ckpt_path, logger):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    logger.info(f'Pretrain ckpt keys: {list(ckpt.keys())}')
    required = ('csi_encoder', 'local_encoder', 'feature_pooling',
                'global_modeler', 'action_classifier')
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise KeyError(f"pretrain ckpt missing keys: {missing}")
    for mod_name in required:
        m, u = getattr(student, mod_name).load_state_dict(ckpt[mod_name], strict=False)
        logger.info(f'  {mod_name}: missing={len(m)} unexpected={len(u)}')
        if m or u:
            raise RuntimeError(f'KEY MISMATCH {mod_name}: missing={m[:5]} unexpected={u[:5]}')
    logger.info(f'Loaded pretrained: {list(required)}')


def build_optimizer(student, proj, lr_backbone, lr_head, weight_decay):
    backbone_params, head_params = [], []
    for name, p in student.named_parameters():
        top = name.split('.', 1)[0]
        (backbone_params if top in BACKBONE_MODULES else head_params).append(p)
    head_params.extend(list(proj.parameters()))
    return AdamW([
        {'params': backbone_params, 'lr': lr_backbone, 'name': 'backbone'},
        {'params': head_params,     'lr': lr_head,     'name': 'head+proj'},
    ], weight_decay=weight_decay)


# ----------------------------------------------------------------------
# Dataloaders: train + held-out val (E01-E03 按 subject 划) + E04 监控
# ----------------------------------------------------------------------
def split_by_subject(dataset, val_ratio, seed):
    groups = defaultdict(list)
    for i in range(len(dataset)):
        groups[dataset.samples[i]['subject']].append(i)
    keys = sorted(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_ratio)))
    val_keys = set(keys[:n_val])
    tr, va = [], []
    for k, idxs in groups.items():
        (va if k in val_keys else tr).extend(idxs)
    return sorted(tr), sorted(va), sorted(val_keys)


def build_loaders(args, logger):
    common = dict(seq_len=args.seq_len, stride=32, with_depth=True, with_csi=True,
                  depth_img=args.depth_img, depth_clip=args.depth_clip)
    train_full = MMFiDistillDataset(args.data_root, args.train_envs, csi_augment=True, **common)
    val_full   = MMFiDistillDataset(args.data_root, args.train_envs, csi_augment=False, **common)

    tr_idx, va_idx, val_subj = split_by_subject(train_full, args.val_ratio, args.seed)
    logger.info(f'Held-out val subjects ({len(val_subj)}): {val_subj}')
    logger.info(f'Train windows: {len(tr_idx)}  Val windows: {len(va_idx)}')

    train_loader = DataLoader(Subset(train_full, tr_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(val_full, va_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    test_ds = MMFiDistillDataset(args.data_root, [args.test_env], seq_len=args.seq_len,
                                 stride=args.seq_len, with_depth=False, with_csi=True,
                                 depth_img=args.depth_img, depth_clip=args.depth_clip,
                                 csi_augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ----------------------------------------------------------------------
# Train one epoch (每 optimizer step 更新 EMA)
# ----------------------------------------------------------------------
def train_one_epoch(student, proj, teacher, loader, optimizer,
                    total_loss_fn, pose_loss_fn, feat_distill_fn, out_distill_fn,
                    device, epoch, logger, args, ema=None):
    student.train(); proj.train()
    meters = {k: AverageMeter() for k in
              ['loss', 'l_pose_clean', 'l_cons', 'l_action',
               'l_distill_feat', 'l_distill_out', 'l_distill_out_mm',
               'l_struct', 'l_root_res']}
    accum = getattr(args, 'accumulate_grad', 1)
    action_loss_fn = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    use_feat = args.lambda_feat > 0
    use_out  = args.lambda_out  > 0

    for i, batch in enumerate(loader):
        csi = batch['csi'].to(device)
        depth = batch['depth'].to(device)
        pose_3d = batch['pose_3d'].to(device)
        action_labels = torch.tensor([action_to_index(a) for a in batch['action']],
                                     dtype=torch.long, device=device)
        outputs = student.forward_rsc(csi, pose_3d,
                                      loss_fn=lambda p, g: pose_loss_fn(p, g)[0],
                                      action_idx=action_labels)
        action_loss = action_loss_fn(outputs['action_logits'], action_labels)
        base_loss, loss_dict = total_loss_fn(outputs, pose_3d, training=True,
                                             action_loss=action_loss)
        total = base_loss

        # === 蒸馏项 ===
        if use_feat or use_out:
            teacher_out = teacher(depth)
            if use_feat:
                z_s_proj = proj(outputs['z_global'])
                l_feat, fd = feat_distill_fn(z_s_proj, teacher_out['z_global'])
                total = total + args.lambda_feat * l_feat
                meters['l_distill_feat'].update(fd['l_distill_feat'], csi.shape[0])
            if use_out:
                l_out, od = out_distill_fn(outputs['p_final_clean'], teacher_out['p_final'])
                total = total + args.lambda_out * l_out
                meters['l_distill_out'].update(od['l_distill_out'], csi.shape[0])
                meters['l_distill_out_mm'].update(od['l_distill_out_mm'], csi.shape[0])

        # === 结构正则: 只动相对骨架, 不碰 root (PA 杠杆) ===
        l_struct, struct_d = structural_loss(
            outputs['p_final_clean'], pose_3d,
            w_bone=args.w_bone, w_sym=args.w_sym, w_temp=args.w_temp, w_rel=args.w_rel)
        total = total + l_struct
        meters['l_struct'].update(float(l_struct.detach()), csi.shape[0])

        # === 先验 root 残差 L2 惩罚 (在 clean z_global 上重算残差, 梯度干净) ===
        _md = student.module if hasattr(student, 'module') else student
        if args.w_root_res > 0 and hasattr(_md.pose_decoder, 'res'):
            res_c = (torch.tanh(_md.pose_decoder.res(outputs['z_global']))
                     * _md.pose_decoder.residual_scale)
            l_res = res_c.pow(2).mean()
            total = total + args.w_root_res * l_res
            meters['l_root_res'].update(float(l_res.detach()), csi.shape[0])

        (total / accum).backward()
        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(list(student.parameters()) + list(proj.parameters()),
                                         args.grad_clip)
            optimizer.step(); optimizer.zero_grad()
            if ema is not None:
                ema.update(student)
        B = csi.shape[0]
        meters['loss'].update(total.item(), B)
        for k in ['l_pose_clean', 'l_cons', 'l_action']:
            meters[k].update(loss_dict.get(k, 0), B)
        if (i + 1) % args.log_interval == 0:
            msg = (f'Epoch [{epoch}] Batch [{i+1}/{len(loader)}] Loss: {meters["loss"].avg:.4f} '
                   f'Pose(C): {meters["l_pose_clean"].avg:.4f} Act: {meters["l_action"].avg:.4f}')
            msg += f' Struct: {meters["l_struct"].avg:.4f}'
            msg += f" [bone={struct_d.get('bone',0):.3f} rel={struct_d.get('rel',0):.3f}]"
            if args.w_root_res > 0:
                msg += f' RootRes: {meters["l_root_res"].avg:.4f}'
            if use_out:
                msg += (f' Out: {meters["l_distill_out"].avg:.4f}'
                        f' (~{meters["l_distill_out_mm"].avg:.0f}mm)')
            logger.info(msg)
        del outputs, total
        if use_feat or use_out:
            del teacher_out
        torch.cuda.empty_cache()
    return {k: v.avg for k, v in meters.items()}


# ----------------------------------------------------------------------
# Eval
# ----------------------------------------------------------------------
@torch.no_grad()
def _evaluate_builtin(student, loader, device, evaluator, logger):
    student.eval()
    all_preds, all_gts = [], []
    for batch in loader:
        csi = batch['csi'].to(device)
        outputs = student(csi, action_idx=None)
        all_preds.append(outputs['p_final'].cpu()); all_gts.append(batch['pose_3d'])
        del outputs, csi
        torch.cuda.empty_cache()
    preds = torch.cat(all_preds); gts = torch.cat(all_gts)
    m = evaluator.evaluate(preds, gts)
    logger.info(f'  MPJPE: {m["MPJPE (mm)"]:.2f} MPJPE_a: {m["MPJPE_aligned (mm)"]:.2f} '
                f'PA: {m["PA-MPJPE (mm)"]:.2f} P50n: {m["PCK@50_norm (%)"]:.1f}')
    return m


def run_eval(student, loader, device, evaluator, logger):
    if _HAS_EVAL_V2:
        return _evaluate_v2(student, loader, device, evaluator, logger)
    return _evaluate_builtin(student, loader, device, evaluator, logger)


def run_eval_ema(student, ema, loader, device, evaluator, logger):
    backup = copy.deepcopy(student.state_dict())
    ema.copy_to(student)
    m = run_eval(student, loader, device, evaluator, logger)
    student.load_state_dict(backup, strict=True)
    return m


def _save(state, optimizer, epoch, metrics, path):
    torch.save({'epoch': epoch, 'model_state_dict': state,
                'optimizer_state_dict': optimizer.state_dict(), 'metrics': metrics}, path)


# ----------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description='Step B+ v4: distill + EMA + prior-root')
    p.add_argument('--data_root', type=str, default='/home/a123456/PerceptAlign/MMFi')
    p.add_argument('--train_envs', nargs='+', default=['E01', 'E02', 'E03'])
    p.add_argument('--test_env', type=str, default='E04')
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--depth_img', type=int, default=112)
    p.add_argument('--depth_clip', type=float, default=5000.0)
    p.add_argument('--pretrain_ckpt', type=str, required=True)
    p.add_argument('--teacher_ckpt', type=str, required=True)
    p.add_argument('--lambda_feat', type=float, default=0.1)
    p.add_argument('--lambda_out',  type=float, default=1.0)
    p.add_argument('--distill_cos_w', type=float, default=1.0)
    p.add_argument('--distill_sl1_w', type=float, default=1.0)
    p.add_argument('--out_distill_beta', type=float, default=0.05)
    p.add_argument('--out_distill_hip_weight', type=float, default=1.0)   # v3 是 4.0
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
    p.add_argument('--lambda1', type=float, default=1.0)
    p.add_argument('--lambda2', type=float, default=0.5)
    p.add_argument('--lambda3', type=float, default=2.0)
    p.add_argument('--lambda_hip', type=float, default=0.0)   # v3 是 0.3; 关掉 (对 PA 无关, 只加 root 方差)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--beta', type=float, default=2.0)
    p.add_argument('--gamma', type=float, default=0.0)
    p.add_argument('--delta', type=float, default=0.5)
    p.add_argument('--epochs', type=int, default=25)         # v3 是 50; E04 最优在 ~e9
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accumulate_grad', '--accum', type=int, default=8)
    p.add_argument('--lr_backbone', type=float, default=1e-4)
    p.add_argument('--lr_head', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--eval_interval', type=int, default=3)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--save_dir', type=str, default='./checkpoints/distill_priorroot')
    p.add_argument('--val_ratio', type=float, default=0.15)
    p.add_argument('--use_ema', dest='use_ema', action='store_true', default=True)
    p.add_argument('--no_ema', dest='use_ema', action='store_false')
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--ema_no_warmup', action='store_true', default=False)

    # 结构正则 (保持你最优 PA 那次跑用的值; 默认沿用仓库默认, 需要时显式覆盖)
    p.add_argument('--w_bone', type=float, default=1.0)
    p.add_argument('--w_sym',  type=float, default=0.1)
    p.add_argument('--w_temp', type=float, default=0.1)
    p.add_argument('--w_rel',  type=float, default=6.0)

    # 先验 root (本版核心)
    p.add_argument('--root_residual_scale', type=float, default=0.08,
                   help='每帧 root 残差硬上界(米)。越小越安全; 0=纯先验, E04 hip 锁死在动作基线')
    p.add_argument('--w_root_res', type=float, default=0.01,
                   help='残差 L2 惩罚权重, 防饱和')
    p.add_argument('--freeze_root_prior', action='store_true', default=False,
                   help='冻结 action_prior(最硬 E04 保证, 但牺牲室内精度; 默认 False=可训练但用 canonical 初始化)')

    # 周期性快照: 便于对 val 选不出来的早期 epoch 单独跑 faithful 评测
    p.add_argument('--periodic_ckpt', dest='periodic_ckpt', action='store_true', default=True)
    p.add_argument('--no_periodic_ckpt', dest='periodic_ckpt', action='store_false')

    # FK alpha 退火 (透传给 HybridFK base)
    p.add_argument('--fk_alpha_final', type=float, default=0.4)
    p.add_argument('--fk_alpha_warmup', type=int, default=20)

    return p.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = get_args()
    # 把先验 root 的两个超参挂到 args 上, 供 full_model.CSIRSCPoseDG 构造 PriorRootDecoder 时读取
    # (full_model 用 getattr(args, 'root_residual_scale', 0.08) / getattr(args, 'freeze_root_prior', False))
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_run_config(args, args.save_dir, extra={"script": "train_distill_pretrained", "step": "B+v4"})

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('DistillPre', os.path.join(args.save_dir, 'train.log'))
    logger.info('=' * 70)
    logger.info('Step B+ v4: distill + EMA + prior-root, 选点在 E01-E03 held-out val, E04 仅监控')
    logger.info('=' * 70)
    logger.info(f'  lambda_feat={args.lambda_feat} lambda_out={args.lambda_out} '
                f'out_distill_hip_weight={args.out_distill_hip_weight} lambda_hip={args.lambda_hip}')
    logger.info(f'  prior-root: residual_scale={args.root_residual_scale} '
                f'w_root_res={args.w_root_res} freeze_prior={args.freeze_root_prior}')
    logger.info(f'  struct: w_bone={args.w_bone} w_sym={args.w_sym} w_temp={args.w_temp} w_rel={args.w_rel}')
    logger.info(f'  EMA={"ON" if args.use_ema else "OFF"} decay={args.ema_decay} '
                f'val_ratio={args.val_ratio} epochs={args.epochs}')
    logger.info(f'  eval={"evaluate_v2(+hip_err)" if _HAS_EVAL_V2 else "builtin"}')

    train_loader, val_loader, test_loader = build_loaders(args, logger)
    logger.info(f'Train batches: {len(train_loader)}  Val: {len(val_loader)}  E04: {len(test_loader)}')

    args.use_vision_backbone = False
    student = CSIRSCPoseDG(args).to(device)
    logger.info(f'Student params: {count_parameters(student):,}')
    load_pretrained_backbone(student, args.pretrain_ckpt, logger)

    # === 先验 root 初始化: 用源域 canonical hip 初始化 action_prior (必须在建 EMA 之前!) ===
    _md = student.module if hasattr(student, 'module') else student
    if hasattr(_md.pose_decoder, 'action_prior'):
        _canon = build_action_canonical(args.data_root, args.train_envs,
                                        num_actions=args.num_actions).to(device)
        with torch.no_grad():
            _md.pose_decoder.action_prior.data.copy_(_canon)
        logger.info(f'[prior_root] action_prior 已用源域 canonical 初始化 '
                    f'({_canon.shape[0]} actions); '
                    f'requires_grad={_md.pose_decoder.action_prior.requires_grad}')
    else:
        logger.warning('[prior_root] pose_decoder 没有 action_prior —— '
                       '请确认 full_model.py 已把 pose_decoder 换成 PriorRootDecoder')

    teacher = FrozenDepthTeacher(
        args.teacher_ckpt, global_dim=args.global_dim, num_joints=args.num_joints,
        seq_len=args.seq_len, num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads, tcn_channels=tuple(args.tcn_channels),
        tcn_kernel_size=args.tcn_kernel_size, device=device).to(device)
    if sum(p.numel() for p in teacher.parameters() if p.requires_grad) != 0:
        raise RuntimeError('teacher not frozen')
    logger.info('Teacher loaded (frozen).')

    proj = DistillProjection(args.global_dim, args.global_dim).to(device)
    total_loss_fn = TotalLoss(lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
                              alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                              delta=args.delta, lambda_hip=args.lambda_hip)
    pose_loss_fn = PoseLoss(lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
                            lambda_hip=args.lambda_hip)
    feat_distill_fn = FeatureDistillLoss(args.distill_cos_w, args.distill_sl1_w)
    out_distill_fn = OutputDistillLoss(beta=args.out_distill_beta,
                                       hip_weight=args.out_distill_hip_weight,
                                       num_joints=args.num_joints, hip_joint_idx=0).to(device)
    evaluator = PoseEvaluator(unit='meter')

    optimizer = build_optimizer(student, proj, args.lr_backbone, args.lr_head, args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    # EMA 在 action_prior 已用 canonical 初始化之后创建 -> shadow 从先验起步, 不会从 0 慢慢爬
    ema = EMA(student, decay=args.ema_decay, warmup=not args.ema_no_warmup) if args.use_ema else None

    timer = Timer(); timer.start()
    best = {'mpjpe_raw': float('inf'), 'pa_raw': float('inf'),
            'mpjpe_ema': float('inf'), 'pa_ema': float('inf')}
    patience = 0

    for epoch in range(1, args.epochs + 1):
        lrs = [g['lr'] for g in optimizer.param_groups]
        logger.info(f'\n{"="*60}\nEpoch {epoch}/{args.epochs} | LR bb={lrs[0]:.2e} hd={lrs[1]:.2e}')
        _md = student.module if hasattr(student, 'module') else student
        if hasattr(_md.pose_decoder, 'set_alpha'):
            _a = max(args.fk_alpha_final,
                     1.0 - (1.0 - args.fk_alpha_final) * (epoch - 1) / max(1, args.fk_alpha_warmup))
            _md.pose_decoder.set_alpha(_a)
            logger.info(f'[FK] epoch {epoch} alpha={_a:.3f}')
        tm = train_one_epoch(student, proj, teacher, train_loader, optimizer,
                             total_loss_fn, pose_loss_fn, feat_distill_fn, out_distill_fn,
                             device, epoch, logger, args, ema=ema)
        line = f'[Train] Epoch {epoch} | Loss: {tm["loss"]:.4f} Pose(C): {tm["l_pose_clean"]:.4f}'
        if args.lambda_out > 0:
            line += f' Out: {tm["l_distill_out"]:.4f} (~{tm["l_distill_out_mm"]:.0f}mm)'
        line += f' | {timer.elapsed_str()}'
        logger.info(line)
        scheduler.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            improved = False
            v_ema = e_ema = None

            # RAW: 先 val(选点), 再 E04(监控)
            logger.info('  [VAL raw] (selection)')
            v_raw = run_eval(student, val_loader, device, evaluator, logger)
            logger.info('  [E04 raw] (monitor only)')
            e_raw = run_eval(student, test_loader, device, evaluator, logger)
            if v_raw['MPJPE (mm)'] < best['mpjpe_raw']:
                best['mpjpe_raw'] = v_raw['MPJPE (mm)']; improved = True
                _save(student.state_dict(), optimizer, epoch,
                      {'val': v_raw, 'e04': e_raw}, os.path.join(args.save_dir, 'best_mpjpe_raw.pth'))
                logger.info(f'  ** best_mpjpe_raw: val={best["mpjpe_raw"]:.2f}  '
                            f'(E04 MPJPE={e_raw["MPJPE (mm)"]:.2f}) @e{epoch}')
            if v_raw['PA-MPJPE (mm)'] < best['pa_raw']:
                best['pa_raw'] = v_raw['PA-MPJPE (mm)']; improved = True
                _save(student.state_dict(), optimizer, epoch,
                      {'val': v_raw, 'e04': e_raw}, os.path.join(args.save_dir, 'best_pa_raw.pth'))
                logger.info(f'  ** best_pa_raw: val={best["pa_raw"]:.2f} (E04 PA={e_raw["PA-MPJPE (mm)"]:.2f}) @e{epoch}')

            # EMA
            if ema is not None:
                logger.info('  [VAL ema] (selection)')
                v_ema = run_eval_ema(student, ema, val_loader, device, evaluator, logger)
                logger.info('  [E04 ema] (monitor only)')
                e_ema = run_eval_ema(student, ema, test_loader, device, evaluator, logger)
                if v_ema['MPJPE (mm)'] < best['mpjpe_ema']:
                    best['mpjpe_ema'] = v_ema['MPJPE (mm)']; improved = True
                    _save(ema.state_dict(), optimizer, epoch,
                          {'val': v_ema, 'e04': e_ema}, os.path.join(args.save_dir, 'best_mpjpe_ema.pth'))
                    logger.info(f'  ** best_mpjpe_ema: val={best["mpjpe_ema"]:.2f}  '
                                f'(E04 MPJPE={e_ema["MPJPE (mm)"]:.2f}) @e{epoch}  <- 推荐部署')
                if v_ema['PA-MPJPE (mm)'] < best['pa_ema']:
                    best['pa_ema'] = v_ema['PA-MPJPE (mm)']; improved = True
                    _save(ema.state_dict(), optimizer, epoch,
                          {'val': v_ema, 'e04': e_ema}, os.path.join(args.save_dir, 'best_pa_ema.pth'))
                    logger.info(f'  ** best_pa_ema: val={best["pa_ema"]:.2f} (E04 PA={e_ema["PA-MPJPE (mm)"]:.2f}) @e{epoch}')

            # === 周期性快照 (raw + ema), 便于对 val 选不出来的早期 epoch 跑 eval_dtpose_faithful ===
            if getattr(args, 'periodic_ckpt', True):
                _save(student.state_dict(), optimizer, epoch,
                      {'val': v_raw, 'e04': e_raw},
                      os.path.join(args.save_dir, f'epoch{epoch:03d}_raw.pth'))
                if ema is not None:
                    _save(ema.state_dict(), optimizer, epoch,
                          {'val': v_ema, 'e04': e_ema},
                          os.path.join(args.save_dir, f'epoch{epoch:03d}_ema.pth'))

            patience = 0 if improved else patience + 1
            if not improved:
                logger.info(f'  No val improvement. Patience: {patience}/{args.patience}')
            if patience >= args.patience:
                logger.info(f'Early stopping at epoch {epoch}'); break

    logger.info('\n' + '=' * 70)
    logger.info('Step B+ v4 done. (选点在 val; 报告时用对应 ckpt 在 E04 的监控值)')
    logger.info('  部署/对比 DT-Pose: best_mpjpe_ema.pth 或 best_pa_ema.pth, '
                '再用 eval_dtpose_faithful.py --variance 复核')
    logger.info('  另: epochNNN_ema.pth 是周期快照, 可对早期 epoch 单独跑 faithful 评测')
    logger.info(f'  Time: {timer.elapsed_str()}')


if __name__ == '__main__':
    main()