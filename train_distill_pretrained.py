"""
Step B+v3: depth/RGB -> CSI 蒸馏, 从 Stage1B 预训练 backbone 出发。
           结构正则 + Hybrid FK (α 退火) + root anchor (L_anchor)。

口径 (与 DT-Pose / MMFi 主流一致, 见 README §6/§10):
  * E01-E03 全量训练 (无 val 划分)、不早停、跑满 --epochs;
  * archive 每 eval_interval 的 ckpt; 选点在训练后用 eval_dtpose_faithful.py --sweep
    在 E04 上 faithful 逐个评、挑最低 (训练期滑窗监控仅供观察)。

本版新增: --teacher_modality {depth, rgb}
  * depth: FrozenDepthTeacher + batch['depth'] (原行为, 默认)
  * rgb  : FrozenRGBTeacher  + batch['rgb']  (教师由 train_rgb_teacher.py 产出;
           其 ckpt 内含 'backbone' 字段用于重建模型)
  * 蒸馏损失/管线一行不变, 只换教师与教师输入。
  * RGB 教师的 hip 不可信 (单目无米制深度), 建议 --out_distill_hip_weight 1.0
    (depth 默认 4.0 不变; rgb 模态下若未显式指定则自动降为 1.0, 日志会打印)。

============================================================================
本次新增 (打 hip / MPJPE 的三个实验开关, 默认全部保持原行为):
  [1] 输出蒸馏 hip 对齐  --out_distill_align_hip / --no_out_distill_align_hip
      默认随模态: rgb 自动 True (各自减 hip 后只蒸馏相对结构, 不把单目教师那条
      不可信的绝对 hip 灌给学生, 保护学生 PA), depth 默认 False (原行为)。
      需要 distill_loss.OutputDistillLoss 已含 align_hip 参数。
  [2] L_anchor 退火     --anchor_anneal_epochs N (>0 开启) / --w_root_anchor_final F
      前期强按住 root 附近防漂, 后期线性退火到 F, 让网络敢学轨迹。N=0 时为原行为。
  [3] 速度积分 root      --root_mode {absolute(原), velocity} / --vel_scale
      velocity: hip = 锚 + 去均值的速度积分轨迹 (接 fk_decoder.py 的 FKBranch)。
      ★ 需 full_model.py 把 args.root_mode/args.vel_scale 透传给 HybridFKPoseDecoder
        (见文件末尾说明)。未透传时本开关不生效但不报错。
      velocity 模式下 L_anchor 只约束【时间均值 hip = canonical】, 放开轨迹,
      否则逐帧 anchor 会把速度积分学到的轨迹重新摁平。
============================================================================
"""
import os
import sys
import copy
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
from evaluate_v2 import hip_error

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
# Frozen teacher (depth / rgb 统一封装)
# ----------------------------------------------------------------------
class FrozenPoseTeacher(nn.Module):
    """加载并冻结教师。modality='depth' -> DepthPoseTeacher; 'rgb' -> RGBPoseTeacher
    (backbone 从 ckpt 的 'backbone' 字段读, 缺省 resnet18; 重建时不下载预训练权重,
    因为随后会整体加载 ckpt)。forward(x) -> {'p_final','z_global'} 两者一致。"""
    def __init__(self, modality, ckpt_path, global_dim=128, num_joints=17, seq_len=64,
                 num_transformer_layers=3, num_heads=4,
                 tcn_channels=(128, 128), tcn_kernel_size=3, device='cuda'):
        super().__init__()
        assert modality in ('depth', 'rgb'), modality
        self.modality = modality
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'model_state_dict' not in ckpt:
            raise KeyError(f"teacher ckpt missing 'model_state_dict'; got {list(ckpt.keys())}")
        common = dict(global_dim=global_dim, num_joints=num_joints, seq_len=seq_len,
                      num_transformer_layers=num_transformer_layers, num_heads=num_heads,
                      tcn_channels=tcn_channels, tcn_kernel_size=tcn_kernel_size)
        if modality == 'depth':
            self.teacher = DepthPoseTeacher(**common)
        else:
            from models.rgb_teacher import RGBPoseTeacher
            backbone = ckpt.get('backbone', 'resnet18')
            self.teacher = RGBPoseTeacher(backbone=backbone, pretrained=False, **common)
        miss, unex = self.teacher.load_state_dict(ckpt['model_state_dict'], strict=False)
        # px_mean/px_std 等 buffer 在 ckpt 里也存了, strict=False 仅容错命名差异; >0 则报
        if miss or unex:
            raise RuntimeError(f"teacher load mismatch: missing={len(miss)} unexpected={len(unex)} "
                               f"(modality={modality}); 检查 ckpt 与 --teacher_modality 是否配套")
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        out = self.teacher(x)
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
    logger.info('  pose_decoder (Hybrid FK): 不在预训练里, 随机初始化、随训练学。')


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
# Dataloaders (全量训练, 无 val 划分; 教师模态决定 depth/rgb 哪个进 batch)
# ----------------------------------------------------------------------
def build_loaders(args, logger):
    wd = (args.teacher_modality == 'depth')
    wr = (args.teacher_modality == 'rgb')
    common = dict(seq_len=args.seq_len, stride=32, with_depth=wd, with_rgb=wr, with_csi=True,
                  depth_img=args.depth_img, depth_clip=args.depth_clip, rgb_img=args.rgb_img,
                  rgb_root=args.rgb_root)
    if wr and args.rgb_root:
        logger.info(f'[rgb_root] GT/CSI from {args.data_root} ; RGB from {args.rgb_root}')
    train_full = MMFiDistillDataset(args.data_root, args.train_envs, csi_augment=True, **common)
    logger.info(f'[全量训练 / 无 val 划分] {len(train_full)} 窗口 from {args.train_envs} '
                f'(teacher_modality={args.teacher_modality})')

    train_loader = DataLoader(train_full, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    test_ds = MMFiDistillDataset(args.data_root, [args.test_env], seq_len=args.seq_len,
                                 stride=args.seq_len, with_depth=False, with_rgb=False,
                                 with_csi=True,
                                 depth_img=args.depth_img, depth_clip=args.depth_clip,
                                 rgb_img=args.rgb_img, csi_augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    return train_loader, test_loader


# ----------------------------------------------------------------------
# Train one epoch
# ----------------------------------------------------------------------
def train_one_epoch(student, proj, teacher, loader, optimizer, canonical,
                    total_loss_fn, pose_loss_fn, feat_distill_fn, out_distill_fn,
                    device, epoch, logger, args, ema=None):
    student.train(); proj.train()
    meters = {k: AverageMeter() for k in
              ['loss', 'l_pose_clean', 'l_cons', 'l_action',
               'l_distill_feat', 'l_distill_out', 'l_distill_out_mm',
               'l_struct', 'l_anchor']}
    accum = getattr(args, 'accumulate_grad', 1)
    action_loss_fn = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    use_feat = args.lambda_feat > 0
    use_out  = args.lambda_out  > 0
    tkey = args.teacher_modality          # 'depth' or 'rgb' -> batch key
    # L_anchor 当前权重 (epoch 循环里按退火设到 args._w_anchor_cur; 缺省回退到原 w_root_anchor)
    w_anchor_cur = getattr(args, '_w_anchor_cur', args.w_root_anchor)
    vel_mode = (getattr(args, 'root_mode', 'absolute') == 'velocity')

    for i, batch in enumerate(loader):
        csi = batch['csi'].to(device)
        teacher_in = batch[tkey].to(device)
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
            teacher_out = teacher(teacher_in)
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

        # === root anchor (支持退火 + velocity 模式只约束时间均值) ===
        if w_anchor_cur > 0 and canonical is not None:
            pred_hip = outputs['p_final_clean'][:, :, 0, :]      # (B,T,3)
            tgt_hip = canonical[action_labels]                   # (B,3)
            if vel_mode:
                # 只把【时间均值 hip】拉向 canonical, 放开零均值轨迹 (否则会摁平速度积分)
                l_anchor = F.smooth_l1_loss(pred_hip.mean(dim=1), tgt_hip)
            else:
                l_anchor = F.smooth_l1_loss(pred_hip, tgt_hip[:, None, :].expand_as(pred_hip))
            total = total + w_anchor_cur * l_anchor
            meters['l_anchor'].update(float(l_anchor.detach()), csi.shape[0])

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
            if w_anchor_cur > 0:
                msg += f' Anchor: {meters["l_anchor"].avg:.4f}(w={w_anchor_cur:.2f})'
            if use_out:
                msg += (f' Out: {meters["l_distill_out"].avg:.4f}'
                        f' (~{meters["l_distill_out_mm"].avg:.0f}mm)')
            logger.info(msg)
        del outputs, total, teacher_in
        if use_feat or use_out:
            del teacher_out
        torch.cuda.empty_cache()
    return {k: v.avg for k, v in meters.items()}


# ----------------------------------------------------------------------
# E04 在线监控 (滑窗口径, 仅观察; 选点用 faithful sweep)
# ----------------------------------------------------------------------
@torch.no_grad()
def monitor_e04(student, loader, device, evaluator, logger):
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
    m['hip_error (mm)'] = hip_error(preds, gts) * 1000.0
    logger.info(f'  [E04 滑窗监控] MPJPE: {m["MPJPE (mm)"]:.2f} MPJPE_a: {m["MPJPE_aligned (mm)"]:.2f} '
                f'hip: {m["hip_error (mm)"]:.2f} PA: {m["PA-MPJPE (mm)"]:.2f}')
    return m


def monitor_e04_ema(student, ema, loader, device, evaluator, logger):
    backup = copy.deepcopy(student.state_dict())
    ema.copy_to(student)
    m = monitor_e04(student, loader, device, evaluator, logger)
    student.load_state_dict(backup, strict=True)
    return m


def _save(state, optimizer, epoch, metrics, path):
    torch.save({'epoch': epoch, 'model_state_dict': state,
                'optimizer_state_dict': optimizer.state_dict(), 'metrics': metrics}, path)


# ----------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description='Step B+v3: 蒸馏+EMA+全量训练+E04选点(archive); 教师可选 depth/rgb')
    p.add_argument('--data_root', type=str, default='/home/a123456/PerceptAlign/MMFi')
    p.add_argument('--train_envs', nargs='+', default=['E01', 'E02', 'E03'])
    p.add_argument('--test_env', type=str, default='E04')
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--depth_img', type=int, default=112)
    p.add_argument('--depth_clip', type=float, default=5000.0)
    p.add_argument('--rgb_img', type=int, default=112)
    p.add_argument('--rgb_root', type=str, default=None,
                   help='RGB 数据独立根目录 (如机械盘); GT/CSI/depth 仍从 --data_root 读。None=与 data_root 相同')
    p.add_argument('--teacher_modality', type=str, default='depth', choices=['depth', 'rgb'],
                   help="教师模态: depth(原) / rgb(train_rgb_teacher.py 产出的 ckpt)")
    p.add_argument('--pretrain_ckpt', type=str, required=True)
    p.add_argument('--teacher_ckpt', type=str, required=True)
    p.add_argument('--lambda_feat', type=float, default=0.1)
    p.add_argument('--lambda_out',  type=float, default=1.0)
    p.add_argument('--distill_cos_w', type=float, default=1.0)
    p.add_argument('--distill_sl1_w', type=float, default=1.0)
    p.add_argument('--out_distill_beta', type=float, default=0.05)
    p.add_argument('--out_distill_hip_weight', type=float, default=None,
                   help='默认: depth=4.0, rgb=1.0 (RGB 教师 hip 不可信)。显式指定则覆盖。')
    # [1] 输出蒸馏 hip 对齐 (RGB 自动开, 保护学生 PA 不被单目教师那条绝对 hip 毒化)
    p.add_argument('--out_distill_align_hip', dest='out_distill_align_hip',
                   action='store_true', default=None,
                   help='蒸馏前各自减 hip, 只传相对结构。None=随模态(rgb 自动 True, depth False)')
    p.add_argument('--no_out_distill_align_hip', dest='out_distill_align_hip',
                   action='store_false')
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
    p.add_argument('--lambda_hip', type=float, default=0.3)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--beta', type=float, default=2.0)
    p.add_argument('--gamma', type=float, default=0.0)
    p.add_argument('--delta', type=float, default=0.5)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accumulate_grad', '--accum', type=int, default=8)
    p.add_argument('--lr_backbone', type=float, default=1e-4)
    p.add_argument('--lr_head', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--eval_interval', type=int, default=3,
                   help='每 N epoch 在 E04 上监控并 archive 一次')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--save_dir', type=str, default='./checkpoints/distill_fk_anchor')

    p.add_argument('--use_ema', dest='use_ema', action='store_true', default=True)
    p.add_argument('--no_ema', dest='use_ema', action='store_false')
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--ema_no_warmup', action='store_true', default=False)

    p.add_argument('--w_bone', type=float, default=1.0)
    p.add_argument('--w_sym',  type=float, default=0.1)
    p.add_argument('--w_temp', type=float, default=0.1)
    p.add_argument('--w_rel',  type=float, default=6.0)
    p.add_argument('--w_root_anchor', type=float, default=0.5)
    # [2] L_anchor 退火 (默认不退火 = 原行为)
    p.add_argument('--w_root_anchor_final', type=float, default=0.1,
                   help='L_anchor 退火终值(绝对权重下限)')
    p.add_argument('--anchor_anneal_epochs', type=int, default=0,
                   help='>0: L_anchor 从 w_root_anchor 线性退火到 w_root_anchor_final 的 epoch 数; 0=不退火')

    p.add_argument('--fk_alpha_final', type=float, default=0.4)
    p.add_argument('--fk_alpha_warmup', type=int, default=20)

    # [3] 速度积分 root (需 full_model.py 把 root_mode/vel_scale 透传给 HybridFKPoseDecoder)
    p.add_argument('--root_mode', type=str, default='absolute', choices=['absolute', 'velocity'],
                   help='hip root: absolute(原逐帧绝对) / velocity(锚+去均值速度积分轨迹)')
    p.add_argument('--vel_scale', type=float, default=0.12,
                   help='velocity 模式每帧位移限幅(米)')

    p.add_argument('--archive_ckpts', dest='archive_ckpts', action='store_true', default=True)
    p.add_argument('--no_archive_ckpts', dest='archive_ckpts', action='store_false')

    args = p.parse_args()
    # hip 蒸馏权重模态相关默认: depth=4.0, rgb=1.0
    if args.out_distill_hip_weight is None:
        args.out_distill_hip_weight = 4.0 if args.teacher_modality == 'depth' else 1.0
    # hip 对齐蒸馏模态相关默认: rgb=True, depth=False
    if args.out_distill_align_hip is None:
        args.out_distill_align_hip = (args.teacher_modality == 'rgb')
    return args


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = get_args()
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_run_config(args, args.save_dir, extra={"script": "train_distill_pretrained",
                                                "step": f"B+v3-{args.teacher_modality}"})

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('DistillPre', os.path.join(args.save_dir, 'train.log'))
    logger.info('=' * 70)
    logger.info(f'Step B+v3: 蒸馏 + EMA + 全量训练(无val) + E04 选点(archive)  '
                f'[teacher={args.teacher_modality}]')
    logger.info('=' * 70)
    logger.info(f'  lambda_feat={args.lambda_feat} lambda_out={args.lambda_out} '
                f'out_distill_hip_weight={args.out_distill_hip_weight} '
                f'align_hip={args.out_distill_align_hip} lambda_hip={args.lambda_hip}')
    logger.info(f'  struct: w_bone={args.w_bone} w_sym={args.w_sym} w_temp={args.w_temp} w_rel={args.w_rel}')
    logger.info(f'  root anchor: w_root_anchor={args.w_root_anchor} '
                f'final={args.w_root_anchor_final} anneal_epochs={args.anchor_anneal_epochs}')
    logger.info(f'  root_mode={args.root_mode} vel_scale={args.vel_scale}')
    logger.info(f'  FK: alpha_final={args.fk_alpha_final} warmup={args.fk_alpha_warmup}')
    logger.info(f'  EMA={"ON" if args.use_ema else "OFF"} epochs={args.epochs} '
                f'archive_ckpts={args.archive_ckpts}')
    logger.info('  选点: 训练后 eval_dtpose_faithful.py --sweep "<save_dir>/epoch*_ema.pth" (faithful)')

    train_loader, test_loader = build_loaders(args, logger)
    logger.info(f'Train batches: {len(train_loader)}  E04 batches: {len(test_loader)}')

    args.use_vision_backbone = False
    student = CSIRSCPoseDG(args).to(device)
    logger.info(f'Student params: {count_parameters(student):,}')
    load_pretrained_backbone(student, args.pretrain_ckpt, logger)

    # velocity root 透传自检: full_model 是否真把 root_mode 传进了 FK 支
    if args.root_mode == 'velocity':
        _md = student.module if hasattr(student, 'module') else student
        _fk = getattr(getattr(_md, 'pose_decoder', None), 'fk', None)
        if _fk is None or getattr(_fk, 'root_mode', 'absolute') != 'velocity':
            logger.warning('[root_mode=velocity] 但 pose_decoder.fk 未运行在 velocity 模式! '
                           'full_model.py 可能没把 args.root_mode 透传给 HybridFKPoseDecoder。'
                           '请按文件末尾说明加一行, 否则本开关无效。')
        else:
            logger.info(f'[root_mode=velocity] FK 支已确认 velocity (vel_scale={getattr(_fk,"vel_scale",None)})')

    canonical = None
    if args.w_root_anchor > 0:
        canonical = build_action_canonical(args.data_root, args.train_envs,
                                            num_actions=args.num_actions).to(device)
        logger.info(f'[root anchor] canonical 已由源域构建 ({canonical.shape[0]} actions)')

    teacher = FrozenPoseTeacher(
        args.teacher_modality, args.teacher_ckpt,
        global_dim=args.global_dim, num_joints=args.num_joints,
        seq_len=args.seq_len, num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads, tcn_channels=tuple(args.tcn_channels),
        tcn_kernel_size=args.tcn_kernel_size, device=device).to(device)
    if sum(p.numel() for p in teacher.parameters() if p.requires_grad) != 0:
        raise RuntimeError('teacher not frozen')
    logger.info(f'Teacher loaded (frozen, modality={args.teacher_modality}).')

    proj = DistillProjection(args.global_dim, args.global_dim).to(device)
    total_loss_fn = TotalLoss(lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
                              alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                              delta=args.delta, lambda_hip=args.lambda_hip)
    pose_loss_fn = PoseLoss(lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
                            lambda_hip=args.lambda_hip)
    feat_distill_fn = FeatureDistillLoss(args.distill_cos_w, args.distill_sl1_w)
    out_distill_fn = OutputDistillLoss(beta=args.out_distill_beta,
                                       hip_weight=args.out_distill_hip_weight,
                                       num_joints=args.num_joints, hip_joint_idx=0,
                                       align_hip=args.out_distill_align_hip).to(device)
    evaluator = PoseEvaluator(unit='meter')

    optimizer = build_optimizer(student, proj, args.lr_backbone, args.lr_head, args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    ema = EMA(student, decay=args.ema_decay, warmup=not args.ema_no_warmup) if args.use_ema else None

    timer = Timer(); timer.start()

    for epoch in range(1, args.epochs + 1):
        lrs = [g['lr'] for g in optimizer.param_groups]
        logger.info(f'\n{"="*60}\nEpoch {epoch}/{args.epochs} | LR bb={lrs[0]:.2e} hd={lrs[1]:.2e}')
        _md = student.module if hasattr(student, 'module') else student
        if hasattr(_md.pose_decoder, 'set_alpha'):
            _a = max(args.fk_alpha_final,
                     1.0 - (1.0 - args.fk_alpha_final) * (epoch - 1) / max(1, args.fk_alpha_warmup))
            _md.pose_decoder.set_alpha(_a)
            logger.info(f'[FK] epoch {epoch} alpha={_a:.3f}')

        # === L_anchor 退火: 前期强按住防漂, 后期放开让网络学轨迹 (anneal_epochs=0 时恒为 w_root_anchor) ===
        if args.anchor_anneal_epochs > 0:
            _frac = min(1.0, (epoch - 1) / max(1, args.anchor_anneal_epochs))
            args._w_anchor_cur = (args.w_root_anchor
                                  + (args.w_root_anchor_final - args.w_root_anchor) * _frac)
        else:
            args._w_anchor_cur = args.w_root_anchor
        if args.w_root_anchor > 0:
            logger.info(f'[anchor] epoch {epoch} w_root_anchor={args._w_anchor_cur:.3f}'
                        f'{" (velocity: 只约束时间均值hip)" if args.root_mode=="velocity" else ""}')

        tm = train_one_epoch(student, proj, teacher, train_loader, optimizer, canonical,
                             total_loss_fn, pose_loss_fn, feat_distill_fn, out_distill_fn,
                             device, epoch, logger, args, ema=ema)
        line = f'[Train] Epoch {epoch} | Loss: {tm["loss"]:.4f} Pose(C): {tm["l_pose_clean"]:.4f}'
        if args.w_root_anchor > 0:
            line += f' Anchor: {tm["l_anchor"]:.4f}'
        if args.lambda_out > 0:
            line += f' Out: {tm["l_distill_out"]:.4f} (~{tm["l_distill_out_mm"]:.0f}mm)'
        line += f' | {timer.elapsed_str()}'
        logger.info(line)
        scheduler.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            logger.info('  [E04 raw 滑窗监控]')
            e_raw = monitor_e04(student, test_loader, device, evaluator, logger)
            e_ema = None
            if ema is not None:
                logger.info('  [E04 ema 滑窗监控]')
                e_ema = monitor_e04_ema(student, ema, test_loader, device, evaluator, logger)

            if args.archive_ckpts:
                _save(student.state_dict(), optimizer, epoch,
                      {'e04_sliding': e_raw},
                      os.path.join(args.save_dir, f'epoch{epoch:03d}_raw.pth'))
                if ema is not None:
                    _save(ema.state_dict(), optimizer, epoch,
                          {'e04_sliding': e_ema},
                          os.path.join(args.save_dir, f'epoch{epoch:03d}_ema.pth'))
                logger.info(f'  [archive] 已存 epoch{epoch:03d}_raw/ema.pth')

    _save(student.state_dict(), optimizer, args.epochs, {}, os.path.join(args.save_dir, 'last_raw.pth'))
    if ema is not None:
        _save(ema.state_dict(), optimizer, args.epochs, {}, os.path.join(args.save_dir, 'last_ema.pth'))

    logger.info('\n' + '=' * 70)
    logger.info('训练完成。E04 选点请运行 (faithful 口径, 权威):')
    logger.info(f'  python eval_dtpose_faithful.py --data_root {args.data_root} \\')
    logger.info(f'    --sweep "{args.save_dir}/epoch*_ema.pth" --test_env {args.test_env} --seq_len {args.seq_len}')
    logger.info(f'  Time: {timer.elapsed_str()}')


if __name__ == '__main__':
    main()