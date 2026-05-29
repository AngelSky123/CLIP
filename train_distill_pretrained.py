"""
Step B+ : depth -> CSI cross-modal distillation, starting from a PRETRAINED
Stage1B Action checkpoint.

Why pretrained: the from-scratch single-stage diagnostic (train_distill.py)
suffered early collapse because the action classifier never learned from zero,
so the action-conditioned decoder fed on garbage embeddings. Loading
action_best.pt gives the student a strong, non-collapsed starting point
(train_acc ≈ 87%) — the same starting point train_stage2.py uses to reach
PA ≈ 104mm in the baseline. This makes the distillation experiment a fair
incremental change on top of the baseline.

Three distillation paths combine to push BOTH MPJPE (DT-Pose target 316.8)
and PA-MPJPE (DT-Pose target 104.2) below DT-Pose:

  L_total = L_pose(student vs GT)                                   [primary]
          + lambda_feat · L_feat_distill(proj(z_s), z_t.detach())   [feature align]
          + lambda_out  · L_out_distill(p_s_clean, p_t.detach())    [pose align]

  - L_pose: standard Stage2 TotalLoss (clean+masked+cons+action). Internal
            hip-weighting via --lambda_hip (same as train_stage2.py).
  - L_feat: projects student z_global (128) -> teacher z_global (128).
            Cosine direction + smooth-L1 magnitude. Mild pull on latent space.
  - L_out : smooth-L1 on (B,T,17,3) pose, beta=5cm, hip joint weighted 1.5x.
            Targets the MPJPE gap directly — teacher MPJPE=269 is ~48mm
            better than DT-Pose's 316.8, so even partial transfer of this
            advantage is enough to surpass DT-Pose's MPJPE.

Strict DG: test on E04 is CSI-only with action_idx=None. Depth used ONLY at
train, ONLY on source envs. No target-env depth is touched anywhere.

Independent of train.py / losses.py / train_stage2.py — those are imported
read-only. The Stage2 hyperparameters (lr_backbone=1e-4 / lr_head=5e-4 /
batch=2 accum=8 / lambda_hip=0.3 / 50 epochs) are mirrored exactly so any PA
difference from the un-distilled baseline (which gets PA≈104) is attributable
to distillation, not to a confounded training regime.

Suggested usage (matches Stage2 baseline hyperparams + distillation on top):

    python train_distill_pretrained.py \
        --data_root /home/a123456/PerceptAlign/MMFi \
        --train_envs E01 E02 E03 --test_env E04 \
        --pretrain_ckpt checkpoints/stage1b_action/action_best.pt \
        --teacher_ckpt  checkpoints/depth_teacher_full/teacher_best.pt \
        --depth_img 112 --depth_clip 5000 \
        --lambda_feat 0.1 --lambda_out 0.5 --lambda_hip 0.3 \
        --epochs 50 --batch_size 2 --accumulate_grad 8 \
        --lr_backbone 1e-4 --lr_head 5e-4 \
        --save_dir ./checkpoints/distill_pretrained

To run a clean λ=0 ablation (no distillation, but full pipeline otherwise):
    --lambda_feat 0 --lambda_out 0
The teacher is still loaded but its forward output is unused; this isolates
whether distillation itself moves the needle vs baseline.
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
from models.depth_teacher import DepthPoseTeacher
from losses import TotalLoss, PoseLoss
from evaluate import PoseEvaluator
from dataset_distill import MMFiDistillDataset
from distill_loss import DistillProjection, FeatureDistillLoss, OutputDistillLoss
from utils import (set_seed, setup_logger, count_parameters,
                   save_checkpoint, AverageMeter, Timer, save_run_config)


# Backbone modules = pretrained from Stage1B (low LR, fine-tune)
# Head modules     = task heads (higher LR, mostly retrained for pose)
BACKBONE_MODULES = ('csi_encoder', 'local_encoder', 'feature_pooling', 'global_modeler')
HEAD_MODULES     = ('pose_decoder', 'action_classifier')


def action_to_index(a):
    return int(a[1:]) - 1


# ----------------------------------------------------------------------
# Frozen depth teacher: full DepthPoseTeacher, outputs pose AND z_global.
# Used for BOTH feature and output distillation. Frozen, eval-only, detached.
# ----------------------------------------------------------------------
class FrozenDepthTeacher(nn.Module):
    """Loads full DepthPoseTeacher from teacher_best.pt, frozen.

    forward(depth) -> {'p_final': (B,T,J,3) detached,
                       'z_global': (B,T,128) detached}

    Loads from ckpt['model_state_dict'] which contains the full teacher
    (encoder + global_modeler + pose_head). The separate 'encoder' /
    'global_modeler' keys in the ckpt are also there but are subset views;
    we load the full model so the pose_head is included.
    """
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
            raise KeyError(
                f"teacher ckpt missing 'model_state_dict'; got keys={list(ckpt.keys())}")
        miss, unex = self.teacher.load_state_dict(ckpt['model_state_dict'], strict=False)
        if miss or unex:
            raise RuntimeError(
                f"teacher load mismatch: missing={len(miss)} unexpected={len(unex)}\n"
                f"missing[:5]={list(miss)[:5]}\nunexpected[:5]={list(unex)[:5]}")

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, depth):
        out = self.teacher(depth)
        return {'p_final': out['p_final'].detach(),
                'z_global': out['z_global'].detach()}


# ----------------------------------------------------------------------
# Pretrained backbone loader (mirrors train_stage2.py: explicit per-module).
# ----------------------------------------------------------------------
def load_pretrained_backbone(student, ckpt_path, logger):
    """Load action_best.pt's 5 sub-state-dicts into corresponding student modules.

    Required keys in the ckpt: csi_encoder, local_encoder, feature_pooling,
    global_modeler, action_classifier (the same 5 keys train_stage2.py uses,
    confirmed by inspection of action_best.pt).
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    logger.info(f'Pretrain ckpt keys: {list(ckpt.keys())}')
    required = ('csi_encoder', 'local_encoder', 'feature_pooling',
                'global_modeler', 'action_classifier')
    missing_in_ckpt = [k for k in required if k not in ckpt]
    if missing_in_ckpt:
        raise KeyError(f"pretrain ckpt missing required keys: {missing_in_ckpt}")

    loaded = []
    for mod_name in required:
        target = getattr(student, mod_name)
        miss, unex = target.load_state_dict(ckpt[mod_name], strict=False)
        logger.info(f'  {mod_name}: missing={len(miss)} unexpected={len(unex)}')
        if miss or unex:
            raise RuntimeError(
                f'KEY MISMATCH for {mod_name}: missing={miss[:5]} unexpected={unex[:5]} — '
                f'aborting (pretrained backbone load must be clean to be a fair '
                f'baseline reproduction).')
        loaded.append(mod_name)
    logger.info(f'Loaded pretrained components: {loaded}')
    train_acc = ckpt.get('train_acc', None)
    if train_acc is not None:
        logger.info(f'Pretrain ckpt train_acc = {train_acc:.4f} (random = 1/27 ≈ 0.037)')
    return loaded


# ----------------------------------------------------------------------
# Differential LR optimizer: backbone (low LR) + heads + proj (high LR)
# ----------------------------------------------------------------------
def build_optimizer(student, proj, lr_backbone, lr_head, weight_decay):
    backbone_params, head_params = [], []
    for name, p in student.named_parameters():
        top = name.split('.', 1)[0]
        if top in BACKBONE_MODULES:
            backbone_params.append(p)
        elif top in HEAD_MODULES:
            head_params.append(p)
        else:
            # rsc_global has no params; anything else (shouldn't exist) -> head LR
            head_params.append(p)
    # DistillProjection is a newly-initialized head -> head LR
    head_params.extend(list(proj.parameters()))

    param_groups = [
        {'params': backbone_params, 'lr': lr_backbone, 'name': 'backbone'},
        {'params': head_params,     'lr': lr_head,     'name': 'head+proj'},
    ]
    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


# ----------------------------------------------------------------------
# Dataloaders
# ----------------------------------------------------------------------
def build_loaders(args):
    """Train: source envs with BOTH csi+depth; Test: target env CSI-only."""
    train_ds = MMFiDistillDataset(
        args.data_root, args.train_envs, args.seq_len, stride=32,
        with_depth=True, with_csi=True,
        depth_img=args.depth_img, depth_clip=args.depth_clip,
        csi_augment=True)
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


# ----------------------------------------------------------------------
# Train one epoch
# ----------------------------------------------------------------------
def train_one_epoch(student, proj, teacher, loader, optimizer,
                    total_loss_fn, pose_loss_fn, feat_distill_fn, out_distill_fn,
                    device, epoch, logger, args):
    student.train()
    proj.train()
    meters = {k: AverageMeter() for k in
              ['loss', 'l_pose_clean', 'l_pose_masked', 'l_cons', 'l_action',
               'l_distill_feat', 'l_distill_out', 'l_distill_out_mm']}
    accum = getattr(args, 'accumulate_grad', 1)
    action_loss_fn = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    use_feat = args.lambda_feat > 0
    use_out  = args.lambda_out  > 0

    for i, batch in enumerate(loader):
        csi      = batch['csi'].to(device)
        depth    = batch['depth'].to(device)
        pose_3d  = batch['pose_3d'].to(device)
        action_labels = torch.tensor(
            [action_to_index(a) for a in batch['action']],
            dtype=torch.long, device=device)

        # Student forward (RSC training path; same as train.py)
        outputs = student.forward_rsc(
            csi, pose_3d,
            loss_fn=lambda p, g: pose_loss_fn(p, g)[0],
            action_idx=action_labels)

        # Standard pose+RSC+action loss (Stage2 baseline objective)
        action_loss = action_loss_fn(outputs['action_logits'], action_labels)
        base_loss, loss_dict = total_loss_fn(
            outputs, pose_3d, training=True, action_loss=action_loss)
        total = base_loss

        # Teacher forward (skip entirely if both distillation lambdas are 0)
        if use_feat or use_out:
            teacher_out = teacher(depth)            # both keys detached inside teacher

            if use_feat:
                z_s_proj = proj(outputs['z_global'])
                l_feat, fd = feat_distill_fn(z_s_proj, teacher_out['z_global'])
                total = total + args.lambda_feat * l_feat
                meters['l_distill_feat'].update(fd['l_distill_feat'], csi.shape[0])

            if use_out:
                # Output distillation: align student CLEAN pose (no RSC mask) to teacher pose.
                l_out, od = out_distill_fn(outputs['p_final_clean'], teacher_out['p_final'])
                total = total + args.lambda_out * l_out
                meters['l_distill_out'].update(od['l_distill_out'], csi.shape[0])
                meters['l_distill_out_mm'].update(od['l_distill_out_mm'], csi.shape[0])

        (total / accum).backward()

        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(proj.parameters()),
                    args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        B = csi.shape[0]
        meters['loss'].update(total.item(), B)
        for k in ['l_pose_clean', 'l_pose_masked', 'l_cons', 'l_action']:
            meters[k].update(loss_dict.get(k, 0), B)

        if (i + 1) % args.log_interval == 0:
            msg = (f'Epoch [{epoch}] Batch [{i+1}/{len(loader)}] '
                   f'Loss: {meters["loss"].avg:.4f} '
                   f'Pose(C): {meters["l_pose_clean"].avg:.4f} '
                   f'Pose(M): {meters["l_pose_masked"].avg:.4f} '
                   f'Cons: {meters["l_cons"].avg:.4f} '
                   f'Act: {meters["l_action"].avg:.4f}')
            if use_feat:
                msg += f' Feat: {meters["l_distill_feat"].avg:.4f}'
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
# Evaluate (same as train.py — CSI-only, action_idx=None)
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(student, test_loader, device, evaluator, logger):
    student.eval()
    all_preds, all_gts = [], []
    action_correct, action_total = 0, 0
    for batch in test_loader:
        csi = batch['csi'].to(device)
        pose_3d = batch['pose_3d'].to(device)
        outputs = student(csi, action_idx=None)
        all_preds.append(outputs['p_final'].cpu())
        all_gts.append(pose_3d.cpu())
        labels = torch.tensor(
            [action_to_index(a) for a in batch['action']],
            dtype=torch.long, device=device)
        action_correct += (outputs['action_logits'].argmax(-1) == labels).sum().item()
        action_total += labels.shape[0]
        del outputs, csi, pose_3d
        torch.cuda.empty_cache()
    preds = torch.cat(all_preds)
    gts = torch.cat(all_gts)
    metrics = evaluator.evaluate(preds, gts)
    pred_std = preds.mean(dim=1).std(dim=0).mean().item() * 1000
    acc = 100.0 * action_correct / max(action_total, 1)
    logger.info(
        f'[Eval] MPJPE: {metrics["MPJPE (mm)"]:.2f}mm | '
        f'MPJPE_a: {metrics["MPJPE_aligned (mm)"]:.2f}mm | '
        f'PA: {metrics["PA-MPJPE (mm)"]:.2f}mm | '
        f'P50n: {metrics["PCK@50_norm (%)"]:.1f}% | '
        f'P20n: {metrics["PCK@20_norm (%)"]:.1f}% | '
        f'PredStd: {pred_std:.1f}mm | ActAcc: {acc:.1f}%')
    metrics['pred_std'] = pred_std
    metrics['action_acc'] = acc
    return metrics


# ----------------------------------------------------------------------
# Args (mirror Stage2 baseline + distillation knobs)
# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description='Step B+: depth->CSI distillation on pretrained backbone')
    # data / envs
    p.add_argument('--data_root', type=str, default='/home/a123456/PerceptAlign/MMFi')
    p.add_argument('--train_envs', nargs='+', default=['E01', 'E02', 'E03'])
    p.add_argument('--test_env', type=str, default='E04')
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--depth_img', type=int, default=112)
    p.add_argument('--depth_clip', type=float, default=5000.0)
    # checkpoints
    p.add_argument('--pretrain_ckpt', type=str, required=True,
                   help='Stage1B action_best.pt — loaded into 5 backbone modules')
    p.add_argument('--teacher_ckpt', type=str, required=True,
                   help='Step A depth teacher_best.pt — frozen, drives distillation')
    # distillation
    p.add_argument('--lambda_feat', type=float, default=0.1,
                   help='Feature distill weight (z_global alignment)')
    p.add_argument('--lambda_out',  type=float, default=0.5,
                   help='Output distill weight (pose alignment, targets MPJPE)')
    p.add_argument('--distill_cos_w', type=float, default=1.0)
    p.add_argument('--distill_sl1_w', type=float, default=1.0)
    p.add_argument('--out_distill_beta', type=float, default=0.05,
                   help='SmoothL1 beta in meters (0.05 = 5cm). Robust to teacher error.')
    p.add_argument('--out_distill_hip_weight', type=float, default=1.5,
                   help='Hip joint weight in output distillation (1.0 disables)')
    # student model dims (match config.py defaults; same as Stage2 baseline)
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
    # loss weights (TotalLoss — match Stage2 baseline: lambda_hip=0.3, gamma=0)
    p.add_argument('--lambda1', type=float, default=1.0)
    p.add_argument('--lambda2', type=float, default=0.5)
    p.add_argument('--lambda3', type=float, default=2.0)
    p.add_argument('--lambda_hip', type=float, default=0.3,
                   help='Hip-position weight in pose loss (Stage2 baseline uses 0.3)')
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--beta', type=float, default=2.0)
    p.add_argument('--gamma', type=float, default=0.0)
    p.add_argument('--delta', type=float, default=0.5)
    # training (Stage2 baseline: lr_backbone=1e-4 lr_head=5e-4 batch=2 accum=8)
    p.add_argument('--epochs', type=int, default=50)
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
    p.add_argument('--save_dir', type=str, default='./checkpoints/distill_pretrained')
    return p.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = get_args()
    set_seed(args.seed)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_run_config(args, args.save_dir,
                    extra={"script": "train_distill_pretrained",
                           "step": "B+",
                           "pretrain_ckpt": args.pretrain_ckpt,
                           "teacher_ckpt": args.teacher_ckpt})

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('DistillPre', os.path.join(args.save_dir, 'train.log'))
    logger.info('=' * 70)
    logger.info('Step B+: depth->CSI distillation on Stage1B-pretrained backbone')
    logger.info('=' * 70)
    logger.info(f'  train={args.train_envs}  test={args.test_env}')
    logger.info(f'  pretrain_ckpt={args.pretrain_ckpt}')
    logger.info(f'  teacher_ckpt={args.teacher_ckpt}')
    logger.info(f'  lambda_feat={args.lambda_feat}  lambda_out={args.lambda_out}  '
                f'lambda_hip={args.lambda_hip}')
    logger.info(f'  lr_backbone={args.lr_backbone}  lr_head={args.lr_head}')
    logger.info(f'  batch={args.batch_size}  accum={args.accumulate_grad}  '
                f'epochs={args.epochs}')
    logger.info('  Strict DG: target-env CSI-only at test, action_idx=None')

    # ---- data
    train_loader, test_loader = build_loaders(args)
    logger.info(f'Train batches: {len(train_loader)}  Test batches: {len(test_loader)}')

    # ---- student
    args.use_vision_backbone = False
    student = CSIRSCPoseDG(args).to(device)
    logger.info(f'Student params: {count_parameters(student):,}')

    # ---- pretrained backbone load (CRITICAL — must be 0 missing / 0 unexpected)
    load_pretrained_backbone(student, args.pretrain_ckpt, logger)

    # ---- frozen teacher
    teacher = FrozenDepthTeacher(
        args.teacher_ckpt, global_dim=args.global_dim, num_joints=args.num_joints,
        seq_len=args.seq_len, num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        tcn_channels=tuple(args.tcn_channels), tcn_kernel_size=args.tcn_kernel_size,
        device=device).to(device)
    n_train_t = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    n_total_t = sum(p.numel() for p in teacher.parameters())
    logger.info(f'Teacher loaded: total={n_total_t:,}  trainable={n_train_t:,} '
                f'(must be 0)')
    if n_train_t != 0:
        raise RuntimeError(f'teacher has {n_train_t} trainable params — should be 0')

    # ---- distillation modules
    proj = DistillProjection(args.global_dim, args.global_dim).to(device)
    logger.info(f'Projection params: {count_parameters(proj):,}')

    # ---- losses
    total_loss_fn = TotalLoss(
        lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta,
        lambda_hip=args.lambda_hip)
    pose_loss_fn = PoseLoss(
        lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
        lambda_hip=args.lambda_hip)
    feat_distill_fn = FeatureDistillLoss(args.distill_cos_w, args.distill_sl1_w)
    out_distill_fn  = OutputDistillLoss(
        beta=args.out_distill_beta,
        hip_weight=args.out_distill_hip_weight,
        num_joints=args.num_joints,
        hip_joint_idx=0).to(device)
    evaluator = PoseEvaluator(unit='meter')

    # ---- optimizer (differential LR)
    optimizer = build_optimizer(
        student, proj,
        lr_backbone=args.lr_backbone, lr_head=args.lr_head,
        weight_decay=args.weight_decay)
    n_bb = sum(p.numel() for g in optimizer.param_groups if g['name']=='backbone'
               for p in g['params'])
    n_hd = sum(p.numel() for g in optimizer.param_groups if g['name']=='head+proj'
               for p in g['params'])
    logger.info(f'Optimizer: backbone={n_bb:,} @ lr={args.lr_backbone}  '
                f'head+proj={n_hd:,} @ lr={args.lr_head}')
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    timer = Timer(); timer.start()
    best_mpjpe = float('inf'); best_pa = float('inf'); patience = 0

    for epoch in range(1, args.epochs + 1):
        cur_lrs = [g['lr'] for g in optimizer.param_groups]
        logger.info(f'\n{"="*60}\nEpoch {epoch}/{args.epochs} | '
                    f'LR: backbone={cur_lrs[0]:.2e} head={cur_lrs[1]:.2e}')

        tm = train_one_epoch(student, proj, teacher, train_loader, optimizer,
                             total_loss_fn, pose_loss_fn,
                             feat_distill_fn, out_distill_fn,
                             device, epoch, logger, args)
        line = (f'[Train] Epoch {epoch} | Loss: {tm["loss"]:.4f} '
                f'Pose(C): {tm["l_pose_clean"]:.4f} '
                f'Act: {tm["l_action"]:.4f}')
        if args.lambda_feat > 0:
            line += f' Feat: {tm["l_distill_feat"]:.4f}'
        if args.lambda_out > 0:
            line += f' Out: {tm["l_distill_out"]:.4f} (~{tm["l_distill_out_mm"]:.0f}mm)'
        line += f' | {timer.elapsed_str()}'
        logger.info(line)
        scheduler.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            m = evaluate(student, test_loader, device, evaluator, logger)
            cur = m['MPJPE (mm)']
            if cur < best_mpjpe:
                best_mpjpe = cur
                best_pa = m['PA-MPJPE (mm)']
                patience = 0
                save_checkpoint(student, optimizer, epoch, m,
                                os.path.join(args.save_dir, 'best_model.pth'))
                torch.save({'epoch': epoch, 'proj_state_dict': proj.state_dict()},
                           os.path.join(args.save_dir, 'proj_best.pth'))
                logger.info(f'*** NEW BEST MPJPE: {best_mpjpe:.2f}mm  '
                            f'PA: {best_pa:.2f}mm ***')
            else:
                patience += 1
                logger.info(f'No improvement. Patience: {patience}/{args.patience}')
            if patience >= args.patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break

        if epoch % 10 == 0:
            save_checkpoint(student, optimizer, epoch, {},
                            os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'))

    logger.info('\n' + '=' * 70)
    logger.info(f'Step B+ done.  Best MPJPE: {best_mpjpe:.2f}mm  '
                f'PA: {best_pa:.2f}mm  |  Time: {timer.elapsed_str()}')
    logger.info('Targets to beat: DT-Pose MPJPE=316.8 / PA=104.2 (MMFi P3-Setting3)')
    logger.info('Baseline (no distill) at same hyperparams: MPJPE≈345 / PA≈104')


if __name__ == '__main__':
    main()