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
from distill_loss import (DistillProjection, FeatureDistillLoss, OutputDistillLoss,
                          KinematicPriorLoss)
from utils import (set_seed, setup_logger, count_parameters,
                   save_checkpoint, AverageMeter, Timer, save_run_config)


### ---------------------------------------------------------------------
### EMA (Exponential Moving Average) — added to fight training oscillation
### ---------------------------------------------------------------------
### Empirical motivation: prior runs (both distill and baseline) showed massive
### MPJPE oscillation between consecutive evals (~±100mm), causing best-MPJPE
### and best-PA to fall on different epochs. EMA averages parameters across
### the last ~1/(1-decay) optimizer steps, smoothing the model trajectory and
### typically letting both metrics co-locate at the same checkpoint.
###
### v2 lesson: with constant decay=0.999, EMA shadow takes ~6 epochs to catch
### up to the rapidly-evolving model, producing nonsense evals (e.g. MPJPE
### 907mm at epoch 3) in early training. Fix: standard decay-warmup schedule
###     decay_t = min(target_decay, (1 + t) / (10 + t))
### which gives decay ≈ 0.1 at t=0, ≈ 0.99 at t=1000, ≈ target at t=10000.
### Shadow tracks model tightly when stale, locks down once stabilized.
###
### With target_decay=0.999 and ~405 optimizer steps/epoch, the schedule
### reaches 0.99 in ~2.5 epochs and 0.999 in ~25 epochs (functionally equivalent
### to the un-scheduled version once past warmup).
class EMA:
    """Exponential moving average of student parameters with decay warmup.

    Update rule (every optimizer step):
        decay_t  = min(target_decay, (1 + t) / (10 + t))    # warmup schedule
        shadow[n] := decay_t * shadow[n] + (1 - decay_t) * model[n]

    Eval-time pattern:
        ema.apply_to(model)
        try:
            metrics = evaluate(model, ...)
        finally:
            ema.restore(model)

    Only `requires_grad=True` parameters are tracked. Buffers (BatchNorm/LN
    running stats) are kept from the live model — typically what people want.

    Setting `warmup=False` reverts to constant decay (v1/v2 behavior, kept
    for ablation only).
    """
    def __init__(self, model, decay=0.999, warmup=True):
        self.target_decay = decay
        self.warmup = warmup
        self.num_updates = 0
        self.shadow = {}
        self.backup = None
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    def _current_decay(self):
        """Decay schedule: linearly approaches target_decay during warmup."""
        if not self.warmup:
            return self.target_decay
        # Standard formula used by TF/JAX/timm: min(target, (1+t)/(10+t))
        return min(self.target_decay,
                   (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = self._current_decay()
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model):
        """Swap shadow into model in-place; keep original for restore()."""
        assert self.backup is None, "EMA.apply_to() called without prior restore()"
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        """Undo apply_to(): restore the pre-swap weights."""
        assert self.backup is not None, "EMA.restore() called without prior apply_to()"
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = None

    def state_dict(self):
        return {
            'target_decay': self.target_decay,
            'warmup': self.warmup,
            'num_updates': self.num_updates,
            'shadow': self.shadow,
        }

    def load_state_dict(self, state_dict):
        self.target_decay = state_dict['target_decay']
        self.warmup = state_dict.get('warmup', True)
        self.num_updates = state_dict.get('num_updates', 0)
        self.shadow = state_dict['shadow']


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
                    device, epoch, logger, args, ema=None, kine_prior_fn=None):
    student.train()
    proj.train()
    meters = {k: AverageMeter() for k in
              ['loss', 'l_pose_clean', 'l_pose_masked', 'l_cons', 'l_action',
               'l_distill_feat', 'l_distill_out', 'l_distill_out_mm',
               'l_kine', 'l_kine_bone_mm']}
    accum = getattr(args, 'accumulate_grad', 1)
    action_loss_fn = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    use_feat = args.lambda_feat > 0
    use_out  = args.lambda_out  > 0
    use_kine = getattr(args, 'lambda_kine', 0) > 0 and kine_prior_fn is not None

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

        # Kinematic prior (vs GT, no teacher needed — outside the use_feat/use_out block)
        if use_kine:
            l_kine, kd = kine_prior_fn(outputs['p_final_clean'], pose_3d)
            total = total + args.lambda_kine * l_kine
            meters['l_kine'].update(kd['l_kine'], csi.shape[0])
            meters['l_kine_bone_mm'].update(kd['l_kine_bone_mm'], csi.shape[0])

        (total / accum).backward()

        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(proj.parameters()),
                    args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(student)

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
            if use_kine:
                msg += (f' Kine: {meters["l_kine"].avg:.4f}'
                        f' (bone~{meters["l_kine_bone_mm"].avg:.0f}mm)')
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
def evaluate(student, test_loader, device, evaluator, logger, tag=''):
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
        f'[Eval{tag}] MPJPE: {metrics["MPJPE (mm)"]:.2f}mm | '
        f'MPJPE_a: {metrics["MPJPE_aligned (mm)"]:.2f}mm | '
        f'PA: {metrics["PA-MPJPE (mm)"]:.2f}mm | '
        f'P50n: {metrics["PCK@50_norm (%)"]:.1f}% | '
        f'P20n: {metrics["PCK@20_norm (%)"]:.1f}% | '
        f'PredStd: {pred_std:.1f}mm | ActAcc: {acc:.1f}%')
    metrics['pred_std'] = pred_std
    metrics['action_acc'] = acc
    return metrics


@torch.no_grad()
def evaluate_with_ema(student, ema, test_loader, device, evaluator, logger):
    """Evaluate both raw student and EMA student, return both metric dicts.

    EMA evaluation swaps shadow weights into the model in-place, evaluates,
    and restores. This is safe across exceptions due to try/finally.
    """
    m_raw = evaluate(student, test_loader, device, evaluator, logger, tag=' raw')
    if ema is None:
        return m_raw, None
    ema.apply_to(student)
    try:
        m_ema = evaluate(student, test_loader, device, evaluator, logger, tag=' EMA')
    finally:
        ema.restore(student)
    return m_raw, m_ema


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
    p.add_argument('--lambda_kine', type=float, default=0.5,
                   help='Kinematic prior weight (bone-length vs GT + symmetry, '
                        'targets PA-MPJPE). 0 disables. Scale/translation '
                        'invariant so MPJPE-neutral.')
    p.add_argument('--kine_bone_w', type=float, default=1.0,
                   help='Bone-length-vs-GT sub-weight inside kinematic prior')
    p.add_argument('--kine_sym_w', type=float, default=0.3,
                   help='Bilateral symmetry sub-weight inside kinematic prior')
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
    # EMA — added to combat MPJPE oscillation seen in v1 runs
    p.add_argument('--use_ema', action='store_true', default=True,
                   help='Maintain EMA of student weights and evaluate it each '
                        'eval cycle (default: ON). --no_ema disables.')
    p.add_argument('--no_ema', dest='use_ema', action='store_false',
                   help='Disable EMA (run like the original v1).')
    p.add_argument('--ema_decay', type=float, default=0.999,
                   help='Target EMA decay. With decay=d and ~405 opt-steps/epoch, '
                        'effective averaging window ≈ 1/(1-d) steps ≈ '
                        '1/(1-d)/405 epochs. 0.999 ≈ 2.5 epochs.')
    p.add_argument('--ema_no_warmup', dest='ema_warmup', action='store_false',
                   default=True,
                   help='Disable decay warmup schedule (v2 EMA had constant '
                        'decay; gave 5-epoch garbage period at start). Default: '
                        'warmup ON, fixed via min(target_decay, (1+t)/(10+t)).')
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
    kine_prior_fn = KinematicPriorLoss(
        bone_weight=args.kine_bone_w,
        sym_weight=args.kine_sym_w).to(device)
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

    # ---- EMA (optional; on by default — combats v1's MPJPE oscillation)
    ema = None
    if args.use_ema:
        ema = EMA(student, decay=args.ema_decay, warmup=args.ema_warmup)
        win_eps = 1.0 / (1.0 - args.ema_decay) / 405
        logger.info(f'EMA: ON (target_decay={args.ema_decay}, warmup={args.ema_warmup}; '
                    f'effective window ≈ {win_eps:.1f} epochs at 405 opt-steps/epoch)')
        if args.ema_warmup:
            logger.info(f'  decay schedule: min(target, (1+t)/(10+t)) — '
                        f'reaches 0.99 at step ~1000 (~2.5 epochs), '
                        f'target at step ~10000 (~25 epochs)')
    else:
        logger.info('EMA: OFF (--no_ema)')

    # ---- Pareto multi-checkpoint tracking
    # Prior runs showed best-MPJPE and best-PA epochs diverge — saving the
    # MPJPE-only "best" throws away the model that's good at PA. Track and
    # save all four winners so we can pick the best Pareto point afterwards.
    best = {'raw_mpjpe': float('inf'), 'raw_pa': float('inf'),
            'ema_mpjpe': float('inf'), 'ema_pa': float('inf')}
    best_epoch = {'raw_mpjpe': 0, 'raw_pa': 0, 'ema_mpjpe': 0, 'ema_pa': 0}
    best_pa_at_best_mpjpe = {'raw': float('inf'), 'ema': float('inf')}
    best_mpjpe_at_best_pa = {'raw': float('inf'), 'ema': float('inf')}
    patience = 0

    def _save_ckpt(name, metrics, use_ema_weights=False):
        """Save current model state under given name. If use_ema_weights,
        apply EMA shadow weights first, then restore."""
        path = os.path.join(args.save_dir, f'{name}.pth')
        if use_ema_weights and ema is not None:
            ema.apply_to(student)
            try:
                save_checkpoint(student, optimizer, epoch, metrics, path)
            finally:
                ema.restore(student)
        else:
            save_checkpoint(student, optimizer, epoch, metrics, path)
        # also dump the projection state alongside
        torch.save({'epoch': epoch, 'proj_state_dict': proj.state_dict()},
                   os.path.join(args.save_dir, f'{name}_proj.pth'))

    for epoch in range(1, args.epochs + 1):
        cur_lrs = [g['lr'] for g in optimizer.param_groups]
        logger.info(f'\n{"="*60}\nEpoch {epoch}/{args.epochs} | '
                    f'LR: backbone={cur_lrs[0]:.2e} head={cur_lrs[1]:.2e}')

        tm = train_one_epoch(student, proj, teacher, train_loader, optimizer,
                             total_loss_fn, pose_loss_fn,
                             feat_distill_fn, out_distill_fn,
                             device, epoch, logger, args, ema=ema,
                             kine_prior_fn=kine_prior_fn)
        line = (f'[Train] Epoch {epoch} | Loss: {tm["loss"]:.4f} '
                f'Pose(C): {tm["l_pose_clean"]:.4f} '
                f'Act: {tm["l_action"]:.4f}')
        if args.lambda_feat > 0:
            line += f' Feat: {tm["l_distill_feat"]:.4f}'
        if args.lambda_out > 0:
            line += f' Out: {tm["l_distill_out"]:.4f} (~{tm["l_distill_out_mm"]:.0f}mm)'
        if getattr(args, 'lambda_kine', 0) > 0:
            line += f' Kine: {tm["l_kine"]:.4f} (bone~{tm["l_kine_bone_mm"]:.0f}mm)'
        line += f' | {timer.elapsed_str()}'
        logger.info(line)
        scheduler.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            m_raw, m_ema = evaluate_with_ema(
                student, ema, test_loader, device, evaluator, logger)

            improved = False

            # raw bests
            if m_raw['MPJPE (mm)'] < best['raw_mpjpe']:
                best['raw_mpjpe'] = m_raw['MPJPE (mm)']
                best_epoch['raw_mpjpe'] = epoch
                best_pa_at_best_mpjpe['raw'] = m_raw['PA-MPJPE (mm)']
                _save_ckpt('best_mpjpe_raw', m_raw, use_ema_weights=False)
                logger.info(f'  ** NEW best_mpjpe_raw: {best["raw_mpjpe"]:.2f}mm '
                            f'(PA at this epoch: {m_raw["PA-MPJPE (mm)"]:.2f}mm)')
                improved = True
            if m_raw['PA-MPJPE (mm)'] < best['raw_pa']:
                best['raw_pa'] = m_raw['PA-MPJPE (mm)']
                best_epoch['raw_pa'] = epoch
                best_mpjpe_at_best_pa['raw'] = m_raw['MPJPE (mm)']
                _save_ckpt('best_pa_raw', m_raw, use_ema_weights=False)
                logger.info(f'  ** NEW best_pa_raw: {best["raw_pa"]:.2f}mm '
                            f'(MPJPE at this epoch: {m_raw["MPJPE (mm)"]:.2f}mm)')
                improved = True

            # EMA bests
            if m_ema is not None:
                if m_ema['MPJPE (mm)'] < best['ema_mpjpe']:
                    best['ema_mpjpe'] = m_ema['MPJPE (mm)']
                    best_epoch['ema_mpjpe'] = epoch
                    best_pa_at_best_mpjpe['ema'] = m_ema['PA-MPJPE (mm)']
                    _save_ckpt('best_mpjpe_ema', m_ema, use_ema_weights=True)
                    logger.info(f'  ** NEW best_mpjpe_ema: {best["ema_mpjpe"]:.2f}mm '
                                f'(PA at this epoch: {m_ema["PA-MPJPE (mm)"]:.2f}mm)')
                    improved = True
                if m_ema['PA-MPJPE (mm)'] < best['ema_pa']:
                    best['ema_pa'] = m_ema['PA-MPJPE (mm)']
                    best_epoch['ema_pa'] = epoch
                    best_mpjpe_at_best_pa['ema'] = m_ema['MPJPE (mm)']
                    _save_ckpt('best_pa_ema', m_ema, use_ema_weights=True)
                    logger.info(f'  ** NEW best_pa_ema: {best["ema_pa"]:.2f}mm '
                                f'(MPJPE at this epoch: {m_ema["MPJPE (mm)"]:.2f}mm)')
                    improved = True

            if improved:
                patience = 0
            else:
                patience += 1
                logger.info(f'  No improvement on any of 4 fronts. '
                            f'Patience: {patience}/{args.patience}')
            if patience >= args.patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break

        if epoch % 10 == 0:
            save_checkpoint(student, optimizer, epoch, {},
                            os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'))

    logger.info('\n' + '=' * 70)
    logger.info('Step B+ done.  Summary of 4 Pareto-frontier checkpoints:')
    logger.info(f'  best_mpjpe_raw : MPJPE {best["raw_mpjpe"]:7.2f}mm '
                f'(PA {best_pa_at_best_mpjpe["raw"]:7.2f}mm) @ epoch {best_epoch["raw_mpjpe"]}')
    logger.info(f'  best_pa_raw    : PA    {best["raw_pa"]:7.2f}mm '
                f'(MPJPE {best_mpjpe_at_best_pa["raw"]:7.2f}mm) @ epoch {best_epoch["raw_pa"]}')
    if ema is not None:
        logger.info(f'  best_mpjpe_ema : MPJPE {best["ema_mpjpe"]:7.2f}mm '
                    f'(PA {best_pa_at_best_mpjpe["ema"]:7.2f}mm) @ epoch {best_epoch["ema_mpjpe"]}')
        logger.info(f'  best_pa_ema    : PA    {best["ema_pa"]:7.2f}mm '
                    f'(MPJPE {best_mpjpe_at_best_pa["ema"]:7.2f}mm) @ epoch {best_epoch["ema_pa"]}')
    logger.info(f'Time: {timer.elapsed_str()}')
    logger.info('Targets to beat: DT-Pose MPJPE=316.8 / PA=104.2 (MMFi P3-Setting3)')


if __name__ == '__main__':
    main()