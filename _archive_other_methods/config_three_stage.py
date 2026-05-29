"""
Reference hyperparameter configuration for the three-stage pipeline.

Copy the relevant blocks into your existing config.py / args parser.
"""

# ============================================================
# Stage 1A: MAE Pretraining
# ============================================================
STAGE_1A_CONFIG = {
    'epochs': 300,
    'batch_size': 4,
    'accum': 4,                  # effective batch = 16
    'lr': 1.5e-4,                # MAE-recommended LR for small batches
    'weight_decay': 0.05,
    'warmup_epochs': 20,
    'mask_ratio': 0.75,
    'patch_t': 3,
    'patch_s': 19,               # 114 / 19 = 6 patches along subcarrier
    'patch_a': 5,                # 10 / 5 = 2 patches along antenna
    # patches per sample = (30/3) * (114/19) * (10/5) = 10 * 6 * 2 = 120
    # masked per sample = 120 * 0.75 = 90 (these are reconstruction targets)
    'optimizer': 'AdamW',
    'betas': (0.9, 0.95),
    'grad_clip': 1.0,
    'save_every': 20,            # save checkpoint every 20 epochs
}

# ============================================================
# Stage 1B: Action Pretraining (optional)
# ============================================================
STAGE_1B_CONFIG = {
    'epochs': 50,
    'batch_size': 8,
    'accum': 2,
    'lr_backbone': 1e-4,         # MAE-pretrained, slow LR
    'lr_head': 5e-4,             # fresh action head, fast LR
    'weight_decay': 1e-4,
    'warmup_epochs': 3,
    'dropout': 0.1,
    'label_smoothing': 0.1,
    'n_classes': 27,
    'optimizer': 'AdamW',
    'betas': (0.9, 0.999),
    'grad_clip': 1.0,
}

# ============================================================
# Stage 2: Pose Fine-tuning
# ============================================================
STAGE_2_CONFIG = {
    'epochs': 50,
    'batch_size': 2,             # match your current setup
    'accum': 8,                  # effective batch = 16
    'lr_backbone': 1e-4,
    'lr_head': 5e-4,
    'weight_decay': 1e-3,        # match your Plan A
    'warmup_epochs': 3,
    'dropout': 0.3,              # match your Plan A
    # Loss weights — note lambda_hip downscaled from 1.0 to 0.3 (Plan A+B learned 1.0 was too high)
    'lambda_pose': 1.0,
    'lambda_bone': 0.05,
    'lambda_action': 0.1,
    'lambda_rsc': 0.5,
    'lambda_hip': 0.3,           # ⬇ from 1.0
    'lambda3': 2.0,
    # RSC
    'use_rsc': True,
    'rsc_drop_f': 0.33,
    'rsc_drop_b': 0.33,
    # Action dropout
    'action_dropout': 0.5,
    'optimizer': 'AdamW',
    'betas': (0.9, 0.999),
    'grad_clip': 1.0,
    'eval_every': 3,
}


# ============================================================
# Estimated wall-clock (per your hardware: ~12min/epoch as in previous runs)
# ============================================================
TIMING_ESTIMATES = {
    'Stage 1A (300 ep, bs=4, accum=4, no GCN/RSC, smaller model)': '~50 hours',
    'Stage 1B (50 ep, bs=8, accum=2, classifier only)':            '~10 hours',
    'Stage 2  (50 ep, bs=2, accum=8, full model)':                 '~10 hours',
    'TOTAL':                                                       '~70 hours',
}


# ============================================================
# HMSF parameters (used in all stages)
# ============================================================
HMSF_CONFIG = {
    'coarse_size': 1,            # global pool
    'medium_size': 2,            # 4 regions
    'fine_size': 4,              # 16 regions
    # adds ~0.16M params vs original single-scale pooling
}


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 1A:", STAGE_1A_CONFIG)
    print("=" * 60)
    print("Stage 1B:", STAGE_1B_CONFIG)
    print("=" * 60)
    print("Stage 2:", STAGE_2_CONFIG)
    print("=" * 60)
    print("Timing:", TIMING_ESTIMATES)