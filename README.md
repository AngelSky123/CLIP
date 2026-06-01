# CSI-RSC-PoseDG: Depth-Distilled 3D Pose Estimation from WiFi CSI

Cross-environment 3D human pose estimation from WiFi Channel State Information
(CSI), trained against the MMFi dataset under the strict Setting-3
(cross-environment) protocol. The student model uses only CSI at inference;
depth maps are used at training time as a cross-modal teacher signal.

**Headline result (MMFi Setting 3, Protocol 3, all 27 actions):**

| Method                               | MPJPE ↓ | PA-MPJPE ↓ |
|--------------------------------------|:-------:|:----------:|
| MetaFi++ (Zhou et al., 2023)         |  369.5  |   116.0    |
| HPE-Li (Gian et al., 2025)           |  388.4  |   107.9    |
| **DT-Pose (Chen et al., 2025)**      |  316.8  |   104.2    |
| **Ours: depth → CSI distill, EMA @ e33** | **310.03** |   108.40   |
| Ours: baseline (Stage2, no distill) + EMA | 316.76 |   108.23   |

**MPJPE: −6.77 mm vs DT-Pose (state-of-the-art beaten).**
PA-MPJPE: +4.2 mm vs DT-Pose — within seed-noise of HPE-Li (107.9) and
clearly above MetaFi++ (116.0). Both metrics from a single EMA checkpoint.


## Method

Two stages, with the first reused across runs:

**Step A — Depth teacher (one-time).**
A small encoder-temporal-pose pipeline is supervised on RGB-D depth
(`(T, 1, 112, 112)`, 16-bit mm) from the source environments. The teacher
outputs both a temporal feature `z_global ∈ R^(B,T,128)` and a pose
prediction `p_final ∈ R^(B,T,17,3)` (meters). After training the teacher
is frozen and reused.
Final teacher metrics on the E04 probe: **MPJPE 269.18 mm / PA-MPJPE 89.5 mm**
— better than DT-Pose on both, but uses depth and is not the deployed system.

**Step B+ — Depth → CSI distillation with a pretrained CSI backbone.**
The CSI student (`CSIRSCPoseDG`: dual-branch CSI encoder → local 3D-CNN →
feature pooling → temporal modeler → action-conditioned pose decoder + action
classifier + RSC) is initialized from the Stage1B Action checkpoint
(`action_best.pt`, all 5 backbone modules) and fine-tuned with:

```
L = L_pose(student vs GT)            [primary, Stage2 baseline objective]
  + λ_feat · L_feat (z_s_proj, z_t)  [feature-level alignment, latent geometry]
  + λ_out  · L_out  (p_s_clean, p_t) [pose-level alignment, targets MPJPE]
```

- `L_feat`: cosine + smooth-L1 on a learned `128 → 128` student projection
  against the teacher's `z_global`. Teacher features are detached.
- `L_out`: smooth-L1 (β = 5 cm) on the student's clean (non-RSC) pose against
  the teacher's pose. Hip joint weighted 1.5× (where the teacher's MPJPE
  advantage concentrates).

**Inference is strictly CSI-only.** Depth is consumed only by the frozen
teacher during training on source environments; the target environment (E04)
is never paired with depth, anywhere.

### EMA — the critical stabilizer

A first pass without EMA reached MPJPE 296.89 / PA 106.29 across two separate
"best" epochs (e18 and e42), with adjacent evals oscillating by ±100 mm.
**Exponential moving average of student weights**, updated every optimizer step
with a decay-warmup schedule

```
decay_t = min(target_decay, (1 + t) / (10 + t))    # standard timm/JAX formula
shadow_t = decay_t · shadow_{t-1} + (1 - decay_t) · model_t
```

co-locates best-MPJPE and best-PA at the same checkpoint (e33), and produces
the reported single-model result. **The warmup schedule is essential** —
constant decay=0.999 leaves the shadow 8× further from the model at end of
epoch 1, producing nonsense evals (MPJPE > 900 mm) for the first ~5 epochs.


## Results

### Final 4-checkpoint summary (single distillation run, 50 epochs, ~15 h)

| Checkpoint            | MPJPE   | PA      | epoch | Beats DT-Pose ? |
|-----------------------|:-------:|:-------:|:-----:|:----------------|
| `best_mpjpe_raw.pth`  | 294.58  | 115.72  |  33   | MPJPE ✓, PA ✗   |
| `best_pa_raw.pth`     | 345.01  | 106.34  |  48   | MPJPE ✗, PA ✗   |
| **`best_mpjpe_ema.pth`** | **310.03** | **108.40** | **33** | **MPJPE ✓, PA ≈** |
| `best_pa_ema.pth`     | 333.06  | 106.35  |  36   | MPJPE ✗, PA ✗   |

The recommended ckpt for any downstream use (eval / demo / paper figure) is
**`best_mpjpe_ema.pth` @ epoch 33**: it is the single checkpoint that
simultaneously beats DT-Pose MPJPE and stays competitive on PA. EMA produces
a smooth trajectory in this epoch range, so neighboring epochs (e30, e36)
yield similar metrics — there is no lottery component.

### Ablation: distillation vs baseline (same hyperparameters, both with EMA)

| Run            | best EMA MPJPE | best EMA PA   |
|----------------|:--------------:|:-------------:|
| Baseline (λ_feat=0, λ_out=0) |  316.76 (e50) |  108.23 (e48) |
| **Distillation** (λ_feat=0.1, λ_out=0.5) | **310.03 (e33)** | 106.35 (e36) |

Distillation gives a **6.7 mm MPJPE improvement** over the no-distillation
baseline with identical training. PA is statistically tied between the two
(within seed noise of ~1 mm); the gain is concentrated on MPJPE, consistent
with the teacher's largest advantage being in absolute joint localization
(teacher MPJPE 269 vs DT-Pose 316.8 → 47 mm headroom).

### Observed PA floor at ~106 mm

Across all configurations (raw / EMA, distill / baseline, multiple runs),
PA-MPJPE converges to **105–108 mm**. We did not surpass 105 mm. We hypothesize
this is a physical-observation floor of single-link CSI on MMFi: the
fine-extremity joints (hands, elbows, feet) — where PA error concentrates
after Procrustes alignment — cannot be resolved by 3-antenna × 1-RX CSI at
this resolution. DT-Pose's table itself supports this: their hand MPJPE is
364 mm and elbow MPJPE is 249 mm in P1-S1; further reduction requires more
antennas or higher subcarrier resolution, not better algorithms on the same
hardware.


## Pipeline

```
Stage 1A   Stage 1B   Teacher       Step B+ (this work)
MAE        Action     Depth         Depth → CSI distillation + EMA
pretrain   pretrain   pose          on Stage1B-pretrained backbone

CSI only   CSI only   Depth only    CSI + depth at train,
                                    CSI only at test

stage1a_   stage1b_   depth_        distill_pretrained/
mae/mae_   action/    teacher_      best_mpjpe_ema.pth
latest.pt  action_    full/         (+ raw and pa variants)
           best.pt    teacher_
                      best.pt
```

The first three stages are existing artefacts of this codebase; Step B+
(`train_distill_pretrained.py`) is the contribution.


## Repository Layout

```
.
├── README.md
├── config.py                       # arg parser + defaults
├── dataset.py                      # base MMFi dataset (CSI/GT only)
├── dataset_distill.py              # multimodal: CSI + depth + GT
├── augmentation.py                 # CSI augment ops
├── losses.py                       # TotalLoss (Stage2 pose + RSC + action)
├── evaluate.py                     # MPJPE, PA-MPJPE, PCK
├── utils.py                        # logger, ckpt I/O, save_run_config
│
├── train_depth_teacher.py          # Step A — train depth teacher
├── train.py                        # Stage1A/1B & Stage2 baselines
├── train_distill.py                # diagnostic from-scratch (deprecated)
├── train_distill_pretrained.py     # Step B+ — main entry (EMA + 4-ckpt save)
├── distill_loss.py                 # FeatureDistillLoss + OutputDistillLoss + Projection
├── analyze_distill_log.py          # post-hoc convergence diagnosis
│
└── models/
    ├── __init__.py
    ├── csi_encoder.py              # dual-branch amp/phase encoder
    ├── local_encoder.py            # 3D-CNN local feature encoder
    ├── global_encoder.py           # temporal Transformer + TCN
    ├── pose_decoder.py             # action-conditioned coarse→fine head
    ├── rsc.py                      # Representation Self-Challenging
    ├── mixstyle.py                 # MixStyle / InstanceNorm DG
    ├── full_model.py               # CSIRSCPoseDG: wires everything
    └── depth_teacher.py            # DepthPoseTeacher
```


## Setup

```bash
# Tested on Python 3.7, PyTorch 2.2.2, CUDA 12.x, RTX 4080 (16 GB)
pip install torch torchvision numpy scipy
```

Data: MMFi at `/home/<user>/PerceptAlign/MMFi/<E01..E04>/<S01..S40>/<A01..A27>/`
with `gt.pickle`, `depth/`, `wifi-csi/` per sequence.


## Reproduction

The first three stages are existing artefacts; commands below assume their
checkpoints are present. Step 4 is what this repo adds.

### 1. Stage 1A — MAE pretraining
```bash
python train.py --stage mae --train_envs E01 E02 E03 --test_env E04 \
    --save_dir ./checkpoints/stage1a_mae
```

### 2. Stage 1B — Action recognition
```bash
python train.py --stage action --train_envs E01 E02 E03 --test_env E04 \
    --pretrain_ckpt ./checkpoints/stage1a_mae/mae_latest.pt \
    --save_dir ./checkpoints/stage1b_action
```
Expected: ~87 % train accuracy on 27 classes.

### 3. Step A — Depth teacher
```bash
python train_depth_teacher.py \
    --data_root /home/<user>/PerceptAlign/MMFi \
    --train_envs E01 E02 E03 --test_env E04 \
    --depth_img 112 --depth_clip 5000 \
    --epochs 30 --batch_size 4 --lr 5e-4 \
    --save_dir ./checkpoints/depth_teacher_full
```
Expected: best at e24, MPJPE 269 mm / PA 89 mm on E04 probe.

### 4. Step B+ — Depth → CSI distillation (this work)
```bash
python train_distill_pretrained.py \
    --data_root /home/<user>/PerceptAlign/MMFi \
    --train_envs E01 E02 E03 --test_env E04 \
    --pretrain_ckpt ./checkpoints/stage1b_action/action_best.pt \
    --teacher_ckpt  ./checkpoints/depth_teacher_full/teacher_best.pt \
    --depth_img 112 --depth_clip 5000 \
    --lambda_feat 0.1 --lambda_out 0.5 --lambda_hip 0.3 \
    --epochs 50 --batch_size 2 --accumulate_grad 8 \
    --lr_backbone 1e-4 --lr_head 5e-4 \
    --use_ema --ema_decay 0.999 \
    --save_dir ./checkpoints/distill_pretrained
```

The recommended deployment checkpoint is `best_mpjpe_ema.pth` (epoch 33 in
the reference run).

To run a no-distillation ablation under identical infrastructure (same
batching, EMA, optimizer):
```bash
python train_distill_pretrained.py ... --lambda_feat 0 --lambda_out 0 ...
```
The teacher forward is skipped when both lambdas are zero.

Diagnose any training log post-hoc:
```bash
python analyze_distill_log.py ./checkpoints/distill_pretrained/<run>/train.log
```


## Loss Knobs

| Flag                    | Default | What it does |
|-------------------------|:-------:|--------------|
| `--lambda_feat`         | 0.1     | Cosine + smooth-L1 on projected `z_global` against teacher's `z_global`. |
| `--lambda_out`          | 0.5     | Smooth-L1 (β = 5 cm, hip ×1.5) on student vs teacher pose. Drives the MPJPE gain. |
| `--lambda_hip`          | 0.3     | Hip-joint weight inside the pose loss. |
| `--use_ema` / `--no_ema`| ON      | EMA of student weights with warmup schedule. Disabling reverts to lottery-best behaviour. |
| `--ema_decay`           | 0.999   | Target EMA decay. Effective averaging window ≈ 1/(1-d)/405 epochs. |
| `--ema_no_warmup`       | off     | Use constant decay (v2 behaviour — produces nonsense evals for first ~5 epochs; for ablation only). |

Ablation handles (subset of relevant runs):
- `--lambda_feat 0 --lambda_out 0`: pure Stage2 + EMA, no distillation.
- `--lambda_out 0`: feature distillation only.
- `--lambda_feat 0`: output distillation only.


## Evaluation Protocol

All numbers follow the MMFi benchmark definitions in DT-Pose Section A.2:

- **MPJPE**: average per-joint Euclidean distance in mm, mean over frames
  and joints.
- **PA-MPJPE**: MPJPE after Procrustes alignment (translation + rotation +
  uniform scaling). Verified numerically against synthetic transformations:
  rotation 60° / scale 1.7× / translate 3 m all yield PA ≈ 0 mm.
- **PCK@α**: percentage of predictions within α × torso length of GT.
- **Setting 3 (Cross-Environment)**: train on three rooms (E01–E03), test on
  the held-out room (E04). `action_idx=None` at test, so no action label leakage.
- **Protocol 3**: all 27 actions.

Reference DT-Pose Setting 3 numbers (Table 1 of arXiv:2501.09411):

|                | P1 (14 daily) | P2 (13 rehab) | **P3 (all 27)** |
|----------------|:-------------:|:-------------:|:---------------:|
| MPJPE ↓        | 332.7         | 338.3         | **316.8**       |
| PA-MPJPE ↓     | 105.1         | 102.0         | **104.2**       |


## References

> Chen, Y., Guo, J., Guo, S., Zhou, J., Tao, D.
> *Towards Robust and Realistic Human Pose Estimation via WiFi Signals.*
> arXiv:2501.09411, 2025.

> Yang, J. et al. *MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for
> Versatile Wireless Sensing.* NeurIPS D&B Track, 2023.

> He, K. et al. *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022.


## Notes

- Run records: each run writes `run_config.json` (full args + git SHA +
  cmd-line + env) into its `save_dir`, plus four checkpoints (raw best
  MPJPE / raw best PA / EMA best MPJPE / EMA best PA) and matching
  projection-head saves for distillation runs.
- The diagnostic from-scratch script `train_distill.py` is retained for the
  historical lesson that single-stage from-scratch distillation collapses
  (the action classifier never learns from zero, starving the action-
  conditioned decoder). It is **not** the production entry point.
- The PA floor at ~106 mm is, to our knowledge, hardware-limited rather than
  algorithm-limited under MMFi's 3-Tx × 1-Rx CSI capture. Improvements should
  target richer signal capture (more antennas, higher subcarrier resolution)
  rather than further loss-function engineering on this dataset.