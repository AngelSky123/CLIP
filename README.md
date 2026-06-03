# CSI-RSC-PoseDG: Depth-Distilled 3D Pose Estimation from WiFi CSI

Cross-environment 3D human pose estimation from WiFi Channel State Information
(CSI), evaluated on MMFi under the strict **Setting 3 (Cross-Environment) ×
Protocol 3 (all 27 actions)** protocol. The student model uses **only CSI at
inference**; depth maps are used solely at training time as a cross-modal
teacher signal.

> **Evaluation honesty note.** All numbers below use a **frame-faithful
> evaluation** that matches DT-Pose's protocol exactly (every frame of every
> E04 sequence scored once, no window padding, `action_idx=None`). An earlier
> sliding-window evaluation under-counted frames and produced optimistic numbers
> (~310 mm); those have been retracted. See *Evaluation Protocol* below.


## Headline result (MMFi Setting 3, Protocol 3, frame-faithful)

| Method                                  | MPJPE ↓ | PA-MPJPE ↓ |
|-----------------------------------------|:-------:|:----------:|
| MetaFi++ (Zhou et al., 2023)            |  369.5  |   116.0    |
| HPE-Li (Gian et al., 2025)              |  388.4  |   107.9    |
| **DT-Pose (Chen et al., 2025)**         | **316.8** | **104.2** |
| Ours (depth→CSI distill, EMA @ e48)     |  366.6  |   106.2    |

**Honest standing:**
- **PA-MPJPE 106.2 ≈ DT-Pose 104.2** — pose *structure* is essentially on par
  (within ~2 mm).
- **MPJPE 366.6 vs DT-Pose 316.8** — we trail by ~50 mm. The entire gap is in
  **global localization** (root/hip placement), not skeletal structure.

This is the current state, reported faithfully. We do **not** claim to beat
DT-Pose on MPJPE.


## Where the error lives (diagnosis)

Decomposing the E04 result of the deployed checkpoint:

| Component                          | Value    |
|------------------------------------|:--------:|
| MPJPE (absolute)                   | 366.6 mm |
| MPJPE_aligned (hip-aligned, structure only) | 127.9 mm |
| **hip_error (global root)**        | **337.7 mm** |
| PA-MPJPE (Procrustes)              | 106.2 mm |

Two facts drive everything:

1. **Structure is fine.** MPJPE_aligned on the unseen room (E04, 127.9 mm) is
   nearly identical to held-out in-domain val (~122 mm). Procrustes PA (106.2)
   is within ~2 mm of DT-Pose. The CSI encoder, temporal modeler, and pose loss
   are **not** the bottleneck.
2. **Global localization is the whole gap.** `hip_error` of 337.7 mm accounts
   for almost the entire MPJPE shortfall vs DT-Pose. Locating a person's
   absolute position in an unseen room from single-link (3 Tx × 1 Rx) CSI is the
   hard, partly physical limitation here — the depth teacher itself has
   E04 hip_error ≈ 236 mm, so the geometry is hard even with depth.

**Implication:** further loss-tuning of `lambda_out` / `hip_weight` on top of
teacher distillation does not move localization (a run with hip distill weight
4.0 moved E04 hip_error by only ~3 mm over 50 epochs), because the teacher's own
localization is weak. Localization must be supervised directly against GT and
**decoupled** from skeletal structure — see *Root-decoupled decoder*.


## Method

Two stages, with the first reused across runs:

**Step A — Depth teacher (one-time).**
A small encoder-temporal-pose pipeline is supervised on RGB-D depth
(`(T, 1, 112, 112)`, 16-bit mm) from the source environments. Outputs a temporal
feature `z_global ∈ R^(B,T,128)` and pose `p_final ∈ R^(B,T,17,3)` (meters).
Frozen after training. Teacher E04 probe: **MPJPE 269 / PA 89.5** — better than
DT-Pose on this probe, but it uses depth and is not the deployed system, and its
hip_error (~236 mm) shows localization is hard even with depth.

**Step B+ — Depth → CSI distillation on a pretrained CSI backbone.**
The CSI student (`CSIRSCPoseDG`: dual-branch CSI encoder → local 3D-CNN →
feature pooling → temporal modeler → action-conditioned pose decoder + action
classifier + RSC) is initialized from the Stage1B Action checkpoint
(`action_best.pt`) and fine-tuned with:

```
L = L_pose(student vs GT)            [primary]
  + λ_feat · L_feat (z_s_proj, z_t)  [feature-level alignment]
  + λ_out  · L_out  (p_s_clean, p_t) [pose-level structural alignment]
```

- `L_feat`: cosine + smooth-L1 on a learned 128→128 student projection vs the
  teacher's `z_global` (teacher detached).
- `L_out`: smooth-L1 (β = 5 cm) on the student's clean pose vs teacher pose.
  **Now configured for structure only** (`hip_weight = 1.0`); global hip is
  supervised by GT, not by the teacher.

**Inference is strictly CSI-only.** Depth is consumed only by the frozen teacher
during training on source environments; E04 is never paired with depth.

### EMA stabilizer

Exponential moving average of student weights with a decay-warmup schedule:
```
decay_t  = min(target_decay, (1 + t) / (10 + t))
shadow_t = decay_t · shadow_{t-1} + (1 - decay_t) · model_t
```
The warmup is essential — constant decay=0.999 produces nonsense evals
(MPJPE > 900 mm) for the first ~5 epochs.


## Root-decoupled decoder (current improvement, in progress)

Motivated by the diagnosis (all gap = global localization), the pose decoder is
being restructured to **decouple** the two tasks:

- **PoseRelHead** — root-relative skeleton (17×3 with root forced to origin),
  action-conditioned; reuses the structure path that already works.
- **RootHead** — a separate head regressing the global hip xyz, **supervised by
  GT only (no teacher distillation, since the teacher's hip is itself biased)**,
  with light temporal smoothing to suppress per-frame jitter.

Final pose `= pose_rel + root_xyz`. Interface is identical to the original
decoder, so the rest of the pipeline (forward / RSC / distillation) is unchanged.
See `root_decoupled_decoder.py`.

> **Status: training.** Results for this variant are not yet in. The expected
> realistic gain is to pull MPJPE from ~366 toward ~330–345 by reducing hip
> jitter and unblocking structure from translation. It is **not** expected to
> close the full gap to 316.8 — cross-environment absolute localization from
> single-link CSI is a partly physical limit, not a decoder-design problem. This
> section will be updated with frame-faithful numbers once the run completes.


## Repository Layout

```
.
├── README.md
├── config.py
├── dataset.py / dataset_distill.py
├── augmentation.py
├── losses.py
├── distill_loss.py
├── evaluate.py                     # MPJPE / PA-MPJPE / PCK (DT-Pose-aligned formulas)
├── evaluate_v2.py                  # + hip_error, variance tools (online monitor)
├── eval_dtpose_faithful.py         # frame-faithful final eval (matches DT-Pose protocol)
├── root_decoupled_decoder.py       # Root/Pose decoupled decoder (current improvement)
├── utils.py
│
├── train_depth_teacher.py          # Step A
├── train.py                        # Stage1A/1B & Stage2 baselines
├── train_distill_pretrained.py     # Step B+ — main entry (EMA + 4-ckpt save)
│
└── models/
    ├── csi_encoder.py / local_encoder.py / global_encoder.py
    ├── pose_decoder.py / rsc.py / mixstyle.py
    ├── full_model.py               # CSIRSCPoseDG
    └── depth_teacher.py
```


## Setup

```bash
# Tested on Python 3.7, PyTorch 2.2.2, CUDA 12.x, RTX 4080 (16 GB)
pip install torch torchvision numpy scipy
```

Data: MMFi at `/home/<user>/PerceptAlign/MMFi/<E01..E04>/<S01..S40>/<A01..A27>/`
with `ground_truth.npy`, `depth/`, `wifi-csi/` per sequence.


## Reproduction

### Step B+ — Depth → CSI distillation (root-decoupled, current config)
```bash
python train_distill_pretrained.py \
    --data_root /home/<user>/PerceptAlign/MMFi \
    --train_envs E01 E02 E03 --test_env E04 \
    --pretrain_ckpt checkpoints/stage1b_action/action_best.pt \
    --teacher_ckpt  checkpoints/depth_teacher_full/teacher_best.pt \
    --depth_img 112 --depth_clip 5000 \
    --lambda_feat 0.1 --lambda_out 0.5 \
    --out_distill_hip_weight 1.0 --lambda_hip 1.0 --gamma 0.01 \
    --val_ratio 0.15 \
    --epochs 50 --batch_size 2 --accumulate_grad 8 \
    --lr_backbone 1e-4 --lr_head 5e-4 \
    --use_ema --ema_decay 0.999 \
    --save_dir ./checkpoints/distill_rootdecoupled
```
(Requires swapping `self.pose_decoder` to `RootDecoupledPoseDecoder` in
`models/full_model.py`; keep the `ActionClassifier` import.)

### Final evaluation (frame-faithful, the only number to report)
```bash
python eval_dtpose_faithful.py \
    --data_root /home/<user>/PerceptAlign/MMFi \
    --ckpt ./checkpoints/distill_rootdecoupled/best_mpjpe_ema.pth \
    --test_env E04 --seq_len 64 --variance
```


## Evaluation Protocol

Numbers follow the MMFi benchmark definitions used by DT-Pose (Section A.2).

- **MPJPE / PA-MPJPE / MPJPE_aligned**: implemented in `evaluate.py`, verified
  numerically equivalent to DT-Pose's `calulate_error` /
  `compute_similarity_transform`.
- **Frame-faithful protocol** (`eval_dtpose_faithful.py`): each frame of every
  E04 sequence is predicted **exactly once** (non-overlapping windows + a tail
  window covering only uncovered frames), **no edge-padding**, `action_idx=None`
  (no test-time label leakage). All frames pooled, metric computed once — this
  matches DT-Pose's per-frame, equal-weight averaging.
  - Difference from DT-Pose that **remains by design**: our model consumes 64
    frames of temporal context per prediction; DT-Pose is per-frame. This is a
    method-level difference and is reported as such.
- **Variance floor**: `multi_stride_variance` quantifies evaluation-protocol
  noise. For the deployed checkpoint, σ(MPJPE) = 0.03 mm — i.e. the gap to
  DT-Pose is real, not measurement noise.
- **Setting 3 / Protocol 3**: train E01–E03 (S01–S30), test E04 (S31–S40), all
  27 actions, `action_idx=None` at test.

Reference DT-Pose Setting 3 numbers (Table 1, arXiv:2501.09411):

|                | P1 (14) | P2 (13) | **P3 (all 27)** |
|----------------|:-------:|:-------:|:---------------:|
| MPJPE ↓        | 332.7   | 338.3   | **316.8**       |
| PA-MPJPE ↓     | 105.1   | 102.0   | **104.2**       |


## Known limits

- **PA-MPJPE floor ~106 mm.** Across all configurations PA converges to
  105–108 mm. This is hypothesized to be a hardware limit of single-link
  (3 Tx × 1 Rx) CSI on MMFi: fine-extremity joints (hands, elbows, feet) — where
  PA error concentrates after Procrustes — are not resolvable at this antenna /
  subcarrier resolution. Not worth further loss engineering.
- **Global localization is the open problem.** ~50 mm of the MPJPE gap to
  DT-Pose is hip placement in unseen rooms. This is the active research target;
  it is partly physical (single-link CSI) and not closable by decoder tweaks
  alone.


## References

> Chen, Y., Guo, J., Guo, S., Zhou, J., Tao, D. *Towards Robust and Realistic
> Human Pose Estimation via WiFi Signals.* arXiv:2501.09411, 2025.

> Yang, J. et al. *MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for
> Versatile Wireless Sensing.* NeurIPS D&B Track, 2023.

> He, K. et al. *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022.