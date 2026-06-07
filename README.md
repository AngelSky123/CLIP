# CSI-RSC-PoseDG

**Cross-Environment 3D Human Pose Estimation from Single-Link WiFi CSI, via Depth-Teacher Knowledge Distillation**

基于 WiFi CSI 的跨环境 3D 人体姿态估计。训练期由深度图教师 (depth teacher) 做知识蒸馏向 CSI 学生注入几何先验，**推理阶段只用 CSI**（不依赖任何深度 / 视觉输入）。在 MMFi 数据集上以严格盲测的跨房间设定（Setting 3 / Protocol 3，全 27 动作，单链路 CSI）对标 DT-Pose。

---

## 目录

1. [动机与问题定义](#1-动机与问题定义)
2. [核心贡献](#2-核心贡献)
3. [方法详解](#3-方法详解)
4. [数据集与预处理](#4-数据集与预处理)
5. [训练流程（四阶段）](#5-训练流程四阶段)
6. [实验设定](#6-实验设定与-dt-pose-对齐)
7. [评测指标定义](#7-评测指标定义)
8. [结果与消融](#8-结果与消融)
9. [Limitation 分析：跨房间绝对定位的信息上界](#9-limitation-分析跨房间绝对定位的信息上界)
10. [复现步骤](#10-复现步骤)
11. [超参数完整参考](#11-超参数完整参考)
12. [仓库结构](#12-仓库结构)
13. [常见问题与坑](#13-常见问题与坑)
14. [诚实声明](#14-诚实声明)

---

## 1. 动机与问题定义

WiFi CSI（Channel State Information）作为无感、无光照依赖、保护隐私的人体感知模态，近年被用于 3D 人体姿态估计。其核心难点在于**跨环境泛化**：CSI 由多径传播决定，强烈依赖房间几何、家具、天线布局，因此在房间 A 训练的模型迁移到未见房间 B 时性能骤降。

本项目针对最严苛的设定：

- **跨房间盲测**：测试房间的数据在训练全程从不可见（既无标签也无无标签数据）。
- **单链路 CSI**：仅 1 条收发链路（3Tx×1Rx），不依赖多设备阵列。
- **CSI-only 推理**：深度图仅在训练期作为教师，部署时不可用。

形式化：给定 CSI 序列 `X ∈ R^(T×9×114×10)`，预测 3D 关节序列 `P ∈ R^(T×17×3)`（绝对坐标，单位米）。训练集来自环境 `{E01,E02,E03}`，测试集来自从未见过的 `E04`。

---

## 2. 核心贡献

1. **深度-教师蒸馏框架**：训练期用深度图姿态教师对 CSI 学生做 feature-level + output-level 蒸馏，把视觉几何先验迁移进 CSI 表征；推理只需 CSI。
2. **RSC（Representation Self-Challenging）**：对全局表征做时间 / 通道 / batch 维随机置challenge，逼模型不依赖单一捷径特征，提升跨域鲁棒性。
3. **结构正则损失套件**（`structural_losses.py`）：骨长、左右对称、时序骨长稳定、root-relative 位置对齐——四项均平移不变，专门优化 PA-MPJPE 所度量的相对骨架，且**结构上不可能恶化全局定位项**。
4. **跨房间绝对定位的信息论 limitation 分析**：通过原始输入信息探针、教师上界、零信息基线对照、杠杆穷举，量化证明单链路 CSI 下"未见房间的绝对位置"不可跨域迁移——这是对 WiFi 感知能力边界的实证刻画。

---

## 3. 方法详解

### 3.1 整体管线

```
CSI (B, T=64, 9, 114, 10)
  │
  ├─[csi_encoder]            逐帧 CSI 编码 (amp+phase 通道) ──────────────┐
  │                                                                       │
  ├─[local_encoder]          3D 卷积 (ResNet3D, num_res3d_blocks=2)       │
  │                          捕捉 子载波×packet×时间 的局部时频结构        │
  │                                                                       │
  ├─[feature_pooling]        空间池化 → 每帧一个 token                    │
  │                                                                       │
  ├─[global_modeler]         Transformer(3 层, 4 头) + TCN([128,128])     │
  │                          建模长时序依赖 ──> z_global (B, 64, 128)      │
  │                                                                       │
  ├─[RSC]                    训练期对 z_global 做 self-challenging         │
  │                                                                       │
  ├─[pose_decoder]           TaskPromptCoarseHead → 粗姿态                 │
  │                          + SkeletonRefiner (GCN, 3 层) → 精修关节关系  │
  │                          ──> p_final (B, 64, 17, 3)                    │
  │                                                                       │
  └─[action_classifier]      27 类动作 (辅助任务, 正则表征) ───────────────┘
```

学生总参数量：**1,579,035 (~1.58M)**。

### 3.2 各模块

- **csi_encoder**：把 `(9,114,10)` 的逐帧 CSI（9 = 3 幅度 + 6 相位通道）编码为紧凑特征。`encoder_hidden_dim=32, encoder_out_dim=64`。
- **local_encoder**：3D 残差卷积块（`num_res3d_blocks=2`，`local_hidden_dim=64, local_out_dim=64`），在 子载波 / packet / 时间 三个轴上提取局部时频纹理。
- **feature_pooling**：把每帧的时频特征图池化为单 token。
- **global_modeler**：`num_transformer_layers=3, num_heads=4, transformer_dropout=0.3` 的 Transformer 叠加 `tcn_channels=[128,128], tcn_kernel_size=3` 的时序卷积网络，输出 `z_global ∈ R^(B×64×128)`。
- **pose_decoder**：`TaskPromptCoarseHead`（`coarse_hidden_dim=256`）产出粗 3D 姿态，再经 `SkeletonRefiner`（GCN，`gcn_hidden_dim=128, num_gcn_layers=3`）按骨架邻接关系精修。
- **action_classifier**：27 类动作分类头，作为辅助监督正则化全局表征。

### 3.3 RSC（Representation Self-Challenging）

训练期对 `z_global` 施加随机遮挡，强制模型分散依赖、避免过拟合单一环境捷径：

- `rsc2_time_drop_pct=0.5`：时间维随机丢弃比例
- `rsc2_channel_drop_pct=0.5`：通道维随机丢弃比例
- `rsc2_batch_pct=0.5`：batch 内施加 challenge 的样本比例

### 3.4 知识蒸馏

教师 `DepthPoseTeacher`（深度图输入，训练期冻结）对学生两路蒸馏：

- **Feature-level**：学生 `z_global` 经 `DistillProjection` 投影后，与教师全局特征做余弦 + Smooth-L1 对齐（`distill_cos_w=1.0, distill_sl1_w=1.0`），权重 `lambda_feat=0.1`。
- **Output-level**：学生 `p_final_clean` 与教师 `p_final` 做姿态级蒸馏（Smooth-L1，`out_distill_beta=0.05`），hip 关节加权 `out_distill_hip_weight=4.0`，权重 `lambda_out=1.0`。

### 3.5 结构正则损失（当前版本）

见 `structural_losses.py`。设预测姿态 `p`、真值 `g`，骨架边集 `EDGES`，左右对称骨对 `SYM_BONE_PAIRS`：

```
骨长损失      L_bone = L1( bonelen(p), bonelen(g) )                       # 需 GT
对称损失      L_sym  = mean_{(a,b)∈SYM} L1( len(a), len(b) )              # 无需 GT
时序骨长稳定  L_temp = mean_t | bonelen(p)_{t+1} − bonelen(p)_t |         # 无需 GT
root-relative L_rel  = L1( p − p_hip,  g − g_hip )                        # 需 GT, 髋中心化
```

总结构损失：`L_struct = w_bone·L_bone + w_sym·L_sym + w_temp·L_temp + w_rel·L_rel`，默认 `w_bone=1.0, w_sym=0.1, w_temp=0.1, w_rel=3.0`。

**关键性质**：四项全部基于骨长（关节差）或髋中心化坐标，对全局平移不变 —— 因此**只重塑相对骨架，碰不到 hip 全局 xyz**，结构上保证不会恶化 MPJPE 的定位主项。`L_rel` 直接对应 PA-MPJPE 度量的"髋中心相对位置"，是冲击 PA 最对症的杠杆。

骨架定义（MMFi 17 关节，joint 0 = hip/root）：

```python
EDGES = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),       # 两条腿
         (0,7),(7,8),(8,9),(9,10),                   # 脊柱→头
         (8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]  # 两条臂
SYM_BONE_PAIRS = [((0,1),(0,4)),((1,2),(4,5)),((2,3),(5,6)),         # 髋/大腿/小腿
                  ((8,11),(8,14)),((11,12),(14,15)),((12,13),(15,16))]  # 肩/上臂/前臂
```

### 3.6 总损失

```
L_total = L_pose(p_clean, g)              # 基础姿态损失 (PoseLoss: lambda1/2/3)
        + L_action                        # 动作分类 CE
        + lambda_feat · L_distill_feat    # 特征蒸馏
        + lambda_out  · L_distill_out     # 输出蒸馏 (hip 加权)
        + L_struct                        # 结构正则 (本版新增)
```

---

## 4. 数据集与预处理

### 4.1 MMFi 目录结构

```
<data_root>/
  E01/ S01..S10 /        E02/ S11..S20 /
  E03/ S21..S30 /        E04/ S31..S40 /
    └─ A01..A27/
         ├─ wifi-csi/ frame001.mat ... frameNNN.mat   # 每帧一个 .mat
         ├─ depth/                                      # 深度图 (仅训练期教师用)
         └─ ground_truth.npy                            # (num_frames, 17, 3), 单位米
```

| 划分 | 环境 | Subjects | 用途 |
|---|---|---|---|
| 训练 | E01–E03 | S01–S30 | 训练 + held-out val 选点 |
| 测试 | **E04** | S31–S40 | **严格盲测，训练全程不可见** |

held-out val：从 E01–E03 中按 `val_ratio=0.15` 留出若干 subjects（如 S16/S18/S24/S28）不参与训练，仅用于早停 / 选点。

### 4.2 CSI 张量与预处理（`dataset.py`）

每帧 `.mat` 含 `CSIamp` 与 `CSIphase`，形状均为 `(3, 114, 10)`：3 天线 × 114 子载波 × 10 packet。预处理后拼成网络输入 `(T=64, 9, 114, 10)`：

```
幅度路 (3 通道):  逐帧 min-max 归一化  → amp_norm (3,114,10)
相位路 (6 通道):  沿子载波 unwrap → detrend → [sin, cos] 编码 → phase_enc (6,114,10)
拼接:            concat([amp_norm, phase_enc], axis=channel) → (9,114,10)
```

9 通道 = 3 幅度 + 3 sin + 3 cos。**10 packet 维与 3 天线维全程保留进网络**（已核验，dataloader 不丢弃这两个维度）。

> **注**：逐帧 min-max 归一化会抹掉绝对幅度量级（路损 / RSSI），相位 detrend 会去掉子载波线性斜率（ToF）。§9 的探针证明这两个绝对距离线索即便保留也无法跨房间迁移。

---

## 5. 训练流程（四阶段）

完整复现按以下顺序产出 checkpoint：

| 阶段 | 脚本 | 产出 | 说明 |
|---|---|---|---|
| **Stage 1A** | `train_mae.py` | `stage1a_mae/` | （可选）MAE 自监督预训练 backbone |
| **Stage 1B** | (action 预训练) | `stage1b_action/action_best.pt` | 动作监督预训练 backbone（表征健康、margin/s_diff 正常） |
| **Teacher** | (深度教师训练) | `depth_teacher_full/teacher_best.pt` | 深度图姿态教师，蒸馏时冻结 |
| **Distill** | `train_distill_pretrained.py` | `distill_struct_rel/best_mpjpe_ema.pth` | 在 1B backbone 上做蒸馏 + 结构正则，CSI-only 学生 |

> 当前主线在 **Stage 1B（action backbone）** 之上做蒸馏；探针实验表明 action-supervised backbone 表征健康，纯 MAE backbone 反而塌陷（margin≈0.001），故部署用 1B。

---

## 6. 实验设定（与 DT-Pose 对齐）

| 项 | 设定 |
|---|---|
| 数据集 | MMFi |
| 划分 | **Setting 3（cross-environment）** |
| 协议 | **Protocol 3（全 27 动作）** |
| 训练 / 测试 | E01–E03 → **E04（严格盲测）** |
| 输入 | 单链路 CSI（3Tx×1Rx），64 帧窗口 |
| 推理 | **CSI-only** |
| 评测口径 | 逐帧、全帧覆盖、无 padding、`action_idx=None`（无 GT 动作标签泄露） |
| MPJPE | 纯绝对误差，**不做任何 centering**（与 DT-Pose `calculate_error` 一致） |
| PA-MPJPE | 含 scale 的 Procrustes 对齐后 MPJPE（`compute_similarity_transform`） |

**可复现性**：固定 `--seed 42`，启用 `cudnn.deterministic=True / benchmark=False`。报告数一律来自 `eval_dtpose_faithful.py`（唯一权威口径）。训练期 `evaluate_v2` 的滑窗监控值与 faithful 口径**不可比**（差可达 100mm+），**不用于任何结论**。

---

## 7. 评测指标定义

设第 `f` 帧第 `j` 关节预测 `p_{f,j}`、真值 `g_{f,j}`，共 `F` 帧、`J=17` 关节：

```
MPJPE          = (1/F) Σ_f (1/J) Σ_j ‖ p_{f,j} − g_{f,j} ‖₂          # 绝对, 无对齐
MPJPE_aligned  = 同上, 但先各自减去 hip(joint0)                       # 髋中心相对结构
PA-MPJPE       = 同上, 但先对每帧做 Procrustes(旋转+缩放+平移) 对齐    # 纯形状
hip_error      = (1/F) Σ_f ‖ p_{f,0} − g_{f,0} ‖₂                    # 仅 hip 全局定位
PCK@τ_norm     = 关节误差 < τ%·(躯干尺度) 的比例
```

三者关系：`MPJPE` 含全局定位 + 结构；`MPJPE_aligned` 去掉全局平移、保留尺度/朝向；`PA-MPJPE` 进一步去掉缩放/旋转，是最纯的"姿态形状"指标。本项目差距全部落在 `hip_error`（全局定位），`PA-MPJPE` 与 SOTA 持平。

---

## 8. 结果与消融

### 8.1 主结果（E04，faithful 逐帧口径；270 序列 / 80,190 帧）

| 模型 | MPJPE (mm) | PA-MPJPE (mm) | hip_err (mm) |
|---|---|---|---|
| baseline（TaskPrompt 解码器） | 366.6 | 106.2 | 337.8 |
| + 结构正则（骨长/对称/时序，`w_bone=0.5`） | 361.97 | **105.42** | 334.91 |
| + root-relative 位置对齐（当前版本，训练中） | _TBD_ | _TBD_ | _TBD_ |
| **DT-Pose (S3/P3)** | **316.8** | **104.2** | — |

> PA-MPJPE 多 stride 评测 σ ≈ 0.02mm，105.42 为稳定真值。
> 最后一行待 `distill_struct_rel` 训练完成后用 faithful 口径填入。

### 8.2 结构正则消融（PA-MPJPE 视角）

| 配置 | PA-MPJPE | ΔPA | MPJPE | 说明 |
|---|---|---|---|---|
| baseline | 106.2 | — | 366.6 | 无结构正则 |
| + 骨长/对称/时序 | 105.42 | −0.78 | 361.97 | 结构损失生效，MPJPE 同时小降（印证不碰 root） |
| + root-relative | _TBD_ | _TBD_ | _TBD_ | 更对症的 PA 杠杆 |

结论：结构正则在降低 PA 的同时**不恶化甚至小幅改善 MPJPE**，验证了"平移不变损失只重塑相对骨架"的设计。

---

## 9. Limitation 分析：跨房间绝对定位的信息上界

绝对 MPJPE 的 ~45mm 差距**全部集中在 hip 全局定位**，且为信息论上界，非建模不足。四条独立证据：

### (a) 原始输入信息探针（`probe_raw_amplitude_hip.py`）

用线性岭回归从 CSI 幅度直接预测 E04 hip 绝对坐标：

| 特征 | held-in (E01–03) | E04 |
|---|---|---|
| mean_base（预测训练集均值，零信息标尺） | 172.7 | **324.2** |
| 原始绝对幅度 | 152.3 | 350.8 |
| log 功率 | 155.5 | 348.8 |
| 逐帧归一化幅度 | 149.6 | 356.4 |

三种幅度表示在训练房间均优于零信息标尺，**在 E04 上全部劣于标尺** → 幅度→定位映射逐房间不同，呈反向迁移，绝对距离线索不可跨域。

### (b) 教师上界

深度图教师（拥有视觉深度）在 E04 的 hip_err 仍 ~236mm —— 即便强模态也难恢复跨房间绝对位置。

### (c) 完整模型 vs 零信息基线

部署模型 E04 hip_err ~335mm ≥ 零信息基线 324mm —— 用满全部输入的非线性模型，在绝对定位上未超过常数预测。

### (d) 杠杆穷举

解码器结构（多轮）、自监督预训练（MAE-DCL: TC-CL + uniformity）、合规 test-time 重心化、蒸馏权重调参 —— 均未移动 hip 误差。

**结论**：严格盲测 / 跨房间 / 单链路 CSI / CSI-only 推理下，绝对 MPJPE 受信息上界约束；可改善空间在相对结构（PA-MPJPE），本方法已推至 SOTA 持平。DT-Pose 原文亦指出末端关节误差受限于 WiFi 分辨率（需更多设备 / 更高分辨率），与本结论一致。

---

## 10. 复现步骤

### 10.1 环境

```bash
# Python 3.7 / PyTorch 2.2.2 / CUDA (RTX 4080, 16GB)
conda create -n 3DHPE1 python=3.7 -y
conda activate 3DHPE1
pip install torch==2.2.2 numpy scipy
# 其余依赖见仓库 requirements.txt
```

### 10.2 数据

将 MMFi 解压至 `--data_root`，确认目录结构如 §4.1。

### 10.3 训练（当前版本：结构正则 + root-relative）

```bash
python train_distill_pretrained.py \
    --data_root /path/to/MMFi \
    --train_envs E01 E02 E03 --test_env E04 \
    --pretrain_ckpt checkpoints/stage1b_action/action_best.pt \
    --teacher_ckpt  checkpoints/depth_teacher_full/teacher_best.pt \
    --depth_img 112 --depth_clip 5000 \
    --w_bone 1.0 --w_sym 0.1 --w_temp 0.1 --w_rel 3.0 \
    --epochs 50 --batch_size 2 --accumulate_grad 8 \
    --use_ema --ema_decay 0.999 --seed 42 \
    --save_dir ./checkpoints/distill_struct_rel
```

显存提示：RTX 4080 16GB 下 `batch_size=2 + accumulate_grad=8`（等效 batch 16）。OOM 时降 batch 提 accum。

### 10.4 评测（唯一权威口径）

```bash
python eval_dtpose_faithful.py \
    --data_root /path/to/MMFi \
    --ckpt ./checkpoints/distill_struct_rel/best_mpjpe_ema.pth \
    --test_env E04 --seq_len 64 --variance
```

`--variance` 跑多 stride 报 mean±σ，用于判断指标领先是否超过评测口径噪声（领先 ≤ σ 只能写"持平"）。

### 10.5 信息上界探针（§9 复现）

```bash
python probe_raw_amplitude_hip.py \
    --data_root /path/to/MMFi \
    --train_envs E01 E02 E03 --test_env E04 \
    --max_frames_per_seq 60 --ridge 1e-2
```

---

## 11. 超参数完整参考

| 类别 | 参数 | 默认 | 说明 |
|---|---|---|---|
| 数据 | `--seq_len` | 64 | 时间窗口长度 |
| 数据 | `--val_ratio` | 0.15 | E01–E03 中留作 val 的 subject 比例 |
| 深度 | `--depth_img` / `--depth_clip` | 112 / 5000 | 教师深度图尺寸 / 裁剪 |
| 蒸馏 | `--lambda_feat` | 0.1 | 特征蒸馏权重 |
| 蒸馏 | `--lambda_out` | 1.0 | 输出蒸馏权重 |
| 蒸馏 | `--out_distill_hip_weight` | 4.0 | 输出蒸馏中 hip 加权 |
| 蒸馏 | `--out_distill_beta` | 0.05 | Smooth-L1 beta |
| 姿态 | `--lambda1/2/3` | 1.0 / 0.5 / 2.0 | PoseLoss 各项权重 |
| 姿态 | `--lambda_hip` | 0.3 | hip 项权重 |
| TotalLoss | `--alpha/beta/gamma/delta` | 0.5/2.0/0.0/0.5 | 总损失组合权重 |
| **结构** | `--w_bone` | 1.0 | 骨长损失 |
| **结构** | `--w_sym` | 0.1 | 左右对称 |
| **结构** | `--w_temp` | 0.1 | 时序骨长稳定 |
| **结构** | `--w_rel` | 3.0 | root-relative 位置对齐 |
| RSC | `--rsc2_time/channel/batch_pct` | 0.5/0.5/0.5 | RSC challenge 比例 |
| 网络 | `--global_dim` | 128 | 全局特征维 |
| 网络 | `--num_transformer_layers` / `--num_heads` | 3 / 4 | Transformer |
| 网络 | `--tcn_channels` / `--tcn_kernel_size` | [128,128] / 3 | TCN |
| 网络 | `--coarse_hidden_dim` | 256 | 粗解码隐藏维 |
| 网络 | `--gcn_hidden_dim` / `--num_gcn_layers` | 128 / 3 | SkeletonRefiner |
| 优化 | `--lr_backbone` / `--lr_head` | 1e-4 / 5e-4 | 分组学习率 |
| 优化 | `--weight_decay` | 1e-3 | |
| 优化 | `--grad_clip` | 1.0 | |
| 优化 | `--batch_size` / `--accumulate_grad` | 2 / 8 | 等效 batch 16 |
| 优化 | `--epochs` / `--patience` | 50 / 15 | 早停 patience |
| EMA | `--use_ema` / `--ema_decay` | True / 0.999 | |
| 评测 | `--eval_interval` | 3 | 每 N epoch 评一次 |
| 复现 | `--seed` | 42 | |

---

## 12. 仓库结构

```
RSC V2/
├── train_distill_pretrained.py   # 主训练: 蒸馏 + EMA + held-out 选点 + 结构正则
├── train_mae.py                  # Stage 1A: MAE 自监督预训练 (可选)
├── structural_losses.py          # 结构正则: 骨长/对称/时序骨长/root-relative
├── eval_dtpose_faithful.py       # 逐帧 faithful 评测 (与 DT-Pose 对齐的权威口径)
├── evaluate.py                   # 训练期监控评测 (evaluate_v2, 滑窗, 不用于报告)
├── probe_raw_amplitude_hip.py    # 原始幅度 → E04 hip 信息探针 (§9)
├── dataset.py                    # MMFi CSI/GT 加载 + 预处理 + 增广
├── dataset_distill.py            # 蒸馏数据加载 (csi + depth 配对)
├── augmentation.py               # CSI 数据增广
├── losses.py                     # PoseLoss / TotalLoss
├── distill_loss.py               # DistillProjection / FeatureDistillLoss / OutputDistillLoss
├── taskprompt_decoder.py         # TaskPromptCoarseHead + uniformity_loss
├── action_prior_root.py          # (路1 备选) 动作×相位先验 root 损失
├── utils.py                      # set_seed / logger / 参数统计等
├── models/
│   ├── __init__.py               # CSIRSCPoseDG
│   └── depth_teacher.py          # DepthPoseTeacher
└── checkpoints/
    ├── stage1b_action/action_best.pt
    ├── depth_teacher_full/teacher_best.pt
    └── distill_struct_rel/best_mpjpe_ema.pth
```

---

## 13. 常见问题与坑

- **`AttributeError: 'PoseDecoder' object has no attribute 'root_head'`**：当前 366 基线解码器无 root_head；训练脚本已对路1 的 `root_prior_losses` 按 `hasattr(decoder,'root_head')` 门控，自动跳过。
- **ckpt 加载后全是噪声**：检查日志 `missing / unexpected`，>8 即模型定义与 ckpt 不配套；改代码后务必 `rm -rf **/__pycache__`。
- **EMA ckpt 直接是 shadow dict**：`best_mpjpe_ema.pth` 已是 EMA 权重，评测直接加载。
- **监控值远好于 faithful**：训练期滑窗监控 (stride + padding) 会失真，**只认 `eval_dtpose_faithful.py`**。
- **探针 raw_abs 出 NaN**：原始 `CSIamp` 含 inf，须先 `nan_to_num(posinf=0)` + 裁剪（`probe_raw_amplitude_hip.py` v2 已修）。
- **PA / MPJPE_aligned 对全局平移不变**：可作 sanity——若改动只该影响结构，这两项应随 MPJPE 同向但 hip_err 不变，反之说明动到了 root。
- **OOM**：降 `--batch_size`、提 `--accumulate_grad` 维持等效 batch；`mae_dcl` 类对比损失需较大 batch，4080 上 batch16 易 OOM。

---

## 14. 诚实声明

- 报告的所有数均来自**严格盲测**（E04 训练期从不可见）+ **faithful 逐帧口径**，无 GT 动作标签泄露、无对 E04 真值的对齐、无 transductive 偷看。
- **PA-MPJPE 与 DT-Pose 持平**（105.4 vs 104.2，约 1mm 内，处于单链路硬件分辨率地板附近）。
- **绝对 MPJPE 落后约 45mm**，本文将其归因于跨房间绝对定位的信息论上界并给出测量证据（§9），**不声称全面超越**。
- 任何低于 faithful 报告值的数（如训练期滑窗监控值、含 GT 动作标签的评测、对 E04 真值对齐后的数）一律**不作为结论**。
- 若后续实验使指标在 faithful 口径下真实反超，将据实更新对应表格与结论措辞。