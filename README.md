# CSI-RSC-PoseDG

**Cross-Environment 3D Human Pose Estimation from WiFi CSI**

基于 WiFi CSI 的跨环境 3D 人体姿态估计系统。训练期由视觉教师（RGB / depth）通过知识蒸馏向 CSI 学生注入几何/结构先验，**推理阶段只使用 CSI**（不依赖任何 RGB、depth 或其他视觉输入）。在 MM-Fi 数据集上以跨房间设定（Setting 3 / Protocol 3，全 27 动作，CSI-only 推理）评测，并与 DT-Pose、Person-in-WiFi 3D 等方法对齐评测协议与选点口径（全量训练 + 在 E04 上选 checkpoint）。

> **硬件配置（MM-Fi 原生）**：1 个发射天线 × 3 个接收天线（1Tx×3Rx），每对天线 114 个子载波，Atheros CSI Tool，一对商用 TP-Link N750 AP，5GHz / 40MHz。每帧 CSI 张量 `(3, 114, 10)`：3 = 接收天线（天线链路），114 = 子载波，10 = 100ms 内时间采样。

> **评测口径**：MPJPE 为绝对误差（不对齐，align=False）；PA-MPJPE 为 Procrustes 对齐后误差；二者均与 DT-Pose `utils.py` 实现对齐。所有报告数来自 `eval_dtpose_faithful.py`（逐帧、全帧覆盖、无 padding、action_idx=None）。

---

## 目录

1. [问题定义](#1-问题定义)
2. [核心组件](#2-核心组件)
3. [方法详解](#3-方法详解)
4. [数据集与预处理](#4-数据集与预处理)
5. [训练流程](#5-训练流程)
6. [实验设定](#6-实验设定)
7. [评测指标定义](#7-评测指标定义)
8. [结果与消融](#8-结果与消融)
9. [DT-Pose 范式复现](#9-dt-pose-范式复现)
10. [乙′：DT-Pose 表征特征蒸馏](#10-乙dt-pose-表征特征蒸馏)
11. [hip 全局定位的实验观察](#11-hip-全局定位的实验观察)
12. [Checkpoint 选择口径](#12-checkpoint-选择口径)
13. [复现步骤](#13-复现步骤)
14. [超参数完整参考](#14-超参数完整参考)
15. [仓库结构](#15-仓库结构)
16. [常见问题与坑](#16-常见问题与坑)
17. [引用](#17-引用)

---

## 1. 问题定义

WiFi CSI（Channel State Information）是一种无感（device-free）、不依赖光照、保护隐私的人体感知模态。CSI 信号由无线多径传播决定，依赖房间几何、家具布置、墙体反射、收发天线位置等环境因素，因此跨环境（不同房间）泛化是该任务的核心难点。

本项目采用跨房间设定，并与对比方法对齐口径：

- **跨房间（cross-environment）**：训练用 `{E01, E02, E03}`，测试用结构不同的 `E04`。
- **硬件（MM-Fi 原生配置）**：1Tx×3Rx，每对天线 114 子载波。每帧 CSI `(3, 114, 10)`。
- **CSI-only 推理**：视觉图像仅在训练阶段作为教师参与蒸馏，部署/推理时不可用。
- **全量训练 + E04 选点**：E01–E03 全部 subject 进训练（不另划 val），checkpoint 在 E04 上按指标选取（与 DT-Pose `val=E04` 同口径，见 §12）。

形式化：给定 CSI 时序窗口 `X ∈ R^(T×9×114×10)`，预测 3D 关节序列 `P ∈ R^(T×17×3)`（绝对世界坐标，单位米）。`ENV_SUBJECTS`：E01=S01–10, E02=S11–20, E03=S21–30, E04=S31–40。

---

## 2. 核心组件

1. **双分支 CSI 编码 + 时空建模 backbone**：DualBranchCSIEncoder（幅度 3ch + 相位 6ch 双分支）→ LocalSpatioTemporalEncoder（3D 残差卷积）→ LocalFeaturePooling → GlobalTemporalModeler（Transformer + TCN），输出 `z_global (B, 64, 128)`。
2. **RSC（Representation Self-Challenging）**：训练期对全局表征施加时间维 / 通道维 / batch 维随机遮挡。
3. **MixStyle**：在 CSI 编码器残差块之间、以及时序特征上混合实例统计量。
4. **结构正则损失套件**（`structural_losses.py`）：骨长一致、左右对称、时序骨长稳定、root-relative 位置对齐——四项均平移不变。
5. **Hybrid FK 解码器**（`fk_decoder.py`）：在现有解码器外挂一条正运动学（FK）分支，以 α 退火融合；支持 `absolute` / `velocity` 两种 root 模式。
6. **Root anchor**（`structural_losses.py`）：以损失项 `L_anchor` 把预测 hip 往按动作的源域 canonical 先验拉。
7. **知识蒸馏**：训练期用 RGB / depth 教师对 CSI 学生做 feature-level + output-level 蒸馏；教师模态可切换。
8. **DT-Pose 表征特征蒸馏（乙′，可选）**：把 DT-Pose 式自监督预训练编码器作为冻结特征教师，将其表征蒸馏进 student `z_global`（§10）。

---

## 3. 方法详解

### 3.1 整体管线

```
CSI 输入 (B, T=64, 9, 114, 10)
   │
   ├─[csi_encoder]        逐帧 CSI 编码 (9 = 3 幅度 + 6 相位通道)
   ├─[local_encoder]      3D 残差卷积 (num_res3d_blocks=2)
   ├─[feature_pooling]    空间池化, 每帧聚合为单 token
   ├─[global_modeler]     Transformer(layers=3, heads=4) + TCN([128,128])
   │                      ──> z_global (B, 64, 128)
   ├─[RSC]                训练期对 z_global 做 representation self-challenging
   ├─[pose_decoder]       HybridFKPoseDecoder (结构支 + FK 支 + α 融合)
   │                      ──> p_coarse, p_final ∈ (B, 64, 17, 3)
   └─[action_classifier]  27 类动作分类 (辅助任务)

训练期教师 (推理不用):
   RGB (B,T,3,112,112)  ──[RGBPoseTeacher: ImageNet ResNet18 + GlobalTemporalModeler]──>
   {p_final, z_global}  ──蒸馏──> 对齐学生 z_global / p_final
   (可选) DT-Pose 预训练编码器 ──冻结──> z_teacher (B,T,256) ──蒸馏──> 对齐学生 z_global
```

学生网络总参数量：约 **1.63M**。

### 3.2 各模块详解（学生，源码见 `models/`）

- **csi_encoder**（`csi_encoder.py::DualBranchCSIEncoder`）：每帧 `(9,114,10)` → `(64,114,10)`。9 通道 = 3 幅度 + 3 sin + 3 cos。幅度分支与相位分支各自经 EnvironmentNormalization + ResBlock2D（BN/IN 可学习门控混合）+ MixStyle2D，再用 GatedFusion（门控 + 交叉注意力）融合。`encoder_hidden_dim=32, encoder_out_dim=64`。
- **local_encoder**（`local_encoder.py::LocalSpatioTemporalEncoder`）：3D 残差卷积（`num_res3d_blocks=2`），时间/子载波/packet 三轴提取局部时频纹理。
- **feature_pooling**（`local_encoder.py::LocalFeaturePooling`）：AdaptiveAvgPool2d + Linear-LayerNorm-GELU，每帧 → 单 token，输出 `(B,T,128)`。
- **global_modeler**（`global_encoder.py::GlobalTemporalModeler`）：PatchEmbedding + 正弦位置编码 + MixStyleTemporal + TransformerBlock×3（4 头，pre-norm）+ TemporalConvBlock×2（空洞 TCN，dilation 1/2），输出 `z_global (B,64,128)`。
- **pose_decoder**（`fk_decoder.py::HybridFKPoseDecoder`）：结构支 `PoseDecoder`（`TaskPromptCoarseHead` + `SkeletonRefiner` GCN×3）+ FK 支 `FKBranch`，α 融合。
- **action_classifier**（`pose_decoder.py::ActionClassifier`）：27 类动作分类头 + 动作嵌入表（训练用 GT 硬查表，推理用预测概率软加权）。

### 3.3 RSC（Representation Self-Challenging）+ Action Dropout

训练期对 `z_global` 随机遮挡（时间/通道/batch 维各 0.5，`rsc.py::RSCGlobalChallenger`），mask 基于梯度选 top-k 并保留 backbone 梯度路径。配合 Action Dropout（训练时 50% 概率把 action_emb 置零）。

### 3.4 知识蒸馏（RGB / depth 教师）

教师（训练期冻结）对学生做两路蒸馏（`distill_loss.py`）：

- **Feature-level**：学生 `z_global` 经 `DistillProjection` 投影后，与教师 `z_global` 做 cosine + Smooth-L1 对齐（`lambda_feat`）。
- **Output-level**：学生 `p_final_clean` 与教师 `p_final` 做 Smooth-L1 姿态蒸馏（`lambda_out`）。`align_hip=True` 时各自减 hip 后只蒸馏相对结构（RGB 教师默认开），`align_hip=False` 时直接对齐绝对坐标、hip 关节可加权（depth 教师默认）。

`out_distill_hip_weight` 默认：depth=4.0，rgb=1.0。`align_hip` 默认：rgb=True，depth=False。

### 3.5 Hybrid FK 解码器（`fk_decoder.py`）

FK 把姿态参数化为「root + 每条骨长度 + 每条骨方向单位向量」，沿运动学树合成绝对坐标：

```
joints[0] = root
for (parent, child) in EDGES:
    joints[child] = joints[parent] + bone_dir[child] · bone_len[child]
```

```python
EDGES = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),
         (0,7),(7,8),(8,9),(9,10),
         (8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]
```

**Hybrid（外挂式）**：`p_final = α·p_struct + (1-α)·p_fk`，`α` 注册为 buffer，逐 epoch 从 1.0 退火到 `fk_alpha_final`（默认 0.4），`fk_alpha_warmup`（默认 20）内完成。

**root 模式**（`root_mode`）：
- `absolute`（默认）：`root_t = root_head(z_t)` 逐帧绝对回归。
- `velocity`：`v_t = tanh(vel_head(z_t)) · vel_scale`，`traj = cumsum(v) - mean_t(traj)`，`root_t = anchor + traj`；root 时间均值 = anchor，由 `L_anchor` 约束。
- `axis_split`：x 轴走 velocity 路径（`vel_head_x` → cumsum，不去均值），y/z 轴走 absolute 路径（`root_head_yz` 逐帧回归）。此模式下 `L_anchor` 只约束 y/z 两轴（见 §3.6），x 轴不加 anchor 约束。

### 3.6 结构正则与 root anchor（`structural_losses.py`）

```
L_bone   = L1( bonelen(p), bonelen(g) )                  # 需 GT (源域)
L_sym    = mean L1( len(a), len(b) )                     # 无需 GT
L_temp   = mean_t | bonelen(p)_{t+1} − bonelen(p)_t |    # 无需 GT
L_rel    = L1( p − p_hip, g − g_hip )                    # 髋中心相对位置
L_anchor = SmoothL1( p_hip, canonical[action] )          # root 往源域按动作先验拉
```

前四项对全局平移不变。`L_anchor` 用源域按动作平均 hip（`build_action_canonical` 预扫训练集 GT，与 E04 无关）做先验。`velocity` 模式下 `L_anchor` 只约束时间均值 hip。`axis_split` 模式下 `L_anchor` 只约束 y/z 两轴（`p_hip[...,1:]` 对齐 canonical），x 轴放开。

### 3.7 总损失

```
L_total = L_pose + L_action
        + lambda_feat·L_distill_feat + lambda_out·L_distill_out
        + w_bone·L_bone + w_sym·L_sym + w_temp·L_temp + w_rel·L_rel
        + w_root_anchor·L_anchor
        + w_dtpose_feat·L_dtpose_feat   (乙′, 可选, §10)
```

### 3.8 DT-Pose x 轴轨迹蒸馏（B1，可选，配合 axis_split）

把 §9 阶段二训出的 DT-Pose 解码器作为冻结教师，逐帧输出 hip 的 x 坐标，取其去均值轨迹作为蒸馏目标，监督 axis_split 下 student FK 支的 x 轨迹。

- 组件（`dtpose_rootx_teacher.py::DTPoseRootXTeacher`）：复用乙′的口径转换（`dtpose_feature_teacher._amp_to_dtpose_input`，取 CSI 幅度 3 通道、逐帧 DT-Pose 式 min-max），过 DT-Pose 完整解码器（`train_pose_dtpose_style.ViT_Pose_Decoder`）得 `(B,T,17,3)`，取 hip(joint 0) 的 x、逐序列减时间均值，输出 `(B,T)`（detach）。
- 损失（`train_distill_pretrained.py`）：student `p_final_clean` 的 hip.x 逐序列去均值后，与教师 x 轨迹做 Smooth-L1（beta=0.05），权重 `--w_dtx`。
- 参数：`--w_dtx`（默认 0，>0 启用）、`--dtx_ckpt`（DT-Pose 预训练整对象）、`--dtx_decoder_ckpt`（DT-Pose 阶段二解码器）。`--w_dtx 0` 时与原 axis_split 一致。

---

## 4. 数据集与预处理

### 4.1 MM-Fi 目录结构与划分

```
<data_root>/                              <rgb_root>/  (可在独立磁盘)
  E01/ S01..S10 / A01..A27/                 E01/ S01..S10 / A01..A27/
    ├─ wifi-csi/ frameNNN.mat                 └─ rgb/ frameNNN.png  (640x480 RGB)
    └─ ground_truth.npy (num_frames,17,3)        (或 A01/ 下直接放 frameNNN.png, 自动探测)
```

| 划分 | 环境 | Subjects | 用途 |
|---|---|---|---|
| 训练 | E01–E03 | S01–S30 | 全部进训练（不另划 val） |
| 测试 / 选点 | E04 | S31–S40 | 跨房间测试；checkpoint 在其上选取 |

### 4.2 CSI 张量与预处理（`dataset.py::CSIPreprocessor`）

每帧 `.mat` 含 `CSIamp`/`CSIphase`，形状 `(3, 114, 10)`。预处理拼成网络输入 `(T=64, 9, 114, 10)`：

- 幅度路：逐帧 min-max 归一化（3 通道）。
- 相位路：unwrap → detrend → [sin, cos]（6 通道）。
- 3 天线维与 10 packet 维全程保留进网络。

### 4.3 RGB 预处理与工程（`dataset_distill.py`）

- RGB 逐帧 PNG → `[0,1]`，resize 到 `rgb_img`（默认 112）。ImageNet mean/std 归一化在 RGB 教师模型内部做。
- 建议预缓存为 112×112（`cache_rgb_112.py`），避免机械盘随机小文件 I/O 瓶颈。
- `--rgb_root` 支持 RGB 独立放盘，自动探测 `E/S/A/rgb/` 与 `E/S/A/` 两种布局。

### 4.4 CSI 数据增强（`augmentation.py::CSIAugmentor`，仅训练）

幅度缩放、相位噪声、子载波 dropout、频带 mask（SpecAugment 风格）、天线 dropout/shuffle、时序 jitter、高斯噪声。`p=0.8`。

---

## 5. 训练流程

| 阶段 | 脚本 | 产出 | 说明 |
|---|---|---|---|
| Stage 1A | `train_mae.py` | `stage1a_mae/` | （可选）MAE 自监督预训练 backbone |
| Stage 1B | action 预训练 | `stage1b_action/action_best.pt` | 动作监督预训练 backbone |
| Teacher | `train_rgb_teacher.py` | `rgb_teacher_dg/teacher_best.pt` | RGB 姿态教师（ImageNet ResNet18 + MixStyle），蒸馏时冻结 |
| Distill | `train_distill_pretrained.py` | `distill_*/epoch*_ema.pth` | 在 1B backbone 上蒸馏 + 结构正则 + FK + anchor（+ 可选乙′）；全量训练、全程 archive ckpt |

> depth 教师（`train_depth_teacher.py` + `models/depth_teacher.py`）作为历史变体保留，可用 `--teacher_modality depth`。

DT-Pose 范式复现是一条独立的实验线（§9），不属于上述主管线。

---

## 6. 实验设定

| 项 | 设定 |
|---|---|
| 数据集 / 划分 / 协议 | MM-Fi / Setting 3（cross-env）/ Protocol 3（全 27 动作） |
| 训练环境 | E01–E03（S01–S30），全部 subject 进训练 |
| 测试 / 选点 | E04（S31–S40），跨房间测试 + 在其上选 checkpoint |
| 训练 | 全量、不早停、跑满 `--epochs`（默认 50） |
| 输入 | MM-Fi 原生 CSI（1Tx×3Rx，114 子载波），64 帧窗口 |
| 推理 | CSI-only（视觉仅训练期教师） |
| 评测口径 | 逐帧、全帧覆盖、无 padding、action_idx=None |
| MPJPE / PA-MPJPE | 绝对误差不 centering / Procrustes 对齐后 |

报告数一律来自 `eval_dtpose_faithful.py`；训练期滑窗监控值与 faithful 不可比，仅供观察。

---

## 7. 评测指标定义（`evaluate.py::PoseEvaluator`）

```
MPJPE          = 绝对, 无对齐 (align=False)
MPJPE_aligned  = 每帧减 hip 后 (相对结构)
PA-MPJPE       = Procrustes(旋转+缩放+平移) 对齐后 (纯形状)
hip_error      = 仅 hip(joint 0) 全局定位误差
PCK@τ_norm     = 关节误差 < τ%·(LShoulder-LHip 距离) 的占比
```

---

## 8. 结果与消融

### 8.0 RGB 教师强度（E04 sanity；教师可用 RGB，推理学生不用）

RGB 教师（`rgb_teacher_dg`，ResNet18 + MixStyle）E04 best：

| 指标 | MPJPE | MPJPE_aligned | PA | hip | 选点 |
|---|---:|---:|---:|---:|---|
| RGB 教师 | 291.2 | 118.4 | 96.65 | 258.2 | E04 by MPJPE |

### 8.1 主结果（E04，faithful 逐帧口径；全量训练 + E04 选点）

| 模型 | MPJPE (mm) | PA-MPJPE (mm) | hip_err (mm) | 选点 |
|---|---:|---:|---:|---|
| CSI-RSC-PoseDG（depth 教师蒸馏） | 352.23 | 102.75 | 321.88 | E04 faithful sweep（epoch006_ema） |
| CSI-RSC-PoseDG（RGB 教师 + 乙′，训练中滑窗 ep3 ema） | 351.68 | 102.84 | 321.04 | E04 滑窗监控（非 faithful，训练中） |
| **DT-Pose (S3 / P3)** | 316.8 | 104.2 | — | — |
| **Person-in-WiFi 3D (S3 / P3)** | 302.5 | 101.1 | — | TPAMI 2026 |
| MetaFi++ | 369.5 | 116.0 | — | — |
| HPE-Li | 388.4 | 107.9 | — | — |

> 乙′ 行为训练过程中的 E04 滑窗监控值，非 faithful sweep 选点结果，仅供过程观察。

### 8.2 PA-MPJPE 结构杠杆消融（历史趋势，源域 val 选点口径）

| 配置 | PA-MPJPE | ΔPA | MPJPE | MPJPE_aligned |
|---|---:|---:|---:|---:|
| baseline | 106.2 | — | 366.6 | — |
| + 骨长/对称/时序 | 105.42 | −0.78 | 361.97 | 122.24 |
| + root-relative | 104.73 | −0.69 | 363.74 | 120.24 |
| + FK + anchor | 103.92 | −0.81 | — | — |

### 8.3 Step B 实验矩阵（已运行的 hip / MPJPE 杠杆）

| 杠杆 | 现象 |
|---|---|
| 解码器结构（多轮） | hip_err 在 320–335 区间 |
| MAE-DCL 预训练（旧实现） | 与无预训练 baseline 相当 |
| 合规 TTA | hip ~354 |
| hip 蒸馏权重 / lambda_hip | hip 340→337 |
| raw / log 幅度输入 | E04 上探针结果见 §11a |
| 保尺度幅度残差支路（scale=0.3） | faithful 9 ckpt 高于纯先验，已归档 |
| 乙′ DT-Pose 特征蒸馏（w=0.1, 训练中） | DTfeat loss 0.77→0.71；hip 321 / PA 102.8（ep3） |

### 8.4 axis_split + B1（E04 faithful 逐帧口径）

配置：`--root_mode axis_split --w_dtx 1.0`，RGB 教师，其余同 §13.4 默认。全 12 epoch archive，faithful sweep 逐 ckpt 评测。各 seed best-MPJPE 的 checkpoint（raw）：

| seed | best-MPJPE epoch | MPJPE | hip | PA |
|---|---|---:|---:|---:|
| 42 | ep4 | 311.48 | 273.87 | 103.59 |
| 0  | ep6 | 311.70 | 278.39 | 105.26 |
| 1  | ep5 | 327.55 | 295.83 | 105.20 |
| 7  | ep4 | 352.11 | 322.08 | 104.28 |

DT-Pose (S3/P3) 参考值：MPJPE 316.8 / PA 104.2。

同一轮内 best-MPJPE 与 best-PA 落在不同 epoch。以 seed 0 为例（faithful sweep 部分行）：

| epoch | MPJPE | hip | PA |
|---|---:|---:|---:|
| 6  | 311.70 | 278.39 | 105.26 |
| 7  | 355.59 | 322.59 | 103.55 |
| 9  | 348.60 | 314.83 | 103.40 |
| 10 | 350.09 | 319.03 | 102.55 |

观察：各 seed 的 best-MPJPE checkpoint 其 PA 高于该 seed 的 best-PA checkpoint；x 轴 root std 较大的 epoch（如 seed 0 ep6，hip 278.39）其 PA（105.26）高于 x 轴 std 较小、hip 较高的 epoch（如 ep10，hip 319.03 / PA 102.55）。

axis_split 下 E04 raw root 的 x 轴 std 实测在 8（ep1）→ 40~82（ep3 起）区间；y/z 轴 std 维持 12~20。B1 训练中 DTx 蒸馏损失（w_dtx=1.0，seed 42）单调下降：ep1 内 0.163 → 0.031。

> 上述为 raw checkpoint 的 faithful 值；同配置 EMA checkpoint 的 best-MPJPE 在各 seed 上为 349~358（hip 323~328）。

---

## 9. DT-Pose 范式复现

为定位 DT-Pose 在 MM-Fi S3/P3 上 MPJPE 优势的来源，复现其两阶段自监督流程（独立实验线，脚本见仓库根目录）。

### 9.1 阶段一：自监督预训练（`pretrain_dtpose_style.py`）

- 输入：单帧、3 通道纯幅度 `(3,114,10)`，按 DT-Pose `_read_amp`（inf→nan、逐 packet 均值填充、全局 min-max）。
- 编码器：4 层 ViT，emb_dim=256，patch(2,2)，mask_ratio=0.80。
- 损失：掩码重建 MAE + 相邻帧时序对比（InfoNCE）+ uniformity 正则（weight 0.01）。
- 优化：batch（grad-accum）、base_lr 1.5e-4 cosine、warmup 40。
- 运行：100 轮（E04 验证重建 loss ≈ 0.001）；400 轮（`pretrain_dtpose_400.pt`）。

### 9.2 阶段二：拓扑解码（`train_pose_dtpose_style.py`）

- 冻结编码器 + task-prompt + GCN×3（原始 0/1 骨架邻接，无自环、无归一化）+ Transformer×3 + MLP→(17,3)。
- 优化：SGD lr=1e-3 wd=0.01 momentum=0.9，无 scheduler，batch 32，50 epoch，纯 MPJPE 损失，align=False，E04 选 best epoch。

| 预训练 | best MPJPE | PA | hip | epoch |
|---|---:|---:|---:|---:|
| 100 轮 | 350.17 | 109.36 | 330.8 | ep15 |
| 400 轮 | 323.53 | 108.9 | 301.5 | ep08 |

### 9.3 Route B：root/相对分离 + 双 root 策略评测（`train_pose_dtpose_B.py`）

训练时损失拆为 `w_rel·L_rel + w_root·L_root`（默认 1.0 / 0.3）。评测时报两种 root 策略：
- **s1**：模型预测 root（DT-Pose 原样）。
- **s2**：用训练集 root 均值替换 root（常数 root）。

| 策略 | 现象 |
|---|---|
| s2（常数 root） | hip 锁定 324，MPJPE 稳定 ~337 |
| s1（预测 root） | MPJPE 在 313–420 间波动，ep06 最低 313.0 / hip 296 |

---

## 10. 乙′：DT-Pose 表征特征蒸馏

把 §9 阶段一训出的 `pretrain_dtpose_400.pt` 编码器作为**冻结特征教师**，将其表征蒸馏进主系统 student `z_global`。主系统架构、DG 模块、教师蒸馏全部保留。

### 10.1 组件（`dtpose_feature_teacher.py`）

- `DTPoseFeatureTeacher`：加载 `pretrain_dtpose_400.pt` 的 `MAE_Encoder`，冻结。
- 逐帧取 student CSI 的幅度 3 通道，按 DT-Pose 式逐帧 min-max 归一化后过编码器 `feature_extract`，输出 cls 特征 `(B, T, 256)`（detach）。

### 10.2 蒸馏接入（`train_distill_pretrained.py`）

- 新增参数：`--dtpose_feat_ckpt`（教师权重路径，None=不启用，原流程不变）、`--w_dtpose_feat`（权重，默认 0.1）。
- student `z_global (B,T,128)` 经 `proj_dt`（128→256）投影后，与教师 `z_teacher (B,T,256)` 做 cosine + Smooth-L1 对齐（复用 `FeatureDistillLoss`）。
- `proj_dt` 参数以 `lr_head` 加入优化器。

### 10.3 当前运行（w_dtpose_feat=0.1）

| epoch | DTfeat loss | E04 ema MPJPE | hip | PA |
|---|---:|---:|---:|---:|
| 1 | 0.7655 | — | — | — |
| 2 | 0.7085 | — | — | — |
| 3 | 0.7101 | 351.68 | 321.04 | 102.84 |

---

## 11. hip 全局定位的实验观察

主系统 MPJPE 与 PA-MPJPE 的差异主要体现在 hip（joint 0）全局定位项。以下为相关实测数据（均为客观观测值）。

### 11.a 原始输入信息探针（`probe_raw_amplitude_hip.py`）

闭式岭回归从 CSI 幅度直接回归 hip 绝对 xyz，E01–E03 训练、E04 测试：

| 特征 | held-in (E01–03) | E04 |
|---|---:|---:|
| mean_base（预测训练集均值 hip） | 172.7 | 324.2 |
| 原始绝对幅度 raw_abs | 152.3 | 350.8 |
| log 功率 raw_log | 155.5 | 348.8 |
| 逐帧归一化幅度 norm_abs | 149.6 | 356.4 |

### 11.b 跨域 root 统计（`diagnose_hip_motion.py`）

| 量 | 值 |
|---|---|
| 训练域（E01–03）root 站位中心 | x=−0.035, y=0.021, z=3.134 (m) |
| E04 root 站位中心 | x=0.239, y=0.005, z=3.166 (m) |
| 跨域中心差 | x=274.7, y=15.9, z=32.2 (mm) |
| 训练域 root 动作内 std | 121 / 65 / 116 (mm) |
| 用训练域 root 均值预测 E04 站位的误差 | 315.0 (mm) |

### 11.c 完整模型 vs 零信息基线

主系统在 E04 的 hip 误差（root-relative 行 334.29、prior-root 行 321.88）与零信息基线（用训练集均值预测，~315–324mm）处于同一区间。Route B 的 s2（常数 root）策略下 hip 锁定 324。

### 11.d 非线性前端探针（`probe_transformer_hip.py`，val 选点 + 多 seed）

| 前端 | val 选点后 E04 hip |
|---|---|
| Transformer | 341.1 ± 9.5 mm |
| Conv | 354.1 ± 12.5 mm |
| 零信息基线 | 315.9 mm |

---

## 12. Checkpoint 选择口径

1. **全量训练**：E01–E03 全部 subject，不另划 val、不早停，跑满 `--epochs`。
2. **archive**：每 `--eval_interval` epoch 存 `epochNNN_{raw,ema}.pth`。训练期 E04 监控为滑窗口径，仅观察。
3. **E04 faithful 选点**：训练后用 `eval_dtpose_faithful.py --sweep "<dir>/epoch*_ema.pth"` 在 E04 逐帧评测、挑最低（与 DT-Pose `val=E04` 同口径）。
4. **两个 ckpt 都报**：同时给「E04 最低 MPJPE」与「E04 最低 PA」两个 checkpoint 的全套 E04 数。

---

## 13. 复现步骤

### 13.1 环境

```bash
conda create -n 3DHPE1 python=3.7 -y && conda activate 3DHPE1
pip install torch==1.13.0 torchvision==0.14.0 numpy scipy pillow einops
```

### 13.2 RGB 数据准备（建议预缓存 112）

```bash
python cache_rgb_112.py    # 见脚本内路径配置
```

### 13.3 训练 RGB 教师（Stage A）

```bash
python train_rgb_teacher.py \
    --data_root /home/a123456/PerceptAlign/MMFi --rgb_root ~/MMFi_RGB112 \
    --train_envs E01 E02 E03 --test_env E04 \
    --rgb_img 112 --backbone resnet18 \
    --epochs 50 --batch_size 4 --accumulate_grad 4 \
    --save_dir ./checkpoints/rgb_teacher_dg
```

### 13.4 蒸馏（Step B）

```bash
python train_distill_pretrained.py \
    --data_root /home/a123456/PerceptAlign/MMFi --rgb_root ~/MMFi_RGB112 \
    --train_envs E01 E02 E03 --test_env E04 \
    --teacher_modality rgb \
    --pretrain_ckpt checkpoints/stage1b_action/action_best.pt \
    --teacher_ckpt  checkpoints/rgb_teacher_dg/teacher_best.pt \
    --w_bone 1.0 --w_sym 0.1 --w_temp 0.1 --w_rel 6.0 --w_root_anchor 0.5 \
    --fk_alpha_final 0.4 --fk_alpha_warmup 20 \
    --epochs 50 --batch_size 2 --accumulate_grad 8 --use_ema --seed 42 \
    --save_dir ./checkpoints/distill_rgb_teacher
```

启用乙′（加 DT-Pose 表征特征蒸馏）：追加
```bash
    --dtpose_feat_ckpt pretrain_dtpose_400.pt --w_dtpose_feat 0.1
```

启用 axis_split + B1（DT x 轴轨迹蒸馏）：追加
```bash
    --root_mode axis_split \
    --w_dtx 1.0 --dtx_ckpt pretrain_dtpose_400.pt --dtx_decoder_ckpt pose_dtpose.pt
```

### 13.5 E04 选点（faithful sweep，唯一权威口径）

```bash
python eval_dtpose_faithful.py --data_root /home/a123456/PerceptAlign/MMFi \
    --sweep "./checkpoints/distill_rgb_teacher/epoch*_ema.pth" \
    --test_env E04 --seq_len 64
```

### 13.6 DT-Pose 范式复现（独立实验线）

```bash
# 阶段一: 自监督预训练
python pretrain_dtpose_style.py --data_root /home/a123456/PerceptAlign/MMFi \
    --train_envs E01 E02 E03 --total_epoch 400 --save_path pretrain_dtpose_400.pt
# 阶段二: 拓扑解码
python train_pose_dtpose_style.py --data_root /home/a123456/PerceptAlign/MMFi \
    --train_envs E01 E02 E03 --test_env E04 \
    --pretrained pretrain_dtpose_400.pt --epochs 50 --batch_size 32 --lr 1e-3
# Route B: root/相对分离 + 双 root 策略
python train_pose_dtpose_B.py --data_root /home/a123456/PerceptAlign/MMFi \
    --train_envs E01 E02 E03 --test_env E04 \
    --pretrained pretrain_dtpose.pt --epochs 50 --batch_size 32 --lr 1e-3
```

### 13.7 诊断 / 探针

```bash
python probe_raw_amplitude_hip.py --data_root /home/a123456/PerceptAlign/MMFi \
    --train_envs E01 E02 E03 --test_env E04 --max_frames_per_seq 60 --ridge 1e-2
python diagnose_hip_motion.py --data_root /home/a123456/PerceptAlign/MMFi
python probe_transformer_hip.py --data_root /home/a123456/PerceptAlign/MMFi \
    --train_envs E01 E02 E03 --test_env E04
```

---

## 14. 超参数完整参考（与代码默认对齐）

| 类别 | 参数 | 默认 | 说明 |
|---|---|---|---|
| 数据 | `--seq_len` | 64 | 时间窗口 |
| RGB | `--rgb_img` / `--rgb_root` | 112 / None | RGB resize / 独立根目录 |
| 教师 | `--teacher_modality` | depth | rgb / depth |
| 教师(RGB) | `--backbone` | resnet18 | resnet18 / scratch |
| 教师(RGB) | `--lr` / `--lr_backbone` | 5e-4 / 1e-4 | 新增层 / 预训练主干 |
| 蒸馏 | `--lambda_feat` / `--lambda_out` | 0.1 / 1.0 | 特征 / 输出蒸馏权重 |
| 蒸馏 | `--out_distill_hip_weight` | depth=4.0 / rgb=1.0 | 模态相关默认 |
| 蒸馏 | `--out_distill_align_hip` | rgb=True / depth=False | 模态相关默认 |
| 乙′ | `--dtpose_feat_ckpt` / `--w_dtpose_feat` | None / 0.1 | DT-Pose 特征蒸馏 |
| 姿态 | `--lambda1/2/3` / `--lambda_hip` | 1/0.5/2 / 0.3 | PoseLoss |
| 结构 | `--w_bone/--w_sym/--w_temp/--w_rel` | 1.0/0.1/0.1/6.0 | 结构正则 |
| FK | `--fk_alpha_final` / `--fk_alpha_warmup` | 0.4 / 20 | α 终值 / 退火 |
| FK | `--root_mode` / `--vel_scale` | absolute / 0.12 | root 模式（absolute / velocity / axis_split） |
| B1 | `--w_dtx` | 0 | DT x 轴轨迹蒸馏权重（>0 启用，配 axis_split） |
| B1 | `--dtx_ckpt` / `--dtx_decoder_ckpt` | None / None | DT-Pose 预训练 / 阶段二解码器 |
| root | `--w_root_anchor` | 0.5 | L_anchor 强度 |
| root | `--anchor_anneal_epochs` / `--w_root_anchor_final` | 0 / 0.1 | anchor 退火 |
| RSC | `--rsc2_*_pct` | 0.5 | challenge 比例 |
| 优化 | `--batch_size`/`--accumulate_grad` | 2/8（蒸馏）, 4/4（RGB 教师） | 等效 batch 16 |
| 优化 | `--epochs` | 50 | 跑满 |
| 选点 | `--eval_interval`/`--archive_ckpts` | 3 / True | archive 供 sweep |
| EMA | `--use_ema`/`--ema_decay` | True/0.999 | |
| 复现 | `--seed` | 42 | |

---

## 15. 仓库结构

```
RSC V2/
├── train_distill_pretrained.py   # 主训练: 蒸馏(rgb/depth)+全量+FK+anchor+archive (+乙′)
├── train_rgb_teacher.py          # Stage A (主线): RGB 教师 (ResNet18+MixStyle)
├── train_depth_teacher.py        # Stage A (历史): 深度教师
├── train_mae.py                  # Stage 1A: MAE 自监督预训练 (主 backbone)
├── fk_decoder.py                 # Hybrid FK 解码器 (absolute / velocity / axis_split root)
├── dtpose_rootx_teacher.py       # B1: DT-Pose 解码器作冻结教师, 出 hip.x 去均值轨迹
├── structural_losses.py          # 骨长/对称/时序/root-relative/root-anchor + canonical
├── eval_dtpose_faithful.py       # 逐帧 faithful 评测 + E04 选点 sweep (权威)
├── evaluate.py / evaluate_v2.py  # 指标 / 训练期监控
├── distill_loss.py / losses.py   # 蒸馏损失 / PoseLoss / TotalLoss
├── dtpose_feature_teacher.py     # 乙′: DT-Pose 预训练编码器作冻结特征教师
├── pretrain_dtpose_style.py      # DT-Pose 复现 阶段一: 单帧幅度 MAE+对比+uniformity
├── train_pose_dtpose_style.py    # DT-Pose 复现 阶段二: 冻结encoder+GCN+Transformer
├── train_pose_dtpose_B.py        # DT-Pose 复现 Route B: root/相对分离 + 双root策略
├── probe_raw_amplitude_hip.py    # 原始幅度 -> E04 hip 信息探针 (§11a)
├── probe_transformer_hip.py      # 非线性前端 hip 探针 (§11d)
├── diagnose_hip_motion.py        # 跨域 root 统计 (§11b)
├── dataset.py / dataset_distill.py   # MM-Fi CSI/GT/RGB 加载 + 预处理
├── augmentation.py / rgb_augment.py  # CSI / RGB 数据增强
├── domain_balanced_sampler.py    # 同动作跨环境 batch 采样
├── config.py / utils.py
├── models/
│   ├── full_model.py             # CSIRSCPoseDG (pose_decoder = HybridFKPoseDecoder)
│   ├── csi_encoder.py / local_encoder.py / global_encoder.py
│   ├── pose_decoder.py / taskprompt_decoder.py / fk (root)
│   ├── rgb_teacher.py / depth_teacher.py
│   ├── rsc.py / mixstyle.py / mae_pretrain.py
└── checkpoints/
    ├── stage1b_action/action_best.pt
    ├── rgb_teacher_dg/teacher_best.pt
    └── distill_*/epoch*_{raw,ema}.pth
```

---

## 16. 常见问题与坑

- **0 samples / 索引为空**：GT 从 `--data_root`、RGB 从 `--rgb_root`，二者可不同盘；0 样本时日志打印实际查找路径。
- **机械盘直读原图卡死**：每 epoch 几十万张随机小文件读，预缓存 112 到本地 SSD（§4.3）。
- **教师 ckpt 加载 mismatch（missing/unexpected 上百）**：teacher_best.pt 与当前 `models/rgb_teacher.py` 结构不配套（例如旧版 backbone）。确认用 `rgb_teacher_dg/teacher_best.pt`（resnet18），其 state_dict 键以 `encoder.enc.stem`/`layer` 开头。
- **乙′ 教师加载 `AttributeError: Can't get attribute '_Attention'`**：`pretrain_dtpose_*.pt` 是整对象存盘，`dtpose_feature_teacher.py` 会把 `pretrain_dtpose_style.py` 的类注入 `__main__`；确保该脚本与 ckpt 同目录。
- **监控值 ≠ faithful**：训练期滑窗监控失真可达 100mm+，报告与选点只认 `eval_dtpose_faithful.py`。
- **加载旧 ckpt strict 报错**：改代码后 `rm -rf **/__pycache__`。

---

## 17. 引用

- DT-Pose: *Towards Robust and Realistic Human Pose Estimation via WiFi Signals* (arXiv:2501.09411)
- MM-Fi: *Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing* (NeurIPS 2023 D&B; arXiv:2305.10345)
- Person-in-WiFi 3D: *Unified Model for 3D WiFi Perception*, IEEE TPAMI 2026