# CSI-RSC-PoseDG

**Cross-Environment 3D Human Pose Estimation from WiFi CSI, via RGB-Teacher Knowledge Distillation**

基于 WiFi CSI 的跨环境 3D 人体姿态估计系统。训练期由 **RGB 图像教师**（ImageNet 预训练骨干）通过知识蒸馏向 CSI 学生注入几何/结构先验，**推理阶段只使用 CSI**（不依赖任何 RGB 或视觉输入）。在 MMFi 数据集上以跨房间设定（Setting 3 / Protocol 3，全 27 动作，CSI-only 推理）对标 DT-Pose，**与其对齐评测协议与选点口径**（全量训练 + 在 E04 上选 checkpoint）。

> **硬件配置（客观）**：采用 MMFi 原生 WiFi CSI 配置——**1 个发射天线 × 3 个接收天线（1Tx×3Rx）**，每对天线 114 个子载波，Atheros CSI Tool，一对商用 TP-Link N750 AP，5GHz / 40MHz。每帧 CSI 张量 `(3, 114, 10)`：3 = 接收天线（天线链路），114 = 子载波，10 = 100ms 内时间采样。

> **主线说明**：当前主线为 **RGB 教师蒸馏**。早期曾用深度图教师，因 RGB 可用 ImageNet 预训练骨干而改用 RGB；深度教师作为历史变体保留（§5 备注、§13），不再是主线。

> **当前主攻点**：在 **PA-MPJPE（相对骨架结构）** 上追平/反超 DT-Pose；绝对 **MPJPE** 与 DT-Pose 仍有差距，差距集中在 hip 全局定位（见 §9）。本文不就此差距作"信息论上界"之类的强断言——同条件下已有工作（Person-in-WiFi 3D, TPAMI 2026）将其缩小，详见 §9。

> **进度提示（重要）**：RGB 教师已**训练完成并收敛**（E04 best MPJPE 291.20mm @ epoch 51，§8.0）。**RGB 教师 → CSI 学生的蒸馏（Step B）正在进行中**（distill_A_base 等多组实验已启动）；§8.1 主表的 RGB 学生 faithful 结果待 sweep 选点后回填，暂记 `进行中`。当前唯一已落地的 CSI 学生 faithful 结果（352.23 / 102.75 / 321.88）来自**深度教师**蒸馏，作为前序对照行保留（§8.1），不冒充 RGB 主线结果。

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
9. [Limitation 分析：hip 全局定位差距](#9-limitation-分析hip-全局定位差距)
10. [Checkpoint 选择口径](#10-checkpoint-选择口径)
11. [复现步骤](#11-复现步骤)
12. [超参数完整参考](#12-超参数完整参考)
13. [仓库结构](#13-仓库结构)
14. [常见问题与坑](#14-常见问题与坑)
15. [诚实声明与引用](#15-诚实声明与引用)

---

## 1. 动机与问题定义

WiFi CSI（Channel State Information）作为一种无感（device-free）、不依赖光照、保护隐私的人体感知模态，近年被广泛用于 3D 人体姿态估计。其根本难点在于**跨环境泛化**：CSI 信号由无线多径传播决定，强烈依赖房间几何、家具布置、墙体反射、收发天线位置等环境因素；因此在房间 A 采集数据训练的模型，迁移到结构不同的未见房间 B 时性能会显著下降。

本项目针对该领域的跨房间设定，并**与对比方法（DT-Pose 等）对齐口径**以保证可比性：

- **跨房间（cross-environment）**：训练用 `{E01, E02, E03}`，测试用结构不同的 `E04`。
- **硬件（MMFi 原生配置）**：1 发射天线 × 3 接收天线（1Tx×3Rx），每对天线 114 子载波，Atheros CSI Tool，一对 TP-Link N750 AP。每帧 CSI `(3, 114, 10)`。
- **CSI-only 推理**：RGB 图像仅在训练阶段作为教师参与蒸馏，部署/推理时不可用。
- **全量训练 + E04 选点**：与 DT-Pose（其验证集即 E04、`if val_mpjpe<best:save`）及 MMFi 上多数工作一致——E01–E03 全部 subject 进训练（不另划 val），checkpoint 在 E04 上按指标选取（见 §10）。

形式化定义：给定 CSI 时序窗口 `X ∈ R^(T×9×114×10)`，预测 3D 关节序列 `P ∈ R^(T×17×3)`（绝对世界坐标，单位米）。

---

## 2. 核心贡献

1. **RGB-教师蒸馏框架**：训练期用 RGB 图像姿态教师（ImageNet 预训练 ResNet18 骨干）对 CSI 学生做 feature-level + output-level 双路蒸馏，把视觉结构先验迁移进 CSI 表征；推理只需 CSI。教师模态可切换（`--teacher_modality {rgb,depth}`），RGB 为当前主线。
2. **RSC（Representation Self-Challenging）**：基于梯度的 Top-k 特征 Masking，对全局表征施加时间维 / 通道维 / batch 维随机遮挡（`RSCGlobalChallenger`），强迫模型分散依赖、避免过拟合单一环境捷径特征，提升跨域鲁棒性。
3. **结构正则损失套件**（`structural_losses.py`）：骨长一致、左右对称、时序骨长稳定、root-relative 位置对齐——四项损失全部**平移不变**，专门优化 PA-MPJPE 所度量的相对骨架，**结构上不可能恶化全局定位项**。
4. **Hybrid FK 解码器**（`fk_decoder.py`）：在不替换现有解码器的前提下，外挂一条正运动学（FK）分支（root + 骨长 + 骨方向单位向量 → FK 合成），以 α 退火与原解码器融合；骨架合法性由**构造保证**，是比惩罚项更彻底的 PA 杠杆。FK 支的 root 支持 `absolute`（逐帧绝对）与 `velocity`（锚点 + 速度积分轨迹）两种模式（§3.5、§9）。
5. **Root anchor + 退火**（修复 MPJPE）：以损失项 `L_anchor` 把预测 hip 往按动作的源域 canonical 先验拉，抑制源域过拟合导致的 root 漂移，使预测 root 不致比常数先验更差；支持训练中线性退火（`--anchor_anneal_epochs`），让网络后期能端到端学 root（§3.6）。
6. **输出蒸馏 hip 对齐**（保护学生 PA）：RGB 教师为单目无米制深度，其绝对 hip 不可信。`OutputDistillLoss(align_hip=True)`（RGB 模态自动启用）在蒸馏前对学生/教师各自减 hip，**只迁移相对结构**，不把教师的坏 hip 灌给学生（§3.4）。
7. **hip 全局定位差距的实证分析**：通过原始输入信息探针、教师误差上界、零信息基线对照、hip 运动诊断、对所有候选杠杆的系统性穷举，刻画**本文方法**在跨房间绝对定位上的局限，并诚实对照同条件 SOTA（Person-in-WiFi 3D）以界定该差距的性质（方法局限而非信息论极限，见 §9）。

---

## 3. 方法详解

### 3.1 整体管线

```
CSI 输入 (B, T=64, 9, 114, 10)
   │
   ├─[csi_encoder]        DualBranchCSIEncoder: 幅度支/相位支 (各 EnvNorm+ResBlock2D+MixStyle2D) + GatedFusion
   ├─[local_encoder]      LocalSpatioTemporalEncoder: 3D 残差卷积 (num_res3d_blocks=2)
   ├─[feature_pooling]    LocalFeaturePooling: 空间池化, 每帧聚合为单 token (-> global_dim)
   ├─[global_modeler]     GlobalTemporalModeler: PatchEmbed + MixStyleTemporal +
   │                      Transformer(layers=3, heads=4) + TCN([128,128], 膨胀卷积)
   │                      ──> z_global (B, 64, 128)
   ├─[RSC]                训练期 RSCGlobalChallenger 对 z_global 做基于梯度的 self-challenging
   ├─[pose_decoder]       HybridFKPoseDecoder (结构支 PoseDecoder + FK 支 FKBranch + α 融合)
   │                      ──> p_coarse, p_final ∈ (B, 64, 17, 3)
   └─[action_classifier]  ActionClassifier: 27 类动作分类 (辅助任务) + 产生 action_emb

训练期教师 (推理不用):
   RGB (B,T,3,112,112)  ──[RGBPoseTeacher: ImageNet ResNet18 + MixStyle + GlobalTemporalModeler]──>
   {p_final, z_global}  ──蒸馏──> 对齐学生 z_global (feature) / p_final (output, align_hip)
```

学生网络总参数量：约 **1.63M**（实测 `Student params: 1,629,790`）。当 α=1 时纯走结构支，保证 FK 分支安全冷启动。

### 3.2 各模块详解（学生，源码见 `models/`）

- **csi_encoder（`DualBranchCSIEncoder`）**：输入 `(B,T,9,114,10)`。幅度支（3 通道）与相位支（6 通道）各自 `EnvironmentNormalization`（per-sample InstanceNorm，去环境统计偏移）→ `ResBlock2D`（BN/IN 可学习门控融合）→ `MixStyle2D`（混合域风格）→ `ResBlock2D`，再由 `GatedFusion`（门控 + cross-attention）融合两支。逐帧分 chunk 处理（chunk_size=16）。`encoder_hidden_dim=32, encoder_out_dim=64`。
- **local_encoder（`LocalSpatioTemporalEncoder`）**：`num_res3d_blocks=2` 个 `Res3DConvBlock`（3D 卷积，沿 时间/子载波/packet 三轴），`(B,T,64,114,10) → (B,T,64,114,10)`。
- **feature_pooling（`LocalFeaturePooling`）**：`AdaptiveAvgPool2d(1)` 把每帧 `(114,10)` 池成单 token，线性投到 `global_dim=128`，输出 `(B,T,128)`。
- **global_modeler（`GlobalTemporalModeler`）**：`PatchEmbedding` → `PositionalEncoding` → `MixStyleTemporal` → 3 层 `TransformerBlock`（4 头，dropout=0.3）→ 2 层 `TemporalConvBlock`（TCN，膨胀 `2^i`，残差）→ `LayerNorm`。输出 `z_global (B,T,128)`——全模型核心特征，也是蒸馏对齐对象。
- **pose_decoder（`HybridFKPoseDecoder`）**：结构支 `PoseDecoder`（其 coarse_head 为 `TaskPromptCoarseHead`，每关节带可学习 task prompt + 共享回归器；refiner 为 `SkeletonRefiner` GCN，`gcn_hidden_dim=128`，`num_gcn_layers=3`）+ FK 支 `FKBranch`，α 融合（§3.5）。
- **action_classifier（`ActionClassifier`）**：对 `z_global` 时间池化后分 27 类，输出 `action_logits`；并提供 `action_embed`（训练用 GT 硬查表，推理用预测软概率加权）供解码器做动作条件化。

### 3.3 RSC（Representation Self-Challenging）+ Action Dropout

训练期 `forward_rsc` 在 `z_global` 上做基于梯度的 self-challenging（`RSCGlobalChallenger`）：在分离图上对 `pose_loss` 求 `z_global` 梯度 → 选梯度最大（最"主导"）的 Top-k 时间/通道位置遮挡掉（`rsc2_time_drop_pct/channel_drop_pct/batch_pct=0.5`），迫使模型不依赖单一线索。Masking 作用在保留计算图的 `z_global_raw` 上，梯度仍经未遮挡元素回传到 backbone。

**Action Dropout**：`forward_rsc` 中训练时以 **50% 概率把 `action_emb` 置零**，迫使解码器不过度依赖动作标签，提升跨域鲁棒性（避免 E04 动作分类一旦出错就连带姿态崩）。

### 3.4 知识蒸馏（RGB 教师）

教师 `RGBPoseTeacher`（RGB 输入，训练期由 `FrozenPoseTeacher` 冻结封装）对学生做两路蒸馏（`distill_loss.py`）：

- **Feature-level（`FeatureDistillLoss`）**：学生 `z_global` 经 `DistillProjection` 投影后，与教师 `z_global` 做余弦相似 + Smooth-L1 对齐（`lambda_feat=0.1`）。教师与学生共享 `GlobalTemporalModeler` 几何，`z_global` 维度同为 128，直接对齐。
- **Output-level（`OutputDistillLoss`）**：学生 `p_final_clean` 与教师 `p_final` 做姿态级 Smooth-L1 蒸馏（`beta=0.05`=5cm，对教师残差噪声鲁棒），由 `lambda_out=1.0` 加权。
  - **`align_hip`（RGB 模态自动 True）**：蒸馏前学生/教师各自减 hip（参考点 detach），只蒸馏**相对结构**，hip 关节从均值中排除（hip 零梯度）。已验证：全局平移不变、学生=教师+平移时 loss≈0。这样教师那条不可信的绝对 hip 不会传给学生，学生绝对 hip 由 `L_pose` 坐标项 + `L_anchor` 负责，与本项解耦。depth 模态 `align_hip=False`（原行为）。

> **教师 hip 权重（`out_distill_hip_weight`）模态相关默认**：RGB=1.0（hip 不可信，不强监督），depth=4.0。当 `align_hip=True` 时 hip 已被排除，该权重不起作用。

### 3.5 Hybrid FK 解码器（PA 主杠杆 + root 模式）

FK 把姿态参数化为「root + 每条骨长度 + 每条骨方向单位向量」，沿运动学树合成绝对坐标，使**骨架合法性成为构造保证**。

```
joints[0] = root
for e, (parent, child) in enumerate(EDGES):
    joints[child] = joints[parent] + bone_dir[e] · bone_len[e]
```

```python
EDGES = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),
         (0,7),(7,8),(8,9),(9,10),
         (8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]
```

FK 支（`FKBranch`）：`z_global → trunk → {dir_head (单位向量), len_head (softplus+clamp[0.02,0.8]), root}`。

**root 两种模式**（`--root_mode`，由 `full_model.py` 透传给 decoder）：
- `absolute`（默认，原行为）：`root_t = root_head(z_t)`，逐帧绝对回归。
- `velocity`（新）：`v_t = tanh(vel_head(z_t))·vel_scale`（每帧限幅位移）→ `traj = cumsum(v)`，再 `traj -= mean_t(traj)`（去时间均值，绕锚波动不漂移）→ `anchor = anchor_head(mean_t z)`（单一绝对锚）→ `root_t = anchor + traj`。其性质是 **`root` 的时间均值恰好 == anchor**，因此 `L_anchor`（约束时间均值 hip）监督的正是 anchor，无需新接线。

**Hybrid（不替换、只外挂）**：`p_final = α·p_struct + (1-α)·p_fk`，`α` 注册为 buffer，逐 epoch 从 1.0 退火到 `fk_alpha_final`（默认 0.4），`fk_alpha_warmup`（默认 20）内完成；早期 α≈1 安全冷启动。EMA 对浮点 buffer（含 α）做平均，评测自动用最终 α。

### 3.6 结构正则与 root anchor

```
L_bone   = L1( bonelen(p), bonelen(g) )                  # 需 GT (源域)
L_sym    = mean L1( len(a), len(b) )                     # 无需 GT, 左右同名骨等长
L_temp   = mean_t | bonelen(p)_{t+1} − bonelen(p)_t |    # 无需 GT, 同窗骨长恒定
L_rel    = L1( p − p_hip, g − g_hip )                    # 髋中心相对位置, PA 最对症杠杆
L_anchor = SmoothL1( p_hip, canonical[action] )          # root 往按动作源域先验拉
```

前四项（`structural_loss`）对全局平移不变 → 只重塑相对骨架、不碰 hip 全局 xyz。蒸馏配置 `w_rel=6.0`（比教师训练的 2.0 更重，因学生形状更难学）。

`L_anchor` 用源域按动作平均 hip（`build_action_canonical` 预扫训练集 GT，与 E04 无关）做稳健先验。**退火**（`--anchor_anneal_epochs N`，0=原行为不退火）：权重从 `w_root_anchor` 线性退到 `w_root_anchor_final`，前期强按住防漂、后期放开让网络端到端学 root。**velocity 模式下** `L_anchor` 自动只约束**时间均值 hip = canonical**（`smooth_l1(pred_hip.mean(t), tgt)`），放开零均值轨迹，否则逐帧 anchor 会把速度积分学到的轨迹重新摁平。

### 3.7 总损失

```
L_total = L_pose(clean) + α·L_pose(masked) + β·L_cons + γ·(反塌陷项) + δ·L_action   # TotalLoss
        + lambda_feat·L_distill_feat + lambda_out·L_distill_out
        + w_bone·L_bone + w_sym·L_sym + w_temp·L_temp + w_rel·L_rel                 # structural_loss
        + w_root_anchor(t)·L_anchor                                                 # 可退火
```

其中 `L_pose`（`PoseLoss`）= `L_coord + λ1·L_bone + λ2·L_vel + λ3·L_motion + λ_hip·L_hip`；`L_motion`（`MotionGuidanceLoss`）用 detach 解耦相邻帧、给每帧独立的 GT 位移方向梯度，专治时序塌缩。蒸馏配置 `λ_hip=0.3`（RGB 模态）。TotalLoss 的反塌陷项（`gamma`）默认 0。

---

## 4. 数据集与预处理

### 4.1 MMFi 目录结构与划分

```
<data_root>/                              <rgb_root>/  (可在独立磁盘)
  E01/ S01..S10 / A01..A27/                 E01/ S01..S10 / A01..A27/
    ├─ wifi-csi/ frameNNN.mat                 └─ rgb/ frameNNN.png  (640x480 RGB)
    ├─ ground_truth.npy (num_frames,17,3)        (或 A01/ 下直接放 frameNNN.png, 自动探测)
    └─ depth/ frameNNN.png (历史, 深度教师用)
```

| 划分 | 环境 | Subjects | 用途 |
|---|---|---|---|
| 训练 | E01–E03 | S01–S30 | **全部进训练（不另划 val）** |
| 测试 / 选点 | **E04** | S31–S40 | 跨房间测试；checkpoint 在其上选取（与 DT-Pose 同口径，§10） |

> GT/CSI 从 `--data_root`，RGB 从 `--rgb_root`（可不同盘，如 `/home/a123456/PerceptAlign/MMFi_Defaced_RGB`，人脸已打码）。本次 RGB 蒸馏实测：训练 6480 窗口（stride=32），E04 测试 1080 窗口（stride=64 无重叠）。

### 4.2 CSI 张量与预处理（`dataset.py::CSIPreprocessor`）

每帧 `.mat` 含 `CSIamp`/`CSIphase`，形状 `(3, 114, 10)`，对应 MMFi 原生配置：`3` = 3 接收天线链路，`114` = 子载波，`10` = 100ms 时间窗采样。

预处理拼成网络输入 `(T=64, 9, 114, 10)`：
- **幅度路（3 通道）**：`normalize_amplitude` 逐帧 min-max 归一化（沿子载波/packet 维），NaN/Inf 清零。
- **相位路（6 通道）**：`process_phase` = 沿子载波 `unwrap` → `scipy.signal.detrend`（线性去趋势，消除 SFO/STO 时钟不同步引入的线性相位污染）→ `[sin, cos]` 编码（避免相位环绕不连续）。3 天线 × 2 = 6 通道。
- 3 天线维与 10 packet 维全程保留进网络。

> 评测脚本 `evaluate_v2.py` 用纯 numpy 闭式最小二乘重写了 detrend（`_detrend_linear_np`），与 `scipy.signal.detrend` 逐元素一致（误差 ~1e-15），以规避某些 scipy/BLAS 构建上 LAPACK 崩溃，保证训练/评测预处理口径一致。

### 4.3 RGB 预处理与工程（`dataset_distill.py`）

- RGB 逐帧 PNG → `[0,1]`，`F.interpolate` 到 `rgb_img`（默认 112）。**ImageNet mean/std 归一化在 RGB 教师模型内部做**（`RGBEncoder` 的 `px_mean/px_std` buffer），dataloader 只出 `[0,1]`。
- **磁盘/IO 工程**：原始 640×480 RGB 全集约 90G+；网络输入本就是 112，故可**预缓存为 112×112**（整套约几 G），既省盘又解决机械盘随机小文件 I/O 瓶颈（原图直读机械盘约 30–60h/epoch，缓存到本地 SSD 后约 7min/epoch）。缓存后 `--rgb_root` 指向缓存目录、`--rgb_img 112` 不变。
- `--rgb_root` 支持 RGB 独立放盘，并自动探测 `E/S/A/rgb/` 与 `E/S/A/` 两种布局（`_resolve_rgb_dir`）。

### 4.4 CSI 数据增强（`augmentation.py::CSIAugmentor`，仅训练）

训练集 `csi_augment=True` 时启用，模拟跨环境信道差异：幅度随机缩放、相位高斯噪声、子载波 dropout、频带 mask（SpecAugment 式）、天线 dropout/shuffle、时序 jitter。测试集不增强。

---

## 5. 训练流程（四阶段）

| 阶段 | 脚本 | 产出 | 说明 |
|---|---|---|---|
| **Stage 1A** | `train_mae.py` | `stage1a_mae/` | （可选）MAE 自监督预训练 backbone；含 DCL 域一致表征项 + 域平衡采样。经验上对跨环境无增益（§8.3），非必需 |
| **Stage 1B** | action 预训练 | `stage1b_action/action_best.pt` | 动作监督预训练 backbone（5 模块：csi_encoder/local_encoder/feature_pooling/global_modeler/action_classifier） |
| **Teacher** | `train_rgb_teacher.py` | `rgb_teacher_dg/teacher_best.pt` | **RGB 姿态教师**（ImageNet ResNet18 + MixStyle + 光度增强 + 冻结浅层 + EMA），蒸馏时冻结 |
| **Distill** | `train_distill_pretrained.py` | `distill_*/epoch*_ema.pth` | 在 1B backbone 上 RGB 蒸馏 + 结构正则 + FK + root anchor；全量训练、archive 全程 ckpt |

> 备注（depth 历史变体）：早期主线为深度教师（`train_depth_teacher.py` + `models/depth_teacher.py`），其 CSI 学生 faithful 结果 352.23/102.75/321.88 见 §8.1 对照行。改用 RGB 后深度教师保留可用（`--teacher_modality depth`），但本地深度图已删（机械盘有备份），复现需先取回。

---

## 6. 实验设定（与 DT-Pose 对齐）

| 项 | 设定 |
|---|---|
| 数据集 / 划分 / 协议 | MMFi / Setting 3（cross-env）/ Protocol 3（全 27 动作） |
| 训练环境 | E01–E03（S01–S30），**全部 subject 进训练** |
| 测试 / 选点 | **E04（S31–S40）**，跨房间测试 + 在其上选 checkpoint（与 DT-Pose `val=E04` 同口径） |
| 训练 | 全量、不早停、跑满 `--epochs`（默认 50） |
| 输入 | MMFi 原生 CSI（1Tx×3Rx，每对天线 114 子载波），64 帧窗口 |
| 推理 | **CSI-only**（RGB 仅训练期教师，`action_idx=None`） |
| 评测口径 | 逐帧、全帧覆盖、无 padding、无重叠铺窗 + 尾窗补、每帧恰好预测一次 |
| MPJPE / PA-MPJPE | 绝对误差不 centering（与 DT-Pose 一致）/ Procrustes（含 optimal scaling）对齐后 MPJPE |

报告数一律来自 `eval_dtpose_faithful.py`（唯一权威口径）；训练期 `monitor_e04` 滑窗监控值（含 raw / ema 两套）与 faithful 不可比，仅供观察、不用于选点与报告。

---

## 7. 评测指标定义（`evaluate.py::PoseEvaluator`）

```
MPJPE          = 绝对, 无对齐
MPJPE_aligned  = 每帧减 hip 后 (相对结构), 对应 DT-Pose calulate_error(align=True)
PA-MPJPE       = Procrustes(旋转+缩放+平移) 对齐后 (纯形状), 含 optimal scaling
hip_error      = 仅 hip(joint 0) 全局定位误差 (evaluate_v2.hip_error)
PCK@τ_norm     = 关节误差 < τ·(LShoulder-LHip 身长) 的占比 (DT-Pose 风格)
```

本项目差距集中在 `hip_error`（全局定位）；`PA-MPJPE` 与 SOTA 持平——"形状对了，绝对位置还差"。

---

## 8. 结果与消融

### 8.0 RGB 教师强度（E04 sanity；教师可用 RGB，推理学生不用）

RGB 教师（ImageNet ResNet18 + DG，`train_rgb_teacher.py`）已**跑满 50 epoch 并收敛**，E04 best（EMA 选点）：

| 检查点 | MPJPE | MPJPE_aligned | PA | hip | PCK@50n | 备注 |
|---|---:|---:|---:|---:|---:|---|
| RGB 教师 @e3（早期，未收敛） | 293.28 | 119.31 | 102.77 | 263.0 | 67.2% | 仅 3 epoch |
| **RGB 教师 best @e51（收敛）** | **291.20** | 118.4 | 96.7 | 258.2 | 67.4% | `teacher_best.pt` |

判读：RGB 教师**结构强且收敛干净**——DG 套件（MixStyle + 光度增强 + 冻结浅层 + EMA）把过拟合按住，E04 MPJPE 从训练初期单调下降到 291 后平台，不再像无 DG 版那样反弹（旧版从 293 涨到 369）。形状（PA/MPJPE_aligned）约在 epoch 33 收敛，后续 MPJPE 微降主要靠 hip。对蒸馏（align_hip 只迁形状）而言，该教师已是强形状教师，**Step B 值得做**。

### 8.1 主结果（E04，faithful 逐帧口径；全量训练 + E04 选点）

> RGB 学生（RGB 教师蒸馏）为当前主线，**Step B 进行中**，故为 `进行中`。depth 学生为前序对照。

| 模型 | MPJPE (mm) | PA-MPJPE (mm) | hip_err (mm) | 选点 |
|---|---:|---:|---:|---|
| **RGB 教师蒸馏（主线）** | _进行中_ | _进行中_ | _进行中_ | E04 faithful sweep |
| depth 教师蒸馏（前序对照） | 352.23 | 102.75 | 321.88 | E04 faithful sweep（epoch006_ema） |
| **DT-Pose (S3 / P3)** | 316.8 | 104.2 | — | — |
| **Person-in-WiFi 3D (S3 / P3)** | **302.5** | **101.1** | — | TPAMI 2026，同条件参考 |

- depth 对照行：**PA 102.75 优于 DT-Pose 104.2**。MPJPE 352.23 仍落后；差距集中在 hip（§9）。
- **Person-in-WiFi 3D 行**：同为 MMFi S3/P3、原生 CSI，MPJPE 302.5 / PA 101.1，**同条件下 MPJPE 与 PA 均优于 DT-Pose**。这说明本文与 DT-Pose 的 MPJPE 差距**并非信息论极限，而是建模空间**（见 §9）。
- RGB 学生目标：在 PA 上压低 depth 学生的 102.75，并尽量缩小 MPJPE/hip 差距。

### 8.2 PA-MPJPE 结构杠杆消融（历史趋势，口径见注）

> 下表为历史口径的相对趋势，仅说明结构正则方向性有效（平移不变损失只重塑相对骨架、不碰 root）；绝对值待新口径重测。

| 配置 | PA-MPJPE | ΔPA | MPJPE | MPJPE_aligned |
|---|---:|---:|---:|---:|
| baseline | 106.2 | — | 366.6 | — |
| + 骨长/对称/时序 | 105.42 | −0.78 | 361.97 | 122.24 |
| + root-relative | 104.73 | −0.69 | 363.74 | 120.24 |

### 8.3 当前 Step B 实验矩阵（hip / MPJPE 杠杆）

教师统一用 RGB DG 教师（291mm）`teacher_best.pt`，预训练用 `stage1b_action/action_best.pt`，RGB 走 `align_hip=True`：

| 实验 | 关键参数 | 目的 | 依赖 |
|---|---|---|---|
| **A** baseline | 默认（anchor=0.5, absolute root） | 确认管线（含 align_hip）正常跑通 | distill_loss 带 align_hip |
| **B** 端到端 root | `--w_root_anchor 0.0` | 验证松开 anchor 能否释放 hip（最高优先级，零外部依赖） | 无 |
| **C** anchor 退火 | `--anchor_anneal_epochs 20 --w_root_anchor_final 0.0` | 前期稳后期放，渐进释放 hip | 无 |
| **D** 速度积分 root | `--root_mode velocity --vel_scale 0.12 --anchor_anneal_epochs 20 --w_root_anchor_final 0.1` | 捞 hip 轨迹动态 | 已透传（full_model 含 root_mode） |

判读：每组训练后用 faithful sweep 选点，只看 **MPJPE + hip**。B 的 hip 若掉到 307 以下、MPJPE 往 330 以下走 → anchor 钉死了 hip，顺这条挖；若 hip 反涨 → 常数先验确是最优，转向"PA 达 SOTA + 诚实写 hip limitation"。

### 8.4 已穷举、本文方法下未能改善 hip 的杠杆

| 杠杆 | 现象 | 结论 |
|---|---|---|
| 解码器结构（多轮） | hip_err 不动 | 本文方法下无效 |
| MAE-DCL 预训练 | 健康 backbone 无增益 | 停止 |
| 合规 TTA | 杠杆耗尽 ~354 | 已用满 |
| 加大 hip 蒸馏 / lambda_hip | hip 仅 340→337 | 无效 |
| raw / log 幅度输入救 MPJPE | E04 反向迁移（§9a） | 停止 |
| 保尺度幅度残差支路（非线性） | scale=0.3 全部 9 ckpt 劣于纯先验 | 停止（§9d，已归档 `_archive/`） |
| 相位去噪 / 共轭 / spotfi 救 hip | 所有相位特征 < 零信息基线 | 封闭（§9a） |
| 全局 velocity-root 作"银弹" | GT 诊断理论上界仅 ~7.6mm | 降级为诚实小增量（§9e） |

> 注：这些是**本文这套方法/架构**下试过且无效的杠杆，不代表 hip 不可改善——Person-in-WiFi 3D 用 DETR 式端到端 set prediction + 相位去噪在同条件下取得了更低 MPJPE（§9）。

---

## 9. Limitation 分析：hip 全局定位差距

本文 CSI 学生与 DT-Pose 的绝对 MPJPE 差距，**集中在 hip 全局定位**；`PA-MPJPE` 已与 SOTA 持平。以下实测刻画**本文方法在跨房间 hip 定位上的局限**。

> **重要更正（不再主张"信息论上界"）**：本文早期版本曾据下列实测推断"跨房间绝对定位是单设备 CSI 的信息论上界、不可逾越"。该断言**过强，现予撤回**。证据：Person-in-WiFi 3D（TPAMI 2026）在**完全相同**的 MMFi S3/P3、相同的原生单设备 CSI 下，取得 MPJPE 302.5 / PA 101.1，**同条件下 MPJPE 优于 DT-Pose 的 316.8**。若该跨房间定位信息真的不存在于 CSI 中，此结果不可能取得。故正确结论是：**该差距是本文方法的建模局限，而非信息论极限。** 下列实测因此重新定性为"本文方法未能提取该信息"，而非"信息不存在"。

### (a) 原始输入信息探针（`probe_raw_amplitude_hip.py`）

| 特征 | held-in (E01–03) | E04 |
|---|---:|---:|
| mean_base（零信息标尺） | 172.7 | 324.2 |
| 原始绝对幅度 | 152.3 | 350.8 |
| log 功率 | 155.5 | 348.8 |
| 逐帧归一化幅度 | 149.6 | 356.4 |

**线性**岭回归从幅度直接回归 E04 hip，在 E04 上劣于零信息标尺 → 说明**线性手段**无法从幅度跨房间迁移定位。相位特征同理（detrend/conj/spotfi 在 E04 上均 ≥ 324 零信息基线）。这是线性/手工特征探针的局限，非信息不存在（Person-in-WiFi 3D 的非线性端到端模型在同数据上做到了更低 MPJPE）。

### (b) 教师误差上界（本文教师）

本文深度教师在 E04 hip ~236mm、RGB 教师 ~258mm —— 即本文**教师本身**的 hip 监督质量有限，故 output distill 难以靠教师把学生 hip 拉准。这正是 RGB 蒸馏改用 `align_hip`（不迁教师绝对 hip）的原因。这是本文蒸馏路线的上界，不是 WiFi 模态的上界。

### (c) 本文完整模型 vs 零信息基线

本文模型在 E04 的 hip（prior-root 行 321.88）接近零信息基线（324）。把 root 钉到按动作先验后 faithful hip 收至 ~322 ≈ 324 —— 说明**本文方法**在 hip 上未超过常数先验。

### (d) hip 运动诊断（`diagnose_hip_motion.py`，velocity-root 的理论上界）

用 GT 对 E04 做 hip 运动分解（轨迹 RMS / 锚偏移 / oracle 收益）：整体 velocity-root 的**理论最大收益仅 ~7.6mm**（且这是用 GT 轨迹的上界，真实更少），因为 27 个动作里约 11 个 hip 轨迹 RMS≈0（原地动作）。收益高度集中在少数走动类（A15 理论收益 83.9mm、A16 29mm）。**结论**：velocity-root 是诚实的小增量，不是把 MPJPE 打到个位数的银弹；35mm 差距的主体在锚偏移（~307mm，跨房间绝对定位），那是本文方法未能提取的部分。

### (e) 杠杆穷举（本文方法下）

解码器结构、MAE-DCL、合规 TTA、蒸馏权重、raw/log 幅度、相位去噪、action-prior root、root 解耦解码器、保尺度幅度残差支路、全局 velocity-root —— 均未在**本文这套架构**下显著移动 hip 误差。这界定了本文方法的边界，不构成信息论结论。

**结论（修订）**：本文 CSI 学生在 PA-MPJPE 上追平/反超 DT-Pose，但绝对 MPJPE 仍落后约 35mm，差距集中在 hip。该差距是**本文方法的局限**。**同条件 SOTA Person-in-WiFi 3D（TPAMI 2026）已将 MMFi S3/P3 的 MPJPE 推进到 302.5 / PA 101.1**，其关键不同在于 DETR 式端到端 set prediction 架构与 CSI 相位去噪。因此缩小本文 hip 差距是**明确可行的未来方向**——引入端到端 set-prediction 解码、CSI 相位去噪等，而非本文路线的小修小补。本文不就此差距作任何"信息论不可能"的断言。

---

## 10. Checkpoint 选择口径

1. **全量训练**：E01–E03 全部 subject，不另划 val、不早停，跑满 `--epochs`。
2. **archive**：每 `--eval_interval` epoch 存 `epochNNN_{raw,ema}.pth`（`--archive_ckpts` 默认开）。训练期 E04 `monitor_e04` 为滑窗口径，仅观察、不选点。
3. **E04 faithful 选点**：训练后用 `eval_dtpose_faithful.py --sweep "<dir>/epoch*_ema.pth"` 在 E04 对所有 archive ckpt faithful 逐帧评测、挑最低。与 DT-Pose（`val=E04`）同口径，且用 faithful 而非滑窗。
4. **两个 ckpt 都报**：脚本同时给「E04 最低 MPJPE」与「E04 最低 PA」两个 checkpoint，报告两者全套 E04 数。

---

## 11. 复现步骤

### 11.1 环境

```bash
conda create -n 3DHPE1 python=3.7 -y && conda activate 3DHPE1
pip install torch==1.13.0 torchvision==0.14.0 numpy scipy pillow
```

### 11.2 RGB 数据准备（建议预缓存 112）

```bash
python - << 'EOF'
import os, glob
from PIL import Image
SRC = "/path/to/MMFi_Defaced_RGB"; DST = os.path.expanduser("~/MMFi_RGB112"); SIZE=112
for env in ['E01','E02','E03','E04']:
    for adir in sorted(glob.glob(f"{SRC}/{env}/S*/A*")):
        rgb = f"{adir}/rgb" if os.path.isdir(f"{adir}/rgb") else adir
        out = f"{DST}/{os.path.relpath(rgb, SRC)}"; os.makedirs(out, exist_ok=True)
        for fp in glob.glob(f"{rgb}/frame*.png"):
            op=f"{out}/{os.path.basename(fp)}"
            if not os.path.exists(op):
                Image.open(fp).convert('RGB').resize((SIZE,SIZE),Image.BILINEAR).save(op)
EOF
```

### 11.3 训练 RGB 教师（Stage A）

```bash
python train_rgb_teacher.py \
    --data_root /path/to/MMFi --rgb_root ~/MMFi_RGB112 \
    --train_envs E01 E02 E03 --test_env E04 \
    --rgb_img 112 --backbone resnet18 \
    --dg_mixstyle --freeze_stages 2 --rgb_aug \
    --w_bone 1.0 --w_sym 0.1 --w_temp 0.1 --w_rel 2.0 \
    --use_ema --select_metric MPJPE \
    --epochs 50 --batch_size 4 --accumulate_grad 4 --num_workers 8 \
    --save_dir ./checkpoints/rgb_teacher_dg
```

### 11.4 RGB 蒸馏（Step B）

```bash
python train_distill_pretrained.py \
    --data_root /path/to/MMFi --rgb_root ~/MMFi_RGB112 \
    --train_envs E01 E02 E03 --test_env E04 \
    --teacher_modality rgb \
    --pretrain_ckpt checkpoints/stage1b_action/action_best.pt \
    --teacher_ckpt  checkpoints/rgb_teacher_dg/teacher_best.pt \
    --rgb_img 112 \
    --w_bone 1.0 --w_sym 0.1 --w_temp 0.1 --w_rel 6.0 --w_root_anchor 0.5 \
    --fk_alpha_final 0.4 --fk_alpha_warmup 20 \
    --epochs 50 --batch_size 2 --accumulate_grad 8 --use_ema --seed 42 \
    --save_dir ./checkpoints/distill_rgb_teacher
```

RGB 模态会自动启用 `align_hip=True`（保护学生 PA）。hip / MPJPE 实验开关（§8.3）：
```bash
# B: 关 anchor, root 端到端学
--w_root_anchor 0.0
# C: anchor 退火
--anchor_anneal_epochs 20 --w_root_anchor_final 0.0
# D: 速度积分 root
--root_mode velocity --vel_scale 0.12 --anchor_anneal_epochs 20 --w_root_anchor_final 0.1
```

### 11.5 E04 选点（faithful sweep，唯一权威口径）

```bash
python eval_dtpose_faithful.py --data_root /path/to/MMFi \
    --sweep "./checkpoints/distill_rgb_teacher/epoch*_ema.pth" \
    --test_env E04 --seq_len 64
```

### 11.6 诊断 / 探针

```bash
# 原始幅度 -> E04 hip 信息探针 (§9a)
python probe_raw_amplitude_hip.py --data_root /path/to/MMFi \
    --train_envs E01 E02 E03 --test_env E04 --max_frames_per_seq 60 --ridge 1e-2
# hip 运动诊断: velocity-root 理论收益 (§9d)
python diagnose_hip_motion.py --data_root /path/to/MMFi \
    --train_envs E01 E02 E03 --test_env E04 --seq_len 64
```

---

## 12. 超参数完整参考（与代码默认对齐）

| 类别 | 参数 | 默认 | 说明 |
|---|---|---|---|
| 数据 | `--seq_len` | 64 | 时间窗口 |
| RGB | `--rgb_img` / `--rgb_root` | 112 / None | RGB resize 尺寸 / 独立根目录 |
| 教师 | `--teacher_modality` | depth | **rgb=主线** / depth=历史 |
| 教师(RGB) | `--backbone` | resnet18 | resnet18(ImageNet) / scratch(兜底) |
| 教师(RGB) | `--freeze_stages` / `--mixstyle_p` | 2 / 0.5 | 冻结浅层 / MixStyle 概率 |
| 蒸馏 | `--lambda_feat` / `--lambda_out` | 0.1 / 1.0 | 特征 / 输出蒸馏权重 |
| 蒸馏 | `--out_distill_hip_weight` | depth=4.0 / rgb=1.0 | 模态相关默认 |
| 蒸馏 | `--out_distill_align_hip` | rgb=True / depth=False | 蒸馏前减 hip 只传相对结构 |
| 蒸馏 | `--out_distill_beta` | 0.05 | Smooth-L1 拐点 (5cm) |
| 姿态 | `--lambda1/2/3` / `--lambda_hip` | 1/0.5/2 / **0.3**（蒸馏） | PoseLoss（教师训练 lambda_hip=1.0） |
| 结构 | `--w_bone/--w_sym/--w_temp/--w_rel` | 1.0/0.1/0.1/**6.0**（蒸馏） | root-relative 为 PA 主杠杆（教师 w_rel=2.0） |
| FK | `--fk_alpha_final` / `--fk_alpha_warmup` | 0.4 / 20 | α 终值 / 退火 epoch |
| FK | `--root_mode` / `--vel_scale` | absolute / 0.12 | root 模式 / velocity 每帧位移限幅 |
| root | `--w_root_anchor` | 0.5 | L_anchor 强度 |
| root | `--anchor_anneal_epochs` / `--w_root_anchor_final` | 0 / 0.1 | 0=不退火；>0 线性退火到终值 |
| RSC | `--rsc2_*_pct` | 0.5 | challenge 比例（time/channel/batch） |
| 优化 | `--batch_size`/`--accumulate_grad` | 2/8（蒸馏）, 4/4（RGB 教师） | 等效 batch 16 |
| 优化 | `--lr_backbone` / `--lr_head` | 1e-4 / 5e-4 | 分组学习率（蒸馏） |
| 优化 | `--epochs` | 50 | 跑满, 不早停 |
| 选点 | `--eval_interval`/`--archive_ckpts` | 3 / True | archive 供 E04 faithful sweep |
| EMA | `--use_ema`/`--ema_decay` | True/0.999 | 含 FK α 的浮点 buffer 被跟踪 |
| 复现 | `--seed` | 42 | |

---

## 13. 仓库结构

```
RSC V2/
├── train_distill_pretrained.py   # 主训练: 蒸馏(rgb/depth)+全量+FK+anchor退火+align_hip+archive
├── train_rgb_teacher.py          # Stage A (主线): RGB 教师 (ResNet18 + MixStyle + 增强 + EMA, E04 选点)
├── train_depth_teacher.py        # Stage A (历史): 深度教师
├── train_distill.py              # Step B 早期诊断版 (仅 feature distill, 单阶段)
├── train_mae.py                  # Stage 1A: MAE 自监督预训练 (+ DCL + 域平衡采样)
├── train.py                      # 早期端到端训练 (config.py 驱动, 含早停)
├── fk_decoder.py                 # Hybrid FK 解码器 (FKBranch: absolute/velocity root + α 融合)
├── structural_losses.py          # 骨长/对称/时序/root-relative + root-anchor + build_action_canonical
├── distill_loss.py               # DistillProjection / FeatureDistillLoss / OutputDistillLoss(align_hip) / KinematicPriorLoss
├── losses.py                     # PoseLoss / TotalLoss (含 MotionGuidanceLoss 抗时序塌缩)
├── taskprompt_decoder.py         # TaskPromptCoarseHead (per-joint prompt) + uniformity_loss
├── eval_dtpose_faithful.py       # 逐帧 faithful 评测 + E04 选点 sweep (权威)
├── evaluate.py / evaluate_v2.py  # 指标实现 (PoseEvaluator) / 训练期监控 + hip_error + 方差工具
├── probe_raw_amplitude_hip.py    # 原始幅度 → E04 hip 信息探针 (§9a)
├── diagnose_hip_motion.py        # hip 运动诊断: velocity-root 理论收益 (§9d)
├── dataset.py                    # MMFi CSI/GT 加载 + CSIPreprocessor (amp/phase 预处理) + CSIAugmentor
├── dataset_distill.py            # 蒸馏数据 (csi + rgb/depth; --rgb_root 独立盘 + 布局自探测)
├── augmentation.py               # CSIAugmentor (跨环境 CSI 增强)
├── domain_balanced_sampler.py    # 同动作跨环境 batch 采样 (MAE 的 DCL 用)
├── config.py / utils.py
├── action_prior_root.py / prior_root_decoder.py  # 已证否的 root 方案 (历史, 见 §8.4)
├── _archive/                     # 已证否实验 (§9): prior_root / raw_scale_encoder / viz_eval
├── models/
│   ├── full_model.py             # CSIRSCPoseDG (forward_rsc; pose_decoder=HybridFKPoseDecoder; root_mode 透传)
│   ├── csi_encoder.py            # DualBranchCSIEncoder (幅度/相位双支 + EnvNorm + MixStyle + GatedFusion)
│   ├── local_encoder.py          # LocalSpatioTemporalEncoder (Res3DConv) + LocalFeaturePooling
│   ├── global_encoder.py         # GlobalTemporalModeler (Transformer + TCN + MixStyleTemporal)
│   ├── rsc.py                    # RSCGlobalChallenger / RSCLocalChallenger (基于梯度的 self-challenging)
│   ├── pose_decoder.py           # PoseDecoder / SkeletonRefiner(GCN) / ActionClassifier / TaskPrompt 接入
│   ├── mixstyle.py               # MixStyle / MixStyle2D / MixStyleTemporal (域风格混合)
│   ├── rgb_teacher.py            # RGBPoseTeacher (主线; resnet18/scratch 双骨干 + DG)
│   ├── depth_teacher.py          # DepthPoseTeacher (历史)
│   ├── mae_pretrain.py           # CSIMaeModel (Stage 1A)
│   └── __init__.py
└── checkpoints/
    ├── stage1b_action/action_best.pt
    ├── rgb_teacher_dg/teacher_best.pt
    └── distill_*/epoch*_{raw,ema}.pth
```

---

## 14. 常见问题与坑

- **0 samples / 索引为空**：GT 从 `--data_root`、RGB 从 `--rgb_root`，二者可不同盘；0 样本时日志打印实际查找路径。最常见原因是 RGB 蒸馏漏传 `--rgb_root`（默认回退到 data_root，那里没有 RGB）。
- **`OutputDistillLoss got unexpected keyword 'align_hip'`**：仓库 `distill_loss.py` 不是带 `align_hip` 的版本。更新该文件即可（仅 `OutputDistillLoss` 增加该开关，其余类不变）。
- **`[root_mode=velocity]` WARNING：FK 未运行在 velocity 模式**：`full_model.py` 未把 `args.root_mode/vel_scale` 透传给 `HybridFKPoseDecoder`。当前版本已透传（`getattr(args,'root_mode','absolute')`）；若自定义改动丢了这行需补回。
- **机械盘直读原图卡死 / GPU-Util 长期 0%**：每 epoch 几十万张随机小文件读，机械盘扛不住；瓶颈在数据加载而非计算。**预缓存 112 到本地 SSD**（§4.3）+ 适当加 `--num_workers`，降到 ~7min/epoch。显存富余时可调大 `--batch_size` 并相应降 `--accumulate_grad` 保持等效 batch=16。
- **disk full / 训练中途崩**：原始 RGB 全集 ~90G+；优先用 112 缓存（几 G）。
- **torchvision 装不上 / ResNet 权重下载失败**：`--backbone scratch` 兜底；或手动放权重到 `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`。
- **监控值 ≠ faithful**：训练期滑窗监控失真可达 100mm+，只认 `eval_dtpose_faithful.py`，选点也走它。
- **训练日志里 `Out: ~20000mm` 吓人**：这是 `OutputDistillLoss` 的诊断尺度 `l_distill_out_mm`，**不是 MPJPE**；decoder 随机初始化时学生骨架与教师差很多，此数大且会降，方向对即可。
- **加载旧 ckpt strict 报错**：模型定义与 ckpt 不配套；改代码后 `rm -rf **/__pycache__`。
- **SSH 断导致训练中断**：长训放 `tmux`。RGB 教师支持 `--resume teacher_last.pt`（恢复 model/optimizer/scheduler/EMA/epoch/best/patience）。

---

## 15. 诚实声明与引用

### 15.1 诚实声明

- 硬件按 MMFi 原生配置客观描述：**1Tx×3Rx，每对天线 114 子载波，Atheros CSI Tool，一对 TP-Link N750 AP**。不使用"单链路"等易误解措辞，不作硬件层面的卖点对比。
- 主线为 **RGB 教师蒸馏**；RGB 教师已收敛（291.20mm @e51），**RGB 学生（Step B）faithful 结果仍在产出中**，§8.1 主表为 `进行中`，不提前填数。
- 当前唯一已落地的 CSI 学生 faithful 结果 352.23/102.75/321.88 来自**深度教师**，明确标为前序对照行。
- §8.0 的 RGB 教师数为**训练期教师强度 sanity，非 CSI 学生结果**。
- 选点口径与 DT-Pose/MMFi 主流一致：E01–E03 全量训练 + 在 E04 按 faithful 指标选 checkpoint；报告同时给 E04 最低 MPJPE 与最低 PA 两个 ckpt。
- **关于 hip 差距，本文不主张任何"信息论上界/不可逾越"。** 早期版本曾有此断言，现已撤回——同条件工作 Person-in-WiFi 3D（TPAMI 2026）在 MMFi S3/P3 取得 MPJPE 302.5 / PA 101.1，证明该差距是建模局限、可缩小（§9）。本文将其列为方法局限与未来方向，不夸大、不回避。velocity-root 等 hip 杠杆经 GT 诊断仅有 ~7.6mm 理论上界，作为诚实小增量呈现，不夸大其作用。
- 训练期滑窗监控值与 faithful 不可比，不作结论、不用于选点。

### 15.2 参考

- DT-Pose: *Towards Robust and Realistic Human Pose Estimation via WiFi Signals* (arXiv:2501.09411)
- MM-Fi: *Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing* (NeurIPS 2023 D&B; arXiv:2305.10345)
- Person-in-WiFi 3D: *Unified Model for 3D WiFi Perception*, IEEE TPAMI 2026, DOI 10.1109/TPAMI.2026.3701032（同条件 MMFi S3/P3 取得 MPJPE 302.5 / PA 101.1，§8.1 / §9 引为差距可缩小之证据）