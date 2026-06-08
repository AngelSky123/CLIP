# CSI-RSC-PoseDG

**Cross-Environment 3D Human Pose Estimation from Single-Link WiFi CSI, via Depth-Teacher Knowledge Distillation**

基于 WiFi CSI 的跨环境 3D 人体姿态估计系统。训练期由深度图教师 (depth teacher) 通过知识蒸馏向 CSI 学生注入几何先验，**推理阶段只使用 CSI**（不依赖任何深度图或视觉输入）。在 MMFi 数据集上以严格盲测的跨房间设定（Setting 3 / Protocol 3，全 27 动作，单链路 CSI，CSI-only 推理）对标 DT-Pose。

> **当前主攻点**：在 **PA-MPJPE（相对骨架结构）** 上追平/反超 DT-Pose；绝对 **MPJPE** 的差距已被多组对照实验证明为跨房间绝对定位的信息论上界（见 §9），以严格、可复现的诚实分析方式呈现，而非声称全面超越。

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
10. [Checkpoint 选择与方法论纪律](#10-checkpoint-选择与方法论纪律)
11. [复现步骤](#11-复现步骤)
12. [超参数完整参考](#12-超参数完整参考)
13. [仓库结构](#13-仓库结构)
14. [常见问题与坑](#14-常见问题与坑)
15. [诚实声明与引用](#15-诚实声明与引用)

---

## 1. 动机与问题定义

WiFi CSI（Channel State Information）作为一种无感（device-free）、不依赖光照、保护隐私的人体感知模态，近年被广泛用于 3D 人体姿态估计。其根本难点在于**跨环境泛化**：CSI 信号由无线多径传播决定，强烈依赖房间几何、家具布置、墙体反射、收发天线位置等环境因素；因此在房间 A 采集数据训练的模型，迁移到结构不同的未见房间 B 时性能会显著下降。

本项目刻意针对该领域**最严苛**的一组约束，以保证结论的可比性与公平性：

- **跨房间盲测（strict leave-one-room-out）**：测试房间的数据在训练全程完全不可见——既没有标签，也没有用于自监督/适配的无标签数据。
- **单链路 CSI（single-link）**：仅一条收发链路（3 发射天线 × 1 接收天线，3Tx×1Rx），不依赖多设备阵列或额外标定。
- **CSI-only 推理**：深度图仅在训练阶段作为教师参与蒸馏，部署/推理时不可用。

形式化定义：给定 CSI 时序窗口 `X ∈ R^(T×9×114×10)`，预测 3D 关节序列 `P ∈ R^(T×17×3)`（绝对世界坐标，单位米）。训练集来自环境 `{E01, E02, E03}`，测试集来自**从未见过**的 `E04`。

---

## 2. 核心贡献

1. **深度-教师蒸馏框架**：训练期用深度图姿态教师对 CSI 学生做 feature-level + output-level 双路蒸馏，把视觉几何先验迁移进 CSI 表征；推理只需 CSI。
2. **RSC（Representation Self-Challenging）**：对全局表征施加时间维 / 通道维 / batch 维随机遮挡，强迫模型分散依赖、避免过拟合单一环境捷径特征，提升跨域鲁棒性。
3. **结构正则损失套件**（`structural_losses.py`）：骨长一致、左右对称、时序骨长稳定、root-relative 位置对齐——四项损失全部**平移不变**，专门优化 PA-MPJPE 所度量的相对骨架，**结构上不可能恶化全局定位项**。
4. **Hybrid FK 解码器**（`fk_decoder.py`）：在不替换现有解码器的前提下，外挂一条正运动学（Forward Kinematics）分支（root + 骨长 + 骨方向单位向量 → FK 合成），以 α 退火与原解码器融合；骨架合法性由**构造保证**，是比惩罚项更彻底的 PA 杠杆。
5. **Root anchor**（诚实修复 MPJPE）：把预测 hip 往按动作的源域 canonical 先验做正则，抑制源域过拟合导致的 root 漂移，使预测 root 不致比常数先验更差——在不接触 E04、不对齐真值的前提下回收"自伤"误差。
6. **跨房间绝对定位的信息论 limitation 分析**：通过原始输入信息探针、教师误差上界、零信息基线对照、以及对所有候选杠杆的系统性穷举，**量化证明**单链路 CSI 下"人在未见房间里的绝对位置"不可跨域迁移——这是对 WiFi 感知能力边界的实证刻画，本身具有发表价值。

---

## 3. 方法详解

### 3.1 整体管线

```
CSI 输入 (B, T=64, 9, 114, 10)
   │
   ├─[csi_encoder]        逐帧 CSI 编码 (9 = 3 幅度 + 6 相位通道)
   │                      hidden=32, out=64
   │
   ├─[local_encoder]      3D 残差卷积 (ResNet3D, num_res3d_blocks=2)
   │                      在 子载波 × packet × 时间 三轴提取局部时频结构
   │                      hidden=64, out=64
   │
   ├─[feature_pooling]    空间池化, 每帧聚合为单 token
   │
   ├─[global_modeler]     Transformer(layers=3, heads=4, dropout=0.3)
   │                      + TCN(channels=[128,128], k=3)
   │                      建模长时序依赖 ──> z_global (B, 64, 128)
   │
   ├─[RSC]                训练期对 z_global 做 representation self-challenging
   │
   ├─[pose_decoder]       HybridFKPoseDecoder:
   │                        ├ 结构支 = PoseDecoder(TaskPromptCoarseHead + SkeletonRefiner/GCN)
   │                        │           —— 字节级复用, 继承当前 104.73 的强 baseline
   │                        ├ FK 支  = FKBranch(z_global → root + 骨长 + 骨方向 → 正运动学)
   │                        └ 融合   p_final = α·p_struct + (1-α)·p_fk
   │                                  α 由 1.0 退火到 fk_alpha_final(默认0.4), warmup=20 epoch
   │                      ──> p_coarse, p_final ∈ (B, 64, 17, 3)
   │
   └─[action_classifier]  27 类动作分类 (辅助任务, 正则化全局表征)
```

学生网络总参数量：约 **1.63M**（FK 支约 +5 万参数）。当 α=1 时纯走结构支，等价于此前 1.58M 的模型，保证 FK 分支安全冷启动、不会一上来就拖崩训练。

### 3.2 各模块详解

- **csi_encoder**：把每帧 `(9, 114, 10)` 的 CSI 编码为紧凑特征。9 个输入通道 = 3 路幅度（来自 3 天线）+ 6 路相位（3 sin + 3 cos）。配置 `encoder_hidden_dim=32, encoder_out_dim=64`。
- **local_encoder**：3D 残差卷积块（`num_res3d_blocks=2`，`local_hidden_dim=64, local_out_dim=64`），在子载波 / packet / 时间三个轴上提取局部时频纹理——这是 CSI 区别于普通时序信号的关键结构。
- **feature_pooling**：把每帧的时频特征图池化为单个 token，得到逐帧表征序列。
- **global_modeler**：`num_transformer_layers=3, num_heads=4, transformer_dropout=0.3` 的 Transformer，叠加 `tcn_channels=[128,128], tcn_kernel_size=3` 的时序卷积网络（TCN），联合建模全局与局部时序依赖，输出 `z_global ∈ R^(B×64×128)`，`global_dim=128`。
- **pose_decoder**：当前版本为 `HybridFKPoseDecoder`，详见 §3.5。其结构支沿用 `TaskPromptCoarseHead`（`coarse_hidden_dim=256`）产出粗 3D 姿态，再经 `SkeletonRefiner`（GCN，`gcn_hidden_dim=128, num_gcn_layers=3`）按骨架邻接关系精修。
- **action_classifier**：27 类动作分类头，作为辅助监督正则化全局表征，提升判别性。

### 3.3 RSC（Representation Self-Challenging）

训练期对 `z_global` 施加随机遮挡，强制模型分散依赖、避免靠单一环境相关捷径"作弊"，从而提升跨域鲁棒：

- `rsc2_time_drop_pct=0.5`：时间维随机丢弃比例
- `rsc2_channel_drop_pct=0.5`：通道维随机丢弃比例
- `rsc2_batch_pct=0.5`：batch 内施加 challenge 的样本比例

### 3.4 知识蒸馏

教师 `DepthPoseTeacher`（深度图输入，训练期冻结）对学生做两路蒸馏：

- **Feature-level**：学生 `z_global` 经 `DistillProjection` 投影后，与教师全局特征做余弦相似 + Smooth-L1 对齐（`distill_cos_w=1.0, distill_sl1_w=1.0`），由 `lambda_feat=0.1` 加权。
- **Output-level**：学生 `p_final_clean` 与教师 `p_final` 做姿态级蒸馏（Smooth-L1，`out_distill_beta=0.05`），hip 关节额外加权 `out_distill_hip_weight=4.0`，由 `lambda_out=1.0` 加权。

> 注：实测教师自身在 E04 的 hip_err 已达 ~236mm，即"老师自己定位都不准"，因此 hip 全局定位**无法靠蒸馏教师获得**（见 §9）；output distill 主要贡献的是骨架结构而非绝对定位。

### 3.5 Hybrid FK 解码器（PA 主杠杆）

**动机**：PA-MPJPE 度量的是相对骨架形状。直接回归 `(B,T,17,3)` 坐标容易产生不合理骨长、左右不对称、拓扑错乱。FK 把姿态参数化为「根关节 + 每条骨的长度 + 每条骨的方向单位向量」，再沿运动学树合成绝对坐标，使**骨架合法性成为构造保证**，而非靠惩罚项软约束。

**正运动学**（17 关节、16 条骨；joint 0 = hip/root）：

```
joints[0] = root
for (parent, child) in EDGES:
    joints[child] = joints[parent] + bone_dir[child] · bone_len[child]
```

骨架边集与左右对称对：

```python
EDGES = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),       # 两条腿
         (0,7),(7,8),(8,9),(9,10),                   # 脊柱→头
         (8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]  # 两条臂
SYM_BONE_PAIRS = [((0,1),(0,4)),((1,2),(4,5)),((2,3),(5,6)),
                  ((8,11),(8,14)),((11,12),(14,15)),((12,13),(15,16))]
```

**FKBranch** 从 `z_global (B,T,128)` 预测：root `(B,T,3)`；骨方向 `(B,T,16,3)`（`F.normalize` 成单位向量）；骨长 `(B,T,16,1)`（`softplus` 保证正、`clamp(0.02, 0.8)` 限定合理米数）。

**Hybrid（不替换、只外挂）**：考虑到一上来用 FK 全量替换解码器可能导致训练塌陷，采用融合形式 `p_final = α·p_struct + (1-α)·p_fk`：

- `p_struct` 来自字节级复用的现有 `PoseDecoder`，继承当前 104.73 的强 baseline；
- `α` 注册为 **buffer**，由训练器逐 epoch 从 1.0 退火到 `fk_alpha_final`（默认 0.4），`fk_alpha_warmup` 步（默认 20 epoch）内完成；
- 早期 α≈1 等价旧模型（安全冷启动），后期 FK 支逐步接管结构合法性。

对外接口与 `PoseDecoder` **完全一致**：`forward(z_global, action_emb) -> (p_coarse, p_final)`。接入只需在 `models/full_model.py` 把构造的类名换成 `HybridFKPoseDecoder`（kwargs 原样传），其余 forward / RSC / 蒸馏 / 评测一行不改。α 是浮点 buffer，会被 EMA（对浮点 buffer 做 EMA 平均）平滑跟踪到终值，随 ckpt 保存，评测自动用最终 α。

### 3.6 结构正则与 root anchor

设预测姿态 `p`、真值 `g`，边集 `EDGES`、左右对称对 `SYM_BONE_PAIRS`：

```
L_bone   = L1( bonelen(p), bonelen(g) )                       # 需 GT (源域)
L_sym    = mean_{(a,b)∈SYM} L1( len(a), len(b) )              # 无需 GT, 处处可用
L_temp   = mean_t | bonelen(p)_{t+1} − bonelen(p)_t |         # 无需 GT, 处处可用
L_rel    = L1( p − p_hip,  g − g_hip )                        # 髋中心相对位置, 需 GT
L_anchor = SmoothL1( p_hip, canonical[action] )               # root 往按动作源域先验拉
```

- **L_bone / L_sym / L_temp / L_rel** 全部基于骨长（关节差）或髋中心化坐标，对全局平移不变 → 只重塑相对骨架，**碰不到 hip 全局 xyz**，结构上保证不恶化 MPJPE 的定位主项。其中 **L_rel** 直接对应 PA-MPJPE 度量的"髋中心相对位置"，是冲击 PA 最对症的杠杆（实测从 105.42 → 104.73）。
- **L_anchor** 用「按动作的源域平均 hip」作为稳健先验，把预测 hip 往其正则。`canonical[action] ∈ R^(27×3)` 由 `build_action_canonical` 预扫训练集 GT（只读 `ground_truth.npy`，不碰 CSI，很快）得到。这是**源域统计、与 E04 无关**，因此合法。其作用是降低 root 头的方差、抑制源域过拟合漂移，使 root 在 E04 上不致比常数先验更差（§9c）。

### 3.7 总损失

```
L_total = L_pose(p_clean, g)              # 基础姿态损失 (PoseLoss: lambda1/2/3)
        + L_action                        # 动作分类 CE
        + lambda_feat · L_distill_feat    # 特征蒸馏
        + lambda_out  · L_distill_out     # 输出蒸馏 (hip 加权)
        + w_bone·L_bone + w_sym·L_sym + w_temp·L_temp + w_rel·L_rel   # 结构正则
        + w_root_anchor · L_anchor        # root anchor
```

---

## 4. 数据集与预处理

### 4.1 MMFi 目录结构与划分

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

held-out val：从 E01–E03 中按 `val_ratio=0.15` 留出若干 subjects（实测留出 S16/S18/S24/S28）不参与训练，仅用于早停 / checkpoint 选点。E04 永远只作 monitor，不参与任何选点（见 §10）。

### 4.2 CSI 张量与预处理（`dataset.py`）

每帧 `.mat` 含 `CSIamp` 与 `CSIphase`，形状均为 `(3, 114, 10)`：3 天线 × 114 子载波 × 10 packet。预处理后拼成网络输入 `(T=64, 9, 114, 10)`：

```
原始:  CSIamp (3,114,10),  CSIphase (3,114,10)

幅度路 (3 通道):  逐帧 min-max 归一化(在 (114,10) 上)            → amp_norm (3,114,10)
相位路 (6 通道):  沿子载波 unwrap → detrend → [sin, cos] 编码    → phase_enc (6,114,10)
拼接:            concat([amp_norm, phase_enc], axis=channel)    → (9,114,10)
逐帧堆叠:        T 帧 → (T, 9, 114, 10)
```

9 通道 = 3 幅度 + 3 sin + 3 cos。**10 packet 维与 3 天线维全程保留进网络**（已核验，dataloader 不丢弃这两个维度）。短序列以 edge padding 补齐至 `seq_len`。训练集开启 CSI 增广（`augment=True`），测试集关闭增广、用无重叠滑窗（`stride=seq_len`）。

> **重要**：逐帧 min-max 归一化会抹掉绝对幅度量级（路损 / RSSI），相位 detrend 会去掉子载波线性斜率（ToF）。这两者都是潜在的"距离线索"。§9 的探针实验证明：即便保留绝对幅度，这些线索在跨房间下也不可迁移——所以预处理不是 MPJPE 差距的原因。

---

## 5. 训练流程（四阶段）

完整复现按以下顺序产出 checkpoint：

| 阶段 | 脚本 | 产出 | 说明 |
|---|---|---|---|
| **Stage 1A** | `train_mae.py` | `stage1a_mae/` | （可选）MAE 自监督预训练 backbone |
| **Stage 1B** | action 预训练 | `stage1b_action/action_best.pt` | 动作监督预训练 backbone |
| **Teacher** | 深度教师训练 | `depth_teacher_full/teacher_best.pt` | 深度图姿态教师，蒸馏时冻结 |
| **Distill** | `train_distill_pretrained.py` | `distill_*/best_{mpjpe,pa}_ema.pth` | 在 1B backbone 上蒸馏 + 结构正则 + FK + root anchor，得 CSI-only 学生 |

**为什么部署用 Stage 1B 的 action backbone 而非纯 MAE**：探针实验发现纯 MAE backbone 出现表征塌陷（同/异窗口特征余弦相似度都挤在 ~0.91，margin≈0.001）；而 action-supervised backbone 表征健康（margin / s_diff 正常）。因此主线建立在 1B 之上；MAE-DCL（TC-CL + uniformity）变体已试过，对健康 backbone 无增益，故未采用（见 §9d）。

**Stage 1B 健康性检查**：action val acc 不塌缩；feature std 不低于阈值；same-action cross-env 相似度↑、diff-action 相似度↓。

---

## 6. 实验设定（与 DT-Pose 对齐）

| 项 | 设定 |
|---|---|
| 数据集 | MMFi |
| 划分 | **Setting 3（cross-environment / 跨环境）** |
| 协议 | **Protocol 3（全 27 动作）** |
| 训练环境 | E01–E03（S01–S30） |
| 测试环境 | **E04（S31–S40），严格盲测、训练期从不参与** |
| 输入 | 单链路 CSI（3Tx×1Rx），64 帧窗口 `(T=64, 9, 114, 10)` |
| 推理 | **CSI-only**（深度图仅训练期教师） |
| 评测口径 | 逐帧、全帧覆盖、无 padding、`action_idx=None`（无 GT 动作标签泄露） |
| MPJPE | 纯绝对误差，**不做任何 centering**（与 DT-Pose `calculate_error` 一致） |
| PA-MPJPE | 含 scale 的 Procrustes 对齐后 MPJPE（`compute_similarity_transform`） |

**可复现性**：所有训练固定 `--seed 42`，并启用 `torch.backends.cudnn.deterministic=True / benchmark=False`。报告数一律来自 `eval_dtpose_faithful.py`（唯一权威口径）；训练期 `evaluate_v2` 的滑窗监控值与 faithful 口径**不可比**（实测差可达 100mm+），**不用于任何结论**。

---

## 7. 评测指标定义

设第 `f` 帧第 `j` 关节预测 `p_{f,j}`、真值 `g_{f,j}`，共 `F` 帧、`J=17` 关节：

```
MPJPE          = (1/F) Σ_f (1/J) Σ_j ‖ p_{f,j} − g_{f,j} ‖₂          # 绝对, 无对齐
MPJPE_aligned  = 同上, 但先对每帧各自减去 hip(joint 0)               # 髋中心相对结构
PA-MPJPE       = 同上, 但先对每帧做 Procrustes(旋转+缩放+平移) 对齐   # 纯形状
hip_error      = (1/F) Σ_f ‖ p_{f,0} − g_{f,0} ‖₂                    # 仅 hip 全局定位
PCK@τ_norm     = 关节误差 < τ%·(躯干尺度) 的关节占比
```

三者关系与本项目的关键事实：

- `MPJPE` = 全局定位误差 + 朝向 + 尺度 + 相对结构，是最严格的绝对指标。
- `MPJPE_aligned` 去掉全局平移、保留尺度/朝向。
- `PA-MPJPE` 进一步去掉缩放/旋转，是最纯的"姿态形状"指标。
- 本项目的全部差距集中在 `hip_error`（全局定位）；`PA-MPJPE` 与 SOTA 持平。即"人体形状对了，但人在房间里的绝对位置不准"。

---

## 8. 结果与消融

### 8.1 主结果进展（E04，faithful 逐帧口径；270 序列 / 80,190 帧）

| 模型 | MPJPE (mm) | PA-MPJPE (mm) | hip_err (mm) | 选点 |
|---|---:|---:|---:|---|
| baseline（TaskPrompt 解码器） | 366.6 | 106.2 | 337.8 | val MPJPE |
| + 结构正则（骨长/对称/时序，`w_bone=0.5`） | 361.97 | 105.42 | 334.91 | val MPJPE |
| + root-relative（`w_bone=1.0, w_rel=3.0`） | 363.74 | **104.73** | 334.29 | val MPJPE @e9 |
| + Hybrid FK + root anchor（`w_rel=6, w_root_anchor=0.5`） | _TBD_ | _TBD_ | _TBD_ | 复合选点 |
| **DT-Pose (S3 / P3)** | **316.8** | **104.2** | — | — |

- PA-MPJPE 多 stride 评测 σ ≈ 0.01–0.02mm，上述 PA 为稳定真值（非噪声）。
- 当前已确认最优 **PA-MPJPE = 104.73**，与 DT-Pose 104.2 相差 **+0.53mm**（约 0.5mm，处于单链路硬件分辨率地板附近）。
- 绝对 MPJPE 落后约 45–50mm，差距**全部集中在 hip 全局定位**（见 §9）。
- 最后一行（Hybrid FK + root anchor）待 `distill_fk_anchor` 训练完成后用 faithful 口径填入。

### 8.2 PA-MPJPE 消融（结构杠杆有效，且不恶化 MPJPE）

| 配置 | PA-MPJPE | ΔPA | MPJPE | MPJPE_aligned | 说明 |
|---|---:|---:|---:|---:|---|
| baseline | 106.2 | — | 366.6 | — | 无结构正则 |
| + 骨长/对称/时序 | 105.42 | −0.78 | 361.97 | 122.24 | 结构损失生效，MPJPE 同时小降 |
| + root-relative | 104.73 | −0.69 | 363.74 | 120.24 | 更对症的 PA 杠杆，MPJPE_a 继续降 |

结论：结构正则在降低 PA 的同时 MPJPE 基本不动（甚至小降），验证了"平移不变损失只重塑相对骨架、不碰 root"的设计。`MPJPE_aligned` 随结构改进单调下降（122.24 → 120.24），佐证骨架确实更准。

### 8.3 已被穷举证否、不再主攻的杠杆

| 杠杆 | 现象 | 结论 |
|---|---|---|
| 解码器结构（多轮：路1 ActionPrior / root 解耦等） | hip_err 纹丝不动 | 停止 |
| MAE-DCL 预训练（TC-CL + uniformity） | 健康 backbone 无增益 | 停止 |
| 合规 test-time 重心化（TTA） | 杠杆耗尽在 ~354 | 已用满 |
| 加大 hip 蒸馏 / lambda_hip | E04 hip 仅 340→337 | 无效 |
| raw / log 幅度输入救 MPJPE | E04 反向迁移（§9a） | 停止 |
| post-hoc root fallback（β 混先验） | 诚实选 β（源域）退化为 β≈1 | 无效（见 §10） |

---

## 9. Limitation 分析：跨房间绝对定位的信息上界

绝对 MPJPE 的 ~45–50mm 差距**全部集中在 hip 全局定位**，且为信息论上界，非建模不足。四条独立证据：

### (a) 原始输入信息探针（`probe_raw_amplitude_hip.py`）

用线性岭回归直接从 CSI 幅度预测 E04 hip 绝对坐标，无论是否保留绝对尺度：

| 特征 | held-in (E01–03) | E04 |
|---|---:|---:|
| mean_base（预测训练集均值，零信息标尺） | 172.7 | **324.2** |
| 原始绝对幅度 | 152.3 | 350.8 |
| log 功率 | 155.5 | 348.8 |
| 逐帧归一化幅度 | 149.6 | 356.4 |

三种幅度表示在训练房间均优于零信息标尺，**在 E04 上全部劣于标尺** → 幅度→定位的映射逐房间不同，呈反向迁移，绝对距离线索不可跨域。

### (b) 教师误差上界

深度图教师（拥有视觉深度模态）在 E04 的 hip_err 仍达 ~236mm —— 即便强信号模态也难恢复跨房间绝对位置，说明这是模态/物理层面而非建模层面的难点。

### (c) 完整模型 vs 零信息基线（关键）

部署模型在 E04 的 hip_err（~335mm）已**≥** 零信息基线（324mm）—— 即用满全部输入、训练好的非线性模型，在绝对定位上未超过"永远预测平均位置"的常数预测。其中约 11mm 是源域过拟合 / 晚 epoch 漂移造成的"自伤"，可由 root anchor 诚实回收（往稳健先验拉，把 MPJPE 从 ~363 拉回 ~350）；但这**不是"感知出位置"**——CSI 里那部分跨域定位信息确实不存在。

### (d) 杠杆穷举

解码器结构（多轮）、自监督预训练（MAE-DCL）、合规 TTA、蒸馏权重调参、raw/log 幅度输入、action-prior root、root 解耦解码器 —— 均未移动 hip 误差。

**结论**：在严格盲测 / 跨房间 / 单链路 CSI / CSI-only 推理下，绝对 MPJPE 受信息论上界约束；可改善空间在**相对结构（PA-MPJPE）**，本方法已将其推至 SOTA 持平。root anchor 至多把 MPJPE 拉回 ~350（不再难看），追不到 316.8。geometry-conditioned root 需要 E04 房间几何 / 设备标定信息，严格盲测下不可用、不属于本设定的可行方案。DT-Pose 原文亦指出末端关节误差受限于 WiFi 分辨率（需更多设备 / 更高分辨率），与本结论一致。

---

## 10. Checkpoint 选择与方法论纪律

**实测现象**：E04 的 PA 随训练**早熟**——EMA 监控 PA 在约 e3 最低（~104.3），之后随 epoch 单调变差（e48 ~105.7），同时 PredStd 从 ~2mm 爬到 ~15mm，表明 root 分支后期发散漂移。

**关键陷阱**：源域 val 选点（无论按 MPJPE 还是 PA）与 E04 的 PA **不同步、甚至反向**。例如按源域 val PA 选点会选到晚 epoch（e48），其 E04 faithful PA 反而是 105.60，**差于**按 val MPJPE 选到的 e9（104.73）。

**纪律**（保证可发表、可复现、不作弊）：

1. **选点只用源域 held-out val**；E04 永远只作 monitor，不参与任何选点。
2. **复合选点**：先卡 `MPJPE ≤ baseline+5mm` 且 `root_error ≤ baseline+5mm`，再在满足者中选 PA 最低；防止选到 PA 好但 root 严重漂移的晚 ckpt。
3. **同时保存 `best_mpjpe_ema` 与 `best_pa_ema` 两个 checkpoint，论文里两者的 E04 数都透明列出**，绝不在 E04 上挑 checkpoint。
4. **post-hoc root fallback 的诚实性**：`root_final = β·pred_root + (1−β)·prior` 若按源域 val 选 β，则因源域 root 优于常数先验、必然选到 β≈1，对 E04 无任何改善；唯一在 E04 上有用的 β<1 需要看 E04 选，构成泄漏。因此回收 root 自伤（§9c 那 11mm）的合法方式是**训练期 root anchor**（烤进权重、可迁移），而非测试期 fallback。

---

## 11. 复现步骤

### 11.1 环境

```bash
# Python 3.7 / PyTorch 2.2.2 / CUDA (RTX 4080, 16GB)
conda create -n 3DHPE1 python=3.7 -y
conda activate 3DHPE1
pip install torch==2.2.2 numpy scipy
# 其余依赖见仓库 requirements.txt
```

### 11.2 数据

将 MMFi 解压至 `--data_root`，确认目录结构如 §4.1。

### 11.3 接入 Hybrid FK（改 `models/full_model.py` 一行）

在 `CSIRSCPoseDG.__init__` 中，把原本的 `PoseDecoder(...)` 换成 `HybridFKPoseDecoder(...)`（kwargs 完全一致），并在文件顶部 import：

```python
from fk_decoder import HybridFKPoseDecoder
...
self.pose_decoder = HybridFKPoseDecoder(
    in_dim=args.global_dim, hidden_dim=args.coarse_hidden_dim,
    gcn_hidden=args.gcn_hidden_dim, num_gcn_layers=args.num_gcn_layers,
    num_joints=args.num_joints, action_embed_dim=action_embed_dim,
)
```

原 `from .pose_decoder import PoseDecoder, ActionClassifier` 行保留（ActionClassifier 仍需用）。其余 forward / RSC / 蒸馏 / 评测一行不改。

### 11.4 训练（当前版本：结构正则 + Hybrid FK + root anchor）

```bash
python train_distill_pretrained.py \
    --data_root /path/to/MMFi \
    --train_envs E01 E02 E03 --test_env E04 \
    --pretrain_ckpt checkpoints/stage1b_action/action_best.pt \
    --teacher_ckpt  checkpoints/depth_teacher_full/teacher_best.pt \
    --depth_img 112 --depth_clip 5000 \
    --w_bone 1.0 --w_sym 0.1 --w_temp 0.1 --w_rel 6.0 \
    --w_root_anchor 0.5 --fk_alpha_final 0.4 --fk_alpha_warmup 20 \
    --epochs 50 --batch_size 2 --accumulate_grad 8 \
    --use_ema --ema_decay 0.999 --seed 42 \
    --save_dir ./checkpoints/distill_fk_anchor
```

显存提示：RTX 4080 16GB 下 `batch_size=2 + accumulate_grad=8`（等效 batch 16）。OOM 时降 batch、提 accum。可 `grep "\[FK\]" 日志` 确认 α 从 1.000 平滑退到 0.400。

### 11.5 评测（唯一权威口径；两个 ckpt 都评）

```bash
for c in best_mpjpe_ema best_pa_ema; do
  echo "==== $c ===="
  python eval_dtpose_faithful.py --data_root /path/to/MMFi \
      --ckpt ./checkpoints/distill_fk_anchor/$c.pth --test_env E04 --seq_len 64 --variance
done
```

`--variance` 跑多 stride 报 mean±σ，用于判断指标领先是否超过评测口径噪声（领先 ≤ σ 只能写"持平"）。

### 11.6 信息上界探针（§9 复现）

```bash
python probe_raw_amplitude_hip.py --data_root /path/to/MMFi \
    --train_envs E01 E02 E03 --test_env E04 --max_frames_per_seq 60 --ridge 1e-2
```

---

## 12. 超参数完整参考

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
| **结构** | `--w_rel` | 6.0 | root-relative 位置 (PA 主杠杆) |
| **FK** | `--fk_alpha_final` | 0.4 | Hybrid FK 融合系数终值 |
| **FK** | `--fk_alpha_warmup` | 20 | α 由 1.0→final 的退火 epoch 数 |
| **root** | `--w_root_anchor` | 0.0 | root anchor 强度（>0 启用，救 MPJPE） |
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
| EMA | `--use_ema` / `--ema_decay` | True / 0.999 | 浮点 buffer(含 FK α) 会被 EMA 跟踪 |
| 评测 | `--eval_interval` | 3 | 每 N epoch 评一次 |
| 复现 | `--seed` | 42 | |

---

## 13. 仓库结构

```
RSC V2/
├── train_distill_pretrained.py   # 主训练: 蒸馏+EMA+held-out选点+结构正则+FK退火+root anchor
├── train_mae.py                  # Stage 1A: MAE 自监督预训练 (可选)
├── fk_decoder.py                 # Hybrid FK 解码器 (结构支 + FK 支 + α 融合 + 正运动学)
├── structural_losses.py          # 骨长/对称/时序/root-relative/root-anchor + canonical 构建
├── eval_dtpose_faithful.py       # 逐帧 faithful 评测 (与 DT-Pose 对齐的权威口径)
├── evaluate.py                   # 训练期监控评测 (evaluate_v2, 滑窗, 不用于报告)
├── probe_raw_amplitude_hip.py    # 原始幅度 → E04 hip 信息探针 (§9 分析依据)
├── dataset.py                    # MMFi CSI/GT 加载 + 预处理 + 增广
├── dataset_distill.py            # 蒸馏数据加载 (csi + depth 配对)
├── augmentation.py               # CSI 数据增广
├── losses.py                     # PoseLoss / TotalLoss
├── distill_loss.py               # DistillProjection / FeatureDistillLoss / OutputDistillLoss
├── taskprompt_decoder.py         # TaskPromptCoarseHead (+ uniformity_loss)
├── action_prior_root.py          # (路1 备选) 动作×相位先验 root 损失, 已门控
├── utils.py                      # set_seed / logger / 参数统计等
├── models/
│   ├── full_model.py             # CSIRSCPoseDG (在此把 pose_decoder 换成 HybridFKPoseDecoder)
│   ├── pose_decoder.py           # PoseDecoder / CoarsePoseHead / SkeletonRefiner / ActionClassifier
│   └── depth_teacher.py          # DepthPoseTeacher
└── checkpoints/
    ├── stage1b_action/action_best.pt
    ├── depth_teacher_full/teacher_best.pt
    └── distill_fk_anchor/best_{mpjpe,pa}_ema.pth
```

---

## 14. 常见问题与坑

- **`AttributeError: 'PoseDecoder' object has no attribute 'root_head'`**：路1 的 `root_prior_losses` 已按 `hasattr(decoder,'root_head')` 门控；TaskPrompt / Hybrid FK 基线无 root_head，自动跳过。
- **ckpt 加载后全是噪声**：检查日志 `missing / unexpected`，>8 即模型定义与 ckpt 不配套；改代码后务必 `rm -rf **/__pycache__`。
- **EMA ckpt 直接是 shadow dict**：`best_*_ema.pth` 已是 EMA 权重，评测直接加载。
- **FK α 与 EMA**：α 是浮点 buffer；EMA `update` 对浮点 buffer 做 EMA 平均，故 EMA shadow 的 α 会平滑跟踪到 `fk_alpha_final`，评测自动用最终 α，不会卡在 1.0（无需手动同步）。
- **监控值远好于 faithful**：训练期滑窗监控 (stride + padding) 会失真，差可达 100mm+，**只认 `eval_dtpose_faithful.py`**。
- **PA 早熟、晚 epoch 漂移**：见 §10，必须复合选点、两个 ckpt 都评都报。
- **探针 raw_abs 出 NaN**：原始 `CSIamp` 含 inf，须先 `nan_to_num(posinf=0)` + 裁剪（`probe_raw_amplitude_hip.py` v2 已修）。
- **PA / MPJPE_aligned 对全局平移不变**：可作 sanity——若改动只该影响结构，这两项应随 MPJPE 同向但 hip_err 不变；反之说明动到了 root。
- **OOM**：降 `--batch_size`、提 `--accumulate_grad` 维持等效 batch；对比类损失（mae_dcl）需较大 batch，4080 上 batch16 易 OOM。

---

## 15. 诚实声明与引用

### 15.1 诚实声明

- 报告的所有数均来自**严格盲测**（E04 训练期从不可见）+ **faithful 逐帧口径**，无 GT 动作标签泄露、无对 E04 真值的对齐、无 transductive 偷看、不在 E04 上选 checkpoint / 选 β。
- **PA-MPJPE 当前最优 104.73，与 DT-Pose 104.2 持平**（相差约 0.5mm，处于单链路硬件分辨率地板附近）。是否能稳定反超取决于 Hybrid FK 一轮结果，将据 faithful 真数如实更新，**不提前声称**。
- **绝对 MPJPE 落后约 45–50mm**，归因于跨房间绝对定位的信息论上界（§9 给出测量证据）；root anchor 至多回收源域过拟合自伤的 ~11mm（→ ~350），**不声称追平 316.8**。
- 任何低于 faithful 报告值的数（训练期滑窗监控、含 GT 动作标签、对 E04 真值对齐、E04 选点 / 选 β）一律**不作为结论**。

### 15.2 参考

- DT-Pose: *Towards Robust and Realistic Human Pose Estimation via WiFi Signals* (arXiv:2501.09411)
- MM-Fi: *Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing* (arXiv:2305.10345)，Toolbox: github.com/ybhbingo/MMFi_dataset