# CSI-RSC-PoseDG：基于 WiFi CSI 的 3D 人体姿态估计与域泛化

<p align="center">
  <img src="assets/architecture.png" width="85%" alt="模型架构"/>
</p>

基于 [MMFi](https://github.com/ybhbingo/MMFi) 数据集，利用商用 WiFi 的信道状态信息（CSI）实现**无需摄像头的 3D 人体姿态估计**，并通过域泛化（DG）+ 自监督预训练（MAE）提升跨环境的泛化能力。模型输出 H36M 格式的 17 关节 3D 骨架坐标。

## 主要特点

- **双分支 CSI 编码器**：幅度与相位独立编码，可学习的 BN/IN 门控 + MixStyle 实现环境不变特征提取
- **表示自挑战（RSC）**：训练时遮挡梯度最大的特征维度，梯度回传到整个 backbone（v7.1 修复）
- **动作条件化解码**：decoder 接收动作嵌入，训练时 50% 概率 Action Dropout 防止跨域级联失效
- **由粗到精姿态解码**：MLP 粗回归 + 图卷积网络（GCN）骨架精调
- **三阶段训练（NEW）**：MAE 自监督预训练 → Action 预训练（可选） → Pose 微调
- **严格域泛化协议**：测试时不使用任何 GT 标签（`action_idx=None`），模型必须自行预测动作并回归姿态
- **完整评估体系**：支持 3 种协议 × 3 种划分设定（共 9 组实验）

## 实验结果

### 标准划分（按被试 8:2 划分，所有环境混合）

| 指标 | 数值 |
|------|------|
| MPJPE | 114.12 mm |
| PA-MPJPE | 107.65 mm |
| PCK@50_norm | 41.3% |
| PCK@20_norm | 17.8% |

各环境分项：E01=112.7mm, E02=116.1mm, E03=116.1mm, E04=111.6mm

### 跨环境域泛化（E01-E03 训练，E04 测试，严格 DG 协议）

| 配置 | MPJPE | PA-MPJPE | MPJPE_a | PCK@50_n | 参数量 | 时长 |
|------|-------|----------|---------|----------|--------|------|
| Plan A+B baseline (50ep) | **345.35** | **104.68** | 125.83 | 52.7% | 1.62M | 10h |
| + HMSF ablation (50ep) ⚠ | 343.74 | 108.68 | 132.61 | 52.2% | 1.77M | 10h |
| + MAE pretrain + Stage 2 | _训练中_ | _训练中_ | - | - | - | ~85h |

> **HMSF 是消融实验的负面结果**：在 mmWave 上有效的多尺度池化在 WiFi CSI 上失效，因为 subcarrier-antenna 平面不具备 mmWave range-angle 的空间-物理对应关系。详见 [HMSF 消融研究](#hmsf-消融研究负面结果记录)。

> **PA-MPJPE 持平 SOTA**：DT-Pose 的 PA-MPJPE 是 104.2mm，我们 Plan A+B 拿到 104.68mm，**在不使用 MAE 预训练的情况下持平**。MPJPE gap（28mm）主要来自跨环境绝对 hip 位置预测的物理限制。

## 模型架构

```
CSI 输入 (B, T, 9, 114, 10)
    │
    ├── 幅度 (3通道) ──→ [InstanceNorm → ResBlock2D×2 → MixStyle] ──┐
    │                                                                ├── 门控融合
    └── 相位 (6通道) ──→ [InstanceNorm → ResBlock2D×2 → MixStyle] ──┘
                                         │
                                         ▼
                          局部时空编码器 (2 × Res3DConv)
                                         │
                                         ▼
                         特征池化 (AvgPool2D + Linear)
                        (B, T, 64, 114, 10) → (B, T, 128)
                                         │
                                         ▼
                          全局时序建模器
                    [3层 Transformer + 2层 膨胀TCN]
                                         │
                              ┌──── z_global ────┐
                              │                  │
                         RSC 掩码            动作分类器
                        (仅训练时)         ┌──→ action_logits
                              │            │       │
                              ▼            │    action_emb (B, 32)
                      z_global_masked      │    [训练: GT embedding]
                              │            │    [推理: softmax 加权]
                              │            │    [50% Action Dropout]
                              ▼            ▼
                     粗姿态头 (MLP): (128+32)→256→51
                              │
                              ▼
                   骨架精调器 (GCN): 3→128→128→3
                              │
                              ▼
                    P_final (B, T, 17, 3)
```

**总参数量：~1.62M**

| 模块 | 参数量 | 说明 |
|------|--------|------|
| CSI 编码器 | 168K | 双分支（幅度+相位），BN/IN 门控，MixStyle |
| 局部编码器 | 443K | 2 个 Res3DConv 块，卷积核 (3,3,3) |
| 特征池化 | 9K | 全局平均池化 + 线性投影 |
| 全局建模器 | 810K | 3 层 Transformer (d=128, heads=4) + 2 层膨胀 TCN |
| 姿态解码器 | 148K | 动作条件化粗 MLP + GCN 骨架精调（H36M 17 关节） |
| 动作分类器 | 21K | 分类头 + Embedding(27, 32)，辅助任务（仅训练） |

## 数据集

使用 [MMFi 数据集](https://github.com/ybhbingo/MMFi)：

- **4 个环境**（E01–E04），不同房间布局和 WiFi 部署
- **40 个被试**（每环境 10 人，S01–S40）
- **27 类动作**：A01–A14 日常动作，A15–A27 康复动作
- **每序列约 297 帧**，每帧 CSI 格式 `(3, 114, 10)`（3 接收天线 × 114 子载波 × 10 packets）
- 真值标注 `ground_truth.npy` 形状 `(F, 17, 3)`，单位米

### 数据目录结构

```
MMFi/
├── E01/
│   ├── S01/
│   │   ├── A01/
│   │   │   ├── wifi-csi/
│   │   │   │   ├── frame001.mat
│   │   │   │   ├── frame002.mat
│   │   │   │   └── ...
│   │   │   └── ground_truth.npy
│   │   ├── A02/
│   │   └── ...
│   ├── S02/
│   └── ...
├── E02/
├── E03/
└── E04/
```

## 安装

```bash
git clone https://github.com/YOUR_USERNAME/CSI-RSC-PoseDG.git
cd CSI-RSC-PoseDG

# 创建环境
conda create -n csi-pose python=3.9 -y
conda activate csi-pose

# 安装依赖
pip install torch torchvision  # 根据 CUDA 版本选择
pip install scipy numpy matplotlib
```

## 快速开始

### 方式 1：标准跨环境训练（单阶段，约 10 小时）

```bash
# 在 E01+E02+E03 上训练，E04 上测试（严格 DG，测试时不用 GT 动作标签）
python train.py
```

模型和日志保存在 `checkpoints/run_YYYYMMDD_HHMMSS/`，每次运行自动创建独立目录。预期 MPJPE ≈ 345mm。

### 方式 2：三阶段训练（推荐，约 85 小时）

```bash
# Stage 1A: MAE 自监督预训练（300 ep，~75h）
python train_mae.py \
    --data_root /home/a123456/PerceptAlign/MMFi \
    --train_envs E01 E02 E03 \
    --epochs 300 --batch_size 4 --accumulate_grad 4 \
    --lr 1.5e-4 --mae_mask_ratio 0.75 \
    --save_dir ./checkpoints/stage1a_mae

# Stage 1B: Action 预训练（50 ep，~10h，可选）
python train_action.py \
    --data_root /home/a123456/PerceptAlign/MMFi \
    --train_envs E01 E02 E03 \
    --mae_ckpt ./checkpoints/stage1a_mae/mae_latest.pt \
    --epochs 50 --batch_size 8 --accumulate_grad 2 \
    --lr_backbone 1e-4 --lr_head 5e-4 \
    --save_dir ./checkpoints/stage1b_action

# Stage 2: 姿态微调（50 ep，~10h）
python train_stage2.py \
    --data_root /home/a123456/PerceptAlign/MMFi \
    --train_envs E01 E02 E03 --test_env E04 \
    --pretrain_ckpt ./checkpoints/stage1b_action/action_best.pt \
    --epochs 50 --batch_size 2 --accumulate_grad 4 \
    --lr_backbone 1e-4 --lr_head 5e-4 \
    --save_dir ./checkpoints/stage2_pose
```

详见 [三阶段训练详解](#三阶段训练详解)。

### 方式 3：标准 8:2 划分（约 10 小时）

```bash
# 所有环境混合，按被试 8:2 划分（不区分环境）
python train_standard.py
```

预期 MPJPE ≈ 114mm。

### 可视化

```bash
python visualize.py \
    --checkpoint checkpoints/run_xxx/best_model.pth \
    --env E04 --subject S31 --action A01 \
    --frame 30 --save_dir viz_output
```

生成 4 种可视化：单帧骨架对比（正面+俯视）、多帧骨架网格、逐关节误差热力图、CSI 输入展示。

## 三阶段训练详解

### 动机

跨环境 baseline (Plan A+B) 拿到 MPJPE=345mm vs DT-Pose 316.8mm（差 28mm），但 PA-MPJPE 104.68mm vs 104.2mm（持平）。

差距分析：
- **形状（PA-MPJPE）已持平 SOTA**，模型能学到合理身体比例
- **绝对位置（MPJPE）差 28mm**，主因是跨环境 hip 全局定位的物理可观测性限制
- DT-Pose 的关键差异：使用了 **400 epoch MAE 自监督预训练**，特征质量远超 50 ep 监督训练能学到的

因此采用三阶段 curriculum：先用 MAE 学到模态无关的鲁棒 CSI 表征，再用动作分类进一步对齐语义，最后微调姿态回归。所有阶段**严格 source-only**（不使用目标环境数据）、**非对抗**（无 DANN/GRL）。

### Stage 1A: MAE 自监督预训练

**目标**：学习与下游任务无关的、对 CSI 结构有强归纳偏置的 encoder 特征。

```
Input CSI (B, 64, 9, 114, 10)
    │
    ├── Patchify: 192 patches/sample, patch_dim=3420
    │              (patch_t=4, patch_s=19, patch_a=5)
    │
    ├── Random Mask: 75% of patches → zeros
    │
    ├── Forward 4 backbone modules
    │   (CSIEncoder → LocalEncoder → Pooling → GlobalModeler)
    │
    ├── Lightweight Decoder (per-patch MLP, ~1M params)
    │
    └── Loss: MSE on masked patches only (normalized)
```

- **数据**：仅源域 E01+E02+E03，无标签需求
- **训练时长**：约 75 小时（300 epochs，bs=4×accum=4）
- **保存**：每 20 个 epoch 一份 ckpt，含 4 个 backbone 模块的 state_dict

### Stage 1B: Action 预训练（可选）

**目标**：在 MAE 学到的特征基础上，进一步对齐 action-aware 语义。

```
加载 Stage 1A backbone → 添加 ActionClassifier head
↓
27 类动作 cross-entropy 分类（label smoothing = 0.1）
↓
差别 LR：backbone lr=1e-4（已预训练，慢）/ head lr=5e-4（新增，快）
```

- **数据**：仅源域，使用 `batch['action']` 转换的 int 标签
- **训练时长**：约 10 小时（50 epochs）

### Stage 2: Pose 微调

**目标**：在预训练 backbone 上微调完整 pose decoder + RSC + Action Dropout。

- 加载 Stage 1A 或 1B 的 backbone 权重
- 复用 `train.py` 的 `forward_rsc` 训练逻辑（无任何修改）
- 差别 LR：backbone lr=1e-4 / 新 head lr=5e-4
- 完整 TotalLoss（含 PoseLoss + BoneLoss + ActionLoss + HipLoss + RSC + Anti-collapse）
- 严格 DG 评估：测试时 `action_idx=None`

### 预期收益

| 配置 | 预计 MPJPE (E04) | 状态 |
|------|--------|--------|
| Plan A+B baseline | 345mm | ✅ 已验证 |
| + HMSF only | 343-345mm | ✅ 已验证（无显著收益）|
| + MAE + Stage 2 | **310-325mm** | 🔄 训练中 |
| + MAE + Action + Stage 2 | **305-320mm** | 待定 |

DT-Pose baseline: 316.8mm。理论上有机会**接近或超越 SOTA**。

## HMSF 消融研究（负面结果记录）

我们尝试将 mmWave 论文中的 **HMSF (Hierarchical Multi-Scale Feature Pooling)** 引入 WiFi CSI，作为 `LocalFeaturePooling` 的 drop-in 替换。结果**显示负面**：

| 指标 | Baseline | + HMSF | Δ |
|------|----------|--------|---|
| MPJPE | 345.35 | 343.74 | -1.6mm（噪声范围）|
| PA-MPJPE | 104.68 | 108.68 | **+4.0mm（退步）** |
| MPJPE_a | 125.83 | 132.61 | +6.8mm（退步）|
| Params | 1.62M | 1.77M | +150K |

**根本原因**：HMSF 的设计前提是 range-angle 平面具有"空间-物理"对应关系（人体在不同距离-角度区域产生独特 scattering pattern）。WiFi CSI 的 subcarrier-antenna 平面**没有这种含义**——subcarrier 是频域多径指纹，packets 是时间维度，多尺度池化引入的 fine-scale 特征反而过拟合 source-domain multipath signatures，损害跨环境泛化。

**论文价值**：这是一个 modality-level 的负面发现，说明 mmWave 物理引导方法不能简单迁移到 CSI。代码保留在 `models/hmsf_pooling.py` 用于复现该 ablation。

### 启用 HMSF 进行复现

```bash
# 一键 patch（修改 models/full_model.py 的 import）
python apply_hmsf_patch.py .

# 跑训练
python train.py --epochs 50

# 还原
python apply_hmsf_patch.py . --revert
```

## 项目文件结构

```
CSI-RSC-PoseDG/
├── models/
│   ├── __init__.py
│   ├── csi_encoder.py          # 双分支编码器（BN/IN 门控 + MixStyle）
│   ├── local_encoder.py        # Res3DConv 块 + LocalFeaturePooling
│   ├── hmsf_pooling.py         # HMSF drop-in（用于消融，默认未启用）
│   ├── global_encoder.py       # Transformer + TCN + MixStyle
│   ├── full_model.py           # CSIRSCPoseDG 完整模型（RSC + Action Dropout）
│   ├── pose_decoder.py         # 动作条件化粗 MLP + GCN 骨架精调
│   ├── mae_pretrain.py         # MAE 模型（Stage 1A 用）
│   ├── mixstyle.py             # MixStyle 层（2D / 时序）
│   └── rsc.py                  # RSC 模块定义
├── config.py                   # 跨环境域泛化配置
├── config_standard.py          # 标准 8:2 划分配置
├── dataset.py                  # 域泛化数据加载器
├── dataset_standard.py         # 标准数据加载器
├── train.py                    # 单阶段跨环境训练（Plan A+B baseline）
├── train_standard.py           # 标准 8:2 训练
├── train_experiment.py         # 统一实验脚本（协议 × 设定）
├── run_all_experiments.py      # 全部 9 组实验编排
├── train_mae.py                # Stage 1A: MAE 预训练
├── train_action.py             # Stage 1B: Action 预训练（可选）
├── train_stage2.py             # Stage 2: Pose 微调
├── apply_hmsf_patch.py         # HMSF 一键 patch 工具
├── losses.py                   # 损失函数（PoseLoss, TotalLoss, MotionGuidance...）
├── evaluate.py                 # 评估指标（MPJPE, PA-MPJPE, PCK_norm）
├── augmentation.py             # CSI 数据增强
├── visualize.py                # 3D 骨架可视化
└── utils.py                    # 随机种子、日志、模型保存
```

## 训练超参数

### 单阶段跨环境训练（`train.py`）

| 超参数 | 值 |
|--------|-----|
| 优化器 | AdamW (lr=1e-3, weight_decay=1e-3) |
| 学习率调度 | 余弦退火 (eta_min=1e-6) |
| 批大小 | 8（× 4 梯度累积 = 等效 32） |
| 序列长度 | 64 帧 |
| 滑动窗口步长 | 32 帧 |
| 梯度裁剪 | 1.0 |
| 早停耐心 | 15 次评估 |
| 最大轮数 | 100 |
| RSC drop pct | 0.5（time + channel + batch） |
| Action Dropout | 50% |
| MixStyle p | 0.5 |

### Stage 1A MAE（`train_mae.py`）

| 超参数 | 值 |
|--------|-----|
| Patch 尺寸 | (4, 19, 5) → 192 patches/sample |
| Mask ratio | 0.75 |
| 优化器 | AdamW (lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95)) |
| LR schedule | Linear warmup 20 ep + cosine decay 280 ep |
| 批大小 | 4 × accum 4 = 16 |
| 轮数 | 300 |
| 时长 | ~75 小时 |

### Stage 1B Action / Stage 2 Pose

| 超参数 | Stage 1B | Stage 2 |
|--------|---------|---------|
| 优化器 | AdamW (差别 LR) | AdamW (差别 LR) |
| `lr_backbone` | 1e-4 | 1e-4 |
| `lr_head` | 5e-4 | 5e-4 |
| weight_decay | 1e-4 | 1e-3 |
| 批大小 | 8 × accum 2 = 16 | 2 × accum 4 = 8 |
| 轮数 | 50 | 50 |
| Label smoothing | 0.1 | - |
| 时长 | ~10 小时 | ~10 小时 |

## 损失函数

### 标准模式（`train_standard.py` / `train_experiment.py`）

$$\mathcal{L} = \mathcal{L}_\text{coord} + \lambda_1 \mathcal{L}_\text{bone} + \lambda_2 \mathcal{L}_\text{vel} + \lambda_3 \mathcal{L}_\text{motion} + \lambda_\text{hip} \mathcal{L}_\text{hip} + \delta \mathcal{L}_\text{action}$$

| 损失项 | 权重 | 说明 |
|--------|------|------|
| $\mathcal{L}_\text{coord}$ | 1.0 | 逐关节 L2 距离 |
| $\mathcal{L}_\text{bone}$ | $\lambda_1$=1.0 | 骨骼长度一致性（L1） |
| $\mathcal{L}_\text{vel}$ | $\lambda_2$=0.5 | 速度平滑性 |
| $\mathcal{L}_\text{motion}$ | $\lambda_3$=2.0 | MotionGuidanceLoss（detach 时序解耦，防止时序塌陷） |
| $\mathcal{L}_\text{hip}$ | $\lambda_\text{hip}$=1.0 | Hip 全局位置预测 |
| $\mathcal{L}_\text{action}$ | $\delta$=0.5 | 动作分类辅助任务 |

### 域泛化模式（`train.py`，含 RSC + Action Dropout）

$$\mathcal{L} = \mathcal{L}_\text{pose}^\text{clean} + \alpha \mathcal{L}_\text{pose}^\text{masked} + \beta \mathcal{L}_\text{cons} + \gamma (\mathcal{L}_\text{div} + \mathcal{L}_\text{tdiv} + \mathcal{L}_\text{input}) + \delta \mathcal{L}_\text{action}$$

| 损失项 | 权重 | 梯度流向 | 说明 |
|--------|------|----------|------|
| $\mathcal{L}_\text{pose}^\text{clean}$ | 1.0 | backbone + decoder | 主姿态损失（含 hip） |
| $\mathcal{L}_\text{pose}^\text{masked}$ | α=0.5 | backbone + decoder | RSC 遮挡路径损失（v7.1: 梯度经未遮挡元素回传到 backbone） |
| $\mathcal{L}_\text{cons}$ | β=2.0 | decoder | 遮挡/未遮挡预测一致性（clean 侧 detach） |
| $\mathcal{L}_\text{div}$ | γ=0.0 | backbone + decoder | 惩罚 batch 内预测方差过低（默认关闭） |
| $\mathcal{L}_\text{tdiv}$ | γ=0.0 | backbone + decoder | 惩罚时序动态不足（默认关闭） |
| $\mathcal{L}_\text{action}$ | δ=0.5 | backbone + classifier | 动作分类辅助任务 |

### Stage 1A MAE 损失

$$\mathcal{L}_\text{MAE} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \| \hat{p}_i - \tilde{p}_i \|_2^2$$

其中 $\mathcal{M}$ 是被 mask 的 patch 集合，$\tilde{p}_i$ 是 per-patch 归一化后的 target。

## 域泛化技术汇总

| 技术 | 位置 | 机制 |
|------|------|------|
| **InstanceNorm 门控** | CSI 编码器 | 可学习地混合 BN（共享统计量）和 IN（逐样本统计量），抑制环境特异性统计 |
| **MixStyle** | CSI 编码器 + 全局建模器 | 训练时随机混合样本间特征统计量，合成虚拟域 |
| **CSI 数据增强** | 数据加载 | 幅度缩放、相位噪声、子载波丢弃、频域遮挡，模拟环境变化 |
| **RSC** | z_global | 遮挡梯度最大的 50% 特征维度，梯度回传到整个 backbone（v7.1 修复） |
| **Action Dropout** | forward_rsc | 训练时 50% 概率将动作嵌入置零，防止 decoder 过度依赖动作先验导致跨域级联失效 |
| **MAE 预训练（NEW）** | Stage 1A | 自监督学习模态无关的鲁棒 CSI 表征，源域 only |
| **Action 预训练（NEW）** | Stage 1B | 动作分类对齐 action-aware 语义，源域 only |
| **差别 LR 微调（NEW）** | Stage 1B/2 | 预训练 backbone 慢 LR (1e-4)，新 head 快 LR (5e-4) |
| **动作分类器** | 辅助分支 | 迫使编码器在 z_global 中保留动作区分信息 |

## 完整实验套件

按 MMFi 评估协议运行全部 3 × 3 实验：

**协议：**
- **P1**：A01–A14（14 类日常动作）
- **P2**：A15–A27（13 类康复动作）
- **P3**：A01–A27（全部 27 类动作）

**划分设定：**
- **S1（随机划分）**：序列级 75/25 随机划分
- **S2（跨受试者）**：32 人训练 / 8 人测试（每环境各 2 人测试）
- **S3（跨环境）**：留一环境法（4 次实验取平均）

```bash
# 运行全部 9 组实验（S3 每组 4 次留一实验）
python run_all_experiments.py

# 只运行某个协议或某个设定
python run_all_experiments.py --protocol P1
python run_all_experiments.py --setting S2
python run_all_experiments.py --protocol P3 --setting S1

# 运行单个实验
python train_experiment.py --protocol P1 --setting S1
python train_experiment.py --protocol P3 --setting S3 --test_env E04

# 只汇总已有结果
python run_all_experiments.py --collect_only
```

结果保存在 `experiments/` 下，自动导出 `results_summary.csv`。

## 关键发现

1. **均值姿态塌陷**是核心挑战：当前 CSI 分辨率（每帧 3×114×10 = 3420 个值）携带的动作区分信息有限，模型倾向于收敛到一个平均站姿以最小化期望误差。Action Dropout + RSC + MotionGuidanceLoss 三者协同缓解。

2. **跨环境域偏移**与协议强相关：
   - 标准 8:2 划分: MPJPE ≈ 114mm
   - 严格跨环境（test 不用 GT action）: MPJPE ≈ 345mm
   - 量化了不同 WiFi 部署环境间的分布偏移

3. **PA-MPJPE 持平 SOTA**（104.68 vs 104.2 mm），说明模型能学到合理的身体比例；但 MPJPE 仍差 28mm，主因是绝对 hip 位置预测的物理可观测性限制。

4. **mmWave 物理引导方法（HMSF）不能直接迁移到 CSI**：subcarrier-antenna 平面缺乏 mmWave range-angle 的空间-物理对应。

5. **MAE 预训练是破局关键**：DT-Pose 的 SOTA 主要来自 400 epoch MAE 预训练而非架构。本项目用 300 epoch MAE + 三阶段微调，预期接近或超越 SOTA。

## 版本历史

- **v8.0（当前）**：三阶段训练（MAE → Action → Pose）+ HMSF 消融
- **v7.1**：RSC 梯度修复（保留 backbone 计算图）+ Action Dropout
- **v7.0**：Action 条件化 PoseDecoder
- **v6.x**：MotionGuidanceLoss 替代 VelocitySmoothLoss
- **v5.x**：严格 DG 协议（测试时不用 GT 动作）
- **v4.x**：双分支 CSI Encoder + RSC
- **v1-3**：单分支基线

## 引用

如使用本代码，请引用 MMFi 数据集：

```bibtex
@inproceedings{yang2024mmfi,
  title={MMFi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing},
  author={Yang, Jianfei and Huang, He and Zhou, Yunjiao and Chen, Xinyan and Xu, Yuecong and Yuan, Shenghai and Zou, Han and Lu, Chris Xiaoxuan and Xie, Lihua},
  booktitle={NeurIPS},
  year={2024}
}
```

## 许可

本项目仅供学术研究使用。