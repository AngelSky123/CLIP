# CSI-RSC-PoseDG：基于 WiFi CSI 的 3D 人体姿态估计与域泛化

<p align="center">
  <img src="assets/architecture.png" width="85%" alt="模型架构"/>
</p>

基于 [MMFi](https://github.com/ybhbingo/MMFi) 数据集，利用商用 WiFi 的信道状态信息（CSI）实现**无需摄像头的 3D 人体姿态估计**，并通过域泛化技术提升跨环境的泛化能力。模型输出 H36M 格式的 17 关节 3D 骨架坐标。

## 主要特点

- **双分支 CSI 编码器**：幅度与相位独立编码，可学习的 BN/IN 门控 + MixStyle 实现环境不变特征提取
- **表示自挑战（RSC）**：训练时遮挡梯度最大的特征维度，梯度回传到整个 backbone（v7.1 修复）
- **动作条件化解码**：decoder 接收动作嵌入，训练时 50% 概率 Action Dropout 防止跨域级联失效
- **由粗到精姿态解码**：MLP 粗回归 + 图卷积网络（GCN）骨架精调
- **完整评估体系**：支持 3 种协议 × 3 种划分设定（共 9 组实验）

## 实验结果

### 标准划分（按被试 8:2 划分）

| 指标 | 数值 |
|------|------|
| MPJPE | 114.12 mm |
| PA-MPJPE | 107.65 mm |
| PCK@50 | 41.3% |
| PCK@20 | 17.8% |

各环境分项：E01=112.7mm, E02=116.1mm, E03=116.1mm, E04=111.6mm

### 跨环境域泛化（E01-E03 训练，E04 测试）

| 指标 | 基线 | + RSC + MixStyle |
|------|------|-----------------|
| MPJPE | 119.67 mm | 122.53 mm |
| PA-MPJPE | 109.30 mm | 108.13 mm |
| PCK@50 | 39.1% | 37.8% |
| 预测多样性 PredStd | ~0 mm | 12.1 mm |

> 跨环境设定比标准划分高约 5.5mm，量化了 WiFi CSI 信号的跨环境域偏移。PredStd 衡量预测多样性——数值越高说明模型能区分不同动作，而非塌陷为平均姿态。

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

### 各模块详细说明

**Module 1 — 双分支 CSI 编码器**

幅度分支和相位分支结构对称，每个分支包含：
- `InstanceNorm2d`：逐样本归一化，去除环境导致的全局统计偏移
- 2 个 `ResBlock2D`：每个残差块内部使用 **BN/IN 混合门控**，即 `output = (1-σ(gate))·BN(x) + σ(gate)·IN(x)`，gate 可学习
- `MixStyle2D`：训练时随机混合 batch 内不同样本的特征统计量，模拟虚拟环境

两个分支的 64 维输出通过**门控融合**合并：两个独立的 Sigmoid 门控生成幅度/相位权重，加上交叉注意力捕捉幅度-相位交互。

时序分块处理：T=64 帧按 chunk_size=16 分 4 批送入 2D 网络，避免显存溢出。

**Module 2 — 局部时空编码器**

2 个 Res3DConv 残差块，每个块包含两层 `Conv3d(64→64, k=3×3×3, pad=1)`。3D 卷积同时在时间、子载波、packet 三个维度上提取局部模式。

**Module 3 — 特征池化**

`AdaptiveAvgPool2d` 将 (114, 10) 的空间维度压缩为标量，再通过 `Linear(64→128)` 升维到全局建模维度。

**Module 4 — 全局时序建模器**

3 层 Pre-LN Transformer（d_model=128, 4 头注意力, FFN 扩展比 4×）捕捉长程时间依赖，2 层膨胀 TCN（dilation=1,2）补充局部时序平滑。前面插入 MixStyleTemporal 进一步混合跨环境时序统计量。

**Module 5 — 动作条件化姿态解码器**

动作分类器（`ActionClassifier`）从 z_global 预测动作类别并生成动作嵌入向量（32 维）：
- **训练时**：使用 GT 动作标签查表 `Embedding(27, 32)`
- **推理时**：使用 softmax 概率加权平均所有嵌入向量
- **Action Dropout（v7.1）**：训练时 50% 概率将 action_emb 置零，迫使 decoder 不过度依赖动作先验

粗姿态头（MLP）接收 `[z_global, action_emb]` 拼接输入（128+32=160 维），直接回归 17×3=51 维坐标。骨架精调器（GCN）利用 H36M 骨架邻接矩阵做图卷积传播，以残差方式修正粗预测：`P_final = P_coarse + delta`。

## 数据集

使用 [MMFi 数据集](https://github.com/ybhbingo/MMFi)：

- **4 个环境**（E01–E04），不同房间布局和 WiFi 部署
- **40 个被试**（每环境 10 人，S01–S40）
- **27 类动作**：A01–A14 日常动作，A15–A27 康复动作
- **每序列约 297 帧**，每帧包含 CSI 幅度/相位矩阵

每帧 CSI 格式：`CSIamp (3, 114, 10)` 和 `CSIphase (3, 114, 10)`，其中 3=接收天线数，114=OFDM 子载波数，10=每帧数据包数。

真值标注：`ground_truth.npy`，形状 `(F, 17, 3)`，17 关节 H36M 格式，单位米。

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

### 训练（跨环境域泛化）

```bash
# 在 E01+E02+E03 上训练，E04 上测试
python train.py
```

模型和日志保存在 `checkpoints/run_YYYYMMDD_HHMMSS/`，每次运行自动创建独立目录。

### 训练（标准 8:2 划分）

```bash
# 所有环境混合，按被试 8:2 划分
python train_standard.py
```

### 可视化

```bash
python visualize.py \
    --checkpoint checkpoints_standard/run_xxx/best_model.pth \
    --env E04 --subject S31 --action A01 \
    --frame 30 --save_dir viz_output
```

生成 4 种可视化：单帧骨架对比（正面+俯视）、多帧骨架网格、逐关节误差热力图、CSI 输入展示。

<p align="center">
  <img src="assets/skeleton_example.png" width="80%" alt="骨架可视化示例"/>
</p>

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
# 运行全部 9 组实验（共 18 次训练，S3 每组 4 次留一实验）
python run_all_experiments.py

# 只运行某个协议或某个设定
python run_all_experiments.py --protocol P1
python run_all_experiments.py --setting S2
python run_all_experiments.py --protocol P3 --setting S1

# 运行单个实验
python train_experiment.py --protocol P1 --setting S1
python train_experiment.py --protocol P3 --setting S3 --test_env E04

# 只汇总已有结果，不训练
python run_all_experiments.py --collect_only
```

结果保存在 `experiments/` 下，自动导出 `results_summary.csv`。

## 项目文件结构

```
CSI-RSC-PoseDG/
├── models/
│   ├── __init__.py
│   ├── csi_encoder.py          # 双分支编码器（BN/IN 门控 + MixStyle）
│   ├── local_encoder.py        # Res3DConv 块 + 特征池化
│   ├── global_encoder.py       # Transformer + TCN + MixStyle
│   ├── full_model.py           # 完整模型（RSC 梯度修复 + Action Dropout）
│   ├── pose_decoder.py         # 动作条件化粗 MLP + GCN 骨架精调
│   ├── mixstyle.py             # MixStyle 层（2D / 时序）
│   └── rsc.py                  # RSC 模块定义
├── config.py                   # 跨环境域泛化配置
├── config_standard.py          # 标准 8:2 划分配置
├── dataset.py                  # 域泛化数据加载器
├── dataset_standard.py         # 标准数据加载器
├── train.py                    # 域泛化训练（RSC + 反塌陷损失）
├── train_standard.py           # 标准训练（纯姿态损失）
├── train_experiment.py         # 统一实验脚本（协议 × 设定）
├── run_all_experiments.py      # 全部实验编排脚本
├── losses.py                   # 损失函数（L_pose, L_cons, L_div 等）
├── evaluate.py                 # 评估指标（MPJPE, PA-MPJPE, PCK）
├── augmentation.py             # CSI 数据增强
├── visualize.py                # 3D 骨架可视化
└── utils.py                    # 随机种子、日志、模型保存
```

## 训练细节

| 超参数 | 值 |
|--------|-----|
| 优化器 | AdamW (lr=1e-3, weight_decay=1e-4) |
| 学习率调度 | 余弦退火 (eta_min=1e-6) |
| 批大小 | 2（× 4 梯度累积 = 等效 8） |
| 序列长度 | 64 帧 |
| 滑动窗口步长 | 32 帧 |
| 梯度裁剪 | 1.0 |
| 早停耐心 | 15 次评估 |
| 最大轮数 | 100 |

### 损失函数

**标准模式（`train_standard.py` / `train_experiment.py`）：**

$$\mathcal{L} = \mathcal{L}_\text{coord} + \lambda_1 \mathcal{L}_\text{bone} + \lambda_2 \mathcal{L}_\text{vel} + \delta \mathcal{L}_\text{action}$$

- $\mathcal{L}_\text{coord}$：逐关节 L2 距离
- $\mathcal{L}_\text{bone}$：骨骼长度一致性（L1）
- $\mathcal{L}_\text{vel}$：速度平滑性
- $\mathcal{L}_\text{action}$：动作分类交叉熵

**域泛化模式（`train.py`，含 RSC + forward_rsc）：**

$$\mathcal{L} = \mathcal{L}_\text{pose}^\text{clean} + \alpha \mathcal{L}_\text{pose}^\text{masked} + \beta \mathcal{L}_\text{cons} + \gamma (\mathcal{L}_\text{div} + \mathcal{L}_\text{tdiv} + \mathcal{L}_\text{input}) + \delta \mathcal{L}_\text{action}$$

| 损失项 | 权重 | 梯度流向 | 说明 |
|--------|------|----------|------|
| $\mathcal{L}_\text{pose}^\text{clean}$ | 1.0 | backbone + decoder | 主姿态损失 |
| $\mathcal{L}_\text{pose}^\text{masked}$ | α=0.5 | backbone + decoder | RSC 遮挡路径损失（v7.1: 梯度经未遮挡元素回传到 backbone） |
| $\mathcal{L}_\text{cons}$ | β=2.0 | decoder | 遮挡/未遮挡预测一致性（clean 侧 detach） |
| $\mathcal{L}_\text{div}$ | γ=0.0 | backbone + decoder | 惩罚 batch 内预测方差过低（默认关闭） |
| $\mathcal{L}_\text{tdiv}$ | γ=0.0 | backbone + decoder | 惩罚时序动态不足（默认关闭） |
| $\mathcal{L}_\text{action}$ | δ=0.5 | backbone + classifier | 动作分类辅助任务 |

> **关于 γ=0.0**：`config.py` 默认关闭反塌陷损失。如需启用可设 `--gamma 0.005`。`losses.py` 中 TotalLoss 的初始化默认值 (γ=0.005) 为历史遗留，实际运行时始终由 config 覆盖。

## 域泛化技术汇总

| 技术 | 位置 | 机制 |
|------|------|------|
| **InstanceNorm 门控** | CSI 编码器 | 可学习地混合 BN（共享统计量）和 IN（逐样本统计量），抑制环境特异性统计 |
| **MixStyle** | CSI 编码器 + 全局建模器 | 训练时随机混合样本间特征统计量，合成虚拟域 |
| **CSI 数据增强** | 数据加载 | 幅度缩放、相位噪声、子载波丢弃、频域遮挡，模拟环境变化 |
| **RSC** | z_global | 遮挡梯度最大的 50% 特征维度，梯度回传到整个 backbone（v7.1 修复） |
| **Action Dropout** | forward_rsc | 训练时 50% 概率将动作嵌入置零，防止 decoder 过度依赖动作先验导致跨域级联失效 |
| **多样性损失** | 训练目标 | 惩罚预测方差过低 (L_div)、时序动态不足 (L_tdiv)、输入不敏感 (L_input)（默认关闭） |
| **动作分类器** | 辅助分支 | 迫使编码器在 z_global 中保留动作区分信息 |

## 关键发现

1. **均值姿态塌陷**是核心挑战：当前 CSI 分辨率（每帧 3×114×10 = 3420 个值）携带的动作区分信息有限，模型倾向于收敛到一个平均站姿以最小化期望误差。

2. **跨环境域偏移**造成约 5.5mm MPJPE 性能下降（114→120mm），量化了不同 WiFi 部署环境间的分布偏移。

3. **PA-MPJPE 一致性低于 MPJPE**（标准划分下 108 vs 114mm），说明模型能学到合理的身体比例，但在绝对关节定位上存在困难。

4. 动作分类辅助任务的准确率几乎未超过随机猜测水平（27 类随机=3.30，训练后≈3.00），确认了信号信息瓶颈在输入层面而非模型容量。

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