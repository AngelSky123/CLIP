# CSI-Depth-Distill — 深度跨模态蒸馏用于 WiFi-CSI 跨环境 3D 人体姿态估计

本仓库是 **深度跨模态蒸馏（depth cross-modal distillation）** 方案的独立代码库，
目标是用「深度图教师（depth teacher）」在训练期蒸馏「CSI 学生（CSI student）」，
把深度模态携带的几何信息注入纯 CSI 分支，从而在**严格跨环境域泛化（cross-environment DG）**
下提升 3D 人体姿态估计。**测试期只用 CSI**，不使用任何深度数据，符合 DG 协议。

> 本仓库已从原多方案工程中剥离，仅保留蒸馏方案所需文件，便于独立优化与修改。
> 其它历史方案（image-rendering / ImageNet-vision / standard-split / CSI-baseline 三阶段）
> 不在此仓库内。

---

## 方法概览

整个方案分两步：

**Step A — 深度教师（已完成）**
从零训练一个单通道深度 CNN + 时序建模器（复用 `GlobalTemporalModeler`），
在源域 E01–E03 上用 GT 姿态监督。训练完成后**冻结**，其逐帧特征
`z_global (B,T,128)` 作为 Step B 蒸馏的对齐目标。

**Step B — 蒸馏（待实现）**
CSI 学生（即主模型 `CSIRSCPoseDG`）在 E01–E03 上训练，每个 batch 同时把
对齐的 CSI 与 depth 分别喂给学生与冻结的教师，对齐学生的 `z_global` 到教师的
`z_global`（特征级蒸馏：投影头 + cosine/smooth-L1，权重 `λ_distill`）。
测试期丢弃深度，学生仅凭 CSI 推理（`action_idx=None`，严格 DG）。

> Step B 的训练脚本尚未加入本仓库；当前仓库包含 Step A 的完整实现，
> 以及 Step B 学生所需的全部主模型代码。

---

## Step A 教师：实测结果（诚实记录）

在 MMFi 上，源域 E01–E03 训练、E04 评测探针（E04 深度**仅用于这一步的评测探针，
绝不进入任何训练/蒸馏**）：

| 指标 | 教师（全量 E01-E03→E04） | CSI baseline 参考 |
|---|---|---|
| MPJPE | **269 mm** (best, epoch 24) | 345 mm |
| PA-MPJPE | **89–90 mm** (epoch 30+ 稳定) | ~104 mm |
| PredStd | 100–130 mm（未塌缩） | — |

**如何解读（重要）：**
- 教师 MPJPE（269）显著低于 CSI baseline（345），说明**深度携带了 CSI 学不到的几何信息**。
- 但教师真正的强项不是绝对定位，而是**姿态结构**：PA-MPJPE ≈ 89，比 baseline 的 ~104
  好约 15mm（PA 把全局位置/旋转/尺度对齐后剩下的纯肢体配置）。
- E04 的全局位置分布与 E01–E03 系统性不同，导致教师的绝对定位（MPJPE）在 E04 上
  transfer 受限、PredStd 偏大。

**因此 Step B 蒸馏的主攻目标是 PA-MPJPE**（把教师的姿态结构信息传给 CSI 学生），
而非寄望于 MPJPE 大幅下降。这是一个比「压 hip 全局定位」更稳、更可辩护的目标。

> 注意：教师强 ≠ 蒸馏一定有效。深度里有几何信息，不代表该信息一定能通过特征对齐
> 注入 CSI 分支——CSI 物理上可能观测不到那些线索。Step B 需用诊断实验（看 PA-MPJPE
> 是否从 104 向 90 移动）来验证，而非假设。

---

## 目录结构

```
.
├── train_depth_teacher.py      # Step A 入口：训练深度教师
├── dataset_distill.py          # 蒸馏数据集：depth(+可选 CSI) + GT，逐帧对齐
├── models/
│   ├── depth_teacher.py        # 深度教师：DepthEncoder + GlobalTemporalModeler + pose_head
│   ├── full_model.py           # CSI 学生主模型 CSIRSCPoseDG（Step B 用）
│   ├── csi_encoder.py          # 双分支 CSI 编码器（InstanceNorm + MixStyle）
│   ├── local_encoder.py        # 局部时空编码 + 特征池化
│   ├── global_encoder.py       # 全局时序建模器（教师与学生共用）
│   ├── pose_decoder.py         # 动作条件化姿态解码器 + H36M 骨架定义
│   ├── rsc.py                  # Representation Self-Challenging（DG 正则）
│   ├── mixstyle.py             # MixStyle 域风格混合
│   └── __init__.py
├── losses.py                   # PoseLoss / TotalLoss 等训练目标
├── evaluate.py                 # MPJPE / PA-MPJPE / PCK 评测（DT-Pose 对齐）
├── dataset.py                  # 原生 CSI 数据集 + CSIPreprocessor（学生与蒸馏 CSI 分支用）
├── augmentation.py             # CSI 数据增强（跨环境鲁棒性）
├── config.py                   # 主模型/训练超参（学生构造用）
├── train.py                    # CSI 学生的 DG 训练循环（Step B 参照/复用）
└── utils.py                    # 日志、随机种子、checkpoint、run_config 存档等
```

---

## 数据假设

MMFi 数据集，目录形如：
```
<data_root>/<Env>/<Subject>/<Action>/
    ├── wifi-csi/frame###.mat        # CSIamp, CSIphase: (3, 114, 10)
    ├── depth/frame###.png           # 16-bit 毫米深度图 (480, 640)
    └── ground_truth.npy             # (帧数, 17, 3) 米
```
- 环境-被试映射：E01→S01-10, E02→S11-20, E03→S21-30, E04→S31-40。
- 每序列内 depth / csi / gt **帧数一致且逐帧对齐**（同步采集）。
- 深度归一化用**固定物理尺度**：`clip(d, 0, depth_clip) / depth_clip`（默认 depth_clip=5000mm），
  **不用逐图 min-max**——逐图归一化会抹掉绝对距离，正是要保留给教师的全局定位线索。

> **DG 红线**：E04（目标环境）的深度仅可用于 Step A 的评测探针。
> 任何训练 / 蒸馏都**不得**使用 E04 深度。测试期学生仅用 CSI。

---

## 运行

### Step A：训练深度教师

```bash
python train_depth_teacher.py \
    --data_root /path/to/MMFi \
    --train_envs E01 E02 E03 --test_env E04 \
    --depth_img 112 --depth_clip 5000 \
    --epochs 50 --batch_size 16 --accumulate_grad 1 --lr 5e-4 \
    --num_workers 12 \
    --save_dir ./checkpoints/depth_teacher_full
```

产物：`./checkpoints/depth_teacher_full/teacher_best.pt`，包含
`model_state_dict` 以及供蒸馏单独加载的 `encoder` 与 `global_modeler`。

> **IO 提示**：深度图是 480×640 的 PNG，逐帧读取 + resize 是 IO 瓶颈，
> 训练时 GPU 利用率低（接近 0%）是**正常现象**——瓶颈在磁盘/CPU 而非算力。
> 适当增大 `--num_workers` 与 `--batch_size` 可缓解（显存占用很小）。

### Step B：蒸馏（脚本待加入）

设计要点（实现时遵循）：
- 数据用 `dataset_distill.py`，`with_depth=True, with_csi=True`，CSI 与 depth 由
  同一 `(start, length)` 切片 → 天然逐帧对齐。
- 加载 `teacher_best.pt` 的 `encoder` + `global_modeler`，`eval()` 且冻结（不回传梯度）。
- 学生侧加投影头 `proj: 128→128`，对齐 `proj(z_student)` 与 `z_teacher.detach()`，
  损失用 cosine + smooth-L1。
- 总损失 = 学生原有 `TotalLoss` + `λ_distill · L_align`，`λ_distill` 默认 0.1，
  建议扫 {0.05, 0.1, 0.5}。
- 先在单阶段训练上做**诊断版**（看 PA-MPJPE 是否改善），有效再上完整三阶段。

---

## 评测指标

`evaluate.py` 输出与 DT-Pose 严格对齐的指标：MPJPE、MPJPE_aligned（hip 对齐）、
PA-MPJPE（Procrustes 含尺度对齐）、PCK@50/@20（按身长归一化）。所有 mm 指标内部 ×1000。

---

## 训练配置存档（复现）

`utils.save_run_config(args, save_dir)` 会在 `save_dir` 写入 `run_config.json`，
记录完整 args、git commit/branch/dirty 标志、运行环境（python/torch/cuda/gpu）、
完整命令行与时间戳。`train_depth_teacher.py` 已接入。该函数包了 try/except，
存档失败绝不会中断训练。

> 提示：训练前先 `git commit`，配合 `run_config.json` 里的 commit hash 才能精确复现；
> `dirty: true` 表示训练时有未提交改动，复现性会打折扣。

---

## 环境

- Python 3.7，PyTorch 2.2.2，CUDA（单卡，显存需求低）
- numpy / scipy / Pillow
- 单卡 RTX 4080（16GB）可跑

---

## 备注

- 本仓库的 CSI 学生使用**原生 CSI 编码链**（`use_vision_backbone=False`，默认）；
  蒸馏方案**不使用** vision backbone，相关代码已从本仓库移除。
- `train.py` / `config.py` / `dataset.py` 保留是因为 Step B 的 CSI 学生需要它们
  （主模型构造、CSI 预处理、训练循环参照）。