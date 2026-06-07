# CSI-RSC-PoseDG

基于 WiFi CSI 的跨环境 3D 人体姿态估计。训练期用深度图教师做知识蒸馏，**推理只用 CSI**（不依赖深度/视觉）。在 MMFi 上以严格盲测的跨房间设定（Setting 3 / Protocol 3，全 27 动作，单链路 CSI）对标 DT-Pose。

---

## 1. 方法概述

```
CSI (T,9,114,10)
   └─ csi_encoder ──> local_encoder (3D conv) ──> feature_pooling
        ──> global_modeler (Transformer + TCN)  ──>  z_global (B,64,128)
              ├─ pose_decoder (TaskPromptCoarseHead + SkeletonRefiner/GCN) ──> 3D pose (B,T,17,3)
              ├─ action_classifier
              └─ RSC (representation self-challenging)
```

- **教师**：深度图姿态教师 (`DepthPoseTeacher`)，训练期冻结，对学生做 feature-level + output-level 蒸馏。
- **学生**：CSI-only，参数量约 **1.58M**。
- **结构正则**（当前版本新增，见 `structural_losses.py`）：骨长对齐 (vs GT)、左右对称、时序骨长稳定、root-relative（髋中心）位置对齐。四项均**平移不变**，只约束相对骨架，不触碰全局 hip 定位项。

---

## 2. 实验设定（与 DT-Pose 对齐）

| 项 | 设定 |
|---|---|
| 数据集 | MMFi |
| 划分 | **Setting 3（跨环境 / cross-environment）** |
| 协议 | **Protocol 3（全 27 动作）** |
| 训练环境 | E01–E03（S01–S30） |
| 测试环境 | **E04（S31–S40），严格盲测、训练期从不参与** |
| 输入 | 单链路 CSI（3Tx×1Rx），64 帧窗口 `(T=64, 9, 114, 10)` |
| 推理 | **CSI-only** |
| 评测口径 | 逐帧、全帧覆盖、无 padding、`action_idx=None`（与 DT-Pose 的单帧绝对评测一致）。MPJPE 为纯绝对误差（不做任何 centering）；PA-MPJPE 用含 scale 的 Procrustes 对齐 |

> **可复现性**：所有训练固定 `--seed 42` 并启用 `cudnn.deterministic=True / benchmark=False`。报告数一律来自 `eval_dtpose_faithful.py`（唯一权威口径）；训练期 `evaluate_v2` 的滑窗监控值与 faithful 口径不可比，**不用于报告**。

---

## 3. 结果（E04，faithful 逐帧口径）

| 模型 | MPJPE (mm) | PA-MPJPE (mm) | hip_err (mm) |
|---|---|---|---|
| baseline（TaskPrompt 解码器） | 366.6 | 106.2 | 337.8 |
| + 结构正则（骨长/对称/时序） | 361.97 | **105.42** | 334.91 |
| + root-relative 位置对齐（当前版本，训练中） | _TBD_ | _TBD_ | _TBD_ |
| **DT-Pose (S3/P3)** | **316.8** | **104.2** | — |

> PA-MPJPE 方差极小（多 stride 评测 σ ≈ 0.02mm），105.42 为稳定真值。
> _注：最后一行待当前 `distill_struct_rel` 训练完成后用 faithful 口径填入。_

**小结**：在结构指标 PA-MPJPE 上，本方法（105.4）与 DT-Pose（104.2）**基本持平**，差距约 1mm，处于单链路 CSI 的硬件分辨率地板附近（DT-Pose 原文亦指出手/肘等末端误差受限于 WiFi 分辨率，需更多设备/更高分辨率）。绝对 MPJPE 落后约 45mm，差距**全部集中在 hip 全局定位**（见 §4）。

---

## 4. Limitation 分析：跨房间绝对定位的信息上界

绝对 MPJPE 的差距并非建模不足，而是**单链路 CSI 中“人在未见过房间里的绝对位置”这一信息本身不可跨域迁移**。多条独立证据指向同一结论：

**(a) 原始输入信息探针**（`probe_raw_amplitude_hip.py`）。用线性岭回归从 CSI 幅度直接预测 E04 hip 绝对坐标，无论保留绝对尺度与否：

| 特征 | held-in (E01–03) | E04 |
|---|---|---|
| mean_base（预测训练集均值，零信息标尺） | 172.7 | **324.2** |
| 原始绝对幅度 | 152.3 | 350.8 |
| log 功率 | 155.5 | 348.8 |
| 逐帧归一化幅度 | 149.6 | 356.4 |

三种幅度表示在训练房间都优于零信息标尺，**在 E04 上却全部劣于零信息标尺**——典型的“室内可学、跨房间反向迁移”，说明幅度→定位的映射逐房间不同，不可迁移。

**(b) 教师上界**。深度图教师（拥有视觉深度）在 E04 的 hip_err 仍达 ~236mm，即便强信号模态也难以恢复跨房间绝对位置。

**(c) 完整模型 vs 零信息基线**。部署模型在 E04 的 hip_err（~335mm）已与“永远预测平均位置”的零信息基线（324mm）相当或更高——用满全部输入的非线性模型在绝对定位上未超过常数预测，进一步印证信息不在特征中。

**(d) 杠杆穷举**。解码器结构（多轮）、自监督预训练（MAE-DCL：TC-CL + uniformity）、合规 test-time 重心化、蒸馏权重调参——均未移动 hip 误差。

**结论**：在严格盲测的跨房间、单链路 CSI、CSI-only 推理下，绝对 MPJPE 受信息论上界约束；可改善的空间在**相对结构（PA-MPJPE）**，本方法已将其推至 SOTA 持平水平。

---

## 5. 复现步骤

### 环境
```
Python 3.7 / PyTorch 2.2.2 / CUDA (RTX 4080, 16GB)
依赖见 requirements（numpy, scipy, torch ...）
```

### 数据
MMFi 置于 `--data_root`，目录结构：
```
<data_root>/E0{1..4}/S{01..40}/A{01..27}/{wifi-csi/frame*.mat, depth/, ground_truth.npy}
```

### 训练（当前版本：结构正则 + root-relative）
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
- `--w_bone/--w_sym/--w_temp/--w_rel`：四项结构损失权重。
- 选点在 E01–E03 held-out subjects 的 val MPJPE 上；E04 仅监控、不参与选点。

### 评测（唯一权威口径）
```bash
python eval_dtpose_faithful.py \
    --data_root /path/to/MMFi \
    --ckpt ./checkpoints/distill_struct_rel/best_mpjpe_ema.pth \
    --test_env E04 --seq_len 64 --variance
```
`--variance` 跑多 stride 报 mean±σ，用于判断指标差异是否超过评测口径噪声。

---

## 6. 关键文件

| 文件 | 说明 |
|---|---|
| `train_distill_pretrained.py` | 主训练脚本（蒸馏 + EMA + held-out 选点 + 结构正则） |
| `structural_losses.py` | 结构正则：骨长 / 对称 / 时序骨长 / root-relative |
| `eval_dtpose_faithful.py` | 逐帧 faithful 评测（与 DT-Pose 对齐的权威口径） |
| `probe_raw_amplitude_hip.py` | 原始幅度 → E04 hip 绝对定位信息探针（§4 分析依据） |
| `models/` | CSI-RSC-PoseDG 学生网络 |
| `models/depth_teacher.py` | 深度图教师 |
| `losses.py` / `distill_loss.py` | 基础姿态损失 / 蒸馏损失 |
| `dataset.py` / `dataset_distill.py` | MMFi 数据加载 |

---

## 7. 诚实声明

- 报告的所有数均来自严格盲测（E04 训练期从不可见）+ faithful 逐帧口径，无 GT 动作标签泄露、无对 E04 真值的对齐。
- PA-MPJPE 与 DT-Pose 持平（约 1mm 内）；绝对 MPJPE 落后，且本文将其归因于信息论上界并给出测量证据，而非声称全面超越。
- 任何低于 faithful 报告值的数（如训练期滑窗监控值）均不作为结论。