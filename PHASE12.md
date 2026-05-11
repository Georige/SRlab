# Phase 12：小波流匹配超分辨率（Wavelet Flow Matching SR）

## 概述

Phase 12 的核心思想：**用离散小波变换（DWT）将超分问题分解到频域子带，再对流匹配（Flow Matching）做 ODE 积分生成高频细节**。

对于 X4 超分（512×1024 HR ← 128×256 LR）：

- LR → bicubic↑ → lr_up (512×1024) → DWT → **LL_lr (3ch, 256×512)** + **H_lr (9ch, 256×512)**
- HR → DWT → **LL_hr (3ch, 256×512)** + **H_hr (9ch, 256×512)**

网络需要预测 H（高频细节），DWT 已将空间分辨率降半（512→256），网络在 256×512 上运行，显存友好。推理时用 LL_lr（来自 LR）做 IDWT 重建。

---

## 核心数学原理

### Haar DWT

纯 PyTorch 实现，无外部依赖，GPU 高效、可微分。

2×2 patch [a00, a01, a10, a11] → Haar 矩阵 H/2：

```
LL = (a00 + a01 + a10 + a11) / 2    低频近似（图像结构）
LH = (a00 - a01 + a10 - a11) / 2    水平高频细节
HL = (a00 + a01 - a10 - a11) / 2    垂直高频细节
HH = (a00 - a01 - a10 + a11) / 2    对角高频细节
```

关键性质：Haar 矩阵是**对称正交**的（H@H = 4I），正/逆变换用同一个矩阵：

```
dwt:   y[b,c,o,h,w] = Σ_p x[b,c,p,h,w] · H[o,p] / 2
idwt:  x[b,c,p,h,w] = Σ_o y[b,c,o,h,w] · H[p,o]
```

其中 `p` 索引 4 个像素位置（a00, a01, a10, a11），`o` 索引 4 个输出子带（LL, LH, HL, HH）。

H_cat = concat(LH, HL, HH) 将 3 个子带拼成 9 通道（3 RGB × 3 高频子带）。

### 流匹配（Flow Matching）

这是最核心的数学创新。将超分视为从 LR 高频到 HR 高频的**连续确定性传输过程**。

#### 训练时

定义两个分布之间的**直线插值路径**：

```
X0 = H_lr   起点（LR 高频子带）
X1 = H_hr   终点（HR 高频子带）
x_t = t·X1 + (1-t)·X0 = X0 + t·(X1 - X0)    t ∈ [0,1] 随机采样
```

目标速度场（恒定的直线速度）：

```
v_target = dx/dt = X1 - X0
```

网络学习预测这个向量场：

```
L_mse = MSE(v_pred, v_target)
L_l1  = L1(v_pred, v_target)
Loss  = L_mse + 0.1 · L_l1
```

**直观理解**：网络学会在得知"当前位置 x_t、低频结构 LL_lr、全图导航 lr_up"后，预测还需要朝 HR 高频方向走多少。

#### 推理时（欧拉积分）

```
x_0 = H_lr
for step = 0, 1, ..., N-1:
    t = step / N
    v = model(x_t, LL_lr, lr_up, t)
    x_{t+dt} = x_t + v · dt

H_pred = x_{t=1}
SR = IDWT(LL_lr, H_pred)
```

默认 N=4 步，步长 dt = 1/4。

#### Teacher Forcing 差距桥接

训练时 x_t 是干净的直线插值，但推理时 x_t 包含模型自己的预测误差，导致训练/推理分布不一致。两项修复：

1. **TF noise**（`tf_noise_std > 0`）：训练时向 x_t 注入高斯噪声，模拟推理时的误差
2. **Pixel loss**（`pixel_weight > 0`）：对 2 步欧拉中间结果计算像素域 MSE 辅助损失

---

## 神经网络架构

### WaveletVFE（小波向量场估计器）

流匹配路线的主模型。

```
lr_up [B,3,H,W]
      │
      ▼
CondEncoder: Conv2d(3→32, stride=2) → GELU → Conv2d(32→64)
      │
      ▼
cond_feat [B,64,H/2,W/2]  ←── 像素域"全息导航图"
      │
x_t [B,9] + LL_lr [B,3] + cond_feat [B,64]
      │
      ▼ concat
[B, 76, H/2, W/2]
      │
      ▼ in_proj Conv2d(76 → base_ch, 3×3)
[B, 64, H/2, W/2]  (假设 base_ch=64)
      │
      ▼
┌──────────────────────────────────────────┐
│  N × FiLMResBlock                        │
│                                          │
│  每个块：                                 │
│  Conv3×3 → GroupNorm → SiLU              │
│      │                                    │
│      ▼ FiLM modulation                   │
│  h = h × (1 + scale) + shift             │
│      │                                    │
│  Conv3×3 → GroupNorm → +skip → SiLU     │
│                                          │
│  scale = time_scale + cond_scale          │
│  shift = time_shift + cond_shift          │
└──────────────────────────────────────────┘
      │
      ▼ Output Head
GroupNorm → SiLU → Conv2d(base_ch → 9, 3×3)
      │
      ▼
v_pred [B, 9, H/2, W/2]
```

#### 关键组件

| 模块 | 功能 | 实现 |
|------|------|------|
| **CondEncoder** | 将 bicubic↑ LR 编码为 DWT 分辨率下的 64ch 特征图 | 2× Conv2d (stride=2) |
| **SinusoidalTimeEmbedding** | t ∈ [0,1] → sin/cos 编码 → MLP → 每块 FiLM 参数 | 标准 DDPM 式时间嵌入 |
| **FiLMResBlock** | 残差块 + 特征调制 | `h = h·(1+scale) + shift` |
| **CondFiLM** | 全局池化 cond_feat → MLP → 每块 FiLM 参数 | 与 time FiLM 相加组合 |

### WaveletDirectSR（直接小波回归）

简化路线：去掉时间、流匹配、x_t 状态，**单次前向传播**直接预测 H_hr。

```
LL_lr [B,3] + cond_feat [B,64] → in_proj → N×FiLMResBlock → Head → H_pred [B,9]
```

架构 = CondEncoder + CondFiLM + N×FiLMResBlock（无 time FiLM），无欧拉积分。

推理：`SR = IDWT(LL_lr, model(LL_lr, lr_up))`

### 参数量

| base_ch | n_blocks | 总参数量 |
|---------|----------|---------|
| 64 | 8 | ~1.9M |
| 128 | 12 | ~8M |

模型没有 U-Net 编解码器——DWT 已经将空间分辨率降半，因此网络在 H/2×W/2 上用平坦的 FiLM 残差块堆叠。

---

## 全景图数据增强（Phase-Aligned Augmentation）

针对 360° 等距柱状投影图的专门增强，所有操作保持 DWT 网格对齐。

| 增强 | 概率 | 方法 | DWT 安全性 |
|------|------|------|-----------|
| 水平循环滚动 | 50% | `torch.roll` 纯平移 | 零插值，完全安全 |
| 垂直裁剪 + 缩放 | 90% | 裁剪量对齐到 `scale×2=8` 的网格 | nearest 缩放，不引入新像素值 |
| 亮度/对比度抖动 | 70% | ±10% 亮度，±10% 对比度 | 逐点操作，安全 |
| HR 高斯噪声 | 30% | HR 目标加性噪声 σ=0.005 | 正则化项，安全 |

---

## 实验路线

### 两阶段探索

#### 阶段 0：单图 Sanity Check

| 实验 | 描述 | 状态 |
|------|------|------|
| Stage 0 (wf_sanity) | 单图 Wavelet Flow，无增强 | 验证 DWT+IDWT 管线 |
| Stage 0 (direct) | 单图 Direct Wavelet Regression | 验证直接回归可行性 |

#### 阶段 1：多图迭代优化

| Stage | 配置文件名 | 改进 | 目的 |
|-------|-----------|------|------|
| 1 (wf_multi) | `phase12_stage1.yaml` | 16 图 + 全增强 + lr=5e-5 | 多图基线 |
| 1.5a (naked) | `phase12_stage1_naked.yaml` | 16 图 + 无增强 + lr=2e-4 | 对照：证明模型能过拟合 16 图 |
| 1.5b (aligned) | `phase12_stage1_aligned.yaml` | +相位对齐增强 + lr=2e-4 | 更好的增强策略 |
| 1.8 (condenc) | `phase12_stage1.8_condenc.yaml` | +CondEncoder (64ch 特征图) | 注入像素域导航信息 |
| 1.9 (tfgap) | `phase12_stage1.9_tfgap.yaml` | +TF noise + pixel loss + 8步 | 缩小 TF→AR 差距 |
| 1.10 (hc) | `phase12_stage1.10_hc.yaml` | base_ch=128, n_blocks=12 | 高容量模型 ~8M |
| 1.11 (condfilm) | `phase12_stage1.11_condfilm.yaml` | +CondFiLM 每块调制 | 防止条件被"洗掉" |
| 1.12 (direct) | `phase12_stage1.12_direct.yaml` | 完全去掉时间/流/欧拉 | 直接回归对照 |

---

## 文件结构

| 文件 | 角色 |
|------|------|
| [model/wavelet_flow.py](model/wavelet_flow.py) | WaveletVFE + WaveletDirectSR 模型定义 |
| [model/dwt_utils.py](model/dwt_utils.py) | Haar DWT/IDWT 工具函数 |
| [factory/trainer_wavelet_flow.py](factory/trainer_wavelet_flow.py) | WaveletFlowTrainer + WaveletDirectTrainer |
| [factory/registry.py](factory/registry.py) | 模型/trainer 注册（wavelet_flow, wavelet_direct） |
| [factory/configs/](factory/configs/) | 10 个阶段 YAML 配置文件 |

---

## 与 Phase 6（SHT 球谐路线）的本质区别

| 维度 | Phase 6 (SHT) | Phase 12 (DWT) |
|------|--------------|----------------|
| 变换 | 球谐变换（**全局**频域） | Haar 小波（**局部**频域） |
| 网络输入 | SHT 系数（频域复数） | DWT 子带 + 像素特征 |
| 生成方式 | 残差回归（一次前向） | **流匹配（4 步 ODE 积分）** |
| 空间分辨率 | 512×1024（全分辨率） | 256×512（DWT 降半） |
| 频域特性 | 全局带限（L_max=255 截断） | 局部多分辨率 |
| 可逆性 | **有截断误差不可逆**（天然图像非带限） | **Haar 完美可逆** |
| 条件注入 | ISHT 重建（频域→空域） | CondEncoder（空域特征编码） |

Phase 12 的核心优势：
1. **DWT 完美可逆**——无截断误差（SHT 即使 L=255 仍有 ~0.006 不可约 MSE）
2. **流匹配 ODE 路径**——从 LR 高频到 HR 高频的平滑插值，物理意义清晰
3. **更小的特征图**——DWT 降半分辨率，显存和计算量更低
