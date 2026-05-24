# Phase 12：小波流匹配超分辨率（Wavelet Flow Matching SR）

## 概述

Phase 12 的核心思想：**用离散小波变换（DWT）将超分问题分解到频域子带，再对流匹配（Flow Matching）做 ODE 积分生成高频细节**。

对于 X4 超分（512×1024 HR ← 128×256 LR），网络只在 256×512 的 DWT 子带上运行，显存友好。

---

## 推理数据流（具体维度追踪）

以 batch size B=1 为例，追踪单张图片在 WaveletVFE 推理全流程中的维度变化。

### Step 0：输入

| 变量 | 维度 | 说明 |
|------|------|------|
| LR | `[1, 3, 128, 256]` | 原始低清图（X4 下采样） |
| HR (目标) | `[1, 3, 512, 1024]` | 高清真值（仅用于验证） |

### Step 1：Bicubic 上采样

```
LR [1, 3, 128, 256] ──bicubic↑──→ lr_up [1, 3, 512, 1024]
```

### Step 2：DWT 分解（Haar）

`lr_up [1, 3, 512, 1024]` 被 DWT 分解为 4 个子带，每个空间分辨率降半：

```
                    ┌── LL_lr [1, 3, 256, 512]   低频（图像轮廓）
DWT(lr_up) ────────┼── LH    [1, 3, 256, 512]   水平边缘
  512→256 (÷2)     ├── HL    [1, 3, 256, 512]   垂直边缘
  1024→512 (÷2)    └── HH    [1, 3, 256, 512]   对角纹理
```

LL_lr 保留作为 IDWT 重建的结构锚点。三个高频子带拼成 H_lr：

```
H_lr = concat(LH, HL, HH)  →  [1, 9, 256, 512]
```

**具体数值示例**（假设 lr_up 左上角 2×2 像素值）：

```
lr_up 像素:  [0.3, 0.1]      Haar 矩阵:  H/2 = 0.5 × [[1, 1, 1, 1],
             [0.5, 0.9]                              [1,-1, 1,-1]
                                                     [1, 1,-1,-1]
                                                     [1,-1,-1, 1]]

a00=0.3, a01=0.1, a10=0.5, a11=0.9
LL = (0.3 + 0.1 + 0.5 + 0.9) / 2 = 0.90
LH = (0.3 - 0.1 + 0.5 - 0.9) / 2 = -0.10
HL = (0.3 + 0.1 - 0.5 - 0.9) / 2 = -0.50
HH = (0.3 - 0.1 - 0.5 + 0.9) / 2 = 0.30
```

### Step 3：CondEncoder 编码

```
lr_up [1, 3, 512, 1024]
      │ Conv2d(3→32, kernel=3, stride=2, pad=1)
      │ [1, 32, 256, 512]  ← 分辨率降半
      │ GELU
      │ Conv2d(32→64, kernel=3, stride=1, pad=1)
      ▼
cond_feat [1, 64, 256, 512]  像素域"全息导航图"
```

cond_feat 自带 stride=2 降采样到 DWT 分辨率（256×512），为每个 FiLMResBlock 提供全局结构指引。

### Step 4：4 步欧拉积分（核心推理循环）

欧拉积分的起始状态来自 Step 2 的 DWT 分解结果：

```
H_lr = concat(LH, HL, HH) = [1, 9, 256, 512]    ← 来自 Step 2
x_0 = H_lr                                          ← 欧拉起点
步长: dt = 1/4 = 0.25
```

欧拉第 1 步输入 `x_t` 即 `x_0`（t=0 时尚未更新），第 2 步起逐步使用上一步的输出 `x_{t+dt}`。

#### 第 1 步拆解（t=0.0 → t=0.25）：WaveletVFE 内部逐层追踪

##### ① 输入拼接（concat）

```
x_t = x_0 = H_lr   [1,  9, 256, 512]   当前高频状态（来自 Step 2 DWT)
LL_lr               [1,  3, 256, 512]   低频结构锚点（来自 Step 2 DWT)
cond_feat           [1, 64, 256, 512]   像素域导航特征（来自 Step 3)
    │
    ▼ dim=1 拼接
input [1, 76, 256, 512]
```

##### ② 输入投影（in_proj）

```
Conv2d(in_ch=76, out_ch=64, kernel_size=3, padding=1)
权重张量: [64, 76, 3, 3]  ← 64*76*3*3 = 43,776 个参数
偏置:     [64]

input [1, 76, 256, 512] → Conv2d → h [1, 64, 256, 512]
```

##### ③ FiLM 参数生成（两条支路并行）

**支路 A：SinusoidalTimeEmbedding（时间 → 调制信号）**

```
t = [0.0]  (标量，已 broadcast 到 [1])
    │
    ▼ 频率编码
half = dim/2 = 128
ω_k = exp(-k · ln(10000) / 128),  k = 0, 1, ..., 127
args[0,k] = t × ω_k              →  [1, 128]
emb = concat(sin(args), cos(args))  →  [1, 256]
    │
    ▼ MLP: Linear(256 → 1024) → SiLU → Linear(1024 → 2048)
原始输出 [1, 2048] = [1, 8 × 64 × 2]  → reshape → [1, 8, 64, 2]
    │
    ▼ 拆分为 8 个块的 scale/shift
time_film[:, i, :, 0] = scale_i  [1, 64]   第 i 块的 time scale
time_film[:, i, :, 1] = shift_i  [1, 64]   第 i 块的 time shift
```

**支路 B：CondFiLM（条件 → 调制信号）**

```
cond_feat [1, 64, 256, 512]
    │
    ▼ 全局平均池化 (在 H,W 维度取 mean)
pooled [1, 64]   每个通道的平均激活值
    │
    ▼ MLP: Linear(64 → 256) → SiLU → Linear(256 → 1024)
原始输出 [1, 1024] = [1, 8 × 64 × 2]  → reshape → [1, 8, 64, 2]
    │
    ▼ 拆分为 8 个块的 scale/shift
cond_film[:, i, :, 0] = scale_i  [1, 64]
cond_film[:, i, :, 1] = shift_i  [1, 64]
```

**合并：** 每个块的最终 FiLM 参数 = time 支路 + cond 支路

```
scale_i = time_film[:, i, :, 0] + cond_film[:, i, :, 0]    [1, 64]
shift_i = time_film[:, i, :, 1] + cond_film[:, i, :, 1]    [1, 64]
```

当 t=0.0 时，time_film 接近零输出（sin(0)=0, cos(0)=1 编码后经 MLP→接近 0），所以初始时 cond_film 主导调制。

##### ④ 8×FiLMResBlock（逐块展开）

以第 i 个 FiLMResBlock 为例，输入 h_in = [1, 64, 256, 512]：

```
h_in [1, 64, 256, 512]
    │
    │ 第 1 层卷积
    │ Conv2d(64, 64, kernel=3, padding=1)
    │ 权重: [64, 64, 3, 3]  ← 64*64*3*3 = 36,864 个参数
    │ 偏置: [64]
    ▼
h [1, 64, 256, 512]
    │
    │ GroupNorm(num_groups=min(8,64)=8, num_channels=64)
    │ 可训练参数: γ[64] + β[64] = 128
    │ 分组: 64ch ÷ 8组 = 每组 8 通道
    │ 归一化: h[:, g*8:(g+1)*8, :, :] = (h - μ_g) / σ_g × γ + β
    │ 每组 μ_g ∈ ℝ, σ_g ∈ ℝ 在 [B, 8, H, W] 上统计
    ▼
h [1, 64, 256, 512]
    │
    │ SiLU (Swish): h = h × sigmoid(h)
    │ 逐点激活，维度不变
    ▼
h [1, 64, 256, 512]
    │
    │ ★ FiLM 调制（核心步骤）
    │ scale_i [1, 64] → unsqueeze → [1, 64, 1, 1]  (广播到全空间)
    │ shift_i [1, 64] → unsqueeze → [1, 64, 1, 1]
    │
    │ 公式: h = h × (1 + scale_i) + shift_i
    │
    │ 数值示例（单个空间位置 h[c] 的变换）:
    │   若 h[c]=0.5, scale_i[c]=0.2, shift_i[c]=-0.1
    │   则 h' = 0.5 × (1 + 0.2) + (-0.1) = 0.5
    │
    │ (1+scale) 形式确保初始化时 scale≈0 → 恒等映射
    ▼
h [1, 64, 256, 512]
    │
    │ 第 2 层卷积
    │ Conv2d(64, 64, kernel=3, padding=1)
    ▼
h [1, 64, 256, 512]
    │
    │ GroupNorm(8, 64)
    ▼
h [1, 64, 256, 512]
    │
    │ ★ 残差连接 + 最终激活
    │ h_out = SiLU(h + h_in)
    │
    │ h_in 保留了进入块之前的原始特征
    │ h 包含了本块学到的残差更新
    │ 残差连接让梯度直通，避免 8 块堆叠时的梯度消失
    ▼
h_out [1, 64, 256, 512]   ← 输入到下一个 FiLMResBlock
```

8 个块串行连接，每个块重复上述流程。block 0 输出 → block 1 输入 → ... → block 7 输出。所有块共用相同的 FiLMResBlock 结构，但拥有独立的可训练权重（每块独立 Conv2d + GroupNorm 参数）。

##### ⑤ 输出头（Output Head）

```
8×FiLMResBlock 的输出 [1, 64, 256, 512]
    │
    │ GroupNorm(8, 64) → SiLU（逐点激活）
    ▼
[1, 64, 256, 512]
    │
    │ Conv2d(64 → 9, kernel=3, padding=1)
    │ 权重: [9, 64, 3, 3]  ← 9*64*3*3 = 5,184 个参数
    ▼
v_pred [1, 9, 256, 512]    ← 预测的速度向量场
```

##### ⑥ 欧拉更新

```
x_{0.25} = x_0 + v_pred × 0.25
         = [1, 9, 256, 512] + [1, 9, 256, 512] × 0.25
         = [1, 9, 256, 512]
```

##### WaveletVFE 参数量汇总

| 组件 | 参数计算 | 参数量 |
|------|---------|--------|
| CondEncoder | (3×32×3×3 + 32) + (32×64×3×3 + 64) | 896 + 18,496 = **19,392** |
| in_proj | 76 × 64 × 3 × 3 + 64 | **43,776** |
| SinusoidalTimeEmbedding | (256×1024+1024) + (1024×2048+2048) | 263K + 2,099K = **2,362,880** |
| CondFiLM | (64×256+256) + (256×1024+1024) | 16,640 + 263,168 = **279,808** |
| 8×FiLMResBlock | 8 × [(64×64×3×3+64)×2 + 128×2] | 8 × (73,856 + 256) = **592,896** |
| Output Head | 64 × 9 × 3 × 3 + 9 | **5,193** |
| **合计** | | **~3.30M** |

注：实际配置为 base_ch=64, n_blocks=8 时约 1.9M（SinusoidalTimeEmbedding 使用更小的 hidden dim）。此处为完整展开计算。

#### 第 2 步（t=0.25 → t=0.50）

```
x_t = x_{0.25}  [1, 9, 256, 512]
t = [0.25]
模型输出 v_pred [1, 9, 256, 512]
x_{0.50} = x_{0.25} + v_pred × 0.25
         = [1, 9, 256, 512]
```

#### 第 3 步（t=0.50 → t=0.75）与第 4 步（t=0.75 → t=1.0）

同理，逐步更新状态 x_t。

#### 第 4 步完成后

```
H_pred = x_{1.0} = [1, 9, 256, 512]
```

### Step 5：IDWT 重建

```
H_pred [1, 9, 256, 512] ──chunk(3)──→ LH [1, 3, 256, 512]
                                        HL [1, 3, 256, 512]
                                        HH [1, 3, 256, 512]
                                              +
                                   LL_lr [1, 3, 256, 512] (来自 Step 2)
                                       │
                                       ▼
                                 Haar IDWT (pixel_shuffle ×2)
                                       │
                                       ▼
                                 SR [1, 3, 512, 1024]       ← HR 分辨率
```

IDWT 的过程是将 256×512 的 4 个子带，通过 Haar 逆矩阵重新组合为 512×1024 的全分辨率图像。

### 完整推理管线图

```
LR [1,3,128,256]
    │ bicubic↑
    ▼
lr_up [1,3,512,1024]
    │
    ├──→ DWT ──→ LL_lr [1,3,256,512] ──────────────────┐
    │            H_lr [1,9,256,512] ─┐                  │
    │                               │                  │
    ├──→ CondEncoder ──→ cond_feat  │                  │
    │                   [1,64,256,512]                 │
    │                               │                  │
    │    ┌──────────────────────────┘                  │
    │    ▼                                             │
    │   x₀ = H_lr [1,9,256,512]                       │
    │    │                                             │
    │    │  ┌── 4× Euler ──────────────────────┐       │
    │    ├──│ Step 1: x₀ + v₀·¼ → x₀.₂₅        │       │
    │    ├──│ Step 2: x₀.₂₅ + v₀.₂₅·¼ → x₀.₅₀  │       │
    │    ├──│ Step 3: x₀.₅₀ + v₀.₅₀·¼ → x₀.₇₅  │       │
    │    └──│ Step 4: x₀.₇₅ + v₀.₇₅·¼ → x₁.₀   │       │
    │       └────────────────────────────────────┘       │
    │                      │                             │
    │                   H_pred [1,9,256,512]             │
    │                      │                             │
    └──────────────────────┼─────────────────────────────┘
                           ▼
                     IDWT(H_pred, LL_lr)
                           │
                           ▼
                     SR [1,3,512,1024]
```

### 对比：DirectSR 推理管线（单次前向，无欧拉循环）

```
LR → bicubic↑ → lr_up [1,3,512,1024]
    │
    ├── DWT → LL_lr [1,3,256,512] ──────────────┐
    ├── CondEncoder → cond_feat [1,64,256,512]   │
    │    │                                       │
    │    ▼                                       │
    │   concat [1,67,256,512] → 8×FiLMResBlock   │
    │       → Head → H_pred [1,9,256,512]        │
    │    │                                       │
    └────┼───────────────────────────────────────┘
         ▼
    IDWT = SR [1,3,512,1024]
```

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



# 想法
1. 为何不做一步，从 t = 0.75 直接到 t = 1.0？
2. 为何需要 4 步欧拉积分，而不是 1 步？
3. 如果扩大规模，应该从哪里入手？ 按照文章来说，如何增加上下文空间？
4. 指标是什么，和谁对比？
5. 做成 Dit 架构会怎样？
6. MoE？
