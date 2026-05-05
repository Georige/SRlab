# CLAUDE.md — 球谐域全景图超分辨率重建

## 项目目标

利用球谐变换（SHT）的频域分解能力做全景图超分辨率：

> **低带宽 SHT 系数 → ISHT 结构保真（条件） + 像素空间扩散模型（生成） → 高分辨率全景图**

关键认知：SHT/ISHT 只能做带限重建（保真），不能生成高频细节。**ISHT 作为条件注入，像素空间扩散做最终生成**，突破带宽限制。

## 当前阶段

Phase 3 → Phase 4：过拟合测试发现**训练 loss 稳定下降但模型无法过拟合单张图像**，需要对各模块做诊断性实验，定位架构瓶颈。

## 核心认知

### SHT 带限重建的不可约误差（512×1024, X4）

| L | 系数维度 | 不可约 MSE | vs Bicubic(0.01265) |
|---|---------|-----------|---------------------|
| 64 | 24,576 | 0.01724 | +36.3% |
| 96 | 55,296 | 0.01347 | +6.5% |
| 112 | 75,264 | 0.01214 | -4.0%（首次超越） |
| 128 | 98,304 | 0.01106 | -12.6% |
| 160 | 153,600 | 0.00927 | -26.7% |
| 192 | 221,184 | 0.00790 | -37.6% |
| 224 | 301,056 | 0.00679 | -46.4% |
| 255 | 390,150 | 0.00585 | -53.8% |

- Driscoll-Healy: `L_max = (H-1)//2 = 255`，是 512×1024 的理论上限
- 自然图像不是带限的——即使 L=255，仍有 ~0.006 的截断误差（锐利边缘、纹理）
- **纯 ISHT 重建永远无法生成真正的高清纹理**

### MLP 回归为什么失败

1. **信息不对称**：coeffs_lo 只有 24,576 实值，coeffs_hi(L=255) 有 390,150。93.7% 需要"创造"
2. **MSE 的均值陷阱**：MSE 损失下最优解是条件均值——把所有可能的高频细节平均掉 → 模糊
3. **低频主导梯度**：系数空间 MSE 被大幅度低频系数支配，高频系数（对纹理贡献大）梯度微弱
4. **架构无用**：ResMLP、DualHead、Attention——全部卡在 ~0.022 MSE，无一超越 bicubic

### 当前架构：ISHT 条件 + 像素扩散（扩散目标是残差）

```
                        条件路径
LR [B,3,128×256]
  │ bicubic ↑ → [B,3,512×1024]
  │
  ├─ SHT(L=255) → ISHT(L=255) → base [B,3,512×1024]     ← 带限保真层
  │
  ├─ cond = concat(bicubic, base) → [B,6,512×1024]       ← 主条件
  │   (+3ch HF_residual 若开启 = [B,9,512×1024])
  │
  ├─ residual = HR - base                                  ← ★ 扩散目标！
  │
  ├─ t ~ Uniform(0,T), noise ~ N(0,I)
  │  noisy = sqrt(ᾱt) * residual + sqrt(1-ᾱt) * noise
  │
  └─ U-Net(noisy, cond, t/T, ms_isht) → ε_pred
       loss = MSE(ε_pred, noise)   [+ latitude_weight 可选]
```

**训练**：DDPM 在残差上做噪声预测，残差 = HR 减 ISHT(L=255) 重建。
**推理**：DDIM 50 步从纯噪声生成残差 → SR = base + denoised_residual.

**关键设计决策**：
- 扩散学习的是 **HF 残差**（HR - base），不是完整图像
- ISHT(L=255) 的不可约 MSE ~0.006，残差的方差远小于完整图像
- 条件 = bicubic↑ + ISHT(SHT(bicubic↑, L=255))，提供结构指引
- 像素 U-Net 直接生成 512×1024，可产生任意频率细节

### U-Net 内部结构

```
输入: concat(noisy_residual [B,3], cond [B,6]) = [B,9, H, W]
      可选: + CoordEmbed [B,24] = [B,33, H, W]  (num_freqs=4, 6ch×4freq)

TimeEmbedding: t_norm [B] → sinusoidal(freqs=128) → MLP(256→1024→256)

Encoder:
  enc1:    9→32ch,  pool(↓2) → skip1[32, H, W],     x[32, H/2, W/2]
             ms_isht(enc2): L=128 ISHT → Conv1×1(3→32) 注入到 x
             PolarMoE(32) 可选: 两极/赤道双expert, y-坐标门控

  enc2:   32→64ch,  pool(↓2) → skip2[64, H/2, W/2],  x[64, H/4, W/4]
             ms_isht(enc4): L=64 ISHT → Conv1×1(3→64) 注入到 x
             PolarMoE(64) 可选

  enc3:   64→128ch, pool(↓2) → skip3[128, H/4, W/4], x[128, H/8, W/8]
             ms_isht(enc8): L=32 ISHT → Conv1×1(3→128) 注入到 x

Bottleneck: 128→128ch [H/8, W/8]
             SphericalAttention 可选: axial(W轴+H轴) self-attention

Decoder:
  dec3: x[128] + skip3[128] → upsample(↑2) → 64ch [H/4, W/4]
           head_L2(64→3) 可选 → 辅助预测 [H/4, W/4]

  dec2: x[64] + skip2[64]   → upsample(↑2) → 32ch [H/2, W/2]
           head_L1(32→3) 可选 → 辅助预测 [H/2, W/2]

  dec1: x[32] + skip1[32]   → upsample(↑2) → 32ch [H, W]

输出: GroupNorm + SiLU + Conv(32→3) → ε_pred [B,3, H, W]
```

每级 ResBlock 内部：Conv(3×3) → GroupNorm → SiLU → +time_proj → Conv(3×3) → GroupNorm → +skip → SiLU

### 多尺度 ISHT 注入点

| 注入位置 | L | 分辨率 | 投影 |
|----------|---|--------|------|
| enc2 之后 | 128 | 256×512 | Conv1×1(3→32) |
| enc3 之后 | 64  | 128×256 | Conv1×1(3→64) |
| bottleneck 之前 | 32  | 64×128  | Conv1×1(3→128) |

注入方式：**加法**（x = x + ms_proj[level](isht_img)），不是 concat。

### 配置层级与各实验开关

基座 `diffusion_config.py` 全部开关默认 False，各实验只覆写需要的：

| 配置模块 | 开启的开关 |
|----------|-----------|
| `diffusion_config` (基座) | 全关 |
| `exp_base` | 继承全部默认（=基座） |
| `exp_hf_residual` | USE_HF_RESIDUAL |
| `exp_polar` | USE_LATITUDE_WEIGHT + USE_POLAR_MOE |
| `exp_full` | USE_HF_RESIDUAL + USE_LATITUDE_WEIGHT + USE_POLAR_MOE |
| `exp_laplacian` | USE_LAPLACIAN_PYRAMID |
| `exp_spherical` | USE_CIRCULAR_CONV + USE_COORD_EMBED + USE_SPHERICAL_ATTN |

### 各创新模块

| 模块 | 文件 | 功能 |
|------|------|------|
| `TimeEmbedding` | blocks.py:16-30 | sinusoidal(freqs=128) → MLP(256→1024→256) |
| `ResBlock` | blocks.py:33-50 | Conv+GN+SiLU, time注入(Linear→add), skip |
| `DownBlock` | blocks.py:53-63 | ResBlock + stride-2 Conv |
| `UpBlock` | blocks.py:66-81 | Upsample bilinear + skip concat + ResBlock |
| `PolarMoE` | blocks.py:84-112 | 两极/赤道双expert, gate=1-sin(π·y/H) |
| `CircularConv2d` | spherical.py:14-37 | 宽度方向 circular padding |
| `CoordEmbed` | spherical.py:40-80 | θ,φ→3D笛卡尔→傅里叶特征(6ch×4freq=24ch), 静态buffer |
| `SphericalAttention` | spherical.py:83-154 | 轴向自注意力(W轴+H轴), bottleneck处 |
| `PixelUNet` | unet.py:9-143 | 3级encoder-decoder, ms_isht注入, 可选创新 |
| `PixelDiffusion` | diffusion.py:10-178 | DDPM训练(残差目标) + DDIM推理(50步) |

## Phase 4 诊断结果（已完成 2026-05-05）

### 实验总览

| 实验 | train loss | v_img MSE | 结论 |
|------|-----------|-----------|------|
| **Exp 0** residual stats | — | — | residual var=**0.0042**, SNR<-20dB @ t>249 |
| **Exp 1** bicubic-only cond | 0.03 (震荡) | 1.1~3.8 | cond 去掉 base 后 DDIM 完全失败 |
| **Exp 2** 去扩散+直接预测 | **0.00071** | **0.00067** | U-Net 完美过拟合 ✅ |
| **Exp 3** x0预测+DDIM | 0.004 (停滞) | 0.93~1.7 | 采样全噪声 |
| **Exp 4** x0+一致性正则 | 0.008 | 0.91~1.8 | 无任何改善 |

### 核心结论

1. **U-Net 容量足够** — Exp 2 以 base_ch=64 达到 MSE=0.00067，超越 bicubic(0.01265) 18.8x
2. **扩散 = 唯一瓶颈** — 任何扩散变体（噪声预测/x0预测/一致性）DDIM 采样全失败
3. **根因：信号尺度不匹配** — residual var=0.004 vs 噪声 var=1.0，信号被淹 250:1
4. **x0 预测不比噪声预测好** — 高噪声下预测干净信号比检测噪声更难

### 下一步修复方向

- **方案 A（最简单）**：扩散目标改为完整 HR 图像而非 residual，方差接近 1
- **方案 B**：把 residual 缩放到单位方差再扩散，采样后反缩放
- **方案 C**：用更窄的噪声范围（降低 beta_end），提高低 t 时的 SNR

### 运行命令

```bash
python phase4_diagnosis.py -e 0 -g 7    # residual 统计
python phase4_diagnosis.py -e 1 -g 7    # bicubic-only 条件
python phase4_diagnosis.py -e 2 -g 7    # 去扩散直接预测
python phase4_diagnosis.py -e 3 -g 7    # x0 预测 DDIM
python phase4_diagnosis.py -e 4 -g 7    # 一致性正则
```

## Phase 5 Stage 1：强化基座（已完成 2026-05-05）

### 固定底床

| 组件 | 值 | 理由 |
|------|-----|------|
| Noise schedule | **Cosine** (Nichol & Dhariwal 2021, s=0.008) | SNR 比线性高 3-5x，高 t 不再淹没信号 |
| 预测目标 | **x0** (MSE(x0_pred, HR)) | 直接优化重建质量，非噪声检测 |
| 采样软截断 | **tanh** (每步 DDIM: x0_pred = tanh(x0_pred)) | 约束 [-1,1] 防发散 |
| 扩散目标 | **完整 HR** (非 residual) | HR var=0.27 vs residual var=0.004，65x 差异 |
| DDIM 采样 | **修复版** (linspace T-1→0) | Phase 4 Exp 7 god-model 验证 |

### 实验结果

| 实验 | T | Epochs | Best MSE | NCC | Edge-NCC | vs Bicubic | vs Phase3 |
|------|---|--------|----------|-----|----------|------------|-----------|
| **1.1** | 1000 | 2000 | **0.01006** | 0.9935 | 0.8675 | 1.26x 好 | 16.5x 提升 |
| 1.2 | 500 | 500 | 0.01189 | 0.9897 | 0.7204 | 1.06x 好 | 14.0x 提升 |
| Phase 3 基线 | 1000 | 1000 | 0.166 | — | — | 13x 差 | — |

### 核心结论

1. **扩散流程修复成功** — HR-target + cosine + x0_pred + tanh 四合一，MSE 从 0.166 → 0.01006
2. **首次超越 bicubic** — 扩散采样生成的图像比 bicubic 上采样更接近真实 HR
3. **T=1000 优于 T=500** — Exp 1.1 在 2000 epochs 仍持续收敛（0.01006 < 0.01189），选为强化基座
4. **Edge-NCC 持续提升** — 0.0990 → 0.8675，说明 tanh 软截断有效帮助边缘重建
5. **Batch=1 过拟合成功** — 验证 U-Net(base_ch=32) 容量足以学习扩散去噪

### 强化基座配置（后续 Stage 2+ 继承）

```python
T = 1000
schedule = cosine(s=0.008)
target = x0 (full HR, not residual)
loss = MSE(x0_pred, HR)
sampler = DDIM(50 steps, T-1→0, tanh soft clip)
base_ch = 32
cond_ch = 6 (bicubic + ISHT(L=255))
all innovations = OFF
```

### 运行命令

```bash
python phase5_stage1.py -e 1.1 -g 7    # Exp 1.1: T=1000, 2000epochs
python phase5_stage1.py -e 1.2 -g 6    # Exp 1.2: T=500, 500epochs
```

## 文件结构

```
lab/
├── plan.md                    # 四阶段总体规划
├── plan_laplacian_pyramid.md  # 拉普拉斯金字塔方案
├── plan_spherical_field.md    # 球面隐式神经场方案
├── new.md                     # 20 个创新方向总览
│
├── phase1_verify.py           # SHT/ISHT 验证 + 球谐谱分析
├── phase1_output/             # 谱图输出
│
├── phase2_baseline.py         # MLP 残差基线（已弃用）
├── phase2_dualhead.py         # Attention + Dual Head（已弃用）
├── phase2_diffusion.py        # ★ 全量训练：像素扩散 + ISHT 条件
├── phase2_output/             # 全量训练采样图
│
├── phase3_overfit.py          # ★ 过拟合测试：单图→验证架构容量
├── phase3_output/             # 过拟合采样图 + 进度拼接
│
├── model/
│   ├── blocks.py              # ResBlock, DownBlock, UpBlock, PolarMoE
│   ├── diffusion.py           # PixelDiffusion: DDPM/DDIM + 多尺度条件
│   ├── spherical.py           # CircularConv2d, CoordEmbed, SphericalAttention
│   └── unet.py                # PixelUNet: 3级编码解码 + 多尺度 ISHT 注入
│
├── config/
│   ├── diffusion_config.py    # 基座配置（所有开关默认 False）
│   ├── exp_base.py            # 基线实验
│   ├── exp_hf_residual.py     # 高频残差条件
│   ├── exp_polar.py           # 极区感知训练
│   ├── exp_full.py            # 全部创新组合
│   ├── exp_laplacian.py       # 拉普拉斯金字塔多尺度监督
│   └── exp_spherical.py       # 球面感知 U-Net
│
├── vit/                       # 可视化模块
│   ├── loss_plotter.py        # Phase2: loss 曲线（每 epoch 自动更新）
│   ├── monitor.py             # Phase2: TrainingMonitor + 性能仪表盘
│   ├── overfit_plot.py        # Phase3: loss 曲线 + 图像进度拼接
│   └── plot_logs.py           # 从旧 .log 文件手动生成 loss 图
│
├── utility/
│   ├── data.py                # PanoramaDataset
│   └── schedule.py            # 噪声 schedule (linear beta)
│
├── logs/                      # 训练日志 + 自动生成的可视化图
├── weight/                    # 模型权重 + losses.csv + metrics.csv
│
└── lau_dataset/
    └── sun_test/
        ├── HR/                # 100 张 2048×1024 全景图
        └── LR/
            ├── X2/            # 1024×512
            ├── X4/            # 512×256
            ├── X8/            # 256×128
            └── X16/           # 128×64
            ├── X2/            # 1024×512
            ├── X4/            # 512×256
            ├── X8/            # 256×128
            └── X16/           # 128×64
```

## Phase 2 基准指标

- **Bicubic vs HR MSE**: 0.01265（100 张 sun_test，512×1024，X4）
- **L=255 不可约 MSE**: 0.00585（ISHT 重建的理论下限）
- 模型目标：图像 MSE < 0.01265（超越 bicubic），理想情况逼近 0.00585

## 环境

- Conda: `spectral-sr`, Python 3.10
- PyTorch 2.6.0+cu124（旧 ABI，非 cxx11-abi）
- torch-harmonics: 从 GitHub 源码编译（`--no-build-isolation`），CUDA 12.4
- 编译器: GCC 12 (conda-forge)，nvcc 12.4 (nvidia channel)

**编译 torch-harmonics 的正确命令：**
```bash
conda activate spectral-sr
CUDA_HOME=$CONDA_PREFIX \
CC=$CONDA_PREFIX/bin/gcc \
CXX=$CONDA_PREFIX/bin/g++ \
pip install git+https://github.com/NVIDIA/torch-harmonics.git --no-build-isolation
```

**仅用 conda 装 CUDA 12.4 工具链（不覆盖系统 CUDA）：**
```bash
conda install -c nvidia/label/cuda-12.4.0 cuda-nvcc cuda-cudart-dev -y
conda install -c nvidia/label/cuda-12.4.0 cuda=12.4.0 -y
conda install -c conda-forge gcc=12 gxx=12 -y
```

## 已记录方案（待探索）

### 方案 C：频域可学习变换（SHT → Conv1d → ISHT）

在 SHT 系数上做小型可学习网络，再 ISHT 回去，本质是一个内容自适应的频域 Wiener 滤波器。

```
bicubic↑ → SHT(L) → 小型 Conv1d/MLP（沿 l 维度,相邻 l 描述相近频率）→ ISHT(L) → 增强 base
```

**能学什么**：
- 补偿 bicubic 对高 l 系数的衰减，锐化边缘
- 抑制 bicubic 振铃/混叠 artifact
- 从低 l 强度推断高 l 应有强度（天然图像功率谱统计规律）

**局限**：仍在 L=255 带限内，无法超越 ISHT 的频率上限。收益是逼近 ISHT(HR) 的 L=255 重建质量（MSE ~0.006）。

**定位**：改善 base 质量的辅助方案，不替代扩散生成高频细节。适合在扩散 baseline 跑通后加入。

## 关键注意事项

- SHT 返回**复数系数**，`.view()` 会因非连续内存报错，必须用 `.reshape()`
- `torch-harmonics` 预编译 wheel（0.9.0）用新 ABI 编译，与 PyTorch pip wheel（旧 ABI）不兼容，必须源码编译
- 系统 nvcc 是 CUDA 11.7，PyTorch 用 12.4 编译——需通过 conda 安装 CUDA 12.4 编译器并设 `CUDA_HOME=$CONDA_PREFIX`
