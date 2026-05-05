# CLAUDE.md — 球谐域全景图超分辨率重建

## 项目目标

利用球谐变换（SHT）的频域分解能力做全景图超分辨率：

> **低带宽 SHT 系数 → ISHT 结构保真（条件） + 像素空间扩散模型（生成） → 高分辨率全景图**

关键认知：SHT/ISHT 只能做带限重建（保真），不能生成高频细节。**ISHT 作为条件注入，像素空间扩散做最终生成**，突破带宽限制。

## 当前阶段

Phase 2 — 从纯频域回归转向「ISHT 条件 + 像素扩散生成」混合架构。MLP 回归方案已验证不可行。

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

### 当前架构：ISHT 条件 + 像素扩散

```
                   条件路径（SHT 保真）
LR (128×256)
  │ bicubic ↑ to 512×1024
  │ SHT(L_cond=255) → ISHT(L_cond=255)
  ▼
cond_struct [B,3,512,1024]  ← 带限结构信号
  │
  ├─concat──────────────────┐
  │  cond [B,6,512,1024]    │
  │  (bicubic↑ + struct)    │
  │                         ▼
  │              noisy HR [B,3,512,1024]
  │                         │
  │              ┌──────────┴──────────┐
  │              │   Pixel U-Net (3级) │
  │              │   enc: 512→256→128  │
  │              │   bottleneck: 64    │
  │              │   dec: 64→128→256→512│
  │              │   + time embedding  │
  │              └──────────┬──────────┘
  │                         ▼
  │              ε_pred [B,3,512,1024]
  │                         │
  └──────── MSE ────────────┘  vs 真实噪声

推理：纯噪声 → DDIM 50步（条件注入）→ SR [B,3,512,1024]
```

**关键设计决策**：
- SHT/ISHT 只用做条件（保真），不做输出（生成）
- 扩散在像素空间运行，不受任何 L 带宽限制
- 条件 = bicubic↑ + ISHT(SHT(bicubic↑, L=255))，提供结构指引
- 像素 U-Net 直接生成 512×1024，可产生任意频率细节

## 文件结构

```
lab/
├── plan.md                    # 四阶段总体规划
├── phase1_verify.py           # SHT/ISHT 验证 + 球谐谱分析
├── phase1_output/             # 谱图输出
├── phase2_baseline.py         # MLP 残差基线（已弃用）
├── phase2_dualhead.py         # Attention + Dual Head（已弃用）
├── phase2_diffusion.py        # ★ 当前：像素扩散 + ISHT 条件
├── phase2_output/             # 模型 + 采样图输出
└── lau_dataset/
    └── sun_test/
        ├── HR/                # 100 张 2048×1024 全景图
        └── LR/
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
