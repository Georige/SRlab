# CLAUDE.md — 球谐域全景图超分辨率重建

## 项目目标

利用球谐变换（SHT）的频域分解能力做全景图超分辨率：

> **低带宽 SHT 系数 → ISHT 结构保真（条件） + 直接回归生成 → 高分辨率全景图**

关键认知：SHT/ISHT 只能做带限重建（保真），不能生成高频细节。**ISHT 作为条件注入**，突破带宽限制。

## 当前阶段

**Phase 6 完成**（2026-05-06）：直接回归路线跑通，DirectUNet + ISHT 条件 + 残差预测 + 全景数据增强 → MSE=0.01076，PSNR=25.73dB，首次稳定超越 bicubic（25.02dB）。扩散路线（Phase 3-5）因信号尺度不匹配被搁置。

所有 panoramic 模块（PolarMoE, CoordEmbed, CircularConv, SphericalAttention）在多图训练上未超越纯基线。下一步：扩大训练集 / GAN+perceptual loss。

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

## 实验工厂基础设施

### 架构总览

```
python run.py --cfg factory/configs/<experiment>.yaml -g <GPU>
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
factory/config.py  factory/registry.py  factory/trainer.py
 YAML → DotDict    MODEL_REGISTRY{}     DirectTrainer
                    TRAINER_REGISTRY{}   .run() 训练循环
     │              │              │
     ▼              ▼              ▼
factory/configs/   model/          utility/
 7个 YAML 配置      direct_unet.py  data.py (PanoramaDataset)
                   blocks.py       schedule.py
                   spherical.py   vit/overfit_plot.py
                   losses.py       (loss曲线+进度图)
                   diffusion.py
                   unet.py
```

### 各基础设施目录职责

| 目录 | 角色 | 关键文件 |
|------|------|---------|
| `factory/` | 实验编排层：配置加载、注册中心、训练循环 | `config.py`, `registry.py`, `trainer.py`, `augment.py` |
| `model/` | 模型定义层：网络架构、模块、损失函数 | `direct_unet.py`, `blocks.py`, `spherical.py`, `losses.py`, `unet.py`, `diffusion.py` |
| `utility/` | 数据与工具层：数据集加载、噪声 schedule | `data.py` (PanoramaDataset), `schedule.py` |
| `vit/` | 可视化层：loss 曲线、采样图拼接 | `overfit_plot.py` (update_curves, make_progression) |

### factory/ 详细说明

#### run.py — 唯一入口

```bash
python run.py --list                                    # 列出所有可用配置和注册模块
python run.py --cfg factory/configs/phase6_stage1.yaml -g 7      # 从头训练
python run.py --cfg factory/configs/phase6_stage1.yaml -g 7 --epochs 1000  # 覆写 epochs
python run.py --cfg factory/configs/phase6_stage1.yaml -g 7 --resume --epochs 600  # 从 best_model.pt 恢复
```

CLI 参数：`--cfg`, `--gpu/-g`, `--epochs`, `--resume`, `--start-epoch`, `--output-dir`, `--list`

#### factory/config.py — 配置加载

- `load_config(path)` → `DotDict`：YAML → 属性可访问对象（`cfg.model.base_ch` 而非 `cfg['model']['base_ch']`）
- `REPO_ROOT`：自动解析输出路径到仓库根目录
- DotDict 递归处理嵌套 dict 和 list

#### factory/registry.py — 注册中心

```python
MODEL_REGISTRY = {}     # 'direct_unet' → build_direct_unet(cfg, device)
TRAINER_REGISTRY = {}   # 'direct' → DirectTrainer

@register_model('name')    # 装饰器注册模型 builder
@register_trainer('name')  # 装饰器注册 trainer 类
```

已注册模型：`direct_unet`（DirectUNet，支持所有 panoramic 开关）
已注册 trainer：`direct`（DirectTrainer，残差回归训练循环）

#### factory/trainer.py — 训练循环

`DirectTrainer(cfg, device, output_dir)` 封装：
- `DataBuilder`：内建 SHT/ISHT 变换，构建 train/val 数据集（cond, ms_isht, base, hr）
- `_load_data()` → `_build_model()` → `run()` 训练循环
- 自动：数据增强、残差预测 + L2 正则、PolarMoE balance loss、验证+指标、best model 保存、loss 曲线更新
- 支持 `resume_ckpt` + `start_epoch` 断点续训

#### factory/augment.py — 全景数据增强

`augment_panorama(cond, hr, base, ms_isht, training, **kwargs)`：
1. 水平循环滚动（50% 概率）——360° 全景天然支持
2. 垂直裁剪+缩放（90% 概率，0-10% 裁剪）
3. 亮度/对比度抖动（70% 概率，±10%）
4. HR 目标加性高斯噪声（30% 概率，std=0.005）

#### factory/configs/ — 实验配置（7 个 YAML）

每个 YAML 定义 5 个 section：
```yaml
experiment: {name, phase, stage, desc}
model:      {type, cond_ch, base_ch, latent_ch, use_*, ms_injection, out_ch}
data:       {hr_size, scale, data_dir, n_train, n_val, l_cond, ms_cond}
training:   {lr, epochs, residual_l2_weight, balance_weight?, augmentation}
output:     {dir, log_dir}
```

| 配置 | n_train | latent_ch | 特殊开关 | 用途 |
|------|---------|-----------|---------|------|
| `phase6_stage0.yaml` | 1 | 3 | 无增强 | 单图 sanity |
| `phase6_stage1.yaml` | 16 | 0 | 全增强 | 多图基线 |
| `phase6_stage2_ms_isht.yaml` | 16 | 0 | ms_injection=add | MS ISHT 对照 |
| `phase6_stage2_polar_moe.yaml` | 16 | 0 | PolarMoE, bal=0.01 | 极区专家 |
| `phase6_stage2_coord_embed.yaml` | 16 | 0 | CoordEmbed | 球面位置编码 |
| `phase6_stage2_circular_conv.yaml` | 16 | 0 | CircularConv | 循环卷积 |
| `phase6_stage2_spherical_attn.yaml` | 16 | 0 | SphericalAttn | 球面注意力 |

### model/ 详细说明

| 文件 | 类/函数 | 说明 |
|------|---------|------|
| `direct_unet.py` | `DirectUNet` | 无时间嵌入的 U-Net，cond → 残差，支持全部 panoramic 开关 |
| `blocks.py` | `ResBlock`, `DownBlock`, `UpBlock`, `PolarMoE` | U-Net 基础构件，PolarMoE 有 load_balance_loss |
| `spherical.py` | `CircularConv2d`, `CoordEmbed`, `SphericalAttention` | 球面感知模块 |
| `losses.py` | `VGGLoss`, `PatchGANDiscriminator`, `TVLoss`, `CompositeLoss` | GAN + perceptual loss（待用） |
| `unet.py` | `PixelUNet` | 扩散用 U-Net（含 TimeEmbedding，Phase 3-5 遗留） |
| `diffusion.py` | `PixelDiffusion` | DDPM/DDIM 扩散流程（Phase 3-5 遗留） |

### utility/ 详细说明

| 文件 | 说明 |
|------|------|
| `data.py` | `PanoramaDataset(hr_dir, hr_size, scale)` — 从 HR/ LR/X2/X4/X8/X16 加载全景图，归一化到 [-1,1] |
| `schedule.py` | 噪声 schedule（linear beta），Phase 3-5 遗留 |

### vit/ 详细说明

| 文件 | 函数 | 说明 |
|------|------|------|
| `overfit_plot.py` | `update_curves(exp_name, losses, val_epochs, val_mses, log_dir)` | 双面板：train loss (log) + val MSE |
| | `make_progression(output_dir, max_cols=10)` | 将所有 `e*.png` 拼接为 progression grid |

### 添加新实验的标准流程

```python
# 1. （如有新模型）factory/registry.py
@register_model('my_arch')
def build_my_model(cfg, device):
    from model.xxx import MyModel
    return MyModel(...).to(device)

# 2. factory/configs/my_exp.yaml
# model.type: my_arch
# training.xxx: ...   ← 新参数自动进入 cfg.training.xxx

# 3. 运行
python run.py --cfg factory/configs/my_exp.yaml -g 7
```

## 文件结构

```
lab/
├── run.py                        # ★ 唯一入口，所有实验统一执行
│
├── factory/                      # 实验编排基础设施
│   ├── __init__.py               # 导出
│   ├── config.py                 # YAML → DotDict 配置加载
│   ├── registry.py               # MODEL_REGISTRY + TRAINER_REGISTRY
│   ├── trainer.py                # DirectTrainer: 数据+模型+训练+验证+保存
│   ├── augment.py                # 全景图专属数据增强
│   └── configs/                  # 7 个 YAML 实验配置
│       ├── phase6_stage0.yaml    # 单图 sanity
│       ├── phase6_stage1.yaml    # 多图基线
│       ├── phase6_stage2_ms_isht.yaml
│       ├── phase6_stage2_polar_moe.yaml
│       ├── phase6_stage2_coord_embed.yaml
│       ├── phase6_stage2_circular_conv.yaml
│       └── phase6_stage2_spherical_attn.yaml
│
├── model/                        # 模型定义基础设施
│   ├── direct_unet.py            # ★ DirectUNet: 无时间嵌入, cond→残差
│   ├── blocks.py                 # ResBlock/DownBlock/UpBlock/PolarMoE
│   ├── spherical.py              # CircularConv2d/CoordEmbed/SphericalAttention
│   ├── losses.py                 # VGGLoss/PatchGAN/TVLoss/CompositeLoss
│   ├── unet.py                   # PixelUNet（扩散用，Phase 3-5 遗留）
│   └── diffusion.py              # PixelDiffusion（扩散用，Phase 3-5 遗留）
│
├── utility/                      # 数据与工具基础设施
│   ├── data.py                   # PanoramaDataset
│   └── schedule.py               # 噪声 schedule（扩散用，遗留）
│
├── vit/                          # 可视化基础设施
│   └── overfit_plot.py           # update_curves + make_progression
│
├── phase*_*.py                   # 历史实验脚本（Phase 1-6，保留参考）
├── phase*_output/                # 各阶段实验输出
├── logs/                         # 旧日志
├── weight/                       # 旧权重
│
└── lau_dataset/
    └── sun_test/
        ├── HR/                   # 100 张 2048×1024 全景图
        └── LR/
            ├── X2/               # 1024×512
            ├── X4/               # 512×256
            ├── X8/               # 256×128
            └── X16/              # 128×64
```

## Phase 6 所有实验结果（最终汇总）

基线：Stage 1 v2，DirectUNet(base_ch=64) + ISHT 条件 + 残差预测 + 全景增强

| 实验 | 模块 | Epochs | Best MSE | PSNR | ΔPSNR | 判定 |
|------|------|--------|----------|------|-------|------|
| S1 v2 | Baseline | 1000 | 0.01076 | 25.73 | — | 基座 |
| S2.1 | +MS ISHT | 400 | 0.01198 | 25.27 | -0.5 | 复现基线 |
| S2.2 | +PolarMoE | 1000 | 0.01071 | 25.74 | +0.0 | 持平 |
| S2.3 | +CoordEmbed | 400 | 0.01352 | 24.73 | -1.0 | **丢弃** |
| S2.4 | +CircularConv | 1000 | 0.01083 | 25.70 | -0.0 | 持平 |
| S2.5 | +SphericalAttn | 400 | 0.01244 | 25.08 | -0.6 | **丢弃** |

Bicubic 基线：MSE=0.01267, PSNR=25.02dB, SSIM=0.5838

**结论**：所有 panoramic 模块在 16 图训练集上均未超越纯基线。下一步：扩大训练集 / GAN+perceptual loss。

## 关键成功配方（v2 Recipe）

以下配置是当前唯一稳定超越 bicubic 的方案：
- **latent_ch=0**：不用 learnable latent，模型纯从条件推理
- **残差预测**：SR = base + model(cond)，L2 正则 1e-4
- **全景增强**：水平循环滚动 + 垂直裁剪 + 色彩抖动 + HR 噪声
- **LR=5e-5**：cosine annealing，400+ epochs
- **base_ch=64**：DirectUNet 足以学习 16 图分布

## Phase 3-5 历史总结（扩散路线，已搁置）

- **Phase 3**：DDPM + ISHT 条件过拟合单图，train loss 下降但 DDIM 采样失败
- **Phase 4**：诊断定位根因——U-Net 容量足够（直接回归 MSE=0.00067），DDIM 是唯一瓶颈
- **Phase 5**：尝试修复扩散（cosine schedule, x0 预测, tanh 截断），单图勉强到 MSE=0.010，多图未跑
- **根本问题**：残差 var=0.004 vs 噪声 var=1.0，信号被淹没 250:1。改为完整 HR 目标后略改善，但仍不如直接回归稳定

## 关键注意事项

- SHT 返回**复数系数**，`.view()` 会因非连续内存报错，必须用 `.reshape()`
- `torch-harmonics` 预编译 wheel（0.9.0）用新 ABI 编译，与 PyTorch pip wheel（旧 ABI）不兼容，必须源码编译
- 系统 nvcc 是 CUDA 11.7，PyTorch 用 12.4 编译——需通过 conda 安装 CUDA 12.4 编译器并设 `CUDA_HOME=$CONDA_PREFIX`
- **GPU 0 留给用户自己的任务，所有实验从 GPU 7 往下分配**

## 环境

- Conda: `spectral-sr`, Python 3.10
- PyTorch 2.6.0+cu124（旧 ABI，非 cxx11-abi）
- torch-harmonics: 从 GitHub 源码编译，CUDA 12.4
- 编译器: GCC 12 (conda-forge)，nvcc 12.4 (nvidia channel)
- 新增依赖: `pyyaml`（factory 配置加载）

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

## 待探索方案

### D. 频域可学习变换（SHT → Conv1d → ISHT）

在 SHT 系数上做小型可学习网络，再 ISHT 回去，本质是一个内容自适应的频域 Wiener 滤波器。

```
bicubic↑ → SHT(L) → 小型 Conv1d/MLP（沿 l 维度）→ ISHT(L) → 增强 base
```

- 补偿 bicubic 对高 l 系数的衰减，锐化边缘
- 抑制 bicubic 振铃/混叠 artifact
- 局限：仍在 L=255 带限内，收益是逼近 ISHT(HR) 的 L=255 重建质量（MSE ~0.006）

### E. GAN + Perceptual Loss

直接回归的 MSE 优化导致纹理模糊。加入 VGGLoss + PatchGAN + TVLoss（`model/losses.py` 已实现），可能提升感知质量。风险是 GAN 训练不稳定。

### F. 扩大训练集

当前仅用 16 张训练图，所有 panoramic 模块可能因数据不足无法体现优势。100 张 sun_test 全量训练可能改变 Stage 2 的结论。

## Phase 7 计划：中央渐进式超分（Central Progressive SR）

### 核心思路

放弃 SHT 频域路线，在像素空间用**双锚点 + Teacher Forcing + 中心损失**做从中心向外生长的渐进式超分辨率。

- **输入**：512×1024 低清图（X2 bicubic 下采样）
- **目标输出**：1024×2048 高清图（H×W）
- **宽高比**：2:1（全景图原生比例）
- **训练方式**：对一张或多张固定图像过拟合，验证方案可行性
- **工作空间**：像素空间（纯卷积 U-Net，不引入 VAE）
- **尺度序列**（中央矩形 W×H）：`(6,3) → (12,6) → ... → (1536,768) → (2048,1024)`
- **显存**：base_ch=32 时峰值 6.06 GB（9 个 scale step 完整 epoch）

### 双锚点条件

1. **局部低清块（结构保真）**：从 `I_low_up`（低清图 bicubic↑ 到 1024×2048）中截取中央 w×h 区域，嵌入全零画布
2. **已知高清块（纹理先验）**：上一步的中央高清区域 + 二值 mask，训练时用真值（Teacher Forcing），推理时用模型预测

### 模型架构

`CenterGrowingUNet`（`model/center_growing.py`）：
- 输入：[B, 10, H, W] = concat(全局低清, 局部低清块, 高清已知, mask)
- 输出：[B, 3, H, W] = 残差（SR = I_low_up + residual）
- 4 级 U-Net：H×W → H/2×W/2 → H/4×W/4 → H/8×W/8 → H/16×W/16 bottleneck
- base_ch=32：5.1M 参数，峰值 6.06 GB
- base_ch=64：20.4M 参数，峰值 11.7 GB

### 训练数据流（Teacher Forcing）

```
scales = [(6,3), (12,6), ..., (1536,768), (2048,1024)]

FOR each scale step (w,h) → (next_w, next_h):
    1. local_low = 从 I_low_up 截取中央 w×h → 嵌入全零画布
    2. IF known_high_res is None:
           high_res_known = zeros, mask = zeros
       ELSE:
           high_res_known = 将 known_high_res 置入中央, mask = 对应区域=1
    3. model_input = concat(I_low_up, local_low, high_res_known, mask)  # [B,10,H,W]
    4. residual = model(model_input); SR = I_low_up + residual
    5. loss = MSE(SR 中央 next_w×next_h, I_gt 中央 next_w×next_h)
    6. 反向传播、优化器更新
    7. known_high_res = I_gt 中央 next_w×next_h (Teacher Forcing)
```

### 推理时唯一区别

`known_high_res = pred_crop.detach()` （用模型预测替代真值），`local_low` 始终使用真值低清块。

### 关键设计要点

1. **双锚点**：局部低清块（结构保真）+ 已知高清块（纹理先验）
2. **Teacher Forcing**：训练时用真值高清块，避免误差累积
3. **中心损失**：损失仅作用在 next_w×next_h 区域，强制模型专注"向外生长新的一圈"
4. **尺度递进**：每步宽高同时 ×2，维持 2:1 宽高比
5. **输入通道 = 10**：全局低清(3) + 局部低清块(3) + 高清已知(3) + mask(1)
6. **残差预测**：`SR = I_low_up + residual`，更稳定

### 运行命令

```bash
python run.py --cfg factory/configs/phase7_stage0.yaml -g 7
```

## Phase 7 实验进展

### Stage 0: CenterGrowingUNet 单图过拟合

| 指标 | 值 |
|------|-----|
| 模型 | CenterGrowingUNet, base_ch=32, 5.1M params |
| 数据集 | odisr/training, image index 0, 512x1024 -> 1024x2048 |
| 训练步数 | 200 epochs, 9 scale steps per epoch |
| LR | 1e-4, cosine annealing |
| 输出 | `phase7_output/phase7_stage0_cg_pixel/` |

#### 训练过程

| Epoch | Train Loss | Val MSE | Val PSNR |
|-------|-----------|---------|----------|
| 1 | ~0.015 | 0.0140 | 18.53 dB |
| 100 | ~0.005 | 0.0076 | 21.17 dB |
| 200 | ~0.005 | 0.0076 | 21.18 dB |

Train loss 持续下降，但 autoregressive (AR) 验证 PSNR 在 ~21 dB 左右完全停滞。

#### 关键问题

- **Train loss 下降 vs AR PSNR 停滞**：这是典型的 Teacher Forcing → AR gap。训练时模型看到的是真值高清块（完美条件），推理时看到的是自己的（带误差的）预测，微小误差在 9 步中累积放大。
- **MSE plateau at 0.0076, PSNR 21 dB**：与 bicubic 上采样的 512x1024 相比有提升，但远低于理想结果。
- **可能的根因**：模型容量不足（base_ch=32 仅 5.1M），或者 AR 误差累积是主因（TF 上限可能远高于 21 dB）。

#### 下一步方向

需要进行 TF 上限诊断来区分两种可能性：
- **如果 TF 上限很高（PSNR >> 21 dB）**：模型有拟合能力，问题在 AR 误差累积 → 添加 teacher_noise_std 桥接 TF→AR
- **如果 TF 上限也低（PSNR ~ 21 dB）**：模型容量不足 → base_ch=64, 更高 LR

### Stage 1: Teacher Forcing 上限诊断

配置：`factory/configs/phase7_stage1_tf_bound.yaml`

#### 设计

加载 Stage 0 best model（epoch 200），在验证时使用 GT crops 作为 `known_high_res`（即每个 scale step 的输入条件是完美的高清块），测量模型能达到的 PSNR 上限。

```
AR 模式：known_high_res = model_prediction  → 误差累积 × 9 步
TF 模式：known_high_res = GT_crop           → 完美条件 × 9 步
```

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `training.val_tf` | `true` | 验证时使用 Teacher Forcing |
| `training.resume_ckpt` | `phase7_output/phase7_stage0_cg_pixel/best_model.pt` | 加载已训练 200 轮的模型 |
| `training.epochs` | `1` | 仅做 1 轮训练+验证（诊断用，非实际训练） |
| `training.start_epoch` | `200` | 继续从 epoch 200 编号 |

#### 代码改动

`factory/trainer_cg.py`：
- `_validate(use_tf=False)` 和 `_save_progression(epoch, use_tf=False)`：新增 `use_tf` 参数，当 `True` 时用 `crop_center_rect(I_gt, next_w, next_h)` 替代模型预测作为下一步条件
- `__init__`：读取 `cfg.training.val_tf`
- `run()`：通过 `cfg.training.resume_ckpt` 支持从任意 checkpoint 加载模型

#### 运行

```bash
python run.py --cfg factory/configs/phase7_stage1_tf_bound.yaml -g 7
```

#### 预期结果与决策

| TF PSNR 范围 | 判定 | 下一步 |
|-------------|------|--------|
| > 30 dB | 模型有拟合能力，AR 误差累积是瓶颈 | Stage 2a: `teacher_noise_std=0.01~0.05` + 继续训练 |
| 25-30 dB | 部分容量问题 + AR 误差 | Stage 2b: `base_ch=64` + `teacher_noise_std` |
| < 25 dB | 模型容量严重不足 | Stage 2c: `base_ch=64`, `lr=2e-4`, 训练 500+ epochs |

### 后续阶段规划

| Stage | 场景 | 配置变更 | 目的 |
|-------|------|---------|------|
| 2a | TF 上限高 | `teacher_noise_std: 0.03` | AR→TF桥接 |
| 2b | TF 上限中等 | `base_ch: 64`, `lr: 2e-4` | 提升容量 |
| 2c | TF 上限低 | `base_ch: 64`, `lr: 2e-4` | 大幅提升容量 |
| 3 | 容量+噪声OK | `n_train: 10` | 多图泛化 |

## Phase 11 计划：神经细胞自动机超分（Morpho-SR）

### 核心思想

将超分辨率视为**细胞分裂与生长**的涌现过程，而非传统的前馈映射。每个像素是一个"细胞"，拥有 16 通道内部状态。所有像素共享同一个极小的更新法则（DNA，~5.4K 参数），通过 15-30 步局部迭代，宏观涌现出高清纹理。

### 关键认知

- NCA 是**局部**操作：每个细胞只和 3×3 邻居交换信息
- 更新法则是**共享**的：同一组权重在所有像素、所有步骤上重复使用
- 训练目标是**稳定吸引子**：模型学会在任何步数后保持输出稳定
- 推理是**确定性**的（p_update=1.0），训练是**随机**的（p_update=0.5）

### 架构：MorphoSR

#### 细胞状态 S ∈ R^(B×16×H×W)
- Ch 0-2: 可见 RGB（最终输出）
- Ch 3: Alpha/活性标记
- Ch 4-15: 隐藏化学信使（12 通道）

#### 初始化（细胞分裂）
```
LR [3, h, w] → init_conv (1×1, 3→16) → NN↑ to H×W → S₀ [16, H, W]
```

#### 更新法则（每步重复）
```
Perception: Sobel X + Sobel Y (fixed, per-channel) + DepthwiseConv 3×3 (learnable)
Update MLP: concat(state, sobel_x, sobel_y, perceived, condition)
             → 1×1 Conv → SiLU → 1×1 Conv (zero-init) → ΔS
Stochastic:  mask ~ Bernoulli(p)
              S_{t+1} = S_t + mask ⊙ ΔS
```

#### 条件注入（Stage 2+）
将 LR bicubic↑ 到 HR 分辨率，concat 到 Update MLP 输入作为"生长支架"。

#### 预编码器（Stage 3+）
轻量 CNN（3→32→4×ResBlock→3，residual）在低分辨率空间对真实 LR 去噪。

### 参数统计

| 组件 | 参数量 |
|------|--------|
| init_conv (3→16, 1×1) | 64 |
| perceive_dw (16ch, 3×3) | 160 |
| update_mlp (64/67→64→16, 1×1) | ~5,200 |
| Pre-encoder (Stage 3+, optional) | ~37,000 |
| **核心 NCA 总计** | **~5,400** |

### 文件结构

| 文件 | 说明 |
|------|------|
| `model/morpho_sr.py` | MorphoSR: NCA 细胞更新法则 + PreEncoder |
| `factory/trainer_morpho.py` | MorphoTrainer + RealSRDataset: N 步迭代训练循环 |
| `factory/configs/phase11_stage1.yaml` | Stage 1: 单图 bicubic, 无条件, L1, 15 步固定 |
| `factory/configs/phase11_stage2.yaml` | Stage 2: 12 图 bicubic, 条件, L1+SSIM, 10-20 随机步 |
| `factory/configs/phase11_stage3.yaml` | Stage 3: 真实 LR + 预编码器 + VGG loss |
| `factory/configs/phase11_stage4.yaml` | Stage 4: 真实 LR + 预编码器 + VGG + PatchGAN |

### 四阶段实验路线

| Stage | LR 来源 | 条件 | 预编码器 | Loss | Steps | 训练图数 |
|-------|---------|------|---------|------|-------|---------|
| 1 | Bicubic↓ | 无 | 无 | L1 | 15 fixed | 1 |
| 2 | Bicubic↓ | Concat LR↑ | 无 | L1+SSIM | 10-20 random | 10 |
| 3 | Real LR | Concat LR↑ | 4-ResBlock | L1+SSIM+VGG | 10-20 random | 10 |
| 4 | Real LR | Concat LR↑ | 4-ResBlock | +PatchGAN | 10-20 random | 10 |

### 数据集

RealSR (Canon): `lau_dataset/realsr_canon/`
- 12 对图像，HR 1400×800, LR 700×400 (X2)
- 真实传感器退化：镜头模糊、噪声、轻微不对齐

### 训练策略

- **Crop 训练**：随机 256×256 crops，大幅降低显存（state tensor 仅 4 MB）
- **随机步数**：N ∈ [10, 20]，强迫模型在任何步数下稳定
- **随机更新**：p_update=0.5，异步细胞更新增强鲁棒性
- **溢出惩罚**：L2 正则化隐藏通道（4-15），防止数值爆炸
- **验证**：全图确定性推理（p_update=1.0, n_steps=fixed）

### 运行命令

```bash
# Stage 1: 单图 sanity (500 epochs)
python run.py --cfg factory/configs/phase11_stage1.yaml -g 7

# Stage 2: 多图泛化 (1000 epochs)
python run.py --cfg factory/configs/phase11_stage2.yaml -g 7

# Stage 3: 真实 LR (1500 epochs)
python run.py --cfg factory/configs/phase11_stage3.yaml -g 7

# Stage 4: GAN (2000 epochs, warm-start from Stage 3)
python run.py --cfg factory/configs/phase11_stage4.yaml -g 7
```

### 关键注意事项

- NCA 的 train/val 行为不同：训练时 p_update=0.5（随机），验证时 p_update=1.0（确定性）
- 全图验证（800×1400, 16ch）显存约 68 MB，在 24 GB GPU 上完全可行
- Stage 1 条件关闭（cond_ch=0），Update MLP 输入为 64 通道；Stage 2+ 条件开启（cond_ch=3），输入为 67 通道
- Stage 3/4 预编码器在低分辨率运行（400×700），效率远高于在 HR 分辨率去噪
- 模型极小（5.4K），LR 可用 1e-3，比传统 U-Net 高 5-10 倍
- RealSR 的 LR/HR 不对齐在 Stage 3 通过 VGG perceptual loss 缓解（特征级匹配替代像素级匹配）

### 已有诊断

- Smoke test (Stage 1, 10 epochs): train loss 0.628→0.499, val MSE 0.506→0.273, 前向/反向传播正常，无 NaN
- 模型加载成功：5,424 params, ~1s/epoch on 800×1400 full images

