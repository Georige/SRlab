# 拟合拉普拉斯金字塔 — 实现计划

## 动机

当前 U-Net 只有一个输出：全分辨率 (512×1024) 的噪声预测。扩散模型需要同时学习所有频率的细节——从宏观结构到微观纹理。拉普拉斯金字塔将残差分解为不同频率层级，让各解码器层级各司其职：深层负责粗结构，浅层负责细纹理。

**核心洞察**：U-Net 解码器天然有三个不同分辨率的输出点（dec3: 128×256, dec2: 256×512, dec1: 512×1024），正好匹配拉普拉斯金字塔的三层。不需要新架构，只需加输出头和损失。

## 架构对比

```
当前（单输出）:

  enc1 → enc2 → enc3 → bottleneck → dec3 → dec2 → dec1 → ε_pred [512×1024]
                                                                  ↑
                                                            MSE(ε_pred, ε)

拉普拉斯金字塔（多输出）:

  residual = HR - ISHT_base

  G0 = residual        [512×1024]     L0 = G0 - ↑(G1)   [512×1024]  ← finest
  G1 = ↓(G0)           [256×512]      L1 = G1 - ↑(G2)   [256×512]   ← medium
  G2 = ↓(G1)           [128×256]      L2 = G2            [128×256]   ← coarsest

  enc1 → enc2 → enc3 → bottleneck
                           │
                           ├→ dec3 → head_L2 → ε_L2 [128×256]
                           │              ↑ MSE(ε_L2, ε_noise_L2)
                           │
                           ├→ dec2 → head_L1 → ε_L1 [256×512]
                           │              ↑ MSE(ε_L1, ε_noise_L1)
                           │
                           └→ dec1 → head_final → ε_pred [512×1024]
                                          ↑ MSE(ε_pred, ε_full)

  L_total = L_full + λ2*L_L2 + λ1*L_L1
```

## 数据流

### 训练

```
1. 构建条件（不变）:
   cond, base, ms_isht = make_condition(lr_imgs)

2. 残差:
   residual = hr_imgs - base  [B,3,512,1024]

3. 高斯金字塔（下采样）:
   G0 = residual              [B,3,512,1024]
   G1 = F.interpolate(G0, scale=0.5)  [B,3,256,512]
   G2 = F.interpolate(G1, scale=0.5)  [B,3,128,256]

4. 拉普拉斯金字塔:
   L0 = G0 - F.interpolate(G1, size=G0.shape)  [B,3,512,1024]  高频层
   L1 = G1 - F.interpolate(G2, size=G1.shape)  [B,3,256,512]   中频层
   L2 = G2                                      [B,3,128,256]   低频层

5. 加噪（所有层用同一个 t）:
   noisy_L0 = α_t * L0 + σ_t * ε_L0
   noisy_L1 = α_t * L1 + σ_t * ε_L1
   noisy_L2 = α_t * L2 + σ_t * ε_L2

6. U-Net 前向 + 多尺度监督:
   # 输入: 全分辨率 noisy_L0 + cond（和当前一样）
   x = concat(noisy_L0, cond)  [B,9,512,1024]

   enc1 → enc2 → enc3 → bottleneck

   # dec3: 预测粗糙层
   x = dec3(x, s3, t_emb)  # x: [B,64,128,256]
   pred_L2 = head_L2(x)     # [B,3,128,256]
   loss_L2 = MSE(pred_L2, ε_L2)

   # dec2: 预测中频层
   x = dec2(x, s2, t_emb)  # x: [B,32,256,512]
   pred_L1 = head_L1(x)     # [B,3,256,512]
   loss_L1 = MSE(pred_L1, ε_L1)

   # dec1: 预测全分辨率（主输出，推理时唯一使用）
   x = dec1(x, s1, t_emb)  # x: [B,32,512,1024]
   pred_full = final(x)     # [B,3,512,1024]
   loss_full = MSE(pred_full, ε_L0)  # 主损失，拟合 L0 层噪声

   L_total = loss_full + λ2 * loss_L2 + λ1 * loss_L1
```

### 推理（不变）

推理时只用 `head_final`（dec1 输出），DDIM 采样完全相同。辅助头只在训练时提供多尺度监督。

## 关键设计选择

### 为什么对 L0/L1/L2 分别加噪而不是共享一个噪声图？

各层噪声独立采样，因为不同频率层的噪声统计特性不同。但所有层共享同一个时间步 t，确保一致的去噪进度。

### 为什么推理只用 dec1 输出？

推理时 U-Net 仍然处理全分辨率图像，dec3/dec2 的辅助头在训练时提供了更丰富的梯度信号，但在推理时不使用。这是 "训练时多任务，推理时单任务" 的经典范式。

### 权重 λ1, λ2 的设置

- λ2 (L2 粗糙层): 0.1 — 粗糙结构权重低，因为它信息量少
- λ1 (L1 中频层): 0.3 — 中等权重
- 可通过 config 调整

## 文件改动

| 文件 | 操作 | 说明 |
|------|------|------|
| `model/unet.py` | 修改 | 新增 `LapUNet`：dec3/dec2 处加 `head_L2`/`head_L1`，forward 返回三元组 |
| `model/diffusion.py` | 修改 | forward 中构建拉普拉斯金字塔，计算多尺度 loss |
| `config/diffusion_config.py` | 修改 | 新增开关 `USE_LAPLACIAN_PYRAMID` 和权重 `LP_LAMBDA` |
| `config/exp_laplacian.py` | **新建** | 实验配置 |

## 配置

```python
USE_LAPLACIAN_PYRAMID = False   # 拉普拉斯金字塔多尺度监督
LP_LAMBDAS = (1.0, 0.3, 0.1)    # (L0_full, L1_medium, L2_coarse) 权重
```

## 参数量

- `head_L2`: Conv2d(64, 3, 1) = 195 个参数
- `head_L1`: Conv2d(32, 3, 1) = 99 个参数
- 总计增加 < 300 个参数，可忽略不计

## 实现步骤

### Step 1: 修改 `model/unet.py`
- 新增 `use_laplacian` 开关参数
- dec3 后加 `head_L2 = Conv2d(base_ch*2, 3, 1)`
- dec2 后加 `head_L1 = Conv2d(base_ch, 3, 1)`
- forward 返回 `(pred_full, pred_L1, pred_L2)` 或仅 `pred_full`（取决于开关）

### Step 2: 修改 `model/diffusion.py`
- `__init__` 接收 `use_laplacian` 和 `lp_lambdas`
- `forward` 中：
  - 构建高斯金字塔 `G0, G1, G2`
  - 计算拉普拉斯层 `L0, L1, L2`
  - 对三层分别加噪
  - 用 noisy_L0 喂 U-Net，获取 `pred_full, pred_L1, pred_L2`
  - 分别计算三层 MSE，加权求和

### Step 3: 创建 `config/exp_laplacian.py`

### Step 4: 更新 `phase2_diffusion.py`
- 传入 `use_laplacian` 和 `lp_lambdas` 参数

## 与现有实验兼容性

- 开关默认 False，行为完全不变
- 推理流程完全不变（DDIM 采样只用 dec1 输出）
- 不增加推理时间
- 不增加显存（辅助头在 dec3/dec2 处，是 U-Net 内部已有特征）
- 可与其他创新点（HF residual, polar MoE）共存