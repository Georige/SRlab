明白了，我们要从“全景图超高清”这个目标出发，**完全在频域里用球谐系数完成超分辨率重建**。下面是一个可立即执行的行动方案，分 4 个阶段，每个阶段都有明确的目标、代码路径和验证方式。

---

## 🧭 总体思路

将全景图超分定义为：

> **低带宽 L_lo 的球谐系数 → 神经网络 → 高带宽 L_hi 的球谐系数 → ISHT → 高分辨率全景图**

全程不碰变形的 ERP 像素网格，所有学习都在球谐域完成。

---

## 🔬 第一阶段：环境搭建与频域“无损重建”验证（1–2 天）

**目标**  
确认 SHT/ISHT 在你的数据上可以做到数值级无损，验证带宽选择，为后续训练打下正确性基础。

**具体步骤**

1. **安装 `torch-harmonics`**
   ```bash
   pip install torch-harmonics
   ```

2. **写一个验证脚本，直接验证 L 带宽的重建精度**
   ```python
   import torch
   from torch_harmonics import RealSHT, InverseRealSHT
   
   H, W = 512, 1024          # 目标高分辨率，后续可调整
   L = H                     # 带宽取 512（奈奎斯特条件允许）
   sht = RealSHT(H, W, lmax=L, mmax=L)
   isht = InverseRealSHT(H, W, lmax=L, mmax=L)
   
   img = load_your_panorama()  # [1, 3, H, W]，归一化到 [-1,1] 左右
   coeffs = sht(img)           # [1, 3, L²]
   recon = isht(coeffs)        
   err = (img - recon).abs().max()
   print(f"Max reconstruction error: {err:1.5e}")
   ```
   – 若 `err < 1e-4`，说明该 L 能无损捕获该分辨率。  
   – 用这个脚本确认你的数据分辨率与 L 的对应关系，为后续选择合适的高低带宽做准备。

3. **绘制球谐谱，观察能量集中区域**
   ```python
   # coeffs: [1, 3, L²]，按 l 分组算平均能量
   idx = 0
   energy_per_l = []
   for l in range(L):
       num = 2*l + 1
       block = coeffs[0, :, idx:idx+num]
       energy = (block ** 2).sum().item()
       energy_per_l.append(energy / num)
       idx += num
   # 画出 energy vs l，你会看到能量主要聚集在低 l 区域
   ```
   这一步帮你确定：**低带宽 L_lo 可以取多少就能保留约 99% 的能量**，为后续下采样/压缩倍数提供依据。

---

## 🧱 第二阶段：实现最简频域超分基线（3–5 天）

**目标**  
训练一个网络，将 **L_lo 系数 → L_hi 系数**，在 WS-PSNR 等球面指标上超过 Bicubic 插值。

**关键设计决策**

- **低带宽 L_lo**：建议从 `H/4` 开始，如 H=512 时 L_lo=128（对应 16k 系数）。
- **高带宽 L_hi**：最终目标带宽，初期可设为 256 或 512。
- **网络输入/输出形状**：`[batch, C, N_lo]` → `[batch, C, N_hi]`，其中 `N_lo = L_lo²`，`N_hi = L_hi²`。

**模型设计（先简单后复杂）**

1. **纯 MLP 基线**  
   ```python
   class SpectralSR_MLP(nn.Module):
       def __init__(self, L_lo, L_hi, C=3, hidden=1024):
           super().__init__()
           self.N_lo = L_lo**2
           self.N_hi = L_hi**2
           self.net = nn.Sequential(
               nn.Linear(C * self.N_lo, hidden),
               nn.ReLU(),
               nn.Linear(hidden, hidden),
               nn.ReLU(),
               nn.Linear(hidden, C * self.N_hi)
           )
       def forward(self, coeffs_lo):
           # coeffs_lo: [B, C, N_lo]
           B = coeffs_lo.shape[0]
           x = coeffs_lo.view(B, -1)
           out = self.net(x)
           return out.view(B, C, self.N_hi)
   ```
   直接展平所有通道所有系数喂给 MLP，不做任何频率分组。这能作为**下限基线**。

2. **频率感知的 MLP（分组处理）**  
   利用你先前的知识：同一 l 的 (2l+1) 个系数应该共享变换逻辑。  
   设计一个网络，对每个 l 的输入块单独映射到输出块，并考虑跨 l 的信息交互（例如用 Transformer 在 l 维度建模）。  
   初期可简化：**对每个 l 独立做一个小的线性层**（输入长度 (2l+1)×C_in，输出长度 (2l+1)×C_out，但对应高带宽的 l 分组需要处理非等长问题）。  
   具体做法：将所有 l 块 reshape 到相同长度（如最大长度 2L_hi-1，通过零填充），用共享 MLP 处理，然后裁剪。

**损失函数**

- 在球谐系数上直接计算 **MSE 损失**（频域损失）：
  ```python
  loss = F.mse_loss(coeffs_pred, coeffs_gt)
  ```
- 也可以加一个空间域辅助损失：将预测系数 ISHT 回空间，与 GT 高清图算 WS-PSNR。但训练时 WS-PSNR 计算较慢，初期可只用系数损失。

**数据准备**

下载 **Flickr360** 数据集（或简单用 **SUN360** 子集），按标准方式生成 LR-HR 对：
- HR 图：裁剪/缩放至 512×1024（或其他固定尺寸）
- LR 图：将 HR 图下采样（如 Bicubic 2×/4×），再上采样回相同尺寸，保证对齐，然后分别做 SHT 得到 L_lo 和 L_hi 系数。  
小心：下采样会产生新的频率混叠，建议在 ERP 空间用抗混叠滤波器下采样，然后对 LR 和 HR 各自做 SHT。这样 L_lo 可由 LR 的空间分辨率决定（例如 128×256 的 LR 图，最大带宽约 128）。

**验证指标**

- **WS-PSNR**：在 42个球面均匀分布的采样点上计算 PSNR（`torch-harmonics` 有 `weighted_psnr` 实现可参考，或自己实现）。
- **WS-SSIM**：类似加权结构相似性。
- 可视化：将重建图转为 ERP，检查两极是否出现模糊或伪影。

---

## 🚀 第三阶段：引入频域卷积 / 球面 CNN 组件（1–2 周）

**目标**  
用真正的球面卷积层替换简单的 MLP，让网络在球谐域内具有**旋转等变归纳偏置**，提升泛化能力和重建质量。

**操作方式**

1. 使用 `torch-harmonics` 内置的 **DISCO 卷积层** 或其他球面卷积实现。  
2. 堆叠一个浅层的 U-Net 式架构，但所有操作都在球谐系数上：
   - 下采样：带宽从 L_hi 逐步降低（丢弃高频系数）  
   - 上采样：通过补零高频系数 + 频域卷积恢复到 L_hi。
3. 简单示例（伪代码）：
   ```python
   class SphericalSRNet(nn.Module):
       def __init__(self, L_lo, L_hi):
           super().__init__()
           self.upsample = nn.Linear(L_lo**2, L_hi**2)  # 初期上采样
           # 后续接几个 DISCO 球面卷积层 + 非线性（需要来回 SHT/ISHT）
           self.conv1 = SphericalConv(in_ch=3, out_ch=64, L=L_hi)
           self.conv2 = SphericalConv(64, 64, L=L_hi)
           self.conv3 = SphericalConv(64, 3, L=L_hi)
           
       def forward(self, coeffs_lo):
           x = self.upsample(coeffs_lo.flatten(2))  # 简单上采样到高带宽系数
           x = x.reshape(B, 3, -1)
           # 转换为空间域（球面网格）
           x = ISHT(x, lmax=L_hi)      # [B, 3, H_hi, W_hi]
           x = self.conv1(x)           # 球面卷积，内部会做 SHT→频域缩放→ISHT
           x = F.relu(x)
           x = self.conv2(x)
           x = F.relu(x)
           x = self.conv3(x)
           # 最终输出空间域图，或用 SHT 转回系数另算损失
           return x
   ```
   注意：这种方法需要频繁 SHT/ISHT，但可以保证旋转等变，也容易嵌入其他球面模块。

**实验计划**

- 先用一个简单残差块，只做恒等映射，确认梯度流动和数值稳定性。
- 逐渐加深，监控 WS-PSNR，目标是在同等参数量下显著超越第二阶段 MLP 基线，并在视觉上减轻两极畸变。

---

## 📈 第四阶段：优化与前沿探索（持续迭代）

该阶段可以结合你的创新想法，向更深入的方向推进：

1. **纯频域超分（无需空间域往返）**  
   设计网络全程只在球谐系数上操作，使用 Clebsch-Gordan 张量积实现非线性（如 `e3nn`），避免 SHT/ISHT，追求极致效率且严格等变。  
   - 可参考论文：Clebsch–Gordan Nets (Kondor 2018), e3nn 示例。

2. **高带宽扩展**  
   当 L_hi 很高（如 1024）时，系数长度超过百万，MLP 不可行。采用 **Spectral Attention**：将系数视为长度为 L² 的序列，但使用基于频率位置的编码，应用 Performer 或线性 Transformer 减少复杂度。也可考虑在系数上做稀疏建模（只预测高频的少量系数）。

3. **混合损失与感知质量**  
   加入球面上的感知损失：用预训练的分类或分割网络提取球面特征，计算特征差异。还可尝试球面 GAN。

4. **与现有 SOTA 方法公平对比**  
   选定至少两个强基线：OSRT、SphereSR（或其开源版本），用同一个训练/测试集和 WS-PSNR 评估。关注论文中常报告的 PSNR 增益和视觉两极修复效果。

---

## 🧪 实用建议：快速出第一版结果的核心捷径

- **直接用 `torch-harmonics` 实现一个简单的 “SHT → MLP → ISHT” 端到端训练**，使用 HR 的空间图作为监督信号，避免手动计算系数损失。相当于：
  ```python
  x_lo = sht(lr_img)
  coeffs_pred = mlp(x_lo)
  sr_img = isht(coeffs_pred)
  loss = WS_PSNR(sr_img, hr_img)
  ```
  这个流程代码不超过 100 行，却能让你立刻看到频域方法的潜力，然后再逐步精化。

---

这是你向前一步的具体路线。如果你希望我先帮你写第一阶段那个验证脚本，或者详细规划一个频域 U-Net 的结构，你告诉我，我随时继续。