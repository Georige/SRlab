# 球面隐式神经场全景超分 — 实现计划

## 动机

当前架构将全景图视为普通 2D 图像，用标准 Conv2d + 矩形 padding 处理。全景图本质是球面信号在等距投影下的离散化：
- 左/右边界在球面上是连通的（φ 周期为 2π）
- 顶部/底部各是一个点，但在等距投影中被拉伸为整行
- 标准卷积无法感知这些球面几何特性

目标：让神经网络"知道"它在处理球面信号，而非矩形图像。

## 核心思路

不改变 PixelDiffusion 的训练/推理框架（DDPM/DDIM、条件构建、loss），只将 U-Net 改造为球面感知版本。

```
                    条件路径（完全不变）
LR → bicubic↑ → SHT(L=255) → ISHT(L=255) → base [B,3,512,1024]
  │                                               │
  │   多尺度 ISHT 条件（不变）                      │
  │   ISHT(L=128)@256×512                          │
  │   ISHT(L=64)@128×256                           │
  │   ISHT(L=32)@64×128                            │
  │                                               │
  ▼                                               ▼
        ┌─── SphericalUNet（新） ───┐
        │                            │
        │  noisy_res  [B,3,512,1024] │
        │  + cond     [B,6,512,1024] │
        │  + coord_emb[B,8,512,1024] │  ← 新增：球坐标傅里叶编码
        │                            │
        │  enc1: CircularConv(32)    │  ← 改进：水平循环卷积
        │  enc2: CircularConv(64)    │
        │  enc3: CircularConv(128)   │
        │                            │
        │  bottleneck:                │
        │    ResBlock(128→128)       │
        │    + SphericalAttention    │  ← 新增：大圆距离注意力
        │                            │
        │  dec3/dec2/dec1: 标准UpBlock │
        │                            │
        │  → ε_pred [B,3,512,1024]   │
        └────────────────────────────┘
```

## 三个新组件

### 1. CircularConv2d（水平循环卷积）

问题：标准 Conv2d 在左右边界处用零填充，切断了球面上自然的连通性。

方案：在宽度方向使用循环填充。

```python
class CircularConv2d(nn.Module):
    """Conv2d with circular padding in width dimension."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, **kwargs):
        super().__init__()
        self.pad_w = (kernel_size - 1) // 2 if isinstance(kernel_size, int) else kernel_size[1] // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=(self.pad_w, 0), **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode='circular')  # 只在 W 方向循环
        return self.conv(x)
```

- 高度方向保持零填充（顶部/底部不连通，这是正确的——极点是奇点，不是边界）
- 通过 `USE_CIRCULAR_CONV` 开关控制是否替换标准 Conv

### 2. CoordEmbed（球坐标位置编码）

问题：标准 U-Net 不知道每个像素在球面上的位置。

方案：为每个像素生成球坐标的傅里叶特征，作为额外输入通道。

```
像素坐标 (x, y) → 球坐标 (θ, φ)
  θ = π × (0.5 - y/H)         纬度: [-π/2, π/2]
  φ = 2π × (x + 0.5) / W       经度: [0, 2π]

→ 3D 球面坐标:
  X = cos(θ) × cos(φ)
  Y = cos(θ) × sin(φ)
  Z = sin(θ)

→ 傅里叶特征: sin/cos(2^k × [X,Y,Z]) for k = 0..L-1
  → [B, 6L, H, W] 与输入 concat
```

- 用 3D Cartesian 坐标而非 (θ, φ) 是因为后者在极点有奇异性（θ=±π/2 时 φ 退化）
- 默认 L=4，产生 6×4=24 个额外通道，几乎不增参数
- 通过 `USE_COORD_EMBED` 开关控制

### 3. SphericalAttention（大圆距离注意力，瓶颈层）

问题：标准自注意力的位置编码假设欧氏空间，全景的左右边界在注意力中被视为"很远"。

方案：在瓶颈层用球面大圆距离计算注意力权重。

```
球面大圆距离:
  cos(d_ij) = sin(θ_i)×sin(θ_j) + cos(θ_i)×cos(θ_j)×cos(φ_i - φ_j)

注意力:
  attn_ij = softmax(-d_ij² / temperature)
```

- 只在瓶颈层使用（64×128 = 8192 个 token），计算量可控
- 用局部窗口 + 球形距离加权，不计算全局注意力
- 通过 `USE_SPHERICAL_ATTN` 开关控制

## 文件改动

| 文件 | 操作 | 说明 |
|------|------|------|
| `model/spherical.py` | **新建** | CircularConv2d, CoordEmbed, SphericalAttention |
| `model/blocks.py` | 修改 | DownBlock/UpBlock 增加 `use_circular` 开关 |
| `model/unet.py` | 修改 | 新增 `SphericalUNet`，开关控制是否启用 |
| `config/diffusion_config.py` | 修改 | 新增三个开关，默认全 False |
| `config/exp_spherical.py` | **新建** | 实验配置，全开 |

## 配置开关

```python
# config/diffusion_config.py 新增:
USE_CIRCULAR_CONV = False      # 水平循环卷积
USE_COORD_EMBED = False        # 球坐标位置编码
USE_SPHERICAL_ATTN = False     # 大圆距离注意力

# 整合开关（三步合一）:
USE_SPHERICAL_UNET = USE_CIRCULAR_CONV and USE_COORD_EMBED and USE_SPHERICAL_ATTN
```

## 与现有实验的兼容性

- **不破坏任何现有实验**：三个开关默认 False，行为完全不变
- PixelDiffusion 接口不变：`unet(noisy, cond, t_norm, ms_isht) → ε_pred`
- make_condition 不变：条件构建完全复用
- 训练/推理循环不变
- 参数量增幅 < 5%（主要是 CoordEmbed 的 channel 和 SphericalAttention 的投影矩阵）

## 实现步骤

### Step 1: 创建 `model/spherical.py`（~100 行）
- CircularConv2d: 包装 nn.Conv2d，宽度方向循环填充
- CoordEmbed: 球坐标 → 3D Cartesian → 傅里叶特征
- SphericalAttention: 瓶颈层的大圆距离稀疏注意力

### Step 2: 修改 `model/blocks.py`（~20 行改动）
- DownBlock: pool conv 可选用 CircularConv2d
- UpBlock: upsample conv 可选用 CircularConv2d
- ResBlock: conv1/conv2 可选用 CircularConv2d

### Step 3: 修改 `model/unet.py`（~30 行改动）
- 添加三个开关参数
- 在 forward 入口 concat CoordEmbed
- bottleneck 后添加 SphericalAttention
- encoder/decoder conv 替换为 CircularConv2d

### Step 4: 创建 `config/exp_spherical.py`
```python
from config.diffusion_config import *
USE_CIRCULAR_CONV = True
USE_COORD_EMBED = True
USE_SPHERICAL_ATTN = True
```

### Step 5: 训练验证
```bash
python phase2_diffusion.py -c exp_spherical -g 2 -n spherical
```

## 验证方式

1. **前向测试**：SphericalUNet 输出形状 [B,3,512,1024]
2. **球面对称性**：输入水平翻转，输出应近似一致（左右连通性验证）
3. **定性检查**：生成图在左右边界处应无缝连接
4. **指标对比**：v_img vs base 实验的收敛速度和最终指标
