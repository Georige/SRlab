"""Spherical-aware components for panoramic U-Net.

- CircularConv2d: conv with circular padding in width (left/right boundary connectivity)
- CoordEmbed: spherical coordinate Fourier features as extra input channels
- SphericalAttention: multi-head self-attention at bottleneck for long-range interaction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularConv2d(nn.Module):
    """Conv2d with circular padding in width dimension only.

    Panoramic images are equirectangular projections where left/right edges
    are connected on the sphere (φ wraps 0→2π). Standard zero-padding breaks
    this connectivity. This module applies circular padding in W before conv.
    """

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, **kwargs):
        super().__init__()
        if isinstance(kernel_size, int):
            k = kernel_size
            kw = kh = k
        else:
            kw, kh = kernel_size[1], kernel_size[0]
            k = kernel_size
        pad_h = (kh - 1) // 2
        self.pad_w = (kw - 1) // 2
        # conv handles H padding internally; we add W padding via circular pad
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=(pad_h, 0), **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode='circular')
        return self.conv(x)


class CoordEmbed(nn.Module):
    """Spherical coordinate Fourier feature embedding.

    Maps pixel coordinates (x,y) → spherical coordinates (θ,φ) →
    3D Cartesian (X,Y,Z) → Fourier features, providing each pixel's
    location on the unit sphere as extra input channels.

    Uses Cartesian coordinates (not raw θ,φ) to avoid singularity at poles
    where φ becomes degenerate when θ = ±π/2.
    """

    def __init__(self, H, W, num_freqs=4):
        super().__init__()
        y = torch.arange(H, dtype=torch.float32)
        x = torch.arange(W, dtype=torch.float32)

        theta = math.pi * (0.5 - y / H)          # [-π/2, π/2] latitude
        phi = 2.0 * math.pi * (x + 0.5) / W      # [0, 2π] longitude

        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        cos_p, sin_p = torch.cos(phi), torch.sin(phi)

        # 3D Cartesian coordinates on unit sphere [H, W]
        X = cos_t[:, None] * cos_p[None, :]
        Y = cos_t[:, None] * sin_p[None, :]
        Z = sin_t[:, None].expand(H, W)

        features = []
        for k in range(num_freqs):
            freq = 2 ** k
            for coord in [X, Y, Z]:
                features.append(torch.sin(freq * math.pi * coord))
                features.append(torch.cos(freq * math.pi * coord))

        # [6 * num_freqs, H, W]
        self.register_buffer('encoding', torch.stack(features, dim=0), persistent=False)
        self.out_ch = 6 * num_freqs

    def forward(self, x):
        """x: [B, C, H, W] — used only for batch size and device; returns [B, out_ch, H, W]."""
        return self.encoding.unsqueeze(0).expand(x.shape[0], -1, -1, -1)


class SphericalAttention(nn.Module):
    """Axial self-attention at bottleneck with circular W-padding.

    Decomposes 2D attention into two sequential 1D axial passes:
      1. Width-attention: each row independently (W=128 tokens per row)
      2. Height-attention: each column independently (H=64 tokens per col)

    Complexity: O(HW² + WH²) ≈ 1.0M + 0.5M per head vs O((HW)²) ≈ 67M for full.
    W-attention naturally handles spherical boundary (left↔right connectivity).
    """

    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = ch // num_heads

        self.norm_w = nn.GroupNorm(min(8, ch), ch)
        self.norm_h = nn.GroupNorm(min(8, ch), ch)
        self.w_qkv = nn.Conv2d(ch, ch * 3, 1)
        self.w_proj = nn.Conv2d(ch, ch, 1)
        self.h_qkv = nn.Conv2d(ch, ch * 3, 1)
        self.h_proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        shortcut = x
        x = self.norm_w(x)
        x = shortcut + self._axial_w(x)

        shortcut = x
        x = self.norm_h(x)
        x = shortcut + self._axial_h(x)
        return x

    def _axial_w(self, x):
        """Width-axis attention: each row attends within itself."""
        B, C, H, W = x.shape
        qkv = self.w_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # [B, C, H, W] → [B*H, heads, W, head_dim]
        q = q.reshape(B, self.num_heads, self.head_dim, H, W)
        q = q.permute(0, 3, 1, 4, 2).reshape(B * H, self.num_heads, W, self.head_dim)
        k = k.reshape(B, self.num_heads, self.head_dim, H, W)
        k = k.permute(0, 3, 1, 4, 2).reshape(B * H, self.num_heads, W, self.head_dim)
        v = v.reshape(B, self.num_heads, self.head_dim, H, W)
        v = v.permute(0, 3, 1, 4, 2).reshape(B * H, self.num_heads, W, self.head_dim)

        out = F.scaled_dot_product_attention(q, k, v)  # [B*H, heads, W, head_dim]

        out = out.reshape(B, H, self.num_heads, W, self.head_dim)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, C, H, W)
        return self.w_proj(out)

    def _axial_h(self, x):
        """Height-axis attention: each column attends within itself."""
        B, C, H, W = x.shape
        qkv = self.h_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # [B, C, H, W] → [B*W, heads, H, head_dim]
        q = q.reshape(B, self.num_heads, self.head_dim, H, W)
        q = q.permute(0, 4, 1, 3, 2).reshape(B * W, self.num_heads, H, self.head_dim)
        k = k.reshape(B, self.num_heads, self.head_dim, H, W)
        k = k.permute(0, 4, 1, 3, 2).reshape(B * W, self.num_heads, H, self.head_dim)
        v = v.reshape(B, self.num_heads, self.head_dim, H, W)
        v = v.permute(0, 4, 1, 3, 2).reshape(B * W, self.num_heads, H, self.head_dim)

        out = F.scaled_dot_product_attention(q, k, v)  # [B*W, heads, H, head_dim]

        out = out.reshape(B, W, self.num_heads, H, self.head_dim)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, C, H, W)
        return self.h_proj(out)
