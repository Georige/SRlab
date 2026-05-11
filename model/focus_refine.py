"""FocusRefineUNet — iterative refinement UNet for coarse-to-fine center focus.

Phase 9: Multi-step refinement from full-image blur to sharp center.
Input: [B, 4, H, W] = concat(I_low_up, center_mask)
Output: [B, 3, H, W] = residual (added to input to produce refined image)

Key components:
  - Time embedding (sinusoidal PE) → FiLM injection in bottleneck
  - Spatial bias: fixed Gaussian heatmap at bottleneck resolution (optional, for ablation)
  - No panoramic modules, no ISHT conditioning — pure pixel-space refinement
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Time embedding (sinusoidal PE, same as diffusion models)
# ============================================================

class TimeEmbedding(nn.Module):
    """Sinusoidal positional encoding → 2-layer MLP.

    Input: t_norm [B] in [0, 1] or scalar broadcastable to [B].
    Output: [B, dim].
    """

    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        # t: [B] or scalar
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.dim // 2
        # freq: exp(-log(10000) * i / half) for i in [0, half)
        freq = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)  # [B, half]
        emb = torch.cat([args.sin(), args.cos()], dim=1)    # [B, dim]
        return self.mlp(emb)


# ============================================================
# Gaussian heatmap (fixed spatial prior at bottleneck resolution)
# ============================================================

def make_gaussian_heatmap(h, w, sigma=0.3):
    """Create a 2D Gaussian heatmap [1, 1, h, w] with center=1, edges≈0.

    Args:
        h, w: spatial dimensions (bottleneck resolution, e.g., H/8 × W/8)
        sigma: controls spread relative to half-size. 0.3 → moderate spread.

    Returns:
        [1, 1, h, w] tensor.
    """
    y = torch.arange(h, dtype=torch.float32)
    x = torch.arange(w, dtype=torch.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    # Normalize distance by half-size
    dy = (y.view(-1, 1) - cy) / (h / 2.0)
    dx = (x.view(1, -1) - cx) / (w / 2.0)
    dist_sq = dy ** 2 + dx ** 2
    heatmap = torch.exp(-dist_sq / (2.0 * sigma ** 2))
    return heatmap.view(1, 1, h, w)


# ============================================================
# Building blocks
# ============================================================

class ResBlockFiLM(nn.Module):
    """Residual conv block with FiLM time injection.

    time_proj: time_dim → out_ch*2
    Injection: x = x * (1 + gamma) + beta  (channel-wise, broadcast over spatial dims)
    """

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

        # FiLM projection: time → (gamma, beta)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)

    def forward(self, x, t_emb):
        # t_emb: [B, time_dim]
        gamma, beta = self.time_proj(t_emb).chunk(2, dim=1)  # [B, out_ch], [B, out_ch]

        h = self.act(self.norm1(self.conv1(x)))
        # FiLM injection
        h = h * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class DownBlockDirect(nn.Module):
    """ResBlockDirect (no time) + stride-2 downsampling."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResBlockNoTime(in_ch, out_ch)
        self.pool = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        skip = self.res(x)
        return skip, self.pool(skip)


class UpBlockDirect(nn.Module):
    """Upsample + skip concat + ResBlock (no time in decoder)."""

    def __init__(self, in_ch, out_ch, skip_ch):
        super().__init__()
        self.x_ch = in_ch - skip_ch
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.x_ch, self.x_ch, 3, padding=1),
        )
        self.res = ResBlockNoTime(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


class ResBlockNoTime(nn.Module):
    """Standard residual conv block without time injection."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


# ============================================================
# FocusRefineUNet
# ============================================================

class FocusRefineUNet(nn.Module):
    """Iterative refinement U-Net with time embedding and optional spatial bias.

    Architecture (4-level):
        enc1: in_ch → base_ch       (H→H/2)
        enc2: base_ch → base_ch*2   (H/2→H/4)
        enc3: base_ch*2 → base_ch*4 (H/4→H/8)
        bottleneck: base_ch*4 → base_ch*4 (H/8) with FiLM time injection
        [optional spatial bias: Gaussian heatmap added at bottleneck]
        dec3: base_ch*8 → base_ch*2 (H/8→H/4)
        dec2: base_ch*4 → base_ch   (H/4→H/2)
        dec1: base_ch*2 → base_ch   (H/2→H)
        final: base_ch → out_ch

    Args:
        in_ch: input channels (default 4 = I_low_up 3 + mask 1)
        base_ch: base channel count (default 64)
        out_ch: output channels (default 3 = residual)
        time_dim: time embedding dimension (default 128)
        use_spatial_bias: add Gaussian heatmap at bottleneck (default True)
        hr_size: (H, W) of input images, used to size the heatmap
    """

    def __init__(self, in_ch=4, base_ch=64, out_ch=3, time_dim=128,
                 use_spatial_bias=True, use_time_embedding=True, hr_size=(512, 512)):
        super().__init__()
        self.use_spatial_bias = use_spatial_bias
        self.use_time_embedding = use_time_embedding
        self.base_ch = base_ch

        # Time embedding
        if use_time_embedding:
            self.time_embed = TimeEmbedding(time_dim)
        else:
            self.time_embed = None

        # Encoder (no time injection)
        self.enc1 = DownBlockDirect(in_ch, base_ch)              # → [B, base_ch, H, W]
        self.enc2 = DownBlockDirect(base_ch, base_ch * 2)         # → [B, base_ch*2, H/2, W/2]
        self.enc3 = DownBlockDirect(base_ch * 2, base_ch * 4)     # → [B, base_ch*4, H/4, W/4]

        # Bottleneck: with or without FiLM time injection
        if use_time_embedding:
            self.bottleneck = ResBlockFiLM(base_ch * 4, base_ch * 4, time_dim)
        else:
            self.bottleneck = ResBlockNoTime(base_ch * 4, base_ch * 4)

        # Spatial bias: fixed Gaussian heatmap → learnable projection → add to bottleneck
        if use_spatial_bias:
            h_bn, w_bn = hr_size[0] // 8, hr_size[1] // 8
            self.register_buffer('heatmap', make_gaussian_heatmap(h_bn, w_bn, sigma=0.3))
            self.heatmap_proj = nn.Conv2d(1, base_ch * 4, 1)
        else:
            self.heatmap = None
            self.heatmap_proj = None

        # Decoder (no time injection)
        self.dec3 = UpBlockDirect(base_ch * 8, base_ch * 2, skip_ch=base_ch * 4)
        self.dec2 = UpBlockDirect(base_ch * 4, base_ch, skip_ch=base_ch * 2)
        self.dec1 = UpBlockDirect(base_ch * 2, base_ch, skip_ch=base_ch)

        # Final conv
        self.final = nn.Sequential(
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, out_ch, 3, padding=1),
        )

    def forward(self, x, t_norm=None):
        """Forward pass.

        Args:
            x: [B, in_ch, H, W] input (I_low_up concatenated with mask)
            t_norm: [B] normalized timestep in [0, 1] (ignored if use_time_embedding=False)

        Returns:
            [B, out_ch, H, W] residual
        """
        # Encoder
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)

        # Bottleneck with or without time injection
        if self.use_time_embedding:
            t_emb = self.time_embed(t_norm)
            x = self.bottleneck(x, t_emb)
        else:
            x = self.bottleneck(x)

        # Spatial bias
        if self.use_spatial_bias:
            hmap = self.heatmap_proj(self.heatmap)  # [1, base_ch*4, H/8, W/8]
            x = x + hmap

        # Decoder
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.final(x)


# ============================================================
# DirectSRUNet — baseline: simple 3→3 one-step U-Net
# ============================================================

class DirectSRUNet(nn.Module):
    """Simple one-step U-Net for direct super-resolution (3→3).

    Used as baseline: input I_low_up directly, output residual, full-image MSE loss.
    Same architecture as FocusRefineUNet but no time embedding, no spatial bias.
    """

    def __init__(self, in_ch=3, base_ch=64, out_ch=3):
        super().__init__()
        self.base_ch = base_ch

        self.enc1 = DownBlockDirect(in_ch, base_ch)
        self.enc2 = DownBlockDirect(base_ch, base_ch * 2)
        self.enc3 = DownBlockDirect(base_ch * 2, base_ch * 4)

        self.bottleneck = ResBlockNoTime(base_ch * 4, base_ch * 4)

        self.dec3 = UpBlockDirect(base_ch * 8, base_ch * 2, skip_ch=base_ch * 4)
        self.dec2 = UpBlockDirect(base_ch * 4, base_ch, skip_ch=base_ch * 2)
        self.dec1 = UpBlockDirect(base_ch * 2, base_ch, skip_ch=base_ch)

        self.final = nn.Sequential(
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, out_ch, 3, padding=1),
        )

    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)

        x = self.bottleneck(x)

        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.final(x)
