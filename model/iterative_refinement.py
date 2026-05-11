"""IterativeRefinementUNet — 4-level U-Net with time embedding for same-resolution
iterative refinement with center-first schedule.

Unlike CenterGrowingUNet (which grows the output region step by step), this model
works at a SINGLE resolution throughout, using a time embedding to indicate which
refinement step it's at. Each step REFINES (improves quality) rather than GENERATES
(new content).

Architecture mirrors CenterGrowingUNet but adds sinusoidal time embedding injected
at the bottleneck ResBlocks. The time embedding tells the model which refinement
step it's currently executing.

Input:  [B, 7, H, W] = concat(I_low_up, current_state, mask)
        + t_norm [B] float in [0, 1] (sinusoidal time embedding)
Output: [B, 3, H, W] = refined residual
"""

import math
import torch
import torch.nn as nn


# ============================================================
# Building blocks
# ============================================================

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding -> MLP."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t_norm):
        """t_norm: [B] in [0, 1]"""
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t_norm.device) / half)
        args = t_norm.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ConvBlock(nn.Module):
    """Conv2d -> GroupNorm -> SiLU, with optional stride for downsampling."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """Residual block with time injection via FiLM-style channel addition."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class UpBlock(nn.Module):
    """Bilinear upsample x2 + skip-concat + ConvBlock.

    Accepts t_emb for interface consistency (passed through to conv, which ignores it).
    """

    def __init__(self, in_ch, out_ch, skip_ch, time_dim=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip, t_emb=None):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ============================================================
# Main model
# ============================================================

class IterativeRefinementUNet(nn.Module):
    """4-level U-Net with sinusoidal time embedding for iterative refinement.

    Architecture (same backbone as CenterGrowingUNet):
      Encoder:  H->H/2->H/4->H/8->H/16 (4 ConvBlock stride-2)
      Bottleneck: 2x ResBlock with time injection at H/16
      Decoder:  H/16->H/8->H/4->H/2->H (4 UpBlock + skip connections)

    Time embedding is injected ONLY at the bottleneck ResBlocks. The encoder
    and decoder are the same as CenterGrowingUNet — the time signal flows
    through the feature space from the bottleneck outward.

    Args:
        in_ch:    input channels (default 7: I_low_up(3)+current_state(3)+mask(1))
        out_ch:   output channels (default 3, the refined residual)
        base_ch:  base channel count (default 64)
        time_dim: time embedding dimension (default 128)
    """

    def __init__(self, in_ch=7, out_ch=3, base_ch=64, time_dim=128):
        super().__init__()
        bc = base_ch

        self.time_emb = TimeEmbedding(time_dim)

        # ---- Encoder (same as CenterGrowingUNet) ----
        self.init_conv = ConvBlock(in_ch, bc)
        self.enc1 = ConvBlock(bc,     bc * 2, stride=2)   # H   -> H/2
        self.enc2 = ConvBlock(bc * 2, bc * 4, stride=2)   # H/2 -> H/4
        self.enc3 = ConvBlock(bc * 4, bc * 8, stride=2)   # H/4 -> H/8
        self.enc4 = ConvBlock(bc * 8, bc * 8, stride=2)   # H/8 -> H/16

        # ---- Bottleneck (2x ResBlock with time injection) ----
        self.mid1 = ResBlock(bc * 8, bc * 8, time_dim)
        self.mid2 = ResBlock(bc * 8, bc * 8, time_dim)

        # ---- Decoder (same as CenterGrowingUNet) ----
        # UpBlock(in_ch, out_ch, skip_ch)
        self.dec4 = UpBlock(bc * 8, bc * 8, bc * 8)   # H/16 -> H/8  (skip: e3)
        self.dec3 = UpBlock(bc * 8, bc * 4, bc * 4)   # H/8  -> H/4  (skip: e2)
        self.dec2 = UpBlock(bc * 4, bc * 2, bc * 2)   # H/4  -> H/2  (skip: e1)
        self.dec1 = UpBlock(bc * 2, bc,     bc)       # H/2  -> H    (skip: x0)

        # ---- Output head ----
        self.out_conv = nn.Conv2d(bc, out_ch, 3, padding=1)

    def forward(self, x, t_norm):
        """Forward pass with time conditioning.

        Args:
            x:      [B, in_ch, H, W]  input tensor
            t_norm: [B] float in [0, 1]  normalized refinement step

        Returns:
            [B, out_ch, H, W]  refined residual
        """
        t_emb = self.time_emb(t_norm)

        # Encoder
        x0 = self.init_conv(x)          # [B, bc, H, W]
        e1 = self.enc1(x0)              # [B, bc*2, H/2, W/2]
        e2 = self.enc2(e1)              # [B, bc*4, H/4, W/4]
        e3 = self.enc3(e2)              # [B, bc*8, H/8, W/8]
        e4 = self.enc4(e3)              # [B, bc*8, H/16, W/16]

        # Bottleneck (where time is injected)
        m = self.mid1(e4, t_emb)        # [B, bc*8, H/16, W/16]
        m = self.mid2(m, t_emb)         # [B, bc*8, H/16, W/16]

        # Decoder with skip connections
        d4 = self.dec4(m,  e3, t_emb)   # [B, bc*8, H/8, W/8]
        d3 = self.dec3(d4, e2, t_emb)   # [B, bc*4, H/4, W/4]
        d2 = self.dec2(d3, e1, t_emb)   # [B, bc*2, H/2, W/2]
        d1 = self.dec1(d2, x0, t_emb)   # [B, bc,   H, W]

        return self.out_conv(d1)
