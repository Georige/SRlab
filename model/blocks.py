"""U-Net building blocks: time embedding, residual, down/up sampling."""

import math
import torch
import torch.nn as nn
from model.spherical import CircularConv2d


def _make_conv(in_ch, out_ch, kernel_size, stride=1, padding=1, use_circular=False):
    """Factory: returns CircularConv2d or standard Conv2d based on flag."""
    if use_circular:
        return CircularConv2d(in_ch, out_ch, kernel_size, stride=stride)
    return nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding → MLP."""

    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t_norm):
        """t_norm: [B] in [0, 1]"""
        half = self.mlp[0].in_features // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t_norm.device) / half)
        args = t_norm.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """Residual conv block with time injection."""

    def __init__(self, in_ch, out_ch, time_dim, use_circular=False):
        super().__init__()
        self.conv1 = _make_conv(in_ch, out_ch, 3, use_circular=use_circular)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = _make_conv(out_ch, out_ch, 3, use_circular=use_circular)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class DownBlock(nn.Module):
    """ResBlock + stride-2 downsampling (returns both skip and pooled)."""

    def __init__(self, in_ch, out_ch, time_dim, use_circular=False):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_dim, use_circular=use_circular)
        self.pool = _make_conv(out_ch, out_ch, 3, stride=2, use_circular=use_circular)

    def forward(self, x, t_emb):
        skip = self.res(x, t_emb)
        return skip, self.pool(skip)


class UpBlock(nn.Module):
    """Upsample + skip-concat + ResBlock."""

    def __init__(self, in_ch, out_ch, time_dim, skip_ch, use_circular=False):
        super().__init__()
        x_ch = in_ch - skip_ch
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            _make_conv(x_ch, x_ch, 3, use_circular=use_circular),
        )
        self.res = ResBlock(in_ch, out_ch, time_dim, use_circular=use_circular)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb)


class PolarMoE(nn.Module):
    """2-expert MoE gated by row coordinate (latitude) for panoramic distortion.

    - Expert 0 (equatorial): active near image center (y ≈ H/2)
    - Expert 1 (polar): active near top/bottom (y ≈ 0 or y ≈ H-1)

    Gate: smooth interpolation based on normalized y-coordinate.
    Residual: output = blend(exp0, exp1) + x.

    Load balancing: tracks per-expert output magnitude during forward(),
    exposes load_balance_loss() as CV² penalty to prevent one expert
    from dominating the other (collapse).
    """

    def __init__(self, ch):
        super().__init__()
        self.expert_eq = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.expert_pole = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self._eq_norm = 0.0
        self._pole_norm = 0.0

    def forward(self, x):
        B, C, H, W = x.shape
        y = torch.arange(H, device=x.device, dtype=x.dtype)
        # gate: 0 at equator (y=H/2), 1 at poles (y=0, y=H-1)
        gate = (1.0 - torch.sin(torch.pi * y / H)).view(1, 1, H, 1)  # [1,1,H,1]
        eq = self.expert_eq(x)
        pole = self.expert_pole(x)
        # Track RMS per-channel activation for load balancing
        self._eq_norm = eq.pow(2).mean().sqrt().item()
        self._pole_norm = pole.pow(2).mean().sqrt().item()
        return (1.0 - gate) * eq + gate * pole + x

    def load_balance_loss(self):
        """Coefficient-of-variation² penalty on per-expert output magnitude.

        Returns 0 when both experts contribute equally (CV=0).
        Approaches 1 when one expert dominates completely.
        """
        a, b = self._eq_norm, self._pole_norm
        mean = (a + b) / 2.0
        if mean < 1e-8:
            return 0.0
        var = ((a - mean)**2 + (b - mean)**2) / 2.0
        return var / (mean**2)
