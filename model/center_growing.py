"""CenterGrowingUNet — 4-level U-Net for progressive central growing super-resolution.

Fully convolutional: works with any input size divisible by 16 (e.g. 512×512 or 1024×2048).

Input:  [B, 10, H, W] = concat(I_low_up, local_low, high_res_known, mask)
Output: [B, 3, H, W]   = residual (added to I_low_up in trainer)

Architecture: 4× down + 4× up with skip connections, H→H/2→H/4→H/8→H/16 bottleneck.
Designed for single-image overfitting first, extensible to multi-image training.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d → GroupNorm → SiLU, with optional stride for downsampling."""

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
    """Residual block: two ConvBlocks + skip connection."""

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(ch, ch),
            ConvBlock(ch, ch),
        )

    def forward(self, x):
        return x + self.block(x)


class UpBlock(nn.Module):
    """Upsample (bilinear ×2) → concat skip → ConvBlock."""

    def __init__(self, in_ch, out_ch, skip_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CenterGrowingUNet(nn.Module):
    """4-level U-Net for progressive central growing super-resolution.

    Encoder: H×W → H/2×W/2 → H/4×W/4 → H/8×W/8 → H/16×W/16  (4× down)
    Bottleneck: 2× ResBlock at H/16×W/16
    Decoder: H/16×W/16 → ... → H×W  (4× up + skip connections)

    Works with any input size where H, W are divisible by 16.
    E.g. 512×512 → bottleneck 32×32, or 1024×2048 → bottleneck 64×128.

    Args:
        in_ch:   input channels (default 10)
        out_ch:  output channels (default 3, the residual)
        base_ch: base channel count (default 64; use 32 for single-image overfit)
    """

    def __init__(self, in_ch=10, out_ch=3, base_ch=64):
        super().__init__()
        # ---- Encoder ----
        self.init_conv = ConvBlock(in_ch, base_ch)

        self.enc1 = ConvBlock(base_ch,     base_ch * 2, stride=2)
        self.enc2 = ConvBlock(base_ch * 2, base_ch * 4, stride=2)
        self.enc3 = ConvBlock(base_ch * 4, base_ch * 8, stride=2)
        self.enc4 = ConvBlock(base_ch * 8, base_ch * 8, stride=2)

        # ---- Bottleneck ----
        self.mid = nn.Sequential(
            ResBlock(base_ch * 8),
            ResBlock(base_ch * 8),
        )

        # ---- Decoder ----
        # UpBlock(in_from_prev, out, skip_channels)
        self.dec4 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 8)
        self.dec3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec1 = UpBlock(base_ch * 2, base_ch,     base_ch)

        # ---- Output head ----
        self.out_conv = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x):
        # Encoder
        x0 = self.init_conv(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        m = self.mid(e4)

        # Decoder with skip connections
        d4 = self.dec4(m,  e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x0)

        return self.out_conv(d1)
