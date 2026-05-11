"""MultiScaleConsistencyUNet — DirectUNet with auxiliary decoder heads for
multi-scale consistency training.

Architecture mirrors DirectUNet but drops all panoramic modules and
attaches auxiliary prediction heads at each decoder level. During training,
these heads predict central crops at their native resolution, providing
multi-scale supervision without autoregressive inference.

Input:  I_low_up [B, 3, H, W]         (bicubic-upsampled LR)
Output: dict with 'main' residual and 'aux' list of (name, pred, H, W)
"""

import torch
import torch.nn as nn


# ============================================================
# Building blocks (self-contained, no circular imports)
# ============================================================

class ResBlockDirect(nn.Module):
    """Residual conv block WITHOUT time injection."""

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


class DownBlockDirect(nn.Module):
    """ResBlockDirect + stride-2 downsampling (returns skip and pooled)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResBlockDirect(in_ch, out_ch)
        self.pool = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        skip = self.res(x)
        return skip, self.pool(skip)


class UpBlockDirect(nn.Module):
    """Upsample + skip-concat + ResBlockDirect."""

    def __init__(self, in_ch, out_ch, skip_ch):
        super().__init__()
        x_ch = in_ch - skip_ch
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(x_ch, x_ch, 3, padding=1),
        )
        self.res = ResBlockDirect(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


# ============================================================
# Main model
# ============================================================

class MultiScaleConsistencyUNet(nn.Module):
    """U-Net with multi-scale auxiliary prediction heads at decoder outputs.

    Architecture:
      Encoder:   H->H/2->H/4->H/8  (3 DownBlockDirect)
      Bottleneck: ResBlockDirect at H/8
      Decoder:   H/8->H/4->H/2->H  (3 UpBlockDirect + skip connections)

    Multi-scale auxiliary heads (attached at decoder outputs):
      - head_dec3: after dec3 (H/4, W/4), Conv2d(base_ch*2 -> 3)
      - head_dec2: after dec2 (H/2, W/2), Conv2d(base_ch -> 3)
      - head_dec1: after dec1 (H, W),     Conv2d(base_ch -> 3)
      - head_final: final output,           Conv2d(base_ch -> 3)

    Args:
        in_ch:   input channels (default 3, I_low_up)
        out_ch:  output channels (default 3, residual)
        base_ch: base channel count (default 64)
    """

    def __init__(self, in_ch=3, out_ch=3, base_ch=64):
        super().__init__()
        bc = base_ch

        # Encoder
        self.enc1 = DownBlockDirect(in_ch, bc)       # HxW -> H/2xW/2
        self.enc2 = DownBlockDirect(bc, bc * 2)      # H/2xW/2 -> H/4xW/4
        self.enc3 = DownBlockDirect(bc * 2, bc * 4)  # H/4xW/4 -> H/8xW/8

        # Bottleneck
        self.bottleneck = ResBlockDirect(bc * 4, bc * 4)

        # Decoder
        self.dec3 = UpBlockDirect(bc * 8, bc * 2, skip_ch=bc * 4)  # H/8->H/4
        self.dec2 = UpBlockDirect(bc * 4, bc,     skip_ch=bc * 2)  # H/4->H/2
        self.dec1 = UpBlockDirect(bc * 2, bc,     skip_ch=bc)      # H/2->H

        # Auxiliary prediction heads
        self.head_dec3 = nn.Conv2d(bc * 2, out_ch, 3, padding=1)   # H/4xW/4
        self.head_dec2 = nn.Conv2d(bc,     out_ch, 3, padding=1)   # H/2xW/2
        self.head_dec1 = nn.Conv2d(bc,     out_ch, 3, padding=1)   # HxW

        # Final output head (used at inference)
        self.head_final = nn.Sequential(
            nn.GroupNorm(min(8, bc), bc),
            nn.SiLU(),
            nn.Conv2d(bc, out_ch, 3, padding=1),
        )

    def forward(self, x):
        """Forward pass returning main output and auxiliary predictions.

        Args:
            x: [B, in_ch, H, W] input tensor (I_low_up)

        Returns:
            dict with keys:
              'main': [B, out_ch, H, W] final prediction (residual)
              'aux':  list of (name, prediction, native_H, native_W)
        """
        _, _, H, W = x.shape

        # Encoder
        s1, x = self.enc1(x)    # s1: [B, bc, H, W],        x: [B, bc, H/2, W/2]
        s2, x = self.enc2(x)    # s2: [B, bc*2, H/2, W/2],  x: [B, bc*2, H/4, W/4]
        s3, x = self.enc3(x)    # s3: [B, bc*4, H/4, W/4],  x: [B, bc*4, H/8, W/8]

        # Bottleneck
        x = self.bottleneck(x)  # [B, bc*4, H/8, W/8]

        # Decoder with skip connections
        d3 = self.dec3(x, s3)   # [B, bc*2, H/4, W/4]
        d2 = self.dec2(d3, s2)  # [B, bc,   H/2, W/2]
        d1 = self.dec1(d2, s1)  # [B, bc,   H, W]

        # Auxiliary predictions at each decoder level
        aux = [
            ('dec3', self.head_dec3(d3), H // 4, W // 4),
            ('dec2', self.head_dec2(d2), H // 2, W // 2),
            ('dec1', self.head_dec1(d1), H,      W),
        ]

        # Main output
        main = self.head_final(d1)  # [B, out_ch, H, W]

        return {'main': main, 'aux': aux}
