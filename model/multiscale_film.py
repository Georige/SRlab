"""MultiScaleFiLMUNet — Route A variant 2: multi-scale center mask injection.

At each encoder/decoder level, the center mask is downsampled to match the
feature map resolution and injected via FiLM (Feature-wise Linear Modulation).
This gives every U-Net level explicit spatial awareness of the center region.

Input:  I_low_up [B, 3, H, W]  (center mask built internally per level)
Output: dict with 'main' residual and 'aux' list of (name, pred, H, W)
"""

import torch
import torch.nn as nn

from model.multiscale_consistency import (
    ResBlockDirect, DownBlockDirect, UpBlockDirect,
)


class FiLMBlock(nn.Module):
    """FiLM modulation: uses a conditioning signal (center mask) to produce
    per-channel scale and shift parameters.

    output = x * (1 + scale) + shift
    """

    def __init__(self, ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch * 2, 3, padding=1),
        )

    def forward(self, x, mask):
        """Apply FiLM modulation.

        Args:
            x:    [B, C, H', W'] feature map
            mask: [B, 1, H', W'] downsampled center mask at same resolution
        Returns:
            [B, C, H', W'] modulated feature map
        """
        params = self.proj(mask)
        scale, shift = params.chunk(2, dim=1)
        return x * (1.0 + scale) + shift


class MultiScaleFiLMUNet(nn.Module):
    """U-Net with multi-scale center mask injection via FiLM.

    Same backbone as MultiScaleConsistencyUNet (3-ch input). At each encoder
    skip output and at the bottleneck, a downsampled center mask is injected
    via FiLM to provide explicit spatial position cues at every resolution.

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

        # FiLM blocks at each encoder skip + bottleneck
        # Skip resolutions: s1=H, s2=H/2, s3=H/4, bottleneck=H/8
        self.film_enc1 = FiLMBlock(bc)       # H    (full res skip)
        self.film_enc2 = FiLMBlock(bc * 2)   # H/2
        self.film_enc3 = FiLMBlock(bc * 4)   # H/4
        self.film_bn   = FiLMBlock(bc * 4)   # H/8 bottleneck

        # Bottleneck
        self.bottleneck = ResBlockDirect(bc * 4, bc * 4)

        # Decoder
        self.dec3 = UpBlockDirect(bc * 8, bc * 2, skip_ch=bc * 4)
        self.dec2 = UpBlockDirect(bc * 4, bc,     skip_ch=bc * 2)
        self.dec1 = UpBlockDirect(bc * 2, bc,     skip_ch=bc)

        # Auxiliary prediction heads
        self.head_dec3 = nn.Conv2d(bc * 2, out_ch, 3, padding=1)
        self.head_dec2 = nn.Conv2d(bc,     out_ch, 3, padding=1)
        self.head_dec1 = nn.Conv2d(bc,     out_ch, 3, padding=1)

        # Final output head
        self.head_final = nn.Sequential(
            nn.GroupNorm(min(8, bc), bc),
            nn.SiLU(),
            nn.Conv2d(bc, out_ch, 3, padding=1),
        )

    def _build_multiscale_masks(self, B, H, W, device, dtype):
        """Build center masks at H, H/2, H/4, H/8 resolutions.

        Returns:
            dict mapping 'h1', 'h2', 'h4', 'h8' -> [B, 1, h_i, w_i] mask tensor
        """
        masks = {}
        for scale, name in [(1, 'h1'), (2, 'h2'), (4, 'h4'), (8, 'h8')]:
            h, w = H // scale, W // scale
            h_c, w_c = h // 4, w // 4
            sh, sw = (h - h_c) // 2, (w - w_c) // 2
            m = torch.zeros(B, 1, h, w, device=device, dtype=dtype)
            m[:, :, sh:sh + h_c, sw:sw + w_c] = 1.0
            masks[name] = m
        return masks

    def forward(self, x):
        """Forward pass with multi-scale FiLM center mask injection.

        Args:
            x: [B, 3, H, W] input tensor (I_low_up)

        Returns:
            dict: {'main': [B,3,H,W], 'aux': [(name, pred, H, W), ...]}
        """
        B, _, H, W = x.shape
        masks = self._build_multiscale_masks(B, H, W, x.device, x.dtype)

        # Encoder with FiLM on skip outputs
        s1, x = self.enc1(x)                     # s1: [B, bc, H, W]   full res
        s1 = self.film_enc1(s1, masks['h1'])     #       FiLM at H

        s2, x = self.enc2(x)                     # s2: [B, bc*2, H/2, W/2]
        s2 = self.film_enc2(s2, masks['h2'])     #       FiLM at H/2

        s3, x = self.enc3(x)                     # s3: [B, bc*4, H/4, W/4]
        s3 = self.film_enc3(s3, masks['h4'])     #       FiLM at H/4

        # Bottleneck with FiLM
        x = self.film_bn(x, masks['h8'])         # FiLM before bottleneck
        x = self.bottleneck(x)

        # Decoder
        d3 = self.dec3(x, s3)   # [B, bc*2, H/4, W/4]
        d2 = self.dec2(d3, s2)  # [B, bc,   H/2, W/2]
        d1 = self.dec1(d2, s1)  # [B, bc,   H, W]

        # Auxiliary predictions
        aux = [
            ('dec3', self.head_dec3(d3), H // 4, W // 4),
            ('dec2', self.head_dec2(d2), H // 2, W // 2),
            ('dec1', self.head_dec1(d1), H,      W),
        ]

        main = self.head_final(d1)

        return {'main': main, 'aux': aux}
