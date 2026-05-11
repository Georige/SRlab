"""MultiScaleCenterMaskUNet — Route A variant 1: center mask conditioning.

Extends MultiScaleConsistencyUNet by internally building a center mask and
concatenating it as an extra input channel. The model knows "where the center is"
and can learn to enhance the central region differently from the periphery.

Input:  I_low_up [B, 3, H, W]  (center mask built internally)
Output: dict with 'main' residual and 'aux' list of (name, pred, H, W)
"""

import torch
import torch.nn as nn

from model.multiscale_consistency import (
    ResBlockDirect, DownBlockDirect, UpBlockDirect,
)


class MultiScaleCenterMaskUNet(nn.Module):
    """U-Net with center mask as explicit positional cue (4-channel input).

    Same backbone as MultiScaleConsistencyUNet but the first encoder layer
    takes 4 channels instead of 3: [I_low_up (3), center_mask (1)].

    The center mask is a binary image with 1.0 in the central 1/4 area and 0
    elsewhere, built internally in forward().

    Args:
        in_ch:   input channels (default 3, I_low_up; +1 for mask = 4)
        out_ch:  output channels (default 3, residual)
        base_ch: base channel count (default 64)
    """

    def __init__(self, in_ch=3, out_ch=3, base_ch=64):
        super().__init__()
        bc = base_ch

        # Encoder — first layer takes in_ch+1 (extra center_mask channel)
        self.enc1 = DownBlockDirect(in_ch + 1, bc)      # HxW -> H/2xW/2
        self.enc2 = DownBlockDirect(bc, bc * 2)         # H/2xW/2 -> H/4xW/4
        self.enc3 = DownBlockDirect(bc * 2, bc * 4)     # H/4xW/4 -> H/8xW/8

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

    def forward(self, x):
        """Forward pass with center mask concatenated internally.

        Args:
            x: [B, 3, H, W] input tensor (I_low_up)

        Returns:
            dict: {'main': [B,3,H,W], 'aux': [(name, pred, H, W), ...]}
        """
        B, _, H, W = x.shape

        # Build center mask: 1 in central 1/4 area, 0 elsewhere
        mask = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        h_c, w_c = H // 4, W // 4
        sh, sw = (H - h_c) // 2, (W - w_c) // 2
        mask[:, :, sh:sh + h_c, sw:sw + w_c] = 1.0

        x = torch.cat([x, mask], dim=1)  # [B, 4, H, W]

        # Encoder
        s1, x = self.enc1(x)    # s1: [B, bc, H, W],        x: [B, bc, H/2, W/2]
        s2, x = self.enc2(x)    # s2: [B, bc*2, H/2, W/2],  x: [B, bc*2, H/4, W/4]
        s3, x = self.enc3(x)    # s3: [B, bc*4, H/4, W/4],  x: [B, bc*4, H/8, W/8]

        # Bottleneck
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
