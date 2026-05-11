"""MultiScaleCoordUNet — Route A variant 3: learnable coordinate embedding.

Adds a learnable spatial position encoding at the bottleneck, allowing the
model to learn different enhancement strategies for different spatial locations.

Input:  I_low_up [B, 3, H, W]          (no explicit mask, position is learned)
Output: dict with 'main' residual and 'aux' list of (name, pred, H, W)
"""

import torch
import torch.nn as nn

from model.multiscale_consistency import (
    ResBlockDirect, DownBlockDirect, UpBlockDirect,
)


class CoordEmbed(nn.Module):
    """Learnable 2D position encoding via row + column embeddings.

    The embedding is added to bottleneck features, giving the model spatial
    awareness without explicit mask input. Different spatial positions learn
    different feature offsets during training.
    """

    def __init__(self, ch, max_h=128, max_w=256):
        super().__init__()
        # H/8 = 128, W/8 = 256 for 1024x2048 input
        self.row_embed = nn.Parameter(torch.randn(1, ch, max_h, 1) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(1, ch, 1, max_w) * 0.02)

    def forward(self, x):
        """Add learned position encoding to features.

        Args:
            x: [B, C, H', W'] feature map
        Returns:
            [B, C, H', W'] position-encoded features
        """
        H, W = x.shape[2], x.shape[3]
        embed = self.row_embed[:, :, :H, :] + self.col_embed[:, :, :, :W]
        return x + embed


class MultiScaleCoordUNet(nn.Module):
    """U-Net with learnable coordinate embedding at the bottleneck.

    Same backbone as MultiScaleConsistencyUNet (3-ch input). A learnable
    2D position encoding is added at the bottleneck, giving the model
    spatial awareness through learned features rather than explicit masks.

    This is the lightest-weight variant — only ~50k extra parameters for
    the embedding table at H/8 x W/8 resolution.

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

        # Learnable coord embedding at bottleneck resolution (H/8 x W/8)
        self.coord_embed = CoordEmbed(bc * 4, max_h=128, max_w=256)

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
        """Forward pass with learned coordinate embedding at bottleneck.

        Args:
            x: [B, 3, H, W] input tensor (I_low_up)

        Returns:
            dict: {'main': [B,3,H,W], 'aux': [(name, pred, H, W), ...]}
        """
        _, _, H, W = x.shape

        # Encoder
        s1, x = self.enc1(x)    # s1: [B, bc, H, W],        x: [B, bc, H/2, W/2]
        s2, x = self.enc2(x)    # s2: [B, bc*2, H/2, W/2],  x: [B, bc*2, H/4, W/4]
        s3, x = self.enc3(x)    # s3: [B, bc*4, H/4, W/4],  x: [B, bc*4, H/8, W/8]

        # Bottleneck with learned coordinate embedding
        x = self.bottleneck(x)
        x = self.coord_embed(x)  # add spatial position encoding

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
