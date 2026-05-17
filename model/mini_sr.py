"""MiniSRNet: lightweight U-Net for patch-based super-resolution.

Architecture (from miniSR.md):
  Input 512x512x3 (Image 0)
  -> Conv 3x3, 3->3, stride 1 -> ReLU -> Conv 3x3, 3->320, stride 1 -> Image 1
  -> ReLU -> Conv 3x3, 320->320, stride 1 + Image 1
  -> Conv 3x3, 320->640, stride 2 -> Image 2
  -> ReLU -> Conv 3x3, 640->640, stride 1 + Image 2
  -> Conv 3x3, 640->1280, stride 2 -> Image 3
  -> ReLU -> Conv 3x3, 1280->1280, stride 1 -> ReLU + Image 3
  -> Interp x2 -> Conv 3x3, 1280->640 -> ReLU -> Conv 3x3, 640->640 + Image 2
  -> Interp x2 -> Conv 3x3, 640->320 -> ReLU -> Conv 3x3, 320->320 + Image 1
  -> Conv 3x3, 320->3 + Image 0 -> Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)


class MiniSRNet(nn.Module):
    def __init__(self, zero_init=True):
        super().__init__()

        # --- Encoder Level 1 (512x512) ---
        self.enc1_pre = conv3x3(3, 3)        # 3->3 pre-conv
        self.enc1_expand = conv3x3(3, 320)    # 3->320 channel expansion
        self.enc1_post = conv3x3(320, 320)     # 320->320 + skip (Image 1)

        # --- Encoder Level 2 (256x256) ---
        self.enc2_down = conv3x3(320, 640, stride=2)   # 320->640, downsample
        self.enc2_post = conv3x3(640, 640)              # 640->640 + skip (Image 2)

        # --- Encoder Level 3 / Bottleneck (128x128) ---
        self.enc3_down = conv3x3(640, 1280, stride=2)  # 640->1280, downsample
        self.enc3_post = conv3x3(1280, 1280)            # 1280->1280

        # --- Decoder Level 2 (128x128 -> 256x256) ---
        self.dec2_up = conv3x3(1280, 640)   # 1280->640 after interpolate
        self.dec2_post = conv3x3(640, 640)   # 640->640 + skip (Image 2)

        # --- Decoder Level 1 (256x256 -> 512x512) ---
        self.dec1_up = conv3x3(640, 320)    # 640->320 after interpolate
        self.dec1_post = conv3x3(320, 320)   # 320->320 + skip (Image 1)

        # --- Output ---
        self.out_conv = conv3x3(320, 3)      # 320->3 + skip (Image 0)

        if zero_init:
            self._init_weights()

    def _init_weights(self):
        """Zero-residual initialization: zero only the output layer.

        With out_conv=0, the initial prediction is SR = 0 + Image0 = bicubic.
        Intermediate layers use Kaiming uniform for healthy gradient flow.
        """
        # Kaiming uniform for all intermediate convs
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and name != 'out_conv':
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Conv2d) and name == 'out_conv':
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        img0 = x  # [B, 3, H, W] — Image 0

        # ---- Encoder Level 1 ----
        x = F.relu(self.enc1_pre(x))         # [B, 3, H, W]
        x = F.relu(self.enc1_expand(x))      # [B, 320, H, W]
        img1 = x                              # Image 1
        x = self.enc1_post(F.relu(x)) + img1 # [B, 320, H, W]

        # ---- Encoder Level 2 ----
        x = F.relu(self.enc2_down(x))        # [B, 640, H/2, W/2]
        img2 = x                              # Image 2
        x = self.enc2_post(F.relu(x)) + img2 # [B, 640, H/2, W/2]

        # ---- Encoder Level 3 (Bottleneck) ----
        x = F.relu(self.enc3_down(x))        # [B, 1280, H/4, W/4]
        img3 = x                              # Image 3
        x = self.enc3_post(F.relu(x))        # [B, 1280, H/4, W/4]
        x = F.relu(x) + img3                 # [B, 1280, H/4, W/4]

        # ---- Decoder Level 2 ----
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.dec2_up(x)                   # [B, 640, H/2, W/2]
        x = self.dec2_post(F.relu(x)) + img2 # [B, 640, H/2, W/2]

        # ---- Decoder Level 1 ----
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.dec1_up(x)                   # [B, 320, H, W]
        x = self.dec1_post(F.relu(x)) + img1 # [B, 320, H, W]

        # ---- Output ----
        x = self.out_conv(x) + img0           # [B, 3, H, W]

        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters())
