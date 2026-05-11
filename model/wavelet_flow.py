"""Wavelet Vector Field Estimator (W-VFE) for Flow Matching SR.

Architecture:
  - CondEncoder: lr_up [B,3,H,W] → stride-2 CNN → cond_feat [B,64,H/2,W/2]
  - Input: x_t [B,9,H/2,W/2] + LL_lr [B,3,H/2,W/2] + cond_feat [B,64,H/2,W/2]
    → concat [B,76,H/2,W/2] → in_proj → [B,base_ch,H/2,W/2]
  - Time embedding: Sinusoidal(256) → MLP → FiLM params per block
  - CondFiLM: global-pool cond_feat → MLP → FiLM params per block
    Combined: scale = time_scale + cond_scale, shift = time_shift + cond_shift
  - N × FiLMResBlock (base_ch, base_ch) with time+condition modulation
  - Output head: GroupNorm → SiLU → Conv2d(base_ch, 9, 3)

No U-Net encoder/decoder — DWT already halves spatial resolution,
so the network operates at H/2 × W/2 with flat residual blocks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Time Embedding
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding → MLP → FiLM params for all blocks."""

    def __init__(self, dim=256, n_blocks=8, base_ch=64):
        super().__init__()
        self.dim = dim
        self.n_blocks = n_blocks
        half = dim // 2

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, n_blocks * base_ch * 2),  # scale + shift per block
        )

        # Frequency bands
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half) / half)
        self.register_buffer('freqs', freqs)

    def forward(self, t):
        """t: [B] in [0, 1]. Returns: [B, n_blocks, base_ch, 2] (scale, shift)."""
        B = t.shape[0]
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
        out = self.mlp(emb)  # [B, n_blocks * base_ch * 2]
        out = out.view(B, self.n_blocks, -1, 2)  # [B, n_blocks, base_ch, 2]
        return out


# ============================================================
# FiLM Residual Block
# ============================================================

class FiLMResBlock(nn.Module):
    """Residual block with time-conditioned Feature-wise Linear Modulation.

    Structure:
      Conv3x3 → GroupNorm → SiLU → FiLM(scale, shift) → Conv3x3 → GroupNorm → +skip → SiLU

    FiLM modulates the features AFTER the first activation:
      h = h * (1 + scale) + shift
    The (1 + scale) form initializes to identity modulation at t=0.
    """

    def __init__(self, ch):
        super().__init__()
        ng = min(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(ng, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(ng, ch)
        self.act = nn.SiLU()

    def forward(self, x, scale, shift):
        """x: [B, C, H, W], scale/shift: [B, C]."""
        h = self.act(self.norm1(self.conv1(x)))
        # FiLM modulation: h = h * (1 + scale) + shift
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.norm2(self.conv2(h))
        return self.act(h + x)


# ============================================================
# Conditioning FiLM Generator
# ============================================================

class CondFiLM(nn.Module):
    """Generate per-block FiLM params from cond_feat (global-pooled condition).

    Pool → MLP → [n_blocks, base_ch, 2] scale+shift per block.
    Combined with time FiLM: total_scale = time_scale + cond_scale.
    """

    def __init__(self, cond_ch=64, n_blocks=8, base_ch=64):
        super().__init__()
        self.n_blocks = n_blocks
        self.base_ch = base_ch
        self.mlp = nn.Sequential(
            nn.Linear(cond_ch, cond_ch * 4),
            nn.SiLU(),
            nn.Linear(cond_ch * 4, n_blocks * base_ch * 2),
        )

    def forward(self, cond_feat):
        """cond_feat: [B, cond_ch, H, W] → global pool → FiLM params.
        Returns: [B, n_blocks, base_ch, 2] (scale, shift per block)."""
        B = cond_feat.shape[0]
        pooled = cond_feat.mean(dim=[-2, -1])  # [B, cond_ch]
        out = self.mlp(pooled)  # [B, n_blocks * base_ch * 2]
        return out.view(B, self.n_blocks, self.base_ch, 2)


# ============================================================
# Conditioning Encoder
# ============================================================

class CondEncoder(nn.Module):
    """Lightweight CNN that encodes lr_up into a feature map at DWT resolution.

    Input:  lr_up [B, 3, H, W]          (full-resolution bicubic-upsampled LR)
    Output: feat  [B, 64, H/2, W/2]     (same spatial size as DWT subbands)
    """

    def __init__(self, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, lr_up):
        return self.net(lr_up)


# ============================================================
# Wavelet Vector Field Estimator
# ============================================================

class WaveletVFE(nn.Module):
    """Wavelet Vector Field Estimator for Flow Matching SR.

    Predicts velocity v = dH/dt that flows from LR high-frequency subbands
    (H_lr) to HR high-frequency subbands (H_hr), conditioned on:
      - LL_lr: low-frequency subband (wavelet-domain structure anchor)
      - lr_up: full bicubic-upsampled LR image (pixel-space navigation map)

    The CondEncoder extracts a 64-channel feature map from lr_up at the
    same H/2×W/2 resolution as the DWT subbands. This gives the flow field
    a high-resolution "holographic navigation map" at every FiLMResBlock.

    Input:
      x_t:   [B, 9,  H/2, W/2]  current high-frequency state (LH+HL+HH)
      LL_lr: [B, 3,  H/2, W/2]  LR low-frequency condition
      lr_up: [B, 3,  H,   W  ]  full bicubic-upsampled LR (encoded internally)
      t:     [B]                 time in [0, 1]

    Output:
      v_pred: [B, 9, H/2, W/2]  predicted velocity vector field
    """

    def __init__(self, base_ch=64, n_blocks=8, time_dim=256, cond_ch=64):
        super().__init__()
        self.base_ch = base_ch
        self.n_blocks = n_blocks

        # Conditioning encoder: lr_up → 64-channel feature map at DWT resolution
        self.cond_encoder = CondEncoder(out_ch=cond_ch)

        # Input projection: 9 (x_t) + 3 (LL_lr) + cond_ch (cond_feat)
        total_in = 9 + 3 + cond_ch
        self.in_proj = nn.Conv2d(total_in, base_ch, 3, padding=1)

        # Time embedding → FiLM params per block
        self.time_emb = SinusoidalTimeEmbedding(
            dim=time_dim, n_blocks=n_blocks, base_ch=base_ch)

        # Condition FiLM: global-pooled cond_feat → per-block modulation
        self.cond_film = CondFiLM(cond_ch=cond_ch, n_blocks=n_blocks, base_ch=base_ch)

        # Stack of FiLM residual blocks (no down/up sampling)
        self.blocks = nn.ModuleList([
            FiLMResBlock(base_ch) for _ in range(n_blocks)
        ])

        # Output head: base_ch → 9 (LH+HL+HH velocity)
        ng = min(8, base_ch)
        self.out_head = nn.Sequential(
            nn.GroupNorm(ng, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 9, 3, padding=1),
        )

    def forward(self, x_t, LL_lr, lr_up, t):
        """Forward pass.

        Args:
            x_t:   [B, 9, H/2, W/2] current high-frequency state.
            LL_lr: [B, 3, H/2, W/2] LR low-frequency condition.
            lr_up: [B, 3, H,   W  ] full bicubic-upsampled LR.
            t:     [B] or [1]        time scalar in [0, 1].

        Returns:
            v_pred: [B, 9, H/2, W/2] predicted velocity.
        """
        # Encode lr_up into DWT-resolution feature map
        cond_feat = self.cond_encoder(lr_up)       # [B, cond_ch, H/2, W/2]

        # Concatenate state + wavelet condition + pixel-space navigation
        x = torch.cat([x_t, LL_lr, cond_feat], dim=1)  # [B, 9+3+cond_ch, H/2, W/2]
        x = self.in_proj(x)                              # [B, base_ch, H/2, W/2]

        # Time FiLM params: [B, n_blocks, base_ch, 2]
        time_film = self.time_emb(t)

        # Condition FiLM params: [B, n_blocks, base_ch, 2]
        cond_film = self.cond_film(cond_feat)

        # FiLM residual blocks (time + condition modulation combined)
        for i, block in enumerate(self.blocks):
            scale = time_film[:, i, :, 0] + cond_film[:, i, :, 0]  # [B, base_ch]
            shift = time_film[:, i, :, 1] + cond_film[:, i, :, 1]  # [B, base_ch]
            x = block(x, scale, shift)

        return self.out_head(x)


# ============================================================
# Wavelet Direct Super-Resolution
# ============================================================

class WaveletDirectSR(nn.Module):
    """Direct wavelet-domain super-resolution — no time, no flow, no x_t.

    Architecture:
      CondEncoder: lr_up [B,3,H,W] → cond_feat [B,64,H/2,W/2]
      Input: cat(LL_lr [3], cond_feat [64]) → in_proj(67→base_ch)
      CondFiLM: global-pool cond_feat → MLP → per-block scale+shift
      N × FiLMResBlock with cond-only FiLM modulation
      Output: GN→SiLU→Conv(base_ch, 9) → H_pred

    Inference: single forward pass → IDWT → SR image (no Euler steps).

    Input:
      LL_lr: [B, 3, H/2, W/2]  LR low-frequency (DWT of bicubic-upsampled LR).
      lr_up: [B, 3, H,   W  ]  full bicubic-upsampled LR (encoded internally).

    Output:
      H_pred: [B, 9, H/2, W/2]  predicted high-frequency subbands (LH+HL+HH).
    """

    def __init__(self, base_ch=64, n_blocks=8, cond_ch=64):
        super().__init__()
        self.base_ch = base_ch
        self.n_blocks = n_blocks

        # Conditioning encoder: lr_up → 64-channel feature map at DWT resolution
        self.cond_encoder = CondEncoder(out_ch=cond_ch)

        # Input projection: 3 (LL_lr) + cond_ch (cond_feat)
        total_in = 3 + cond_ch
        self.in_proj = nn.Conv2d(total_in, base_ch, 3, padding=1)

        # Condition FiLM: global-pooled cond_feat → per-block modulation
        self.cond_film = CondFiLM(cond_ch=cond_ch, n_blocks=n_blocks, base_ch=base_ch)

        # Stack of FiLM residual blocks (no down/up sampling)
        self.blocks = nn.ModuleList([
            FiLMResBlock(base_ch) for _ in range(n_blocks)
        ])

        # Output head: base_ch → 9 (LH+HL+HH prediction)
        ng = min(8, base_ch)
        self.out_head = nn.Sequential(
            nn.GroupNorm(ng, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 9, 3, padding=1),
        )

    def forward(self, LL_lr, lr_up):
        """Forward pass — single deterministic prediction.

        Args:
            LL_lr: [B, 3, H/2, W/2] LR low-frequency DWT subband.
            lr_up: [B, 3, H,   W  ] full bicubic-upsampled LR.

        Returns:
            H_pred: [B, 9, H/2, W/2] predicted high-frequency subbands.
        """
        # Encode lr_up into DWT-resolution feature map
        cond_feat = self.cond_encoder(lr_up)       # [B, cond_ch, H/2, W/2]

        # Concatenate wavelet condition + pixel-space navigation
        x = torch.cat([LL_lr, cond_feat], dim=1)   # [B, 3+cond_ch, H/2, W/2]
        x = self.in_proj(x)                          # [B, base_ch, H/2, W/2]

        # Condition FiLM params: [B, n_blocks, base_ch, 2]
        cond_film = self.cond_film(cond_feat)

        # FiLM residual blocks (condition-only modulation)
        for i, block in enumerate(self.blocks):
            scale = cond_film[:, i, :, 0]  # [B, base_ch]
            shift = cond_film[:, i, :, 1]  # [B, base_ch]
            x = block(x, scale, shift)

        return self.out_head(x)
