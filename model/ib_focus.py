"""IB-FocusUNet — Information Bottleneck Focus UNet for iterative refinement.

Phase 10: Route B — spatial adaptive information retention at the bottleneck.
Instead of telling the model "look here", we force it to actively discard
irrelevant information at the bottleneck via a spatial retention map M(u,v).

Core mechanism:
    f' = f ⊙ M + mask_token ⊙ (1-M)
where M is a Gaussian heatmap whose spread shrinks with timestep t.

Input:  [B, 4, H, W] = concat(I_low_up, center_mask)
Output: [B, 3, H, W] = residual (added to input to produce refined image)
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

    Input: t_norm [B] in [0, 1].
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
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        return self.mlp(emb)


# ============================================================
# Gaussian heatmap generator for spatial retention map M
# ============================================================

def make_gaussian_heatmap(h, w, sigma=0.3):
    """Create a 2D Gaussian heatmap [1, 1, h, w] with center=1, edges ≈ 0.

    Args:
        h, w: spatial dimensions (typically bottleneck resolution, e.g., H/8 × W/8)
        sigma: spread in normalized coordinates (distance / half-size).
               0.3 → moderate spread, 0.05 → tight focus.

    Returns:
        [1, 1, h, w] tensor.
    """
    y = torch.arange(h, dtype=torch.float32)
    x = torch.arange(w, dtype=torch.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dy = (y.view(-1, 1) - cy) / (h / 2.0)
    dx = (x.view(1, -1) - cx) / (w / 2.0)
    dist_sq = dy ** 2 + dx ** 2
    heatmap = torch.exp(-dist_sq / (2.0 * sigma ** 2))
    return heatmap.view(1, 1, h, w)


# ============================================================
# Building blocks (reused from FocusRefine)
# ============================================================

class ResBlockFiLM(nn.Module):
    """Residual conv block with FiLM time injection.

    time_proj: time_dim → out_ch*2
    Injection: x = x * (1 + gamma) + beta
    """

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, out_ch * 2)

    def forward(self, x, t_emb):
        gamma, beta = self.time_proj(t_emb).chunk(2, dim=1)
        h = self.act(self.norm1(self.conv1(x)))
        h = h * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class DownBlockDirect(nn.Module):
    """ResBlockNoTime + stride-2 downsampling."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResBlockNoTime(in_ch, out_ch)
        self.pool = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        skip = self.res(x)
        return skip, self.pool(skip)


class UpBlockDirect(nn.Module):
    """Upsample + skip concat + ResBlock."""

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
# IB-Bottleneck — the core innovation
# ============================================================

class IBBottleneck(nn.Module):
    """Information Bottleneck with spatial adaptive retention map M.

    Flow:
        1. FiLM time modulation: x_mod = x * (1+γ) + β
        2. Generate M: Gaussian heatmap with σ = σ_max - (σ_max-σ_min) * t_norm
        3. IB operation: x_ib = x_mod ⊙ M + mask_token ⊙ (1-M)
        4. Residual: output = x + ConvBlock(x_ib)

    The mask_token is a learnable vector that represents "I don't know what
    should be here" — the model's generic fallback for edge regions.

    Information-theoretic meaning: mutual information I(f'; Y) is spatially
    constrained — the model must allocate its "information budget" to the center.

    Args:
        ch: channel count at bottleneck (base_ch * 4)
        time_dim: time embedding dimension
        bn_h, bn_w: bottleneck spatial resolution (H/8, W/8)
        sigma_max: initial Gaussian spread (early steps, wide aperture)
        sigma_min: final Gaussian spread (late steps, tight focus)
    """

    def __init__(self, ch, time_dim, bn_h, bn_w, sigma_max=0.3, sigma_min=0.05):
        super().__init__()
        self.ch = ch
        self.bn_h = bn_h
        self.bn_w = bn_w
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

        # FiLM time modulation
        self.time_embed = TimeEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, ch * 2)

        # Learnable mask token: what the model uses when M → 0 (edges)
        # One value per channel, broadcast over spatial dims
        self.mask_token = nn.Parameter(torch.zeros(1, ch, 1, 1))

        # Residual conv block applied after IB operation
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, ch), ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, ch), ch)
        self.act = nn.SiLU()

        # Cached M for sparsity loss retrieval
        self._M = None  # [1, 1, bn_h, bn_w]

    def _get_M(self, t_norm):
        """Generate spatial retention map M at the current timestep.

        sigma shrinks linearly: σ = σ_max - (σ_max - σ_min) * t_norm

        At t=0: M is wide (most features pass through).
        At t=T-1: M is tight (only center features pass; edges use mask_token).
        """
        if t_norm.dim() == 0:
            t_norm = t_norm.unsqueeze(0)
        t_val = t_norm[0].item()
        sigma = self.sigma_max - (self.sigma_max - self.sigma_min) * t_val
        sigma = max(sigma, self.sigma_min)  # clamp
        return make_gaussian_heatmap(self.bn_h, self.bn_w, sigma)

    def forward(self, x, t_norm):
        """
        Args:
            x: [B, ch, bn_h, bn_w] encoder output at bottleneck
            t_norm: [B] normalized timestep in [0, 1]

        Returns:
            x_out: [B, ch, bn_h, bn_w] bottleneck output
        """
        B = x.shape[0]

        # 1. FiLM time modulation
        t_emb = self.time_embed(t_norm)                       # [B, time_dim]
        gamma, beta = self.time_proj(t_emb).chunk(2, dim=1)    # [B, ch], [B, ch]
        x_mod = x * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) \
                + beta.unsqueeze(-1).unsqueeze(-1)

        # 2. Generate spatial retention map M
        M = self._get_M(t_norm).to(x.device)                  # [1, 1, bn_h, bn_w]
        self._M = M  # cache for sparsity loss

        # 3. IB operation: f' = f ⊙ M + mask_token ⊙ (1-M)
        x_ib = x_mod * M + self.mask_token * (1.0 - M)

        # 4. Residual conv block
        h = self.act(self.norm1(self.conv1(x_ib)))
        h = self.norm2(self.conv2(h))
        return self.act(h + x)

    def ib_sparsity_loss(self):
        """L1 penalty on (1-M) to encourage sparse retention at edges.

        Returns scalar loss (small, weighted externally).
        """
        if self._M is None:
            return torch.tensor(0.0)
        return (1.0 - self._M).mean()  # mean (1-M) across spatial dims


# ============================================================
# IB-FocusUNet
# ============================================================

class IBFocusUNet(nn.Module):
    """Iterative refinement U-Net with Information Bottleneck at the bottleneck.

    Architecture (4-level):
        enc1: in_ch → base_ch       (H → H/2)
        enc2: base_ch → base_ch*2   (H/2 → H/4)
        enc3: base_ch*2 → base_ch*4 (H/4 → H/8)
        bottleneck: IBBottleneck    (H/8) — FiLM + spatial retention M
        dec3: base_ch*8 → base_ch*2 (H/8 → H/4)
        dec2: base_ch*4 → base_ch   (H/4 → H/2)
        dec1: base_ch*2 → base_ch   (H/2 → H)
        final: base_ch → out_ch

    Args:
        in_ch: input channels (default 4 = I_low_up 3 + mask 1)
        base_ch: base channel count (default 64)
        out_ch: output channels (default 3 = residual)
        time_dim: time embedding dimension (default 128)
        hr_size: (H, W) of input images, used to size the heatmap
        sigma_max: initial Gaussian spread (default 0.3)
        sigma_min: final Gaussian spread (default 0.05)
    """

    def __init__(self, in_ch=4, base_ch=64, out_ch=3, time_dim=128,
                 hr_size=(512, 512), sigma_max=0.3, sigma_min=0.05):
        super().__init__()
        self.base_ch = base_ch

        bn_h, bn_w = hr_size[0] // 8, hr_size[1] // 8

        # Encoder (no time injection)
        self.enc1 = DownBlockDirect(in_ch, base_ch)
        self.enc2 = DownBlockDirect(base_ch, base_ch * 2)
        self.enc3 = DownBlockDirect(base_ch * 2, base_ch * 4)

        # IB-Bottleneck
        self.bottleneck = IBBottleneck(
            ch=base_ch * 4,
            time_dim=time_dim,
            bn_h=bn_h,
            bn_w=bn_w,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
        )

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
            t_norm: [B] normalized timestep in [0, 1]

        Returns:
            [B, out_ch, H, W] residual
        """
        # Encoder
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)

        # IB-Bottleneck (FiLM + spatial retention)
        x = self.bottleneck(x, t_norm)

        # Decoder
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.final(x)

    def ib_sparsity_loss(self):
        """Retrieve IB sparsity loss from bottleneck.
        Returns scalar tensor (small, to be weighted externally).
        """
        if hasattr(self.bottleneck, 'ib_sparsity_loss'):
            return self.bottleneck.ib_sparsity_loss()
        return torch.tensor(0.0)
