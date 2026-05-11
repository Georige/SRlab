"""DiT-based focus refinement models for Phase 9.

Shared transformer building blocks + two complete architectures:

  B-1: MultiStepDiT — 5-step iterative refinement with FocusAttention + AdaLN.
       Input:  [B, 7, H, W] = concat(I_low_up, current, mask), t_norm ∈ [0,1]
       Output: [B, 3, H, W] = residual
       Drop-in replacement for FocusRefineUNet in iterative training loop.

  B-2: SingleStepDiT — one-shot refinement with Center-Bias Attention + multi-scale heads.
       Input:  [B, 4, H, W] = concat(I_low_up, center_prior)
       Output: {'center_64': [B,3,64,64], 'center_128': [B,3,128,128],
                'full': [B,3,H,W]}
       Single forward pass, trained with hierarchical center-weighted losses.

Key innovations (both routes):
  - Focus Attention: learnable spatial bias B in attention logits, modulated by step t.
  - Center-Bias Attention: prior heatmap injected as additive bias per token.
  - AdaLN: adaptive layer norm with step-conditioned scale/shift.
  - Multi-scale output heads (B-2 only): different DiT layers feed different region predictors.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared building blocks
# ============================================================

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional encoding → MLP (shared by time and position).

    Input: t [B] or [B, ...] in [0, 1] (or raw index values).
    Output: [B, dim].
    """

    def __init__(self, dim=256, max_period=10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.dim // 2
        freq = torch.exp(-math.log(self.max_period) *
                         torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.float().unsqueeze(-1) * freq.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]
        return self.mlp(emb)


class PatchEmbed2D(nn.Module):
    """Convert 2D image to token sequence via patch-wise convolution.

    Input:  [B, in_ch, H, W]
    Output: [B, L, dim]  where L = (H/patch) * (W/patch)
    """

    def __init__(self, in_ch, dim=512, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] → [B, dim, H/patch, W/patch]
        x = self.proj(x)
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, Hp*Wp, D]
        return x, (Hp, Wp)


class PositionalEncoding2D(nn.Module):
    """Fixed 2D sinusoidal position encoding, added to token sequence.

    Encodes H and W axes separately at different frequencies, concatenates.
    Output: [1, Hp*Wp, dim] ready to add to token embeddings.
    """

    def __init__(self, dim=512, max_H=64, max_W=128):
        super().__init__()
        half = dim // 4  # split equally for H-sin, H-cos, W-sin, W-cos
        assert dim % 4 == 0, f"dim ({dim}) must be divisible by 4"
        self.dim = dim
        self.half = half

        # Precompute for efficiency
        freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer('freq', freq)

    def forward(self, Hp, Wp, device):
        half = self.half
        # H axis
        y = torch.arange(Hp, dtype=torch.float32, device=device)
        y_emb = y.unsqueeze(-1) * self.freq.unsqueeze(0)  # [Hp, half]
        y_sin, y_cos = y_emb.sin(), y_emb.cos()            # [Hp, half] each

        # W axis
        x = torch.arange(Wp, dtype=torch.float32, device=device)
        x_emb = x.unsqueeze(-1) * self.freq.unsqueeze(0)  # [Wp, half]
        x_sin, x_cos = x_emb.sin(), x_emb.cos()            # [Wp, half] each

        # Build grid: for position (h,w), token_emb = concat(y_sin[h], y_cos[h], x_sin[w], x_cos[w])
        # Shape: [Hp, Wp, dim]
        pe = torch.zeros(Hp, Wp, self.dim, device=device)
        pe[:, :, 0:half]        = y_sin.unsqueeze(1)   # broadcast H→W
        pe[:, :, half:2*half]    = y_cos.unsqueeze(1)
        pe[:, :, 2*half:3*half]  = x_sin.unsqueeze(0)  # broadcast W→H
        pe[:, :, 3*half:4*half]  = x_cos.unsqueeze(0)

        return pe.view(1, Hp * Wp, self.dim)


class AdaLN(nn.Module):
    """Adaptive Layer Norm: LN → x = x * (1+scale) + shift.

    scale, shift are produced by an MLP from a conditioning vector (e.g., step embedding).
    Implements adaLN-Zero variant where the modulation is added to the residual path.
    """

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2),
        )

    def forward(self, x, cond):
        # x: [B, L, dim], cond: [B, cond_dim]
        scale, shift = self.scale_mlp(cond).chunk(2, dim=-1)  # [B, dim] each
        x = self.ln(x)
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FocusAttention(nn.Module):
    """Multi-head self-attention with learnable spatial bias for center focusing.

    The attention logits are modified as:
        attn = softmax(Q @ K^T / sqrt(d_k) + B)
    where B is a learnable relative-position bias, optionally modulated by step t.

    Without step modulation (B-2): B is a fixed bias per head that favors
    center-token interactions.
    With step modulation (B-1): B = B_base + t_modulated_bias, where different
    refinement steps learn different center-focus patterns.
    """

    def __init__(self, dim=512, num_heads=8, use_step_modulation=True):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_step_modulation = use_step_modulation

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Learnable spatial bias: parameterized as 2D relative position bias
        # For a grid of patches, we learn bias between all pairs of positions.
        # We use a simpler approach: learn bias based on spatial distance buckets.
        self.bias_num_buckets = 32
        self.bias_table = nn.Parameter(torch.zeros(num_heads, self.bias_num_buckets))

        if use_step_modulation:
            # Step-conditioned modulation of the bias strength
            self.step_scale = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, num_heads),
            )

    def _compute_spatial_bias(self, Hp, Wp, device, step_emb=None):
        """Compute attention bias [1, heads, L, L] from spatial positions.

        Uses Euclidean distance between patch centers, bucketed.
        """
        L = Hp * Wp
        # Create coordinate grid
        y = torch.arange(Hp, dtype=torch.float32, device=device)
        x = torch.arange(Wp, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1)  # [L, 2]

        # Pairwise Euclidean distances
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [L, L, 2]
        dist = (diff ** 2).sum(-1).sqrt()                 # [L, L]

        # Bucket distances linearly
        max_dist = math.sqrt(Hp**2 + Wp**2)
        bucket_idx = (dist / (max_dist + 1e-8) *
                      (self.bias_num_buckets - 1)).long().clamp(0, self.bias_num_buckets - 1)

        bias = self.bias_table[:, bucket_idx]  # [heads, L, L]

        if self.use_step_modulation and step_emb is not None:
            scale = self.step_scale(step_emb).sigmoid()  # [B, heads]
            bias = bias.unsqueeze(0) * scale.unsqueeze(-1).unsqueeze(-1)

        return bias.unsqueeze(0) if step_emb is None else bias

    def forward(self, x, Hp, Wp, step_emb=None):
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention with bias
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, heads, L, L]

        # Add spatial bias
        spatial_bias = self._compute_spatial_bias(Hp, Wp, x.device, step_emb)
        attn = attn + spatial_bias

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, L, D)  # [B, L, D]

        return self.proj(x)


class CenterBiasAttention(nn.Module):
    """Multi-head attention with center-prior bias (for B-2 single-step).

    Uses a fixed/learnable center heatmap to bias attention towards center tokens.
    B = α * (center_prior_i + center_prior_j) where center_prior is a token-grid
    heatmap (center=1, edge=0).

    This is simpler than FocusAttention — the bias is computed from the input
    center_prior rather than learned position embeddings.
    """

    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Learnable strength for center bias
        self.alpha = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

    def forward(self, x, center_prior, Hp, Wp):
        """
        Args:
            x: [B, L, D]
            center_prior: [B, 1, Hp, Wp] heatmap (center=1, edge=0)
        """
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        # Center bias: B_ij = α * (p_i + p_j) where p is the center prior flattened
        prior_flat = center_prior.flatten(2)  # [B, 1, L]
        bias = prior_flat.unsqueeze(-1) + prior_flat.unsqueeze(-2)  # [B, 1, L, L]
        attn = attn + self.alpha * bias

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(x)


class DiTBlock(nn.Module):
    """Single DiT block: AdaLN → Attention → AdaLN → FFN.

    Uses AdaLN-Zero style: the modulation parameters control both the
    attention and FFN residual paths.
    """

    def __init__(self, dim=512, num_heads=8, mlp_ratio=4, cond_dim=512,
                 use_step_modulation=True, use_center_bias=False):
        super().__init__()
        self.use_center_bias = use_center_bias

        self.ln1 = AdaLN(dim, cond_dim)
        if use_center_bias:
            self.attn = CenterBiasAttention(dim, num_heads)
        else:
            self.attn = FocusAttention(dim, num_heads, use_step_modulation)

        self.ln2 = AdaLN(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, Hp, Wp, cond, center_prior=None, step_emb=None):
        # Attention with AdaLN
        if self.use_center_bias and center_prior is not None:
            x = x + self.attn(self.ln1(x, cond), center_prior, Hp, Wp)
        else:
            x = x + self.attn(self.ln1(x, cond), Hp, Wp, step_emb)

        # FFN with AdaLN
        x = x + self.mlp(self.ln2(x, cond))
        return x


class ConvDecoder(nn.Module):
    """Token sequence → 2D feature map → conv decoder → residual image.

    Takes DiT output tokens, reshapes to 2D, then applies conv layers to
    produce the final residual image at the input resolution.
    """

    def __init__(self, dim=512, out_ch=3, base_ch=128):
        super().__init__()
        self.base_ch = base_ch
        self.dim = dim

        # Project token dim → feature channels
        self.token_proj = nn.Conv2d(dim, base_ch * 2, 1)

        # Conv decoder: Hp,Wp → Hp*2,Wp*2 → Hp*4,Wp*4 → ... until target
        # We use a flexible approach: bilinear upsample + conv blocks
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
                nn.GroupNorm(min(8, base_ch), base_ch),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.Conv2d(base_ch, base_ch // 2, 3, padding=1),
                nn.GroupNorm(min(8, base_ch // 2), base_ch // 2),
                nn.SiLU(),
            ),
        ])
        self.final_conv = nn.Conv2d(base_ch // 2, out_ch, 3, padding=1)

    def forward(self, tokens, Hp, Wp, target_H, target_W):
        """
        Args:
            tokens: [B, L, dim]
            Hp, Wp: patch grid dimensions
            target_H, target_W: output image size
        Returns:
            [B, out_ch, target_H, target_W]
        """
        B = tokens.shape[0]
        # Reshape to 2D
        x = tokens.transpose(1, 2).reshape(B, self.dim, Hp, Wp)
        x = self.token_proj(x)  # [B, base_ch*2, Hp, Wp]

        for blk in self.dec_blocks:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = blk(x)

        # Final upsample to target resolution if needed
        if x.shape[2] != target_H or x.shape[3] != target_W:
            x = F.interpolate(x, size=(target_H, target_W),
                              mode='bilinear', align_corners=False)
        return self.final_conv(x)


# ============================================================
# B-1: Multi-Step Iterative Refinement DiT
# ============================================================

class MultiStepDiT(nn.Module):
    """B-1: Multi-step DiT with FocusAttention and AdaLN.

    Replaces U-Net in the Phase 9 iterative refinement pipeline.
    Each refinement step t ∈ {0..T-1} gets a unique attention bias pattern
    via step-conditioned FocusAttention.

    Architecture:
        Input [B, 7, H, W]  →  PatchEmbed (16×16)  →  [B, L, D]
        + PositionalEncoding2D
        + Step embedding → AdaLN conditioning
        × N DiTBlocks (FocusAttention + FFN)
        → ConvDecoder  →  [B, 3, H, W] residual

    Args:
        in_ch: input channels (7 = I_low_up(3) + current(3) + mask(1))
        dim: transformer hidden dimension
        depth: number of DiT blocks
        num_heads: attention heads
        patch_size: pixel size of each patch
        mlp_ratio: FFN hidden expansion ratio
    """

    def __init__(self, in_ch=7, dim=512, depth=8, num_heads=8,
                 patch_size=16, mlp_ratio=4):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size

        # Embedding
        self.patch_embed = PatchEmbed2D(in_ch, dim, patch_size)
        self.pos_embed = PositionalEncoding2D(dim)

        # Step embedding (cond_dim = dim)
        self.step_embed = SinusoidalEmbedding(dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     cond_dim=dim, use_step_modulation=True, use_center_bias=False)
            for _ in range(depth)
        ])

        # Decoder
        self.decoder = ConvDecoder(dim=dim, out_ch=3)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, t_norm):
        """
        Args:
            x: [B, 7, H, W] = concat(I_low_up, current, center_mask)
            t_norm: [B] normalized step index in [0, 1]
        Returns:
            residual: [B, 3, H, W]
        """
        B, _, H, W = x.shape

        # Patch embedding
        tokens, (Hp, Wp) = self.patch_embed(x)  # [B, L, dim]

        # Add positional encoding
        pe = self.pos_embed(Hp, Wp, x.device)
        tokens = tokens + pe

        # Step condition
        step_emb = self.step_embed(t_norm)  # [B, dim]

        # Transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens, Hp, Wp, cond=step_emb, step_emb=step_emb)

        # Decode to residual image
        residual = self.decoder(tokens, Hp, Wp, H, W)
        return residual


# ============================================================
# B-2: Single-Step Forward Focus DiT
# ============================================================

class MultiScaleHead(nn.Module):
    """Extract token features → 2D → predict residual for a specific region size.

    Used by SingleStepDiT to produce region-specific residual predictions
    from intermediate DiT layer outputs.
    """

    def __init__(self, dim=512, region_h=64, region_w=64, out_ch=3, base_ch=64):
        super().__init__()
        self.region_h = region_h
        self.region_w = region_w
        self.dim = dim

        # Feature decoupling: select channels relevant to this region
        self.decouple = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
        )

        # Conv decoder head
        self.conv = nn.Sequential(
            nn.Conv2d(dim, base_ch * 2, 3, padding=1),
            nn.GroupNorm(min(8, base_ch * 2), base_ch * 2),
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, out_ch, 3, padding=1),
        )

    def forward(self, tokens, Hp, Wp):
        """tokens: [B, L, dim] → residual: [B, 3, region_h, region_w]"""
        B = tokens.shape[0]
        x = self.decouple(tokens)
        x = x.transpose(1, 2).reshape(B, self.dim, Hp, Wp)
        x = F.interpolate(x, size=(self.region_h, self.region_w),
                          mode='bilinear', align_corners=False)
        return self.conv(x)


class SingleStepDiT(nn.Module):
    """B-2: Single-step DiT with Center-Bias Attention and multi-scale heads.

    One forward pass produces residuals for multiple center regions + full image.
    Internally simulates the multi-step focus process via:
      - Center-Bias Attention in every layer (soft bias towards center tokens)
      - Multi-scale heads at different depths (shallow→small center, deep→full)

    Input:  [B, 4, H, W] = concat(I_low_up, center_prior)
    Output: dict with keys 'center_64', 'center_128', 'full'
            (region sizes configurable via head_regions param)

    Args:
        in_ch: input channels (4 = I_low_up(3) + center_prior(1))
        dim: transformer hidden dimension
        depth: number of DiT blocks
        num_heads: attention heads
        patch_size: pixel size of each patch
        head_depths: which layer indices feed which heads (e.g., {2:'c64', 4:'c128', 7:'full'})
    """

    def __init__(self, in_ch=4, dim=512, depth=8, num_heads=8,
                 patch_size=16, mlp_ratio=4,
                 head_specs=None):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.depth = depth

        # Patch embedding
        self.patch_embed = PatchEmbed2D(in_ch, dim, patch_size)
        self.pos_embed = PositionalEncoding2D(dim)

        # Fixed condition: a learnable embedding instead of step embedding
        self.cond_embed = nn.Parameter(torch.zeros(1, 1, dim))

        # Transformer blocks with Center-Bias Attention
        self.blocks = nn.ModuleList([
            DiTBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     cond_dim=dim, use_step_modulation=False, use_center_bias=True)
            for _ in range(depth)
        ])

        # Multi-scale heads
        # Default: shallow layers → small centers, deep layers → full image
        if head_specs is None:
            head_specs = {
                2: ('c64', 64, 64),     # layer index → (name, region_H, region_W)
                4: ('c128', 128, 128),
                depth - 1: ('full', None, None),  # full image, from ConvDecoder
            }

        self.head_specs = head_specs
        self.heads = nn.ModuleDict()
        for layer_idx, (name, rh, rw) in head_specs.items():
            if rh is not None and rw is not None:
                self.heads[name] = MultiScaleHead(dim=dim, region_h=rh, region_w=rw,
                                                   out_ch=3)
            else:
                # Full-image head uses ConvDecoder
                self.heads[name] = ConvDecoder(dim=dim, out_ch=3)

    def forward(self, x):
        """
        Args:
            x: [B, 4, H, W] = concat(I_low_up, center_prior)
        Returns:
            dict of residual tensors, e.g.:
                {'c64': [B,3,64,64], 'c128': [B,3,128,128], 'full': [B,3,H,W]}
        """
        B, _, H, W = x.shape

        # Extract center_prior for attention bias
        center_prior = x[:, 3:4, :, :]  # [B, 1, H, W]
        # Downsample to token grid
        center_prior_tokens = F.adaptive_avg_pool2d(center_prior,
                                                     (H // self.patch_size, W // self.patch_size))

        # Patch embedding
        tokens, (Hp, Wp) = self.patch_embed(x)
        pe = self.pos_embed(Hp, Wp, x.device)
        tokens = tokens + pe

        # Condition (learned, not step-dependent)
        cond = self.cond_embed.expand(B, -1, -1).squeeze(1)  # [B, dim]

        # Run transformer, collecting intermediate outputs
        outputs = {}
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens, Hp, Wp, cond=cond, center_prior=center_prior_tokens)

            # Check if this layer feeds a head
            if i in self.head_specs:
                name, rh, rw = self.head_specs[i]
                head = self.heads[name]
                if isinstance(head, ConvDecoder):
                    outputs[name] = head(tokens, Hp, Wp, H, W)
                else:
                    outputs[name] = head(tokens, Hp, Wp)

        return outputs
