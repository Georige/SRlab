"""SSMFocusUNet — U-Net with Mamba (S6) state-space bottleneck for iterative refinement.

Route 3: State Space Focus Propagation.
Replaces the bottleneck ResBlocks with a SSM-Focus module that learns to "focus"
its internal state on the growing central region.

Pure PyTorch implementation — no mamba_ssm dependency.
Uses Hillis-Steele parallel associative scan for O(log L) SSM recurrence.

Input:  [B, 7, H, W] = concat(I_low_up, current_state, mask)
        + t_norm [B] float in [0, 1]
Output: [B, 3, H, W] = refined residual
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Parallel associative scan (Hillis-Steele)
# ============================================================

def associative_scan(a, b):
    """Hillis-Steele parallel inclusive prefix scan.

    Operator: (a1, b1) . (a2, b2) = (a1*a2, a1*b2 + b1)
    where (a1,b1) is the "left" (earlier) segment and (a2,b2) is "right" (later).

    Computes cumulative:
        h_0 = b_0
        h_t = a_t * h_{t-1} + b_t
    for all t in parallel (O(log L) steps).

    Args:
        a: [B, L, D, N]  decay factors per step
        b: [B, L, D, N]  input contributions per step (B * u after discretization)

    Returns:
        h: [B, L, D, N]  cumulative state after each step (inclusive scan of b)
    """
    B, L, D, N = a.shape
    orig_L = L

    # Pad to power of 2 for clean binary-tree traversal
    max_p2 = 1 << (L - 1).bit_length() if L > 1 else 1
    if L < max_p2:
        a = F.pad(a, (0, 0, 0, 0, 0, max_p2 - L), value=1.0)
        b = F.pad(b, (0, 0, 0, 0, 0, max_p2 - L), value=0.0)
        L = max_p2

    n_steps = L.bit_length() - 1

    for d in range(n_steps):
        stride = 1 << d

        a_prev = torch.roll(a, stride, dims=1)
        b_prev = torch.roll(b, stride, dims=1)

        # Only positions >= stride participate (first stride positions stay as-is)
        arange = torch.arange(L, device=a.device)
        mask = (arange >= stride).view(1, L, 1, 1)

        # (right, left) -> combine: a_new = a_right * a_left, b_new = a_right * b_left + b_right
        new_a = torch.where(mask, a * a_prev, a)
        new_b = torch.where(mask, a * b_prev + b, b)
        a, b = new_a, new_b

    return b[:, :orig_L]


# ============================================================
# Time embedding
# ============================================================

class TimeEmbedding(nn.Module):
    """Sinusoidal positional encoding -> 2-layer MLP.

    Input:  t_norm [B] in [0, 1]
    Output: [B, dim]
    """

    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t_norm):
        if t_norm.dim() == 0:
            t_norm = t_norm.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(0, half, dtype=torch.float32, device=t_norm.device) / half)
        args = t_norm.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.mlp(emb)


# ============================================================
# Mamba (S6) block — pure PyTorch
# ============================================================

class MambaBlock(nn.Module):
    """Single Mamba (S6) block with bidirectional SSM scan.

    Standard Mamba architecture:
        1. Input projection: x -> x_delta, z (gate branch)
        2. Causal Conv1d on x_delta -> SiLU
        3. SSM: delta = softplus(Linear(x_delta) + bias_delta)
                B, C = chunk(Linear(x_delta))
                h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
                y_t = C_t * h_t + D * x_t
        4. Output gate: y * SiLU(z)
        5. Linear projection back to d_model

    Diagnostic flags:
        use_fixed_delta: if True, delta is a learnable bias (not input-dependent)
        scan_mode: 'bidirectional', 'forward_only', 'vertical', 'four_direction'

    Args:
        d_model:   feature dimension
        d_state:   SSM state dimension (N in Mamba paper)
        d_conv:    conv1d kernel size
        expand:    inner dimension multiplier (d_inner = d_model * expand)
        use_fixed_delta:  disable input-dependent delta (diagnostic)
        scan_mode: scan direction variant (diagnostic)
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 use_fixed_delta=False, scan_mode='bidirectional'):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand)
        self.use_fixed_delta = use_fixed_delta
        self.scan_mode = scan_mode

        # Input projection: x -> x and z (gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal Conv1d
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=False,
        )

        # SSM: delta projection (only used if not fixed_delta)
        if not use_fixed_delta:
            self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        else:
            self.delta_bias = nn.Parameter(torch.zeros(self.d_inner))

        # SSM: B and C projection
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)

        # SSM: A (state matrix)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.02)

        # SSM: D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.act = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x, H=None, W=None):
        """Forward pass.

        Args:
            x: [B, L, d_model]  input sequence
            H, W: spatial dims (required for 'vertical'/'four_direction' scan modes)

        Returns:
            [B, L, d_model]
        """
        B, L, D = x.shape

        # 1. Input projection
        xz = self.in_proj(x)
        x_delta, z = xz.chunk(2, dim=-1)

        # 2. Causal Conv1d
        x_delta = x_delta.permute(0, 2, 1)
        x_delta = self.conv1d(x_delta)
        x_delta = x_delta[:, :, :L]
        x_delta = self.act(x_delta)
        x_delta = x_delta.permute(0, 2, 1)

        # 3. Delta: fixed or input-dependent
        if self.use_fixed_delta:
            delta = F.softplus(self.delta_bias)
            delta = delta.unsqueeze(0).unsqueeze(0).expand(B, L, -1)
        else:
            delta = F.softplus(self.dt_proj(x_delta))

        # B, C
        bc = self.x_proj(x_delta)
        B_vec, C_vec = bc.chunk(2, dim=-1)

        # A (negative for stability)
        A = -torch.exp(self.A_log.float())

        # 4. SSM scan — dispatch by mode
        y = self._run_scan(x_delta, delta, A, B_vec, C_vec, H, W)

        # 5. Output gate + projection
        y = y * self.act(z)
        y = self.out_proj(y)
        return y

    def _run_scan(self, u, delta, A, B_vec, C_vec, H, W):
        """Dispatch to scan variant based on self.scan_mode."""
        if self.scan_mode == 'bidirectional':
            y_fwd = self._ssm_scan(u, delta, A, B_vec, C_vec, forward=True)
            y_rev = self._ssm_scan(u, delta, A, B_vec, C_vec, forward=False)
            return y_fwd + y_rev
        elif self.scan_mode == 'forward_only':
            return self._ssm_scan(u, delta, A, B_vec, C_vec, forward=True)
        elif self.scan_mode == 'reverse_only':
            return self._ssm_scan(u, delta, A, B_vec, C_vec, forward=False)
        elif self.scan_mode == 'vertical':
            return self._vertical_scan(u, delta, A, B_vec, C_vec, H, W)
        elif self.scan_mode == 'four_direction':
            y_fwd = self._ssm_scan(u, delta, A, B_vec, C_vec, forward=True)
            y_rev = self._ssm_scan(u, delta, A, B_vec, C_vec, forward=False)
            y_vert = self._vertical_scan(u, delta, A, B_vec, C_vec, H, W)
            return y_fwd + y_rev + y_vert
        else:
            raise ValueError(f"Unknown scan_mode: {self.scan_mode}")

    def _vertical_scan(self, u, delta, A, B_vec, C_vec, H, W):
        """Bidirectional scan along columns (H direction) instead of rows."""
        B, L, D = u.shape
        N = self.d_state
        D_inner = self.d_inner

        # Rearrange from row-major [B, H*W, D] to column-major [B, W*H, D]
        def to_col_major(t, ch):
            t2 = t.reshape(B, H, W, ch).permute(0, 2, 1, 3)  # [B, W, H, ch]
            return t2.reshape(B, W * H, ch)

        def to_row_major(t, ch):
            t2 = t.reshape(B, W, H, ch).permute(0, 2, 1, 3)  # [B, H, W, ch]
            return t2.reshape(B, H * W, ch)

        u_col = to_col_major(u, D)
        delta_col = to_col_major(delta, D_inner)
        B_col = to_col_major(B_vec, N)
        C_col = to_col_major(C_vec, N)

        y_fwd = self._ssm_scan(u_col, delta_col, A, B_col, C_col, forward=True)
        y_rev = self._ssm_scan(u_col, delta_col, A, B_col, C_col, forward=False)

        return to_row_major(y_fwd + y_rev, D)

    def _ssm_scan(self, u, delta, A, B_vec, C_vec, forward=True):
        """Diagonal SSM scan in one direction.

        Recurrence: h_t = A_bar_t * h_{t-1} + B_bar_t * u_t
        Output:     y_t = C_t * h_t + D * u_t

        Uses parallel associative scan for O(log L) complexity.

        Args:
            u:       [B, L, d_inner]  input (same as x_delta)
            delta:   [B, L, d_inner]  discretization step
            A:       [d_inner, d_state]  base state matrix
            B_vec:   [B, L, d_state]  input projection
            C_vec:   [B, L, d_state]  output projection
            forward: if True, scan left->right; else right->left

        Returns:
            y: [B, L, d_inner]
        """
        B, L, d_inner = u.shape
        d_state = A.shape[1]

        # Discretize (zero-order hold, Euler approximation for B)
        # A_bar = exp(delta * A)    shape: [B, L, d_inner, d_state]
        A_bar = torch.exp(torch.einsum('bld,dn->bldn', delta, A))

        # B_bar = delta * B     (Euler: B_bar = B * delta)
        # u_bar = B_bar * u = delta.unsqueeze(-1) * B_vec * u.unsqueeze(-1)
        u_bar = torch.einsum('bld,bln,bld->bldn', delta, B_vec, u)

        if not forward:
            # Reverse direction: flip along sequence axis
            A_bar = A_bar.flip(1)
            u_bar = u_bar.flip(1)
            C_vec = C_vec.flip(1)

        # Parallel associative scan
        h = associative_scan(A_bar, u_bar)             # [B, L, d_inner, d_state]

        # Output: y_t = sum(C_t * h_t, dim=-1) + D * u_t
        y = torch.einsum('bldn,bln->bld', h, C_vec)    # [B, L, d_inner]
        y = y + u * self.D                              # skip connection

        if not forward:
            y = y.flip(1)

        return y


# ============================================================
# ConvBlock (shared with other models)
# ============================================================

class ConvBlock(nn.Module):
    """Conv2d -> GroupNorm -> SiLU, with optional stride for downsampling."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Bilinear upsample x2 + skip-concat + ConvBlock."""

    def __init__(self, in_ch, out_ch, skip_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ============================================================
# SSM-Focus Module (bottleneck replacement)
# ============================================================

class SSMFocusModule(nn.Module):
    """SSM-focused bottleneck: time modulation + spatial modulation + Mamba scan.

    Replaces the standard ResBlock bottleneck in the iterative refinement U-Net.

    Pipeline:
        1. Time FiLM:  gamma(t), beta(t) modulate bottleneck features
        2. Spatial:     downsample center_mask, concat, project back (if use_mask=True)
        3. Mamba:       S6 scan over flattened HxW sequence
        4. Residual:    output = input + conv_out(LayerNorm(mamba_output))

    Args:
        ch:        bottleneck channel count (base_ch * 8)
        time_dim:  time embedding dimension
        d_state:   SSM state dimension
        d_conv:    conv1d kernel size in Mamba
        expand:    Mamba inner dimension multiplier
        use_fixed_delta:  if True, delta is a fixed learnable bias (diagnostic)
        scan_mode: 'bidirectional', 'forward_only', 'vertical', 'four_direction'
        use_mask:  if False, skip spatial mask modulation (diagnostic)
    """

    def __init__(self, ch, time_dim=128, d_state=16, d_conv=4, expand=2,
                 use_fixed_delta=False, scan_mode='bidirectional', use_mask=True):
        super().__init__()
        self.ch = ch
        self.use_mask = use_mask

        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)

        # FiLM projection
        self.time_gamma = nn.Linear(time_dim, ch)
        self.time_beta = nn.Linear(time_dim, ch)

        # Spatial: concat mask then project back (only if use_mask)
        if use_mask:
            self.spatial_proj = nn.Conv2d(ch + 1, ch, 1)

        # Layer normalization before Mamba
        self.norm = nn.LayerNorm(ch)

        # Mamba block
        self.mamba = MambaBlock(d_model=ch, d_state=d_state,
                                d_conv=d_conv, expand=expand,
                                use_fixed_delta=use_fixed_delta,
                                scan_mode=scan_mode)

        # Output projection
        self.out_proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x, t_norm, mask):
        B, C, H, W = x.shape
        x_orig = x

        # 1. Time modulation (FiLM)
        t_emb = self.time_embed(t_norm)
        gamma = self.time_gamma(t_emb).view(B, C, 1, 1)
        beta = self.time_beta(t_emb).view(B, C, 1, 1)
        x = x * (1.0 + gamma) + beta

        # 2. Spatial modulation (optional)
        if self.use_mask:
            mask_ds = F.interpolate(mask, size=(H, W),
                                    mode='bilinear', align_corners=False)
            x = self.spatial_proj(torch.cat([x, mask_ds], dim=1))

        # 3. Mamba scan
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm(x_flat)
        x_ssm = self.mamba(x_norm, H=H, W=W)
        x_ssm = x_ssm.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # 4. Residual
        return x_orig + self.out_proj(x_ssm)


# ============================================================
# SSMFocusUNet — Full U-Net with SSM bottleneck
# ============================================================

class SSMFocusUNet(nn.Module):
    """4-level U-Net with SSM-Focus bottleneck for iterative refinement.

    Architecture (identical encoder/decoder to IterativeRefinementUNet):
        Encoder:  init_conv -> enc1(H->H/2) -> enc2(H/2->H/4)
                  -> enc3(H/4->H/8) -> enc4(H/8->H/16)
        Bottleneck: SSMFocusModule (time+spatial modulation + Mamba scan)
        Decoder:  dec4(H/16->H/8) -> dec3(H/8->H/4)
                  -> dec2(H/4->H/2) -> dec1(H/2->H) -> out_conv

    Interface matches IterativeRefinementUNet exactly:
        forward(x, t_norm) where x is [B, 7, H, W] and t_norm is [B] in [0,1]

    The mask (last channel of input) is extracted and passed to the bottleneck
    for spatial modulation.

    Args:
        in_ch:    input channels (default 7: I_low_up(3)+current_state(3)+mask(1))
        out_ch:   output channels (default 3, the refined residual)
        base_ch:  base channel count (default 64)
        time_dim: time embedding dimension (default 128)
        d_state:  SSM state dimension (default 16)
        d_conv:   conv1d kernel size in Mamba (default 4)
        expand:   Mamba inner dimension multiplier (default 2)
    """

    def __init__(self, in_ch=7, out_ch=3, base_ch=64,
                 time_dim=128, d_state=16, d_conv=4, expand=2,
                 use_fixed_delta=False, scan_mode='bidirectional', use_mask=True):
        super().__init__()
        bc = base_ch

        # ---- Encoder ----
        self.init_conv = ConvBlock(in_ch, bc)
        self.enc1 = ConvBlock(bc,      bc * 2, stride=2)   # H   -> H/2
        self.enc2 = ConvBlock(bc * 2,  bc * 4, stride=2)   # H/2 -> H/4
        self.enc3 = ConvBlock(bc * 4,  bc * 8, stride=2)   # H/4 -> H/8
        self.enc4 = ConvBlock(bc * 8,  bc * 8, stride=2)   # H/8 -> H/16

        # ---- SSM-Focus bottleneck ----
        self.bottleneck = SSMFocusModule(
            ch=bc * 8, time_dim=time_dim,
            d_state=d_state, d_conv=d_conv, expand=expand,
            use_fixed_delta=use_fixed_delta, scan_mode=scan_mode,
            use_mask=use_mask,
        )

        # ---- Decoder ----
        self.dec4 = UpBlock(bc * 8, bc * 8, bc * 8)    # H/16 -> H/8  (skip: e3)
        self.dec3 = UpBlock(bc * 8, bc * 4, bc * 4)    # H/8  -> H/4  (skip: e2)
        self.dec2 = UpBlock(bc * 4, bc * 2, bc * 2)    # H/4  -> H/2  (skip: e1)
        self.dec1 = UpBlock(bc * 2, bc,      bc)       # H/2  -> H    (skip: x0)

        # ---- Output head ----
        self.out_conv = nn.Conv2d(bc, out_ch, 3, padding=1)

    def forward(self, x, t_norm):
        """Forward pass.

        Args:
            x:       [B, in_ch, H, W]  concat(I_low_up, current_state, mask)
            t_norm:  [B]  normalized timestep in [0, 1]

        Returns:
            [B, out_ch, H, W]  refined residual
        """
        # Extract mask from input (last channel)
        mask = x[:, -1:]  # [B, 1, H, W]

        # Encoder
        x0 = self.init_conv(x)          # [B, bc, H, W]
        e1 = self.enc1(x0)              # [B, bc*2, H/2, W/2]
        e2 = self.enc2(e1)              # [B, bc*4, H/4, W/4]
        e3 = self.enc3(e2)              # [B, bc*8, H/8, W/8]
        e4 = self.enc4(e3)              # [B, bc*8, H/16, W/16]

        # SSM-Focus bottleneck
        m = self.bottleneck(e4, t_norm, mask)  # [B, bc*8, H/16, W/16]

        # Decoder with skip connections
        d4 = self.dec4(m,  e3)          # [B, bc*8, H/8, W/8]
        d3 = self.dec3(d4, e2)          # [B, bc*4, H/4, W/4]
        d2 = self.dec2(d3, e1)          # [B, bc*2, H/2, W/2]
        d1 = self.dec1(d2, x0)          # [B, bc,   H, W]

        return self.out_conv(d1)
