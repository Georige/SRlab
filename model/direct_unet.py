"""Direct U-Net for one-step panorama super-resolution (no diffusion).

Architecture mirrors PixelUNet but removes time embedding and time injection.
Input: cond [B, C_cond, H, W] → Output: HR or residual [B, 3, H, W].

Optional panoramic modules (all disabled by default):
  - Multi-scale ISHT injection (ms_proj): add/film/scaled_add
  - PolarMoE: coordinate-gated dual-expert per encoder level
  - CoordEmbed: spherical Fourier features as extra input channels
  - CircularConv: circular W-padding for panoramic wrap-around
  - SphericalAttention: axial self-attention at bottleneck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blocks import PolarMoE, _make_conv
from model.spherical import CircularConv2d, CoordEmbed, SphericalAttention


class ResBlockDirect(nn.Module):
    """Residual conv block WITHOUT time injection (for direct regression)."""

    def __init__(self, in_ch, out_ch, use_circular=False):
        super().__init__()
        self.conv1 = _make_conv(in_ch, out_ch, 3, use_circular=use_circular)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = _make_conv(out_ch, out_ch, 3, use_circular=use_circular)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class DownBlockDirect(nn.Module):
    """ResBlockDirect + stride-2 downsampling (returns skip and pooled)."""

    def __init__(self, in_ch, out_ch, use_circular=False):
        super().__init__()
        self.res = ResBlockDirect(in_ch, out_ch, use_circular=use_circular)
        self.pool = _make_conv(out_ch, out_ch, 3, stride=2, use_circular=use_circular)

    def forward(self, x):
        skip = self.res(x)
        return skip, self.pool(skip)


class UpBlockDirect(nn.Module):
    """Upsample + skip-concat + ResBlockDirect."""

    def __init__(self, in_ch, out_ch, skip_ch, use_circular=False):
        super().__init__()
        x_ch = in_ch - skip_ch
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            _make_conv(x_ch, x_ch, 3, use_circular=use_circular),
        )
        self.res = ResBlockDirect(in_ch, out_ch, use_circular=use_circular)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


class DirectUNet(nn.Module):
    """U-Net for direct one-step panorama super-resolution.

    Input:  cond [B, C_cond, H, W]
    Output: prediction [B, 3, H, W] (HR or residual depending on target_mode)

    Encoder: H→H/2→H/4→H/8,  Decoder: H/8→H/4→H/2→H  (skip connections).

    Optional innovations (all disabled by default):
      - Multi-scale ISHT injection (ms_proj)
      - PolarMoE: coordinate-gated dual-expert per encoder level
      - CoordEmbed: spherical Fourier features as extra input channels
      - CircularConv: circular padding in W dimension
      - SphericalAttention: axial self-attention at bottleneck
    """

    def __init__(self, cond_ch=6, base_ch=32, latent_ch=0,
                 use_polar_moe=False,
                 use_circular_conv=False,
                 use_coord_embed=False,
                 use_spherical_attn=False,
                 hr_size=(512, 1024),
                 ms_injection='add',          # 'none', 'add', 'scaled_add', 'film'
                 ms_scale_init=1.0,
                 out_ch=3):
        super().__init__()
        self.latent_ch = latent_ch
        self.use_polar_moe = use_polar_moe
        self.use_circular_conv = use_circular_conv
        self.use_coord_embed = use_coord_embed
        self.use_spherical_attn = use_spherical_attn
        self.ms_injection = ms_injection

        # CoordEmbed: spherical position encoding as extra input channels
        if use_coord_embed:
            H, W = hr_size
            self.coord_embed = CoordEmbed(H, W, num_freqs=4)
            total_in = cond_ch + latent_ch + self.coord_embed.out_ch
        else:
            self.coord_embed = None
            total_in = cond_ch + latent_ch

        # Encoder
        self.enc1 = DownBlockDirect(total_in, base_ch, use_circular=use_circular_conv)
        self.enc2 = DownBlockDirect(base_ch, base_ch * 2, use_circular=use_circular_conv)
        self.enc3 = DownBlockDirect(base_ch * 2, base_ch * 4, use_circular=use_circular_conv)

        # Bottleneck
        self.bottleneck = ResBlockDirect(base_ch * 4, base_ch * 4,
                                         use_circular=use_circular_conv)

        # Spherical attention after bottleneck
        if use_spherical_attn:
            self.spherical_attn = SphericalAttention(base_ch * 4)
        else:
            self.spherical_attn = None

        # Multi-scale ISHT condition injection
        self.ms_proj = None
        self.ms_scales = None
        self.ms_film = None

        if ms_injection == 'none':
            pass
        elif ms_injection == 'film':
            self.ms_film = nn.ModuleDict({
                'enc2': nn.Conv2d(3, base_ch * 2, 1),
                'enc4': nn.Conv2d(3, base_ch * 2 * 2, 1),
                'enc8': nn.Conv2d(3, base_ch * 4 * 2, 1),
            })
        else:
            self.ms_proj = nn.ModuleDict({
                'enc2': nn.Conv2d(3, base_ch, 1),
                'enc4': nn.Conv2d(3, base_ch * 2, 1),
                'enc8': nn.Conv2d(3, base_ch * 4, 1),
            })
            if ms_injection == 'scaled_add':
                self.ms_scales = nn.ParameterDict({
                    'enc2': nn.Parameter(torch.tensor(ms_scale_init)),
                    'enc4': nn.Parameter(torch.tensor(ms_scale_init)),
                    'enc8': nn.Parameter(torch.tensor(ms_scale_init)),
                })

        # Polar MoE
        if use_polar_moe:
            self.polar_moe = nn.ModuleDict({
                'enc1': PolarMoE(base_ch),
                'enc2': PolarMoE(base_ch * 2),
            })
        else:
            self.polar_moe = None

        # Decoder
        self.dec3 = UpBlockDirect(base_ch * 8, base_ch * 2, skip_ch=base_ch * 4,
                                  use_circular=use_circular_conv)
        self.dec2 = UpBlockDirect(base_ch * 4, base_ch, skip_ch=base_ch * 2,
                                  use_circular=use_circular_conv)
        self.dec1 = UpBlockDirect(base_ch * 2, base_ch, skip_ch=base_ch,
                                  use_circular=use_circular_conv)

        # Final conv
        if use_circular_conv:
            self.final = nn.Sequential(
                nn.GroupNorm(min(8, base_ch), base_ch),
                nn.SiLU(),
                CircularConv2d(base_ch, out_ch, 3),
            )
        else:
            self.final = nn.Sequential(
                nn.GroupNorm(min(8, base_ch), base_ch),
                nn.SiLU(),
                nn.Conv2d(base_ch, out_ch, 3, padding=1),
            )

    def _inject_ms(self, x, ms_isht, key):
        """Apply multi-scale ISHT injection at a given encoder level."""
        if self.ms_injection == 'none' or not ms_isht or key not in ms_isht:
            return x

        isht_feat = ms_isht[key]

        if self.ms_injection == 'film':
            film_out = self.ms_film[key](isht_feat)
            gamma, beta = film_out.chunk(2, dim=1)
            return gamma * x + beta
        else:
            proj = self.ms_proj[key](isht_feat)
            if self.ms_injection == 'scaled_add':
                proj = self.ms_scales[key] * proj
            return x + proj

    def forward(self, cond, ms_isht=None, latent=None):
        if latent is not None:
            x = torch.cat([latent, cond], dim=1)
        else:
            x = cond
        if self.use_coord_embed:
            x = torch.cat([x, self.coord_embed(x)], dim=1)

        s1, x = self.enc1(x)
        x = self._inject_ms(x, ms_isht, 'enc2')
        if self.use_polar_moe:
            x = self.polar_moe['enc1'](x)

        s2, x = self.enc2(x)
        x = self._inject_ms(x, ms_isht, 'enc4')
        if self.use_polar_moe:
            x = self.polar_moe['enc2'](x)

        s3, x = self.enc3(x)
        x = self._inject_ms(x, ms_isht, 'enc8')

        x = self.bottleneck(x)
        if self.use_spherical_attn:
            x = self.spherical_attn(x)

        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.final(x)

    def load_balance_loss(self):
        """Sum of CV² penalties across all PolarMoE modules."""
        if self.polar_moe is None:
            return 0.0
        return sum(moe.load_balance_loss() for moe in self.polar_moe.values())
