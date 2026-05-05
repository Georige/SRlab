"""Pixel-space U-Net for 512×1024 diffusion with multi-scale ISHT injection."""

import torch
import torch.nn as nn
from model.blocks import TimeEmbedding, DownBlock, UpBlock, ResBlock, PolarMoE
from model.spherical import CircularConv2d, CoordEmbed, SphericalAttention


class PixelUNet(nn.Module):
    """U-Net for conditional pixel-space denoising.

    Input:  noisy [B, 3, H, W] + cond [B, C_cond, H, W] → concat → [B, 3+C_cond, H, W]
    Output: ε_pred [B, 3, H, W]

    Encoder: H→H/2→H/4→H/8,  Decoder: H/8→H/4→H/2→H  (skip connections).

    Optional innovations (all disabled by default):
      - Multi-scale ISHT injection (ms_proj)
      - Polar MoE: coordinate-gated dual-expert per encoder level (polar_moe)
      - Laplacian pyramid: auxiliary prediction heads at dec3/dec2 (use_laplacian)
      - Spherical UNet: circular conv (use_circular_conv), coord embedding
        (use_coord_embed), bottleneck spherical attention (use_spherical_attn)
    """

    def __init__(self, in_ch=3, cond_ch=6, base_ch=32, time_dim=256,
                 use_polar_moe=False, use_laplacian=False,
                 use_circular_conv=False,
                 use_coord_embed=False,
                 use_spherical_attn=False,
                 hr_size=(512, 1024)):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.use_polar_moe = use_polar_moe
        self.use_laplacian = use_laplacian
        self.use_circular_conv = use_circular_conv
        self.use_coord_embed = use_coord_embed
        self.use_spherical_attn = use_spherical_attn

        # CoordEmbed: spherical position encoding as extra input channels
        if use_coord_embed:
            H, W = hr_size
            self.coord_embed = CoordEmbed(H, W, num_freqs=4)
            total_in = in_ch + cond_ch + self.coord_embed.out_ch
        else:
            self.coord_embed = None
            total_in = in_ch + cond_ch

        # Encoder
        self.enc1 = DownBlock(total_in, base_ch, time_dim, use_circular=use_circular_conv)
        self.enc2 = DownBlock(base_ch, base_ch * 2, time_dim, use_circular=use_circular_conv)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4, time_dim, use_circular=use_circular_conv)

        # Bottleneck
        self.bottleneck = ResBlock(base_ch * 4, base_ch * 4, time_dim,
                                   use_circular=use_circular_conv)

        # Spherical attention after bottleneck
        if use_spherical_attn:
            self.spherical_attn = SphericalAttention(base_ch * 4)
        else:
            self.spherical_attn = None

        # Multi-scale ISHT condition projectors: 3ch → feature_ch (additive)
        self.ms_proj = nn.ModuleDict({
            'enc2': nn.Conv2d(3, base_ch, 1),          # 256×512 level
            'enc4': nn.Conv2d(3, base_ch * 2, 1),      # 128×256 level
            'enc8': nn.Conv2d(3, base_ch * 4, 1),      # 64×128 bottleneck
        })

        # Polar MoE: 2-expert gated by row coordinate per encoder level
        if use_polar_moe:
            self.polar_moe = nn.ModuleDict({
                'enc1': PolarMoE(base_ch),
                'enc2': PolarMoE(base_ch * 2),
            })
        else:
            self.polar_moe = None

        # Decoder
        self.dec3 = UpBlock(base_ch * 8, base_ch * 2, time_dim, skip_ch=base_ch * 4,
                            use_circular=use_circular_conv)
        self.dec2 = UpBlock(base_ch * 4, base_ch, time_dim, skip_ch=base_ch * 2,
                            use_circular=use_circular_conv)
        self.dec1 = UpBlock(base_ch * 2, base_ch, time_dim, skip_ch=base_ch,
                            use_circular=use_circular_conv)

        # Laplacian pyramid heads: predict noise at each decoder level (training only)
        if use_laplacian:
            self.head_L2 = nn.Conv2d(base_ch * 2, in_ch, 1)   # dec3 → L2 (128×256)
            self.head_L1 = nn.Conv2d(base_ch, in_ch, 1)       # dec2 → L1 (256×512)

        # Final conv
        if use_circular_conv:
            self.final = nn.Sequential(
                nn.GroupNorm(min(8, base_ch), base_ch),
                nn.SiLU(),
                CircularConv2d(base_ch, in_ch, 3),
            )
        else:
            self.final = nn.Sequential(
                nn.GroupNorm(min(8, base_ch), base_ch),
                nn.SiLU(),
                nn.Conv2d(base_ch, in_ch, 3, padding=1),
            )

    def forward(self, noisy, cond, t_norm, ms_isht=None):
        x = torch.cat([noisy, cond], dim=1)    # [B, in_ch+cond_ch, H, W]
        if self.use_coord_embed:
            x = torch.cat([x, self.coord_embed(x)], dim=1)
        t_emb = self.time_emb(t_norm)          # [B, time_dim]

        s1, x = self.enc1(x, t_emb)
        if ms_isht and 'enc2' in ms_isht:
            x = x + self.ms_proj['enc2'](ms_isht['enc2'])
        if self.use_polar_moe:
            x = self.polar_moe['enc1'](x)

        s2, x = self.enc2(x, t_emb)
        if ms_isht and 'enc4' in ms_isht:
            x = x + self.ms_proj['enc4'](ms_isht['enc4'])
        if self.use_polar_moe:
            x = self.polar_moe['enc2'](x)

        s3, x = self.enc3(x, t_emb)
        if ms_isht and 'enc8' in ms_isht:
            x = x + self.ms_proj['enc8'](ms_isht['enc8'])

        x = self.bottleneck(x, t_emb)
        if self.use_spherical_attn:
            x = self.spherical_attn(x)

        x = self.dec3(x, s3, t_emb)               # [B, base_ch*2, H/4, W/4]
        pred_L2 = self.head_L2(x) if self.use_laplacian else None

        x = self.dec2(x, s2, t_emb)               # [B, base_ch, H/2, W/2]
        pred_L1 = self.head_L1(x) if self.use_laplacian else None

        x = self.dec1(x, s1, t_emb)               # [B, base_ch, H, W]
        pred_full = self.final(x)

        if self.use_laplacian:
            return pred_full, pred_L1, pred_L2
        return pred_full
