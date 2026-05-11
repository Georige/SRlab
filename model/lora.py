"""LoRA (Low-Rank Adaptation) wrapper for Conv2d layers.

Freeze original weights, train only low-rank A·B decomposition.
B initialized to zero so LoRA output is zero at start (identity pass).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv2d(nn.Module):
    """Wraps a Conv2d with a parallel low-rank branch: out = W(x) + s * B(A(x)).

    - Original weight W: frozen
    - Bias: frozen (or None)
    - A: (r, in_ch, 1, 1) — spatial 1×1 keeps it cheap
    - B: (out_ch, r, 1, 1) — zero-init
    - Scale s: α/r, where α is a fixed scaling factor

    To apply LoRA to an existing Conv2d:
        conv = nn.Conv2d(64, 64, 3, padding=1)
        lora_conv = LoRAConv2d.wrap(conv, r=4, alpha=1.0)
    """

    def __init__(self, conv, r=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.conv = conv  # frozen original
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        out_ch, in_ch = conv.out_channels, conv.in_channels
        device = conv.weight.device

        # LoRA matrices as 1×1 spatial convolutions
        self.lora_A = nn.Conv2d(in_ch, r, 1, bias=False).to(device)
        self.lora_B = nn.Conv2d(r, out_ch, 1, bias=False).to(device)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Init: A ~ Kaiming, B ~ zero
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original
        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.conv(x)
        lora = self.lora_B(self.dropout(self.lora_A(x))) * self.scale
        return base + lora

    @classmethod
    def wrap(cls, conv, r=4, alpha=1.0, dropout=0.0):
        return cls(conv, r=r, alpha=alpha, dropout=dropout)


def inject_lora_to_direct_unet(model, r=4, alpha=1.0):
    """Wrap all Conv2d inside ResBlockDirect with LoRA across the full U-Net hierarchy.

    Targets: enc1.res.{conv1,conv2}, enc2.res.{conv1,conv2}, enc3.res.{conv1,conv2},
             bottleneck.{conv1,conv2},
             dec3.res.{conv1,conv2}, dec2.res.{conv1,conv2}, dec1.res.{conv1,conv2}

    Skips: 1×1 skip convs, pool convs, up-sequence convs, ms_proj, PolarMoE experts,
           coord_embed, spherical_attn, final output conv.

    Returns (model, n_wrapped).
    """
    from model.direct_unet import ResBlockDirect

    n_wrapped = 0

    def _wrap_resblock(res_block):
        nonlocal n_wrapped
        for attr in ['conv1', 'conv2']:
            if hasattr(res_block, attr):
                conv = getattr(res_block, attr)
                if isinstance(conv, nn.Conv2d) and conv.kernel_size != (1, 1):
                    setattr(res_block, attr, LoRAConv2d.wrap(conv, r=r, alpha=alpha))
                    n_wrapped += 1

    # Traverse model hierarchy: downblocks, bottleneck, upblocks
    for blk_name in ['enc1', 'enc2', 'enc3']:
        if hasattr(model, blk_name):
            blk = getattr(model, blk_name)
            if hasattr(blk, 'res') and isinstance(blk.res, ResBlockDirect):
                _wrap_resblock(blk.res)

    if hasattr(model, 'bottleneck') and isinstance(model.bottleneck, ResBlockDirect):
        _wrap_resblock(model.bottleneck)

    for blk_name in ['dec3', 'dec2', 'dec1']:
        if hasattr(model, blk_name):
            blk = getattr(model, blk_name)
            if hasattr(blk, 'res') and isinstance(blk.res, ResBlockDirect):
                _wrap_resblock(blk.res)

    return model, n_wrapped


def get_lora_params(model):
    """Return only the LoRA parameters (lora_A, lora_B) that require grad."""
    params = []
    for n, p in model.named_parameters():
        if 'lora_' in n and p.requires_grad:
            params.append(p)
    return params


def count_lora_params(model):
    """Count trainable LoRA parameters."""
    return sum(p.numel() for p in get_lora_params(model))
