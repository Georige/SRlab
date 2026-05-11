"""Model definitions for panorama super-resolution.

Active:
  - DirectUNet (direct_unet.py): main architecture, no time embedding
  - Building blocks (blocks.py): ResBlock, DownBlock, UpBlock, PolarMoE
  - Spherical modules (spherical.py): CircularConv2d, CoordEmbed, SphericalAttention
  - Losses (losses.py): VGGLoss, PatchGANDiscriminator, TVLoss, CompositeLoss

Legacy (diffusion, Phase 3-5):
  - PixelUNet (unet.py)
  - PixelDiffusion (diffusion.py)
"""

from model.direct_unet import DirectUNet
from model.blocks import ResBlock, DownBlock, UpBlock, PolarMoE
from model.spherical import CircularConv2d, CoordEmbed, SphericalAttention
from model.ib_focus import IBFocusUNet
from model.lora import LoRAConv2d, inject_lora_to_direct_unet, get_lora_params, count_lora_params
