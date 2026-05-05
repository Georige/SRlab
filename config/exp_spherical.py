"""Spherical UNet: all spherical-aware innovations enabled."""
from config.diffusion_config import *
USE_CIRCULAR_CONV = True
USE_COORD_EMBED = True
USE_SPHERICAL_ATTN = True
