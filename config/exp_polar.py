"""Innovation #10: Polar-aware Training (球冠重点采样 & 极区 MoE).

- Latitude-weighted loss: upweight polar rows in loss computation
- Polar MoE: 2-expert (equatorial/polar) per encoder level, gated by row coordinate
"""
from config.diffusion_config import *
USE_LATITUDE_WEIGHT = True
USE_POLAR_MOE = True
