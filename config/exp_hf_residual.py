"""Innovation #1: Frequency Residual Conditioning.

Adds bicubic↑ - ISHT(L=64) as extra 3ch to condition, explicitly showing
the U-Net which frequencies in bicubic are rough/artifact-prone vs reliable.
"""
from config.diffusion_config import *
USE_HF_RESIDUAL = True
