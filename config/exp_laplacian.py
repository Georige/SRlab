"""Innovation #3: Laplacian Pyramid Multi-Scale Supervision.

Builds Laplacian pyramid from HR-ISHT residual and adds noise prediction
heads at dec3 (128x256) and dec2 (256x512), providing multi-scale
gradient signals. Inference unchanged.
"""
from config.diffusion_config import *
USE_LAPLACIAN_PYRAMID = True
