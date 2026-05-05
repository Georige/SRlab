"""Configuration for pixel-space diffusion with ISHT conditioning."""

# ---- Paths ----
DATA_DIR = "lau_dataset/sun_test"
OUTPUT_DIR = "phase2_output"
WEIGHT_DIR = "weight"

# ---- Image / SHT ----
HR_SIZE = (512, 1024)
L_COND = 255              # SHT bandwidth for main structural conditioning
SCALE = 4

# Multi-scale ISHT conditions: (L_bandwidth, spatial_factor)
# factor 2=256×512, 4=128×256, 8=64×128 (relative to 512×1024)
MS_COND = [
    (128, 2),   # L=128 ISHT → enc2 level
    (64,  4),   # L=64  ISHT → enc3 level
    (32,  8),   # L=32  ISHT → bottleneck level
]

# ---- Training ----
BATCH_SIZE = 6            # 512×1024, ~3.5GB per sample on 24GB GPU
EPOCHS = 400
LR_BASE = 2e-4

# ---- Diffusion ----
TRAIN_TIMESTEPS = 1000
INFER_STEPS = 50

# ---- Innovation: Frequency Residual Conditioning ----
# cond gets bicubic↑ - ISHT(L_HF) as extra 3ch, telling U-Net which
# frequencies are "rough bicubic artifacts" vs "reliable ISHT structure".
USE_HF_RESIDUAL = False
L_HF = 64

# ---- Innovation: Polar-aware Training (球冠重点采样 & 极区 MoE) ----
USE_LATITUDE_WEIGHT = False   # row-weighted loss for equirectangular compensation
POLE_WEIGHT = 1.0             # 0=uniform, 1.0=moderate, 2.0=strong
USE_POLAR_MOE = False         # 2-expert (equatorial/polar) per encoder, row-gated

# ---- Innovation: Laplacian Pyramid Multi-Scale Supervision ----
# Builds Laplacian pyramid from HR-ISHT residual, predicts noise at
# three decoder levels. Inference unchanged (uses final output only).
USE_LAPLACIAN_PYRAMID = False
LP_LAMBDAS = (1.0, 0.3, 0.1)   # (L0_finest, L1_medium, L2_coarsest) weights

# ---- Innovation: Spherical UNet (球面感知全景超分) ----
# Three independent switches for spherical-aware processing:
#   USE_CIRCULAR_CONV:  circular padding in width (left/right boundary connected)
#   USE_COORD_EMBED:    spherical coord Fourier features as extra input channels
#   USE_SPHERICAL_ATTN: multi-head self-attention at bottleneck for long-range
USE_CIRCULAR_CONV = False
USE_COORD_EMBED = False
USE_SPHERICAL_ATTN = False

# ---- Model ----
BASE_CH = 32             # U-Net base channels
TIME_DIM = 256
DROPOUT = 0.0
