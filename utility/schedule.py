"""Diffusion noise schedules."""

import math
import torch


def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Linear beta schedule (original DDPM)."""
    return torch.linspace(beta_start, beta_end, T)


def make_cosine_schedule(T, s=0.008):
    """Cosine schedule from 'Improved DDPM' (Nichol & Dhariwal, 2021).

    alpha_cumprod(t) = f(t) / f(0)
    where f(t) = cos((t/T + s) / (1 + s) * pi/2)^2

    Key property: alpha_cumprod stays higher at large t vs linear schedule,
    giving the model better SNR for noise prediction at high noise levels.
    At t=999 (T=1000): alpha ~ 0.02-0.03 vs linear ~ 0.006 → 3-5x larger.
    """
    t = torch.arange(T + 1, dtype=torch.float64)
    # alpha_cumprod at each t (including t=0)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_cumprod = f / f[0]
    # Clip to prevent beta_t at boundaries
    alpha_cumprod = torch.clamp(alpha_cumprod, 0.001, 0.999)
    # beta_t = 1 - alpha_cumprod[t] / alpha_cumprod[t-1]
    alphas = alpha_cumprod[1:] / alpha_cumprod[:-1]
    alphas = torch.clamp(alphas, 0.001, 0.999)
    betas = 1.0 - alphas
    return betas.float()
