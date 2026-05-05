"""Phase 4: Diagnostic experiments to identify why the model can't overfit.

Usage:
  python phase4_diagnosis.py -e 0                         # Exp 0: residual stats
  python phase4_diagnosis.py -e 1 -g 0                    # Exp 1: bicubic-only cond
  python phase4_diagnosis.py -e 2 -g 0                    # Exp 2: direct residual pred
  python phase4_diagnosis.py -e 3 -g 0                    # Exp 3: x0 prediction
  python phase4_diagnosis.py -e 4 -g 0                    # Exp 4: consistency reg
  python phase4_diagnosis.py -e 5 -g 7                    # Exp 5: HR-target diffusion
  python phase4_diagnosis.py -e 6 -g 7 -c exp_base        # Exp 6: HR-target, BASE_CH=32, 200 steps
  python phase4_diagnosis.py -e 7 -g 0                    # Exp 7: God-model DDIM validator
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch_harmonics import RealSHT, InverseRealSHT
from tqdm import tqdm

from utility.data import PanoramaDataset
from utility.schedule import make_beta_schedule
from model.unet import PixelUNet
from model.diffusion import PixelDiffusion


# ============================================================
# Shared setup
# ============================================================

H, W = 512, 1024
L_COND = 255
SCALE = 4
BASE_CH = 64          # increased from 32 for direct prediction tests
TIME_DIM = 256
T = 1000
INFER_STEPS = 50
LR = 2e-4
DATA_DIR = "lau_dataset/sun_test"
OUTPUT_DIR = "phase4_output"


def get_single_image(device):
    dataset = PanoramaDataset(DATA_DIR, (H, W), SCALE)
    lr_img, hr_img = dataset[0]
    return lr_img.unsqueeze(0).to(device), hr_img.unsqueeze(0).to(device)


def to_pil(t):
    a = t[0].cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip((a + 1) * 127.5, 0, 255).astype(np.uint8))


def get_sht_isht(device, L):
    sht = RealSHT(H, W, lmax=L, mmax=L).to(device)
    isht = InverseRealSHT(H, W, lmax=L, mmax=L).to(device)
    return sht, isht


# ============================================================
# Experiment 0: Residual numerical properties
# ============================================================

def exp0_residual_stats(gpu=0):
    """Analyze residual = HR - ISHT(SHT(bicubic(LR))) statistics."""
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    lr_img, hr_img = get_single_image(device)

    bicubic = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)
    sht, isht = get_sht_isht(device, L_COND)
    base = isht(sht(bicubic))

    residual = hr_img - base

    print("=" * 60)
    print("Experiment 0: Residual (HR - ISHT_base) Statistics")
    print("=" * 60)
    print(f"  HR      range: [{hr_img.min().item():.4f}, {hr_img.max().item():.4f}]")
    print(f"  Base    range: [{base.min().item():.4f}, {base.max().item():.4f}]")
    print(f"  Residual range: [{residual.min().item():.4f}, {residual.max().item():.4f}]")
    print(f"  Residual mean:  {residual.mean().item():.6f}")
    print(f"  Residual std:   {residual.std().item():.6f}")
    print(f"  Residual var:   {residual.var().item():.8f}")
    print(f"  |Residual| max:  {residual.abs().max().item():.6f}")

    # Per-channel stats
    for i, name in enumerate(['R', 'G', 'B']):
        ch = residual[0, i]
        print(f"  {name} channel: mean={ch.mean().item():.6f}, std={ch.std().item():.6f}, "
              f"var={ch.var().item():.8f}")

    # What SNR does the diffusion see?
    # At t, noisy = sqrt(alpha_cumprod) * residual + sqrt(1-alpha_cumprod) * noise
    betas = make_beta_schedule(T).to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sigma_T = torch.sqrt(1.0 - alphas_cumprod[-1])
    alpha_T = torch.sqrt(alphas_cumprod[-1])

    # SNR at various timesteps
    print(f"\n  Diffusion SNR analysis (signal=residual, noise~N(0,1)):")
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t_idx = min(int(frac * (T - 1)), T - 1)
        a = torch.sqrt(alphas_cumprod[t_idx])
        s = torch.sqrt(1.0 - alphas_cumprod[t_idx])
        signal_power = (a * residual).var().item()
        noise_power = s.item() ** 2  # Var[noise] = 1
        snr = signal_power / noise_power if noise_power > 0 else float('inf')
        snr_db = 10 * np.log10(snr) if snr > 0 else -float('inf')
        print(f"    t/T={frac:.2f} (t={t_idx}): alpha={a.item():.4f}, sigma={s.item():.4f}, "
              f"SNR={snr_db:.1f} dB")

    print(f"\n  Key insight: residual variance = {residual.var().item():.8f}")
    print(f"  This is {'<< 1 — severe information drowning' if residual.var().item() < 0.1 else 'comparable to noise'}")
    print("=" * 60)


# ============================================================
# Experiment 1: bicubic-only condition (remove base from cond)
# ============================================================

def exp1_bicubic_only(gpu=0, config_module="diffusion_config"):
    """Remove ISHT base from condition, use only bicubic as condition."""
    import importlib
    cfg = importlib.import_module(f"config.{config_module}")

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    lr_img, hr_img = get_single_image(device)

    exp_dir = os.path.join(OUTPUT_DIR, "exp1_bicubic_only")
    os.makedirs(exp_dir, exist_ok=True)

    # Baselines
    to_pil(hr_img).save(os.path.join(exp_dir, "hr.png"))
    lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)
    to_pil(lr_up).save(os.path.join(exp_dir, "bicubic.png"))

    # ISHT setup (still needed for base to compute target residual)
    sht_cond, isht_cond = get_sht_isht(device, L_COND)
    base = isht_cond(sht_cond(lr_up))
    to_pil(base).save(os.path.join(exp_dir, "base.png"))

    # Multi-scale ISHT (still used for injection, unchanged)
    ms_sht_isht = []
    for L_val, factor in cfg.MS_COND:
        sht = RealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        isht = InverseRealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        ms_sht_isht.append((sht, isht, factor))

    # ★ KEY CHANGE: cond = bicubic only (3ch instead of 6ch)
    cond_ch = 3  # was 6

    unet = PixelUNet(
        in_ch=3, cond_ch=cond_ch,
        base_ch=cfg.BASE_CH, time_dim=cfg.TIME_DIM,
        use_polar_moe=cfg.USE_POLAR_MOE,
        use_laplacian=cfg.USE_LAPLACIAN_PYRAMID,
        use_circular_conv=cfg.USE_CIRCULAR_CONV,
        use_coord_embed=cfg.USE_COORD_EMBED,
        use_spherical_attn=cfg.USE_SPHERICAL_ATTN,
        hr_size=cfg.HR_SIZE,
    ).to(device)

    # We need a custom diffusion that builds bicubic-only condition
    # Instead of reusing PixelDiffusion with its make_condition, we build
    # a minimal training loop inline for clarity.

    betas = make_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    optimizer = torch.optim.Adam(unet.parameters(), lr=LR)
    n_params = sum(p.numel() for p in unet.parameters())

    print(f"\n{'='*60}")
    print(f"Experiment 1: Bicubic-only condition (cond_ch={cond_ch})")
    print(f"Params: {n_params:,}  |  base_ch={cfg.BASE_CH}  |  T={T}")
    print(f"Key change: cond = bicubic only, removed ISHT base from input")
    print(f"{'='*60}")

    # Build multi-scale ISHT dict
    ms_isht = {}
    for sht, isht, factor in ms_sht_isht:
        coeffs = sht(lr_up)
        ms_img = isht(coeffs)
        h_t, w_t = H // factor, W // factor
        ms_img = F.interpolate(ms_img, size=(h_t, w_t), mode='bilinear', align_corners=False)
        ms_isht[f'enc{factor}'] = ms_img

    # Pre-compute cond and residual
    cond_bicubic = lr_up  # [1, 3, H, W]
    residual = hr_img - base

    losses = []
    sample_epochs = []
    sample_mses = []

    pbar = tqdm(range(1, 501), desc="[exp1]", unit="ep")
    for epoch in pbar:
        unet.train()
        B = lr_img.shape[0]
        t = torch.randint(0, T, (B,), device=device)
        alpha_t = sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sigma_t = sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)

        noise = torch.randn_like(residual)
        noisy = alpha_t * residual + sigma_t * noise

        # ★ Only bicubic as condition
        pred_noise = unet(noisy, cond_bicubic, t.float() / T, ms_isht)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch == 1 or epoch % 100 == 0 or epoch == 500:
            unet.eval()
            with torch.no_grad():
                # DDIM sampling
                sr = ddim_sample(unet, cond_bicubic, ms_isht, base, device,
                                 T, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                                 steps=INFER_STEPS)
            v_mse = F.mse_loss(sr, hr_img).item()
            sample_epochs.append(epoch)
            sample_mses.append(v_mse)
            to_pil(sr).save(os.path.join(exp_dir, f"e{epoch:04d}.png"))
            pbar.set_postfix(train=f"{loss.item():.4f}", v_img=f"{v_mse:.4f}")
        else:
            pbar.set_postfix(train=f"{loss.item():.4f}")

    print(f"\nFinal: train={losses[-1]:.6f}, best_v_img={min(sample_mses):.6f}")


# ============================================================
# Experiment 2: Direct residual prediction (no diffusion, no timestep)
# ============================================================

def exp2_direct_residual(gpu=0, config_module="diffusion_config"):
    """U-Net directly predicts residual from condition (no noise, no timestep)."""
    import importlib
    cfg = importlib.import_module(f"config.{config_module}")

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    lr_img, hr_img = get_single_image(device)

    exp_dir = os.path.join(OUTPUT_DIR, "exp2_direct_residual")
    os.makedirs(exp_dir, exist_ok=True)

    to_pil(hr_img).save(os.path.join(exp_dir, "hr.png"))
    lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)
    to_pil(lr_up).save(os.path.join(exp_dir, "bicubic.png"))

    sht_cond, isht_cond = get_sht_isht(device, L_COND)
    base = isht_cond(sht_cond(lr_up))
    to_pil(base).save(os.path.join(exp_dir, "base.png"))

    # Build multi-scale ISHT
    ms_isht = {}
    for L_val, factor in cfg.MS_COND:
        sht = RealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        isht = InverseRealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        coeffs = sht(lr_up)
        ms_img = isht(coeffs)
        h_t, w_t = H // factor, W // factor
        ms_img = F.interpolate(ms_img, size=(h_t, w_t), mode='bilinear', align_corners=False)
        ms_isht[f'enc{factor}'] = ms_img

    # ★ KEY CHANGE: U-Net takes cond (bicubic+base=6ch) + learnable latent (3ch)
    #    and directly outputs residual. No time embedding.
    #    We use the full condition (bicubic+base) to test U-Net capacity.
    cond_ch = 6
    cond = torch.cat([lr_up, base], dim=1)

    unet = PixelUNet(
        in_ch=3, cond_ch=cond_ch,
        base_ch=BASE_CH, time_dim=TIME_DIM,
        use_polar_moe=False,
        use_laplacian=False,
        use_circular_conv=False,
        use_coord_embed=False,
        use_spherical_attn=False,
        hr_size=(H, W),
    ).to(device)

    # ★ Zero out ALL time_proj weights — completely remove time influence
    for module in unet.modules():
        if hasattr(module, 'time_proj'):
            nn.init.zeros_(module.time_proj.weight)
            nn.init.zeros_(module.time_proj.bias)

    # Learnable latent
    latent = nn.Parameter(torch.randn(1, 3, H, W, device=device) * 0.02)

    optimizer = torch.optim.Adam(list(unet.parameters()) + [latent], lr=LR)
    n_params = sum(p.numel() for p in unet.parameters())

    print(f"\n{'='*60}")
    print(f"Experiment 2: Direct residual prediction (no diffusion, no timestep)")
    print(f"Params: {n_params:,}  |  base_ch={BASE_CH}  |  time_proj zeroed")
    print(f"Key change: U-Net(latent, cond) → residual_pred, MSE loss")
    print(f"{'='*60}")

    residual_true = hr_img - base

    losses = []
    sample_epochs = []
    sample_mses = []

    # Pre-allocate dummy t_norm (time_proj weights zeroed → time has no effect)
    t_dummy = torch.zeros(1, device=device)

    pbar = tqdm(range(1, 501), desc="[exp2]", unit="ep")
    for epoch in pbar:
        unet.train()

        # ★ time_proj weights are zeroed → U-Net(noisy=latent, cond, t_dummy) → residual_pred
        pred = unet(latent.expand(1, -1, -1, -1), cond, t_dummy, ms_isht)

        if isinstance(pred, tuple):
            pred = pred[0]  # if laplacian returns tuple

        loss = F.mse_loss(pred, residual_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch == 1 or epoch % 100 == 0 or epoch == 500:
            unet.eval()
            with torch.no_grad():
                pred = unet(latent.expand(1, -1, -1, -1), cond, t_dummy, ms_isht)
                if isinstance(pred, tuple):
                    pred = pred[0]
                sr = base + pred
            v_mse = F.mse_loss(sr, hr_img).item()
            sample_epochs.append(epoch)
            sample_mses.append(v_mse)
            to_pil(sr).save(os.path.join(exp_dir, f"e{epoch:04d}.png"))
            to_pil(base + pred).save(os.path.join(exp_dir, f"e{epoch:04d}.png"))
            pbar.set_postfix(train=f"{loss.item():.6f}", v_img=f"{v_mse:.6f}")
        else:
            pbar.set_postfix(train=f"{loss.item():.6f}")

    print(f"\nFinal: train={losses[-1]:.8f}, best_v_img={min(sample_mses):.8f}")


# ============================================================
# Experiment 3: x0 prediction (predict clean residual, not noise)
# ============================================================

def exp3_x0_prediction(gpu=0, config_module="diffusion_config"):
    """Diffusion with x0 (clean residual) prediction target instead of noise."""
    import importlib
    cfg = importlib.import_module(f"config.{config_module}")

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    lr_img, hr_img = get_single_image(device)

    exp_dir = os.path.join(OUTPUT_DIR, "exp3_x0_prediction")
    os.makedirs(exp_dir, exist_ok=True)

    to_pil(hr_img).save(os.path.join(exp_dir, "hr.png"))
    lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)
    to_pil(lr_up).save(os.path.join(exp_dir, "bicubic.png"))

    sht_cond, isht_cond = get_sht_isht(device, L_COND)
    base = isht_cond(sht_cond(lr_up))
    to_pil(base).save(os.path.join(exp_dir, "base.png"))

    # Multi-scale ISHT
    ms_isht = {}
    for L_val, factor in cfg.MS_COND:
        sht = RealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        isht = InverseRealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        coeffs = sht(lr_up)
        ms_img = isht(coeffs)
        h_t, w_t = H // factor, W // factor
        ms_img = F.interpolate(ms_img, size=(h_t, w_t), mode='bilinear', align_corners=False)
        ms_isht[f'enc{factor}'] = ms_img

    # ★ We still use cond = bicubic + base (6ch) as in the original
    cond_ch = 6
    cond = torch.cat([lr_up, base], dim=1)

    unet = PixelUNet(
        in_ch=3, cond_ch=cond_ch,
        base_ch=cfg.BASE_CH, time_dim=cfg.TIME_DIM,
        use_polar_moe=cfg.USE_POLAR_MOE,
        use_laplacian=cfg.USE_LAPLACIAN_PYRAMID,
        use_circular_conv=cfg.USE_CIRCULAR_CONV,
        use_coord_embed=cfg.USE_COORD_EMBED,
        use_spherical_attn=cfg.USE_SPHERICAL_ATTN,
        hr_size=cfg.HR_SIZE,
    ).to(device)

    betas = make_beta_schedule(T).to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    optimizer = torch.optim.Adam(unet.parameters(), lr=LR)
    n_params = sum(p.numel() for p in unet.parameters())

    print(f"\n{'='*60}")
    print(f"Experiment 3: x0 prediction (predict clean residual, not noise)")
    print(f"Params: {n_params:,}  |  base_ch={cfg.BASE_CH}  |  T={T}")
    print(f"Key change: target = true_residual (not noise)")
    print(f"{'='*60}")

    residual_true = hr_img - base

    losses = []
    sample_epochs = []
    sample_mses = []

    pbar = tqdm(range(1, 501), desc="[exp3]", unit="ep")
    for epoch in pbar:
        unet.train()
        B = lr_img.shape[0]
        t = torch.randint(0, T, (B,), device=device)
        alpha_t = sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sigma_t = sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)

        noise = torch.randn_like(residual_true)
        noisy = alpha_t * residual_true + sigma_t * noise

        # ★ Model predicts x0 (clean residual), not noise
        pred_x0 = unet(noisy, cond, t.float() / T, ms_isht)
        if isinstance(pred_x0, tuple):
            pred_x0 = pred_x0[0]

        # ★ Loss: MSE(pred_x0, true_residual)
        loss = F.mse_loss(pred_x0, residual_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch == 1 or epoch % 100 == 0 or epoch == 500:
            unet.eval()
            with torch.no_grad():
                sr = ddim_sample_x0(unet, cond, ms_isht, base, device,
                                    T, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                                    steps=10)  # ★ fewer steps for x0 prediction
            v_mse = F.mse_loss(sr, hr_img).item()
            sample_epochs.append(epoch)
            sample_mses.append(v_mse)
            to_pil(sr).save(os.path.join(exp_dir, f"e{epoch:04d}.png"))
            pbar.set_postfix(train=f"{loss.item():.6f}", v_img=f"{v_mse:.6f}")
        else:
            pbar.set_postfix(train=f"{loss.item():.6f}")

    print(f"\nFinal: train={losses[-1]:.8f}, best_v_img={min(sample_mses):.8f}")


# ============================================================
# Experiment 4: Consistency regularization
# ============================================================

def exp4_consistency(gpu=0, config_module="diffusion_config"):
    """x0 prediction + consistency loss between adjacent timesteps."""
    import importlib
    cfg = importlib.import_module(f"config.{config_module}")

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    lr_img, hr_img = get_single_image(device)

    exp_dir = os.path.join(OUTPUT_DIR, "exp4_consistency")
    os.makedirs(exp_dir, exist_ok=True)

    to_pil(hr_img).save(os.path.join(exp_dir, "hr.png"))
    lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)
    to_pil(lr_up).save(os.path.join(exp_dir, "bicubic.png"))

    sht_cond, isht_cond = get_sht_isht(device, L_COND)
    base = isht_cond(sht_cond(lr_up))
    to_pil(base).save(os.path.join(exp_dir, "base.png"))

    ms_isht = {}
    for L_val, factor in cfg.MS_COND:
        sht = RealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        isht = InverseRealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        coeffs = sht(lr_up)
        ms_img = isht(coeffs)
        h_t, w_t = H // factor, W // factor
        ms_img = F.interpolate(ms_img, size=(h_t, w_t), mode='bilinear', align_corners=False)
        ms_isht[f'enc{factor}'] = ms_img

    cond_ch = 6
    cond = torch.cat([lr_up, base], dim=1)

    unet = PixelUNet(
        in_ch=3, cond_ch=cond_ch,
        base_ch=cfg.BASE_CH, time_dim=cfg.TIME_DIM,
        use_polar_moe=cfg.USE_POLAR_MOE,
        use_laplacian=cfg.USE_LAPLACIAN_PYRAMID,
        use_circular_conv=cfg.USE_CIRCULAR_CONV,
        use_coord_embed=cfg.USE_COORD_EMBED,
        use_spherical_attn=cfg.USE_SPHERICAL_ATTN,
        hr_size=cfg.HR_SIZE,
    ).to(device)

    betas = make_beta_schedule(T).to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    optimizer = torch.optim.Adam(unet.parameters(), lr=LR)
    n_params = sum(p.numel() for p in unet.parameters())

    CONSISTENCY_LAMBDA = 1.0
    DELTA_T = 20  # timestep gap for consistency

    print(f"\n{'='*60}")
    print(f"Experiment 4: Consistency regularization")
    print(f"Params: {n_params:,}  |  base_ch={cfg.BASE_CH}  |  T={T}")
    print(f"lambda_consistency={CONSISTENCY_LAMBDA}  |  delta_t={DELTA_T}")
    print(f"{'='*60}")

    residual_true = hr_img - base

    # Precompute mu (predicted_x0 coefficient) for DDIM-like x0 prediction
    # DDIM x0 prediction: x0_pred = (x_t - sigma_t * eps_pred) / alpha_t
    # For consistency: at t2 < t1, we expect:
    #   x0_pred(t1) [after DDIM step to t2] ≈ x0_pred(t2)

    losses = []
    sample_epochs = []
    sample_mses = []

    pbar = tqdm(range(1, 501), desc="[exp4]", unit="ep")
    for epoch in pbar:
        unet.train()
        B = 1

        # Sample two timesteps: t1 and t2 = max(t1 - delta_t, 0)
        t1 = torch.randint(DELTA_T, T, (B,), device=device)
        t2 = t1 - DELTA_T

        alpha_t1 = sqrt_alphas_cumprod[t1].view(B, 1, 1, 1)
        sigma_t1 = sqrt_one_minus_alphas_cumprod[t1].view(B, 1, 1, 1)
        alpha_t2 = sqrt_alphas_cumprod[t2].view(B, 1, 1, 1)
        sigma_t2 = sqrt_one_minus_alphas_cumprod[t2].view(B, 1, 1, 1)

        # Same noise and target for both
        noise = torch.randn_like(residual_true)
        noisy1 = alpha_t1 * residual_true + sigma_t1 * noise
        noisy2 = alpha_t2 * residual_true + sigma_t2 * noise

        pred_x0_1 = unet(noisy1, cond, t1.float() / T, ms_isht)
        if isinstance(pred_x0_1, tuple):
            pred_x0_1 = pred_x0_1[0]

        pred_x0_2 = unet(noisy2, cond, t2.float() / T, ms_isht)
        if isinstance(pred_x0_2, tuple):
            pred_x0_2 = pred_x0_2[0]

        # Reconstruction losses
        loss_recon = F.mse_loss(pred_x0_1, residual_true) + F.mse_loss(pred_x0_2, residual_true)

        # Consistency loss: x0 predictions should agree
        loss_consistency = F.mse_loss(pred_x0_1, pred_x0_2)

        loss = loss_recon + CONSISTENCY_LAMBDA * loss_consistency

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch == 1 or epoch % 100 == 0 or epoch == 500:
            unet.eval()
            with torch.no_grad():
                sr = ddim_sample_x0(unet, cond, ms_isht, base, device,
                                    T, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                                    steps=10)
            v_mse = F.mse_loss(sr, hr_img).item()
            sample_epochs.append(epoch)
            sample_mses.append(v_mse)
            to_pil(sr).save(os.path.join(exp_dir, f"e{epoch:04d}.png"))
            pbar.set_postfix(train=f"{loss.item():.6f}", v_img=f"{v_mse:.6f}")
        else:
            pbar.set_postfix(train=f"{loss.item():.6f}")

    print(f"\nFinal: train={losses[-1]:.8f}, best_v_img={min(sample_mses):.8f}")


# ============================================================
# Shared sampling utilities
# ============================================================

@torch.no_grad()
def ddim_sample(unet, cond, ms_isht, base, device,
                T, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                steps=50):
    """Standard DDIM noise-prediction sampling."""
    B = cond.shape[0]
    x = torch.randn(B, 3, H, W, device=device)

    indices = torch.linspace(0, T - 1, steps, dtype=torch.long, device=device)
    indices_prev = torch.cat([indices[1:], torch.tensor([-1], device=device)])

    for i in range(steps - 1, -1, -1):
        t, tp = indices[i], indices_prev[i]
        t_norm = torch.full((B,), t.item() / T, device=device)

        out = unet(x, cond, t_norm, ms_isht)
        eps = out[0] if isinstance(out, tuple) else out  # handle laplacian tuple
        alpha_t = sqrt_alphas_cumprod[t]
        sigma_t = sqrt_one_minus_alphas_cumprod[t]
        x0_pred = (x - sigma_t * eps) / alpha_t

        if tp >= 0:
            alpha_p = sqrt_alphas_cumprod[tp]
            sigma_p = sqrt_one_minus_alphas_cumprod[tp]
            x = alpha_p * x0_pred + sigma_p * eps
        else:
            x = x0_pred

    return base + x


@torch.no_grad()
def ddim_sample_x0(unet, cond, ms_isht, base, device,
                   T, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                   steps=10):
    """DDIM sampling for x0-prediction model.

    Model outputs x0_pred (clean residual), not noise.
    From noisy x_t, we compute the noise estimate:
        eps_est = (x_t - alpha_t * x0_pred) / sigma_t
    Then step to x_{t-1}.
    """
    B = cond.shape[0]
    x = torch.randn(B, 3, H, W, device=device)

    indices = torch.linspace(0, T - 1, steps, dtype=torch.long, device=device)
    indices_prev = torch.cat([indices[1:], torch.tensor([-1], device=device)])

    for i in range(steps - 1, -1, -1):
        t, tp = indices[i], indices_prev[i]
        t_norm = torch.full((B,), t.item() / T, device=device)

        out = unet(x, cond, t_norm, ms_isht)
        x0_pred = out[0] if isinstance(out, tuple) else out

        alpha_t = sqrt_alphas_cumprod[t]
        sigma_t = sqrt_one_minus_alphas_cumprod[t]

        # Recover noise estimate from x0 prediction
        eps_est = (x - alpha_t * x0_pred) / sigma_t

        if tp >= 0:
            alpha_p = sqrt_alphas_cumprod[tp]
            sigma_p = sqrt_one_minus_alphas_cumprod[tp]
            x = alpha_p * x0_pred + sigma_p * eps_est
        else:
            x = x0_pred

    return base + x


# ============================================================
# Experiment 5: Path A — diffusion target = full HR (not residual)
# ============================================================

def exp5_hr_target(gpu=0, config_module="diffusion_config"):
    """Diffusion learns full HR image with ISHT base as condition.

    Key change: noise is added to HR (not residual).
    - Training: x_t = alpha_t * HR + sigma_t * noise, predict epsilon
    - Condition: concat(bicubic, base) — unchanged
    - Sampling: DDIM from noise → HR_pred directly (no +base step)
    """
    import importlib
    cfg = importlib.import_module(f"config.{config_module}")

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    lr_img, hr_img = get_single_image(device)

    exp_dir = os.path.join(OUTPUT_DIR, "exp5_hr_target")
    os.makedirs(exp_dir, exist_ok=True)

    # Baselines
    to_pil(hr_img).save(os.path.join(exp_dir, "hr.png"))
    lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)
    to_pil(lr_up).save(os.path.join(exp_dir, "bicubic.png"))

    sht_cond, isht_cond = get_sht_isht(device, L_COND)
    base = isht_cond(sht_cond(lr_up))
    to_pil(base).save(os.path.join(exp_dir, "base.png"))

    # Report HR variance
    print(f"\n  HR var: {hr_img.var().item():.6f} (vs residual var: {(hr_img - base).var().item():.6f})")
    print(f"  HR std:  {hr_img.std().item():.4f}")

    # Multi-scale ISHT
    ms_isht = {}
    for L_val, factor in cfg.MS_COND:
        sht = RealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        isht = InverseRealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        coeffs = sht(lr_up)
        ms_img = isht(coeffs)
        h_t, w_t = H // factor, W // factor
        ms_img = F.interpolate(ms_img, size=(h_t, w_t), mode='bilinear', align_corners=False)
        ms_isht[f'enc{factor}'] = ms_img

    # ★ Same condition as original (bicubic + base = 6ch)
    cond_ch = 6
    cond = torch.cat([lr_up, base], dim=1)

    unet = PixelUNet(
        in_ch=3, cond_ch=cond_ch,
        base_ch=BASE_CH, time_dim=TIME_DIM,
        use_polar_moe=False,
        use_laplacian=False,
        use_circular_conv=False,
        use_coord_embed=False,
        use_spherical_attn=False,
        hr_size=(H, W),
    ).to(device)

    betas = make_beta_schedule(T).to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    optimizer = torch.optim.Adam(unet.parameters(), lr=LR)
    n_params = sum(p.numel() for p in unet.parameters())

    print(f"\n{'='*60}")
    print(f"Experiment 5: HR-target diffusion (noise prediction)")
    print(f"Params: {n_params:,}  |  base_ch={BASE_CH}  |  T={T}")
    print(f"Target: HR (var={hr_img.var().item():.4f}) — NOT residual")
    print(f"Cond: bicubic + base ({cond_ch}ch)  |  DDIM: {INFER_STEPS} steps")
    print(f"{'='*60}")

    losses = []
    sample_epochs = []
    sample_mses = []

    pbar = tqdm(range(1, 501), desc="[exp5]", unit="ep")
    for epoch in pbar:
        unet.train()
        B = 1
        t = torch.randint(0, T, (B,), device=device)
        alpha_t = sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sigma_t = sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)

        # ★ Key change: noise added to HR (not residual)
        noise = torch.randn_like(hr_img)
        noisy = alpha_t * hr_img + sigma_t * noise

        pred_noise = unet(noisy, cond, t.float() / T, ms_isht)
        if isinstance(pred_noise, tuple):
            pred_noise = pred_noise[0]

        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch == 1 or epoch % 100 == 0 or epoch == 500:
            unet.eval()
            with torch.no_grad():
                # DDIM sampling: directly outputs HR (no +base needed)
                sr = ddim_sample_hr(unet, cond, ms_isht, device,
                                    T, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                                    steps=INFER_STEPS)
            v_mse = F.mse_loss(sr, hr_img).item()
            sample_epochs.append(epoch)
            sample_mses.append(v_mse)
            to_pil(sr).save(os.path.join(exp_dir, f"e{epoch:04d}.png"))
            pbar.set_postfix(train=f"{loss.item():.4f}", v_img=f"{v_mse:.4f}")
        else:
            pbar.set_postfix(train=f"{loss.item():.4f}")

    print(f"\nFinal: train={losses[-1]:.6f}, best_v_img={min(sample_mses):.6f}")


@torch.no_grad()
def ddim_sample_hr(unet, cond, ms_isht, device,
                   T, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                   steps=50):
    """DDIM sampling for HR-target model. Output is full HR, not residual."""
    B = cond.shape[0]
    x = torch.randn(B, 3, H, W, device=device)

    indices = torch.linspace(0, T - 1, steps, dtype=torch.long, device=device)
    indices_prev = torch.cat([indices[1:], torch.tensor([-1], device=device)])

    for i in range(steps - 1, -1, -1):
        t, tp = indices[i], indices_prev[i]
        t_norm = torch.full((B,), t.item() / T, device=device)

        out = unet(x, cond, t_norm, ms_isht)
        eps = out[0] if isinstance(out, tuple) else out

        alpha_t = sqrt_alphas_cumprod[t]
        sigma_t = sqrt_one_minus_alphas_cumprod[t]
        x0_pred = (x - sigma_t * eps) / alpha_t

        if tp >= 0:
            alpha_p = sqrt_alphas_cumprod[tp]
            sigma_p = sqrt_one_minus_alphas_cumprod[tp]
            x = alpha_p * x0_pred + sigma_p * eps
        else:
            x = x0_pred

    return x  # ★ directly HR, no +base


# ============================================================
# Experiment 6: Path A clean — HR target, exp_base config, 200 steps
# ============================================================

def exp6_hr_target_clean(gpu=0, config_module="exp_base"):
    """Clean HR-target diffusion using exp_base config (BASE_CH=32).

    Key design:
    - Target: HR (full image, var ~0.27) — NOT residual (var ~0.004)
    - Predict: noise epsilon
    - Condition: concat(bicubic, base) = 6ch
    - Config: exp_base → BASE_CH=32, all innovations OFF
    - 200 training steps, DDIM 50-step sampling
    - Output directly as HR_pred (no +base step)
    """
    import importlib
    cfg = importlib.import_module(f"config.{config_module}")

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    lr_img, hr_img = get_single_image(device)

    exp_dir = os.path.join(OUTPUT_DIR, "exp6_hr_target_clean")
    os.makedirs(exp_dir, exist_ok=True)

    # Baselines
    to_pil(hr_img).save(os.path.join(exp_dir, "hr.png"))
    lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)
    to_pil(lr_up).save(os.path.join(exp_dir, "bicubic.png"))

    sht_cond, isht_cond = get_sht_isht(device, L_COND)
    base = isht_cond(sht_cond(lr_up))
    to_pil(base).save(os.path.join(exp_dir, "base.png"))

    residual = hr_img - base
    print(f"\n  HR var: {hr_img.var().item():.6f},  HR std: {hr_img.std().item():.4f}")
    print(f"  Residual var: {residual.var().item():.6f}  (HR var is {hr_img.var().item()/residual.var().item():.1f}x larger)")

    # Multi-scale ISHT
    ms_isht = {}
    for L_val, factor in cfg.MS_COND:
        sht_ms = RealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        isht_ms = InverseRealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        coeffs = sht_ms(lr_up)
        ms_img = isht_ms(coeffs)
        h_t, w_t = H // factor, W // factor
        ms_img = F.interpolate(ms_img, size=(h_t, w_t), mode='bilinear', align_corners=False)
        ms_isht[f'enc{factor}'] = ms_img

    # Condition: bicubic + base = 6ch
    cond_ch = 6
    cond = torch.cat([lr_up, base], dim=1)

    # ★ Use cfg.BASE_CH (32 from exp_base), not file-level BASE_CH=64
    unet = PixelUNet(
        in_ch=3, cond_ch=cond_ch,
        base_ch=cfg.BASE_CH, time_dim=cfg.TIME_DIM,
        use_polar_moe=cfg.USE_POLAR_MOE,
        use_laplacian=cfg.USE_LAPLACIAN_PYRAMID,
        use_circular_conv=cfg.USE_CIRCULAR_CONV,
        use_coord_embed=cfg.USE_COORD_EMBED,
        use_spherical_attn=cfg.USE_SPHERICAL_ATTN,
        hr_size=cfg.HR_SIZE,
    ).to(device)

    betas = make_beta_schedule(T).to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    optimizer = torch.optim.Adam(unet.parameters(), lr=LR)
    n_params = sum(p.numel() for p in unet.parameters())

    N_EPOCHS = 200

    print(f"\n{'='*60}")
    print(f"Experiment 6: HR-target diffusion — clean (exp_base config)")
    print(f"Params: {n_params:,}  |  base_ch={cfg.BASE_CH}  |  T={T}")
    print(f"Target: HR (not residual)  |  Predict: noise ε")
    print(f"Cond: bicubic+base ({cond_ch}ch)  |  Epochs: {N_EPOCHS}")
    print(f"All innovations: OFF  |  DDIM: {INFER_STEPS} steps")
    print(f"{'='*60}")

    losses = []
    sample_epochs = []
    sample_mses = []

    pbar = tqdm(range(1, N_EPOCHS + 1), desc="[exp6]", unit="ep")
    for epoch in pbar:
        unet.train()
        B = 1
        t = torch.randint(0, T, (B,), device=device)
        alpha_t = sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sigma_t = sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)

        # ★ Noise added to HR (not residual)
        noise = torch.randn_like(hr_img)
        noisy = alpha_t * hr_img + sigma_t * noise

        pred_noise = unet(noisy, cond, t.float() / T, ms_isht)
        if isinstance(pred_noise, tuple):
            pred_noise = pred_noise[0]

        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Sample at epoch 1, 50, 100, 150, 200
        if epoch == 1 or epoch % 50 == 0 or epoch == N_EPOCHS:
            unet.eval()
            with torch.no_grad():
                sr = ddim_sample_hr(unet, cond, ms_isht, device,
                                    T, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                                    steps=INFER_STEPS)
            v_mse = F.mse_loss(sr, hr_img).item()
            sample_epochs.append(epoch)
            sample_mses.append(v_mse)
            to_pil(sr).save(os.path.join(exp_dir, f"e{epoch:04d}.png"))
            pbar.set_postfix(train=f"{loss.item():.4f}", v_img=f"{v_mse:.4f}")
        else:
            pbar.set_postfix(train=f"{loss.item():.4f}")

    print(f"\nFinal: train={losses[-1]:.6f}, best_v_img={min(sample_mses):.6f} (epoch {sample_epochs[sample_mses.index(min(sample_mses))]})")
    if sample_mses:
        print(f"All v_img MSEs: {[f'{m:.4f}' for m in sample_mses]}")


# ============================================================
# Experiment 7: God-model DDIM validator
# ============================================================

class DummyGodModel:
    """A 'god model' that returns perfect noise/target for any timestep.

    Signature MUST match U-Net forward: forward(x, cond, t_norm, ms_isht).
    This eliminates the learned model as a variable: if DDIM sampling with
    this model doesn't perfectly reconstruct HR, the sampler itself is buggy.
    """
    def __init__(self, hr_img, betas, device):
        self.hr_img = hr_img
        self.device = device
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Forward diffuse HR for ALL timesteps with a FIXED noise seed
        T = len(betas)
        self.eps_true = torch.randn(1, 3, H, W, device=device)
        self.T = T

    def forward(self, x, cond, t_norm, ms_isht=None):
        """Match U-Net signature: forward(x, cond, t_norm, ms_isht)."""
        B = t_norm.shape[0]
        t = (t_norm * (self.T - 1)).long()
        eps = torch.stack([self.eps_true[0] for _ in range(B)], dim=0)
        return eps

    def __call__(self, x, cond, t_norm, ms_isht=None):
        return self.forward(x, cond, t_norm, ms_isht)


@torch.no_grad()
def ddim_sample_fixed(unet, cond, ms_isht, device,
                      T, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                      steps=50, x_T=None):
    """CORRECTED DDIM: indices high->low, proper iterative denoising.

    Key fix: indices ordered from T-1 down to 0, so we iteratively
    denoise from x_T -> x_{T-delta} -> ... -> x_0.
    If x_T is provided, use it as starting point (god-model test);
    otherwise start from random noise.
    """
    B = cond.shape[0]
    x = x_T if x_T is not None else torch.randn(B, 3, H, W, device=device)

    indices = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)

    for i in range(steps):
        t = indices[i]
        tp = indices[i + 1] if i + 1 < steps else -1
        t_norm = torch.full((B,), t.item() / T, device=device)

        out = unet(x, cond, t_norm, ms_isht)
        eps = out[0] if isinstance(out, tuple) else out

        alpha_t = sqrt_alphas_cumprod[t]
        sigma_t = sqrt_one_minus_alphas_cumprod[t]
        x0_pred = (x - sigma_t * eps) / alpha_t

        if tp >= 0:
            alpha_p = sqrt_alphas_cumprod[tp]
            sigma_p = sqrt_one_minus_alphas_cumprod[tp]
            x = alpha_p * x0_pred + sigma_p * eps
        else:
            x = x0_pred

    return x


@torch.no_grad()
def ddim_sample_buggy(unet, cond, ms_isht, device,
                      T, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                      steps=50, x_T=None):
    """BUGGY DDIM (current code): indices low->high, loop reversed.

    indices = [0, 20, 41, ..., 999], loop from high to low:
    - First step goes 999->clean (x is random noise, not x_999!)
    - Subsequent steps re-noise then de-noise
    - Final output is at timestep ~20, NOT clean
    """
    B = cond.shape[0]
    x = x_T if x_T is not None else torch.randn(B, 3, H, W, device=device)

    indices = torch.linspace(0, T - 1, steps, dtype=torch.long, device=device)
    indices_prev = torch.cat([indices[1:], torch.tensor([-1], device=device)])

    for i in range(steps - 1, -1, -1):
        t, tp = indices[i], indices_prev[i]
        t_norm = torch.full((B,), t.item() / T, device=device)

        out = unet(x, cond, t_norm, ms_isht)
        eps = out[0] if isinstance(out, tuple) else out

        alpha_t = sqrt_alphas_cumprod[t]
        sigma_t = sqrt_one_minus_alphas_cumprod[t]
        x0_pred = (x - sigma_t * eps) / alpha_t

        if tp >= 0:
            alpha_p = sqrt_alphas_cumprod[tp]
            sigma_p = sqrt_one_minus_alphas_cumprod[tp]
            x = alpha_p * x0_pred + sigma_p * eps
        else:
            x = x0_pred

    return x


def exp7_god_model(gpu=0):
    """God-model DDIM test: if a perfect model can't reconstruct, the sampler is buggy.

    Key design:
    - Forward: x_t = alpha_t * HR + sigma_t * eps_true (same eps at all t)
    - God model returns eps_true regardless of input (perfect predictor)
    - Start DDIM from the actual x_T (not random noise) for fair test
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    lr_img, hr_img = get_single_image(device)

    exp_dir = os.path.join(OUTPUT_DIR, "exp7_god_model")
    os.makedirs(exp_dir, exist_ok=True)

    # Baselines
    to_pil(hr_img).save(os.path.join(exp_dir, "hr.png"))
    lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)
    to_pil(lr_up).save(os.path.join(exp_dir, "bicubic.png"))
    sht_cond, isht_cond = get_sht_isht(device, L_COND)
    base = isht_cond(sht_cond(lr_up))
    to_pil(base).save(os.path.join(exp_dir, "base.png"))

    # Multi-scale ISHT (dummy, god model ignores condition)
    ms_isht = {}
    for L_val, factor in [(L_COND // 8, 8), (L_COND // 4, 4), (L_COND // 2, 2)]:
        sht = RealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        isht = InverseRealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        coeffs = sht(lr_up)
        ms_img = isht(coeffs)
        h_t, w_t = H // factor, W // factor
        ms_img = F.interpolate(ms_img, size=(h_t, w_t), mode='bilinear', align_corners=False)
        ms_isht[f'enc{factor}'] = ms_img

    cond = torch.cat([lr_up, base], dim=1)

    # Build the god model
    betas = make_beta_schedule(T).to(device)
    god_model = DummyGodModel(hr_img, betas, device)

    sqrt_alphas_cumprod = god_model.sqrt_alphas_cumprod
    sqrt_one_minus_alphas_cumprod = god_model.sqrt_one_minus_alphas_cumprod

    print(f"\n{'='*60}")
    print(f"Experiment 7: God-model DDIM validator")
    print(f"God model returns eps_true for all t — zero prediction error")
    print(f"Forward: x_t = alpha_t * HR + sigma_t * eps_true (same eps at all t)")
    print(f"DDIM starts from x_T = x_999 (exact forward noised image)")
    print(f"{'='*60}")

    # --- 1. Verify forward/reverse consistency (1-step) ---
    x_T = sqrt_alphas_cumprod[-1] * hr_img + sqrt_one_minus_alphas_cumprod[-1] * god_model.eps_true
    hr_recovered = (x_T - sqrt_one_minus_alphas_cumprod[-1] * god_model.eps_true) / sqrt_alphas_cumprod[-1]
    fwd_rev_mse = F.mse_loss(hr_recovered, hr_img).item()
    print(f"\n  1-step forward->reverse: MSE = {fwd_rev_mse:.2e}")
    print(f"  x_T var = {x_T.var().item():.4f} (should be ~1.0)")

    # ---- DIAGNOSTIC: Trace buggy DDIM step-by-step ----
    print(f"\n  --- Buggy DDIM trace (T={T}, steps={INFER_STEPS}) ---")
    indices = torch.linspace(0, T - 1, INFER_STEPS, dtype=torch.long, device=device)
    indices_prev = torch.cat([indices[1:], torch.tensor([-1], device=device)])
    print(f"  indices = {indices.tolist()}")
    print(f"  prev    = {indices_prev.tolist()}")

    x = x_T.clone()
    for i in range(INFER_STEPS - 1, -1, -1):
        t, tp = indices[i], indices_prev[i]
        t_norm = torch.full((1,), t.item() / T, device=device)
        eps = god_model(x, cond, t_norm, ms_isht)
        a_t, s_t = sqrt_alphas_cumprod[t], sqrt_one_minus_alphas_cumprod[t]
        x0_pred = (x - s_t * eps) / a_t
        x0_err = F.mse_loss(x0_pred, hr_img).item()
        x_cur_noise = x.var().item()
        if tp >= 0:
            a_p, s_p = sqrt_alphas_cumprod[tp], sqrt_one_minus_alphas_cumprod[tp]
            x = a_p * x0_pred + s_p * eps
        else:
            x = x0_pred
        x_next_noise = x.var().item()
        print(f"  i={i:2d}: t={t.item():4d} tp={tp.item() if tp>=0 else -1:4d}  "
              f"x_var={x_cur_noise:.4f} -> {x_next_noise:.4f}  x0_pred_MSE={x0_err:.6e}")

    mse_buggy = F.mse_loss(x, hr_img).item()
    to_pil(x).save(os.path.join(exp_dir, "recon_buggy.png"))
    print(f"  BUGGY final MSE: {mse_buggy:.6f}")

    # ---- DIAGNOSTIC: Trace fixed DDIM step-by-step ----
    print(f"\n  --- Fixed DDIM trace (T={T}, steps={INFER_STEPS}) ---")
    indices = torch.linspace(T - 1, 0, INFER_STEPS, dtype=torch.long, device=device)
    print(f"  indices = {indices.tolist()}")

    x = x_T.clone()
    for i in range(INFER_STEPS):
        t = indices[i]
        tp = indices[i + 1] if i + 1 < INFER_STEPS else -1
        t_norm = torch.full((1,), t.item() / T, device=device)
        eps = god_model(x, cond, t_norm, ms_isht)
        a_t, s_t = sqrt_alphas_cumprod[t], sqrt_one_minus_alphas_cumprod[t]
        x0_pred = (x - s_t * eps) / a_t
        x0_err = F.mse_loss(x0_pred, hr_img).item()
        x_cur_noise = x.var().item()
        if tp >= 0:
            a_p, s_p = sqrt_alphas_cumprod[tp], sqrt_one_minus_alphas_cumprod[tp]
            x = a_p * x0_pred + s_p * eps
        else:
            x = x0_pred
        x_next_noise = x.var().item()
        if i < 5 or i >= INFER_STEPS - 3:
            print(f"  i={i:2d}: t={t.item():4d} tp={tp.item() if tp>=0 else -1:4d}  "
                  f"x_var={x_cur_noise:.4f} -> {x_next_noise:.4f}  x0_pred_MSE={x0_err:.6e}")
        elif i == 5:
            print(f"  ...")

    mse_fixed = F.mse_loss(x, hr_img).item()
    to_pil(x).save(os.path.join(exp_dir, "recon_fixed.png"))
    print(f"  FIXED final MSE: {mse_fixed:.6e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Buggy DDIM MSE:  {mse_buggy:.6f}")
    print(f"  Fixed DDIM MSE:  {mse_fixed:.6e}")
    print(f"DIAGNOSIS:")
    if mse_fixed < 1e-6:
        print(f"  Corrected DDIM is verified — indexing was the bug")
    elif mse_fixed < 1e-3:
        print(f"  Sampling formulas basically correct, small numerical issues")
    else:
        print(f"  Even corrected DDIM fails → investigate further")
    print(f"{'='*60}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Diagnostic experiments")
    parser.add_argument("--exp", "-e", type=int, required=True, choices=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--config", "-c", type=str, default="diffusion_config")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.exp == 0:
        exp0_residual_stats(args.gpu)
    elif args.exp == 1:
        exp1_bicubic_only(args.gpu, args.config)
    elif args.exp == 2:
        exp2_direct_residual(args.gpu, args.config)
    elif args.exp == 3:
        exp3_x0_prediction(args.gpu, args.config)
    elif args.exp == 4:
        exp4_consistency(args.gpu, args.config)
    elif args.exp == 5:
        exp5_hr_target(args.gpu, args.config)
    elif args.exp == 6:
        exp6_hr_target_clean(args.gpu, args.config)
    elif args.exp == 7:
        exp7_god_model(args.gpu)
