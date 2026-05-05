"""Phase 5 Stage 1: Strengthen base with better training strategy.

Fixed bed (all experiments):
  - Cosine noise schedule (Nichol & Dhariwal 2021)
  - x0 prediction target: U-Net -> x0_pred, loss = MSE(x0_pred, HR)
  - Soft tanh clipping during DDIM sampling
  - HR-target diffusion (per Phase 4: HR var 0.27 vs residual var 0.004, 65x)

Design principle:
  - Single image overfitting, rapid iteration
  - Modules enabled one at a time to avoid interaction confusion
  - Each experiment compared against baseline (0.166 MSE)

Experiment 1.1: Extended training + x0 prediction + soft clip
  - T=1000, cosine schedule, 2000 epochs, sample every 200
  - Baseline: whether MSE < 0.05 (Phase 4 direct pred achieved 0.00067)

Experiment 1.2: Fine-tune cosine schedule
  - T=500, cosine schedule, 500 epochs
  - Compare with 1.1 at 500-epoch mark

Usage:
  python phase5_stage1.py -e 1.1 -g 0
  python phase5_stage1.py -e 1.2 -g 0
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
from utility.schedule import make_cosine_schedule
from model.unet import PixelUNet
from vit.overfit_plot import update_curves, make_progression


# ============================================================
# Shared setup
# ============================================================

H, W = 512, 1024
L_COND = 255
SCALE = 4
BASE_CH = 32
TIME_DIM = 256
LR = 2e-4
DATA_DIR = "lau_dataset/sun_test"
OUTPUT_DIR = "phase5_output"

# Multi-scale ISHT injection levels (L, downscale_factor)
MS_COND = [
    (128, 2),
    (64,  4),
    (32,  8),
]


def get_single_image(device):
    dataset = PanoramaDataset(DATA_DIR, (H, W), SCALE)
    lr_img, hr_img = dataset[0]
    return lr_img.unsqueeze(0).to(device), hr_img.unsqueeze(0).to(device)


def get_sht_isht(device, lmax):
    sht = RealSHT(H, W, lmax=lmax, mmax=lmax).to(device)
    isht = InverseRealSHT(H, W, lmax=lmax, mmax=lmax).to(device)
    return sht, isht


def to_pil(t):
    a = t[0].cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip((a + 1) * 127.5, 0, 255).astype(np.uint8))


# ============================================================
# Metrics: MSE, NCC, Edge-NCC
# ============================================================

@torch.no_grad()
def compute_metrics(pred, target):
    """Compute MSE, NCC, Edge-NCC between pred and target.

    All metrics computed on [-1, 1] range images.
    Returns dict with scalar floats.
    """
    pred_f = pred.float()
    target_f = target.float()

    # MSE
    mse = F.mse_loss(pred_f, target_f).item()

    # NCC (normalized cross-correlation) per channel, then average
    ncc_vals = []
    for c in range(3):
        a = pred_f[:, c]  # [H, W]
        b = target_f[:, c]
        a_mean = a.mean()
        b_mean = b.mean()
        a_std = a.std()
        b_std = b.std()
        if a_std < 1e-8 or b_std < 1e-8:
            ncc_vals.append(0.0)
        else:
            ncc_vals.append(((a - a_mean) * (b - b_mean)).mean().item() / (a_std * b_std).item())
    ncc = float(np.mean(ncc_vals))

    # Edge-NCC: NCC on Sobel gradient magnitudes
    edge_ncc_vals = []
    for c in range(3):
        a = pred_f[0:1, c:c+1]  # [1, 1, H, W]
        b = target_f[0:1, c:c+1]
        # Sobel-x and Sobel-y kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        gx_a = F.conv2d(a, sobel_x, padding=1)
        gy_a = F.conv2d(a, sobel_y, padding=1)
        grad_a = torch.sqrt(gx_a**2 + gy_a**2).view(-1)
        gx_b = F.conv2d(b, sobel_x, padding=1)
        gy_b = F.conv2d(b, sobel_y, padding=1)
        grad_b = torch.sqrt(gx_b**2 + gy_b**2).view(-1)
        a_m = grad_a.mean()
        b_m = grad_b.mean()
        a_s = grad_a.std()
        b_s = grad_b.std()
        if a_s < 1e-8 or b_s < 1e-8:
            edge_ncc_vals.append(0.0)
        else:
            edge_ncc_vals.append(((grad_a - a_m) * (grad_b - b_m)).mean().item() / (a_s * b_s).item())
    edge_ncc = float(np.mean(edge_ncc_vals))

    return {"mse": mse, "ncc": ncc, "edge_ncc": edge_ncc}


# ============================================================
# DDIM x0-prediction sampler with soft tanh clipping
# ============================================================

@torch.no_grad()
def ddim_sample_x0(unet, cond, ms_isht,
                   sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                   T, steps=50, soft_clip=True):
    """DDIM for x0-prediction model with optional soft tanh clipping.

    U-Net outputs x0_pred (clean HR), not noise epsilon.
    At each step:
      1. x0_pred = unet(x, cond, t/T, ms_isht)
      2. if soft_clip: x0_pred = tanh(x0_pred)
      3. eps_est = (x - alpha_t * x0_pred) / sigma_t
      4. x = alpha_{t-1} * x0_pred + sigma_{t-1} * eps_est

    Returns: generated SR image [B, 3, H, W] in [-1, 1] range.
    """
    B = cond.shape[0]
    device = cond.device
    x = torch.randn(B, 3, H, W, device=device)

    indices = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)

    for i in range(steps):
        t = indices[i]
        tp = indices[i + 1] if i + 1 < steps else -1
        t_norm = torch.full((B,), t.item() / T, device=device)

        out = unet(x, cond, t_norm, ms_isht)
        x0_pred = out[0] if isinstance(out, tuple) else out

        # Soft clipping: tanh constrains to [-1, 1] (matching HR range)
        if soft_clip:
            x0_pred = torch.tanh(x0_pred)

        alpha_t = sqrt_alphas_cumprod[t]
        sigma_t = sqrt_one_minus_alphas_cumprod[t]

        # Recover noise estimate from x0 prediction
        eps_est = (x - alpha_t * x0_pred) / (sigma_t + 1e-8)

        if tp >= 0:
            alpha_p = sqrt_alphas_cumprod[tp]
            sigma_p = sqrt_one_minus_alphas_cumprod[tp]
            x = alpha_p * x0_pred + sigma_p * eps_est
        else:
            x = x0_pred  # final step: x0_pred directly

    return x


# ============================================================
# Shared training infrastructure
# ============================================================

def build_condition(lr_img, lr_up, sht_cond, isht_cond, ms_sht_isht_pairs, device):
    """Build condition (bicubic + base) and multi-scale ISHT dict."""
    base = isht_cond(sht_cond(lr_up))
    cond = torch.cat([lr_up, base], dim=1)  # [1, 6, H, W]

    ms_isht = {}
    for L_val, factor in ms_sht_isht_pairs:
        sht, isht = ms_sht_isht_pairs[(L_val, factor)]
        coeffs = sht(lr_up)
        ms_img = isht(coeffs)
        h_t, w_t = H // factor, W // factor
        ms_img = F.interpolate(ms_img, size=(h_t, w_t), mode='bilinear', align_corners=False)
        ms_isht[f'enc{factor}'] = ms_img

    return cond, base, ms_isht


def build_unet(cond_ch=6, base_ch=32):
    """Build a clean PixelUNet with all innovations disabled."""
    return PixelUNet(
        in_ch=3, cond_ch=cond_ch, base_ch=base_ch, time_dim=TIME_DIM,
        use_polar_moe=False,
        use_laplacian=False,
        use_circular_conv=False,
        use_coord_embed=False,
        use_spherical_attn=False,
        hr_size=(H, W),
    )


# ============================================================
# Experiment 1.1: Extended training + x0 prediction + soft clip
# ============================================================

def exp1_1(gpu=0, T=1000, epochs=2000, lr=LR):
    """Experiment 1.1: x0 prediction with cosine schedule, soft clip DDIM.

    Goal: establish whether cosine+x0+soft_clip can beat 0.166 baseline.
    If MSE < 0.05: proceed to 1.2; otherwise consider direct regression fallback.
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    exp_name = "exp1.1"
    exp_dir = os.path.join(OUTPUT_DIR, "stage1", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"[{exp_name}] Loading data...")
    lr_img, hr_img = get_single_image(device)
    lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)

    # Save references
    to_pil(hr_img).save(os.path.join(exp_dir, "hr.png"))
    to_pil(lr_up).save(os.path.join(exp_dir, "bicubic.png"))

    # ISHT conditioning
    sht_cond, isht_cond = get_sht_isht(device, L_COND)
    base = isht_cond(sht_cond(lr_up))
    to_pil(base).save(os.path.join(exp_dir, "base.png"))

    # Multi-scale ISHT
    ms_sht_isht_pairs = {}
    for L_val, factor in MS_COND:
        sht = RealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        isht = InverseRealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        ms_sht_isht_pairs[(L_val, factor)] = (sht, isht)

    cond, base, ms_isht = build_condition(lr_img, lr_up, sht_cond, isht_cond,
                                           ms_sht_isht_pairs, device)
    cond_ch = cond.shape[1]

    # Build U-Net
    unet = build_unet(cond_ch=cond_ch, base_ch=BASE_CH).to(device)
    n_params = sum(p.numel() for p in unet.parameters())

    # Cosine schedule
    betas = make_cosine_schedule(T).to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    # Report
    hr_var = hr_img.var().item()
    residual_var = (hr_img - base).var().item()
    print(f"\n{'='*60}")
    print(f"Experiment 1.1: Extended training + x0 prediction + soft clip")
    print(f"Params: {n_params:,}  |  base_ch={BASE_CH}  |  T={T}")
    print(f"Schedule: cosine (s=0.008)  |  Target: x0 (full HR)")
    print(f"HR var: {hr_var:.4f}  |  Residual var: {residual_var:.6f} "
          f"({hr_var/residual_var:.1f}x)")
    print(f"Condition: bicubic + base ({cond_ch}ch)")
    print(f"Training: {epochs} epochs  |  LR: {lr}")
    print(f"DDIM: 50 steps  |  Soft clip: tanh")
    print(f"Output: {exp_dir}")
    print(f"{'='*60}")

    losses = []
    sample_epochs = []
    sample_metrics = []  # list of dicts

    sample_every = max(200, epochs // 10)  # at least every 200

    pbar = tqdm(range(1, epochs + 1), desc=f"[{exp_name}]", unit="ep")
    for epoch in pbar:
        unet.train()

        # Sample random t
        B = 1
        t = torch.randint(0, T, (B,), device=device)
        alpha_t = sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sigma_t = sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)

        # Forward diffusion on HR directly
        noise = torch.randn_like(hr_img)
        noisy = alpha_t * hr_img + sigma_t * noise

        # ★ x0 prediction: U-Net outputs clean HR estimate
        x0_pred = unet(noisy, cond, t.float() / T, ms_isht)
        if isinstance(x0_pred, tuple):
            x0_pred = x0_pred[0]

        loss = F.mse_loss(x0_pred, hr_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Sample
        if epoch == 1 or epoch % sample_every == 0 or epoch == epochs:
            unet.eval()
            with torch.no_grad():
                sr = ddim_sample_x0(unet, cond, ms_isht,
                                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                                    T, steps=50, soft_clip=True)
            metrics = compute_metrics(sr, hr_img)
            sample_epochs.append(epoch)
            sample_metrics.append(metrics)
            to_pil(sr).save(os.path.join(exp_dir, f"e{epoch:04d}.png"))

            # Also generate no-clip version for comparison (every 400 epochs)
            if epoch % 400 == 0:
                sr_noclip = ddim_sample_x0(unet, cond, ms_isht,
                                           sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                                           T, steps=50, soft_clip=False)
                to_pil(sr_noclip).save(os.path.join(exp_dir, f"e{epoch:04d}_noclip.png"))

            sample_mses_for_plot = [m["mse"] for m in sample_metrics]
            update_curves(exp_name, losses, sample_epochs, sample_mses_for_plot,
                          log_dir=os.path.join(OUTPUT_DIR, "logs"))
            make_progression(exp_dir)

            pbar.set_postfix(
                train=f"{loss.item():.5f}",
                mse=f"{metrics['mse']:.5f}",
                ncc=f"{metrics['ncc']:.4f}",
                encc=f"{metrics['edge_ncc']:.4f}",
            )
        else:
            pbar.set_postfix(train=f"{loss.item():.5f}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Experiment 1.1 Results:")
    print(f"  Final train loss: {losses[-1]:.6f}")
    for ep, m in zip(sample_epochs, sample_metrics):
        print(f"  Epoch {ep:4d}: MSE={m['mse']:.6f}  NCC={m['ncc']:.4f}  Edge-NCC={m['edge_ncc']:.4f}")

    best_idx = min(range(len(sample_metrics)), key=lambda i: sample_metrics[i]["mse"])
    print(f"\n  Best: epoch {sample_epochs[best_idx]}, "
          f"MSE={sample_metrics[best_idx]['mse']:.6f}, "
          f"NCC={sample_metrics[best_idx]['ncc']:.4f}, "
          f"Edge-NCC={sample_metrics[best_idx]['edge_ncc']:.4f}")

    baseline = 0.166
    best_mse = sample_metrics[best_idx]["mse"]
    if best_mse < 0.05:
        print(f"  MSE {best_mse:.6f} < 0.05 -> SUCCESS, proceed to experiment 1.2")
    elif best_mse < baseline:
        print(f"  MSE {best_mse:.6f} < {baseline} -> improved over Phase 3 baseline")
    else:
        print(f"  MSE {best_mse:.6f} >= {baseline} -> NO improvement, consider fallback")
    print(f"{'='*60}")

    return sample_metrics[best_idx]["mse"]


# ============================================================
# Experiment 1.2: Fine-tune cosine schedule
# ============================================================

def exp1_2(gpu=0, T=500, s=0.008, epochs=500, lr=LR):
    """Experiment 1.2: Fine-tune cosine schedule (T=500 or shifted).

    Goal: find whether T=500 + cosine converges faster/better than T=1000.
    Compare against 1.1 at 500-epoch checkpoint.
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    exp_name = f"exp1.2_T{T}_s{s}"
    exp_dir = os.path.join(OUTPUT_DIR, "stage1", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"[{exp_name}] Loading data...")
    lr_img, hr_img = get_single_image(device)
    lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)

    to_pil(hr_img).save(os.path.join(exp_dir, "hr.png"))
    to_pil(lr_up).save(os.path.join(exp_dir, "bicubic.png"))

    sht_cond, isht_cond = get_sht_isht(device, L_COND)
    base = isht_cond(sht_cond(lr_up))
    to_pil(base).save(os.path.join(exp_dir, "base.png"))

    ms_sht_isht_pairs = {}
    for L_val, factor in MS_COND:
        sht = RealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        isht = InverseRealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        ms_sht_isht_pairs[(L_val, factor)] = (sht, isht)

    cond, base, ms_isht = build_condition(lr_img, lr_up, sht_cond, isht_cond,
                                           ms_sht_isht_pairs, device)
    cond_ch = cond.shape[1]

    unet = build_unet(cond_ch=cond_ch, base_ch=BASE_CH).to(device)
    n_params = sum(p.numel() for p in unet.parameters())

    # Cosine schedule with configurable T and s
    if s != 0.008:
        # Allow shifted cosine
        betas = make_cosine_schedule(T, s=s).to(device)
    else:
        betas = make_cosine_schedule(T).to(device)

    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    hr_var = hr_img.var().item()
    residual_var = (hr_img - base).var().item()
    print(f"\n{'='*60}")
    print(f"Experiment 1.2: Fine-tune cosine schedule")
    print(f"Params: {n_params:,}  |  base_ch={BASE_CH}  |  T={T}  |  s={s}")
    print(f"Schedule: cosine  |  Target: x0 (full HR)")
    print(f"HR var: {hr_var:.4f}  |  Residual var: {residual_var:.6f}")
    print(f"Condition: bicubic + base ({cond_ch}ch)")
    print(f"Training: {epochs} epochs  |  LR: {lr}")
    print(f"Output: {exp_dir}")
    print(f"{'='*60}")

    losses = []
    sample_epochs = []
    sample_metrics = []

    sample_every = max(50, epochs // 10)

    pbar = tqdm(range(1, epochs + 1), desc=f"[{exp_name}]", unit="ep")
    for epoch in pbar:
        unet.train()

        B = 1
        t = torch.randint(0, T, (B,), device=device)
        alpha_t = sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sigma_t = sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)

        noise = torch.randn_like(hr_img)
        noisy = alpha_t * hr_img + sigma_t * noise

        x0_pred = unet(noisy, cond, t.float() / T, ms_isht)
        if isinstance(x0_pred, tuple):
            x0_pred = x0_pred[0]

        loss = F.mse_loss(x0_pred, hr_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch == 1 or epoch % sample_every == 0 or epoch == epochs:
            unet.eval()
            with torch.no_grad():
                sr = ddim_sample_x0(unet, cond, ms_isht,
                                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                                    T, steps=50, soft_clip=True)
            metrics = compute_metrics(sr, hr_img)
            sample_epochs.append(epoch)
            sample_metrics.append(metrics)
            to_pil(sr).save(os.path.join(exp_dir, f"e{epoch:04d}.png"))

            sample_mses_for_plot = [m["mse"] for m in sample_metrics]
            update_curves(exp_name, losses, sample_epochs, sample_mses_for_plot,
                          log_dir=os.path.join(OUTPUT_DIR, "logs"))
            make_progression(exp_dir)

            pbar.set_postfix(
                train=f"{loss.item():.5f}",
                mse=f"{metrics['mse']:.5f}",
                ncc=f"{metrics['ncc']:.4f}",
                encc=f"{metrics['edge_ncc']:.4f}",
            )
        else:
            pbar.set_postfix(train=f"{loss.item():.5f}")

    print(f"\n{'='*60}")
    print(f"Experiment 1.2 Results (T={T}, s={s}):")
    print(f"  Final train loss: {losses[-1]:.6f}")
    for ep, m in zip(sample_epochs, sample_metrics):
        print(f"  Epoch {ep:4d}: MSE={m['mse']:.6f}  NCC={m['ncc']:.4f}  Edge-NCC={m['edge_ncc']:.4f}")

    best_idx = min(range(len(sample_metrics)), key=lambda i: sample_metrics[i]["mse"])
    print(f"\n  Best: epoch {sample_epochs[best_idx]}, "
          f"MSE={sample_metrics[best_idx]['mse']:.6f}, "
          f"NCC={sample_metrics[best_idx]['ncc']:.4f}, "
          f"Edge-NCC={sample_metrics[best_idx]['edge_ncc']:.4f}")

    # Compare: at epoch 500, T=1000 vs T=500
    print(f"\n  Decision guidance:")
    print(f"    Compare exp1.1 @ epoch 500 vs exp1.2 @ epoch {epochs}")
    print(f"    Choose the better config as strengthened base for Stage 2+.")
    print(f"{'='*60}")

    return sample_metrics[best_idx]["mse"]


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 Stage 1: Strengthen base")
    parser.add_argument("--exp", "-e", type=str, required=True,
                        choices=["1.1", "1.2"],
                        help="Experiment to run (1.1 or 1.2)")
    parser.add_argument("--gpu", "-g", type=int, default=0,
                        help="GPU device ID")
    # Exp 1.1 options
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override default epochs")
    parser.add_argument("--T", type=int, default=None,
                        help="Override default T")
    parser.add_argument("--lr", type=float, default=LR,
                        help="Learning rate")
    # Exp 1.2 options
    parser.add_argument("--s", type=float, default=0.008,
                        help="Cosine schedule shift parameter")
    args = parser.parse_args()

    os.makedirs(os.path.join(OUTPUT_DIR, "stage1"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)

    if args.exp == "1.1":
        T = args.T if args.T is not None else 1000
        epochs = args.epochs if args.epochs is not None else 2000
        exp1_1(gpu=args.gpu, T=T, epochs=epochs, lr=args.lr)
    elif args.exp == "1.2":
        T = args.T if args.T is not None else 500
        epochs = args.epochs if args.epochs is not None else 500
        exp1_2(gpu=args.gpu, T=T, s=args.s, epochs=epochs, lr=args.lr)
