"""Phase 5 Stage 6: Laplacian pyramid multi-scale supervision.

Fixed bed (inherited from Stage 1):
  - T=1000, cosine schedule, x0 prediction, tanh soft clip, HR target
  - base_ch=32, cond=bicubic+ISHT(L=255)=6ch, ms_injection='add'

Experiment 6.1: Laplacian pyramid auxiliary losses
  - U-Net has 3 output heads at different decoder levels:
    * pred_full: dec1 output → Conv→3ch @ 512×1024 (main prediction)
    * pred_L1:  dec2 output → Conv→3ch @ 256×512  (½ scale)
    * pred_L2:  dec3 output → Conv→3ch @ 128×256  (¼ scale)
  - Multi-scale loss:
    loss = MSE(pred_full, HR) + λ1*MSE(pred_L1, HR↓2) + λ2*MSE(pred_L2, HR↓4)
  - DDIM sampling uses pred_full only (compatible with x0 prediction)
  - Hypothesis: auxiliary gradients at intermediate decoder levels
    improve feature learning, leading to better high-frequency detail

Experiment 6.0: Laplacian with equal weights (control)
  - Same architecture, but λ1=λ2=1.0 (equal contribution from all scales)

Comparison baseline: Stage 1 Exp 1.1 (no laplacian pyramid)
  - Best MSE=0.01006, NCC=0.9935, Edge-NCC=0.8675

Usage:
  python phase5_stage6.py -e 6.1 -g 7    # Laplacian pyramid (decayed weights)
  python phase5_stage6.py -e 6.0 -g 6    # Laplacian pyramid (equal weights)
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
T = 1000
INFER_STEPS = 50
DATA_DIR = "lau_dataset/sun_test"
OUTPUT_DIR = "phase5_output"

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
# Metrics
# ============================================================

@torch.no_grad()
def compute_metrics(pred, target):
    pred_f = pred.float()
    target_f = target.float()
    mse = F.mse_loss(pred_f, target_f).item()

    ncc_vals = []
    for c in range(3):
        a = pred_f[:, c]
        b = target_f[:, c]
        a_m, b_m = a.mean(), b.mean()
        a_s, b_s = a.std(), b.std()
        if a_s < 1e-8 or b_s < 1e-8:
            ncc_vals.append(0.0)
        else:
            ncc_vals.append(((a - a_m) * (b - b_m)).mean().item() / (a_s * b_s).item())
    ncc = float(np.mean(ncc_vals))

    edge_ncc_vals = []
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    for c in range(3):
        a = pred_f[0:1, c:c+1]
        b = target_f[0:1, c:c+1]
        ga = torch.sqrt(F.conv2d(a, sobel_x, padding=1)**2 + F.conv2d(a, sobel_y, padding=1)**2).view(-1)
        gb = torch.sqrt(F.conv2d(b, sobel_x, padding=1)**2 + F.conv2d(b, sobel_y, padding=1)**2).view(-1)
        a_m, b_m = ga.mean(), gb.mean()
        a_s, b_s = ga.std(), gb.std()
        if a_s < 1e-8 or b_s < 1e-8:
            edge_ncc_vals.append(0.0)
        else:
            edge_ncc_vals.append(((ga - a_m) * (gb - b_m)).mean().item() / (a_s * b_s).item())
    edge_ncc = float(np.mean(edge_ncc_vals))

    return {"mse": mse, "ncc": ncc, "edge_ncc": edge_ncc}


# ============================================================
# DDIM x0-prediction sampler
# ============================================================

@torch.no_grad()
def ddim_sample_x0(unet, cond, ms_isht,
                   sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                   T, steps=50, soft_clip=True):
    B = cond.shape[0]
    device = cond.device
    x = torch.randn(B, 3, H, W, device=device)
    indices = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)

    for i in range(steps):
        t = indices[i]
        tp = indices[i + 1] if i + 1 < steps else -1
        t_norm = torch.full((B,), t.item() / T, device=device)

        out = unet(x, cond, t_norm, ms_isht)
        # Laplacian returns (pred_full, pred_L1, pred_L2); use pred_full only
        x0_pred = out[0] if isinstance(out, tuple) else out

        if soft_clip:
            x0_pred = torch.tanh(x0_pred)

        alpha_t = sqrt_alphas_cumprod[t]
        sigma_t = sqrt_one_minus_alphas_cumprod[t]
        eps_est = (x - alpha_t * x0_pred) / (sigma_t + 1e-8)

        if tp >= 0:
            alpha_p = sqrt_alphas_cumprod[tp]
            sigma_p = sqrt_one_minus_alphas_cumprod[tp]
            x = alpha_p * x0_pred + sigma_p * eps_est
        else:
            x = x0_pred
    return x


# ============================================================
# Shared infrastructure
# ============================================================

def build_condition(lr_img, lr_up, sht_cond, isht_cond, ms_sht_isht_pairs):
    base = isht_cond(sht_cond(lr_up))
    cond = torch.cat([lr_up, base], dim=1)

    ms_isht = {}
    for L_val, factor in ms_sht_isht_pairs:
        sht, isht = ms_sht_isht_pairs[(L_val, factor)]
        coeffs = sht(lr_up)
        ms_img = isht(coeffs)
        h_t, w_t = H // factor, W // factor
        ms_img = F.interpolate(ms_img, size=(h_t, w_t), mode='bilinear', align_corners=False)
        ms_isht[f'enc{factor}'] = ms_img

    return cond, base, ms_isht


def make_ms_sht_isht(device):
    pairs = {}
    for L_val, factor in MS_COND:
        sht = RealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        isht = InverseRealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        pairs[(L_val, factor)] = (sht, isht)
    return pairs


# ============================================================
# Experiment runner
# ============================================================

def run_experiment(exp_name, lambda_L1, lambda_L2, gpu, epochs,
                   baseline_mse=0.01006, baseline_ncc=0.9935, baseline_encc=0.8675):
    """Run a single Stage 6 experiment with Laplacian pyramid supervision.

    Args:
        exp_name: experiment name for output directory
        lambda_L1: weight for L1 (½ scale) auxiliary loss
        lambda_L2: weight for L2 (¼ scale) auxiliary loss
        gpu: GPU device ID
        epochs: number of training epochs
        baseline_mse/ncc/encc: Stage 1 baseline for comparison
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(OUTPUT_DIR, "stage6", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"[{exp_name}] Loading data...")
    lr_img, hr_img = get_single_image(device)
    lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)

    to_pil(hr_img).save(os.path.join(exp_dir, "hr.png"))
    to_pil(lr_up).save(os.path.join(exp_dir, "bicubic.png"))

    sht_cond, isht_cond = get_sht_isht(device, L_COND)
    base = isht_cond(sht_cond(lr_up))
    to_pil(base).save(os.path.join(exp_dir, "base.png"))

    ms_sht_isht_pairs = make_ms_sht_isht(device)
    cond, base_img, ms_isht = build_condition(lr_img, lr_up, sht_cond, isht_cond,
                                               ms_sht_isht_pairs)
    cond_ch = cond.shape[1]

    # Pre-compute downsampled HR targets for multi-scale loss
    hr_L1 = F.interpolate(hr_img, size=(H // 2, W // 2), mode='area')
    hr_L2 = F.interpolate(hr_img, size=(H // 4, W // 4), mode='area')

    # Build U-Net with Laplacian pyramid heads
    unet = PixelUNet(
        in_ch=3, cond_ch=cond_ch, base_ch=BASE_CH, time_dim=TIME_DIM,
        use_polar_moe=False,
        use_laplacian=True,
        use_circular_conv=False,
        use_coord_embed=False,
        use_spherical_attn=False,
        hr_size=(H, W),
    ).to(device)
    n_params = sum(p.numel() for p in unet.parameters())

    # Cosine schedule
    betas = make_cosine_schedule(T).to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

    # Print config
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Params: {n_params:,}  |  base_ch={BASE_CH}  |  T={T}")
    print(f"Laplacian pyramid: ON  |  λ_full=1.0  λ_L1={lambda_L1}  λ_L2={lambda_L2}")
    print(f"  pred_full: {H}×{W}  |  pred_L1: {H//2}×{W//2}  |  pred_L2: {H//4}×{W//4}")
    print(f"Schedule: cosine  |  Target: x0 (full HR) + auxiliary scales")
    print(f"Epochs: {epochs}  |  LR: {LR}")
    print(f"Baseline (Stage 1): MSE={baseline_mse:.5f} NCC={baseline_ncc:.4f} ENCC={baseline_encc:.4f}")
    print(f"Output: {exp_dir}")
    print(f"{'='*60}")

    losses = []
    losses_L1 = []
    losses_L2 = []
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

        pred_full, pred_L1, pred_L2 = unet(noisy, cond, t.float() / T, ms_isht)

        # Multi-scale loss
        loss_full = F.mse_loss(pred_full, hr_img)
        loss_L1 = F.mse_loss(pred_L1, hr_L1)
        loss_L2 = F.mse_loss(pred_L2, hr_L2)
        loss = loss_full + lambda_L1 * loss_L1 + lambda_L2 * loss_L2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        losses_L1.append(loss_L1.item())
        losses_L2.append(loss_L2.item())

        if epoch == 1 or epoch % sample_every == 0 or epoch == epochs:
            unet.eval()
            with torch.no_grad():
                sr = ddim_sample_x0(unet, cond, ms_isht,
                                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                                    T, steps=INFER_STEPS, soft_clip=True)
            metrics = compute_metrics(sr, hr_img)
            sample_epochs.append(epoch)
            sample_metrics.append(metrics)
            to_pil(sr).save(os.path.join(exp_dir, f"e{epoch:04d}.png"))

            # Save auxiliary predictions for visual inspection
            if epoch % 200 == 0:
                pred_L1_up = F.interpolate(pred_L1.detach(), size=(H, W), mode='bilinear',
                                           align_corners=False)
                pred_L2_up = F.interpolate(pred_L2.detach(), size=(H, W), mode='bilinear',
                                           align_corners=False)
                to_pil(pred_L1_up).save(os.path.join(exp_dir, f"e{epoch:04d}_L1.png"))
                to_pil(pred_L2_up).save(os.path.join(exp_dir, f"e{epoch:04d}_L2.png"))

            sample_mses_for_plot = [m["mse"] for m in sample_metrics]
            update_curves(exp_name, losses, sample_epochs, sample_mses_for_plot,
                          log_dir=os.path.join(OUTPUT_DIR, "logs"))
            make_progression(exp_dir)

            pbar.set_postfix(
                train=f"{loss.item():.5f}",
                L1=f"{loss_L1.item():.4f}",
                L2=f"{loss_L2.item():.4f}",
                mse=f"{metrics['mse']:.5f}",
                ncc=f"{metrics['ncc']:.4f}",
                encc=f"{metrics['edge_ncc']:.4f}",
            )
        else:
            pbar.set_postfix(
                train=f"{loss.item():.5f}",
                L1=f"{loss_L1.item():.4f}",
                L2=f"{loss_L2.item():.4f}",
            )

    # Final summary
    best_idx = min(range(len(sample_metrics)), key=lambda i: sample_metrics[i]["mse"])
    best = sample_metrics[best_idx]

    print(f"\n{'='*60}")
    print(f"Experiment {exp_name} Results:")
    print(f"  Final train loss: {losses[-1]:.6f}"
          f"  (full={losses[-1]-lambda_L1*losses_L1[-1]-lambda_L2*losses_L2[-1]:.6f},"
          f" L1={losses_L1[-1]:.6f}, L2={losses_L2[-1]:.6f})")
    for ep, m in zip(sample_epochs, sample_metrics):
        marker = " <-- BEST" if ep == sample_epochs[best_idx] else ""
        delta = f"  Δ={m['mse']-baseline_mse:+.5f}" if baseline_mse > 0 else ""
        print(f"  Epoch {ep:4d}: MSE={m['mse']:.6f}  NCC={m['ncc']:.4f}  "
              f"Edge-NCC={m['edge_ncc']:.4f}{delta}{marker}")

    print(f"\n  Best: epoch {sample_epochs[best_idx]}, "
          f"MSE={best['mse']:.6f}, NCC={best['ncc']:.4f}, Edge-NCC={best['edge_ncc']:.4f}")
    print(f"  vs Stage 1 baseline: ΔMSE={best['mse']-baseline_mse:+.5f}, "
          f"ΔNCC={best['ncc']-baseline_ncc:+.4f}, "
          f"ΔENCC={best['edge_ncc']-baseline_encc:+.4f}")

    if best['mse'] < baseline_mse:
        print(f"  MSE improved -> KEEP this module for future stages")
    elif best['mse'] < baseline_mse + 0.0005:
        print(f"  MSE within 0.0005 of baseline -> marginal, consider cost/benefit")
    else:
        print(f"  MSE worse -> DISCARD or adjust")
    print(f"{'='*60}")

    return best['mse'], best['ncc'], best['edge_ncc']


# ============================================================
# Experiment 6.0: Laplacian with equal weights (control)
# ============================================================

def exp6_0(gpu=0, epochs=500):
    """Laplacian pyramid with equal weights on all scales.

    All three scales contribute equally: λ_full=1.0, λ_L1=1.0, λ_L2=1.0.
    Coarser scales have fewer pixels but equal per-pixel weight.
    """
    return run_experiment(
        exp_name="exp6.0_lap_equal",
        lambda_L1=1.0,
        lambda_L2=1.0,
        gpu=gpu,
        epochs=epochs,
    )


# ============================================================
# Experiment 6.1: Laplacian with decayed weights
# ============================================================

def exp6_1(gpu=0, epochs=500):
    """Laplacian pyramid with decayed weights on coarser scales.

    λ_full=1.0, λ_L1=0.5, λ_L2=0.25.
    Full-resolution prediction is the primary objective; auxiliary scales
    provide gradient highways to intermediate decoder levels.
    """
    return run_experiment(
        exp_name="exp6.1_lap_decay",
        lambda_L1=0.5,
        lambda_L2=0.25,
        gpu=gpu,
        epochs=epochs,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 Stage 6: Laplacian pyramid")
    parser.add_argument("--exp", "-e", type=str, required=True,
                        choices=["6.0", "6.1"],
                        help="Experiment: 6.0=equal weights, 6.1=decayed weights")
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(os.path.join(OUTPUT_DIR, "stage6"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)

    if args.exp == "6.0":
        exp6_0(gpu=args.gpu, epochs=args.epochs)
    elif args.exp == "6.1":
        exp6_1(gpu=args.gpu, epochs=args.epochs)
