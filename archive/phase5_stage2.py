"""Phase 5 Stage 2: Condition enhancement — Multi-scale ISHT injection.

Fixed bed (inherited from Stage 1):
  - T=1000, cosine schedule, x0 prediction, tanh soft clip, HR target
  - base_ch=32, cond=bicubic+ISHT(L=255)=6ch

Experiment 2.1: Learnable scaling (init=0.1)
  - MS ISHT injection with per-level learnable scale, starting at 0.1
  - Hypothesis: learnable scale finds optimal injection strength per resolution

Experiment 2.2: FiLM injection
  - ISHT features generate per-channel γ, β for feature modulation
  - Hypothesis: FiLM gives more expressive conditioning than simple addition

Comparison baseline: Stage 1 Exp 1.1 (ms_injection='add', i.e., scale=1.0)
  - Best MSE=0.01006, NCC=0.9935, Edge-NCC=0.8675

Usage:
  python phase5_stage2.py -e 2.1 -g 7    # Learnable scaling
  python phase5_stage2.py -e 2.2 -g 6    # FiLM injection
  python phase5_stage2.py -e 2.0 -g 5    # Control (no MS ISHT)
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
# Shared setup (identical to Stage 1)
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

MS_COND = [(128, 2), (64, 4), (32, 8)]


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
# Metrics (identical to Stage 1)
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
# DDIM x0-prediction sampler (identical to Stage 1)
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
# Shared experiment runner
# ============================================================

def run_experiment(exp_name, ms_injection, ms_scale_init, gpu, epochs,
                   baseline_mse=0.01006, baseline_ncc=0.9935, baseline_encc=0.8675):
    """Run a single Stage 2 experiment with the given MS injection config.

    Returns (best_mse, best_ncc, best_encc) from checkpoint sampling.
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(OUTPUT_DIR, "stage2", exp_name)
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

    # Build U-Net with specified injection type
    unet = PixelUNet(
        in_ch=3, cond_ch=cond_ch, base_ch=BASE_CH, time_dim=TIME_DIM,
        use_polar_moe=False, use_laplacian=False,
        use_circular_conv=False, use_coord_embed=False, use_spherical_attn=False,
        hr_size=(H, W),
        ms_injection=ms_injection, ms_scale_init=ms_scale_init,
    ).to(device)
    n_params = sum(p.numel() for p in unet.parameters())

    # Report learnable scale params if applicable
    has_scales = hasattr(unet, 'ms_scales') and unet.ms_scales is not None

    # Cosine schedule
    betas = make_cosine_schedule(T).to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

    # Print config
    scale_info = f"init={ms_scale_init}" if has_scales else "fixed"
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Params: {n_params:,}  |  base_ch={BASE_CH}  |  T={T}")
    print(f"MS injection: {ms_injection}  |  Scale: {scale_info}")
    print(f"Schedule: cosine  |  Target: x0 (full HR)")
    print(f"Epochs: {epochs}  |  LR: {LR}")
    print(f"Baseline (Stage 1): MSE={baseline_mse:.5f} NCC={baseline_ncc:.4f} ENCC={baseline_encc:.4f}")
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
                                    T, steps=INFER_STEPS, soft_clip=True)
            metrics = compute_metrics(sr, hr_img)
            sample_epochs.append(epoch)
            sample_metrics.append(metrics)
            to_pil(sr).save(os.path.join(exp_dir, f"e{epoch:04d}.png"))

            # Log learned scales
            scale_log = ""
            if has_scales:
                scales = {k: f"{v.item():.4f}" for k, v in unet.ms_scales.items()}
                scale_log = f"  scales={scales}"

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
            if scale_log:
                tqdm.write(f"  [e{epoch}]{scale_log}")
        else:
            pbar.set_postfix(train=f"{loss.item():.5f}")

    # Final summary
    best_idx = min(range(len(sample_metrics)), key=lambda i: sample_metrics[i]["mse"])
    best = sample_metrics[best_idx]

    print(f"\n{'='*60}")
    print(f"Experiment {exp_name} Results:")
    print(f"  Final train loss: {losses[-1]:.6f}")
    for ep, m in zip(sample_epochs, sample_metrics):
        marker = " <-- BEST" if ep == sample_epochs[best_idx] else ""
        delta = ""
        if baseline_mse > 0:
            delta = f"  Δ={m['mse']-baseline_mse:+.5f}"
        print(f"  Epoch {ep:4d}: MSE={m['mse']:.6f}  NCC={m['ncc']:.4f}  "
              f"Edge-NCC={m['edge_ncc']:.4f}{delta}{marker}")

    if has_scales:
        final_scales = {k: f"{v.item():.4f}" for k, v in unet.ms_scales.items()}
        print(f"  Final scales: {final_scales}")

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
# Experiment 2.1: Learnable scaling MS ISHT (init=0.1)
# ============================================================

def exp2_1(gpu=0, epochs=500):
    """MS ISHT with per-level learnable scale initialized at 0.1.

    Each encoder level has a learnable parameter that scales the ISHT
    feature contribution. Starting from 0.1 means ISHT initially has weak
    influence, and the model learns how much to trust the frequency prior.
    """
    return run_experiment(
        exp_name="exp2.1_scaled_add",
        ms_injection="scaled_add",
        ms_scale_init=0.1,
        gpu=gpu,
        epochs=epochs,
    )


# ============================================================
# Experiment 2.2: FiLM injection
# ============================================================

def exp2_2(gpu=0, epochs=500):
    """FiLM-based ISHT modulation.

    ISHT features are projected to per-channel γ (scale) and β (shift),
    then applied to U-Net features: output = γ * x + β.

    More expressive than simple addition since ISHT can both amplify
    and suppress individual feature channels based on frequency content.
    """
    return run_experiment(
        exp_name="exp2.2_film",
        ms_injection="film",
        ms_scale_init=0.0,  # not used for film
        gpu=gpu,
        epochs=epochs,
    )


# ============================================================
# Experiment 2.0: Control — no MS ISHT injection
# ============================================================

def exp2_0(gpu=0, epochs=500):
    """Control: no multi-scale ISHT injection.

    Measures the true contribution of MS ISHT by removing it entirely.
    All other conditions (bicubic + ISHT base) remain unchanged.
    """
    return run_experiment(
        exp_name="exp2.0_no_ms",
        ms_injection="none",
        ms_scale_init=0.0,
        gpu=gpu,
        epochs=epochs,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 Stage 2: MS ISHT injection")
    parser.add_argument("--exp", "-e", type=str, required=True,
                        choices=["2.0", "2.1", "2.2"],
                        help="Experiment: 2.0=control(no MS), 2.1=scaled_add, 2.2=film")
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(os.path.join(OUTPUT_DIR, "stage2"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)

    if args.exp == "2.0":
        exp2_0(gpu=args.gpu, epochs=args.epochs)
    elif args.exp == "2.1":
        exp2_1(gpu=args.gpu, epochs=args.epochs)
    elif args.exp == "2.2":
        exp2_2(gpu=args.gpu, epochs=args.epochs)
