"""Phase 6 Stage 0: Direct UNet single-image sanity check.

Fixed bed (replicating successful Phase 4 Exp 2 precisely):
  - DirectUNet (no diffusion, no time embedding)
  - learnable latent(3ch) + cond(bicubic+base,6ch) → 9ch input
  - MS ISHT injection (add) → residual (3ch), SR = base + residual
  - base_ch=64, no panoramic modules
  - MSE loss, 500 epochs

Purpose: verify that direct regression pipeline works and can overfit
a single image (expected MSE < 0.001).

Usage:
  python phase6_stage0.py -g 7
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch_harmonics import RealSHT, InverseRealSHT
from tqdm import tqdm

from utility.data import PanoramaDataset
from model.direct_unet import DirectUNet
from vit.overfit_plot import update_curves, make_progression


# ============================================================
# Shared setup
# ============================================================

H, W = 512, 1024
L_COND = 255
SCALE = 4
BASE_CH = 64
LR = 2e-4
DATA_DIR = "lau_dataset/sun_test"
OUTPUT_DIR = "phase6_output"
EXP_NAME = "s0_direct_sanity"

MS_COND = [
    (128, 2),
    (64,  4),
    (32,  8),
]


# ============================================================
# Data helpers
# ============================================================

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

    # PSNR
    psnr = 20 * np.log10(2.0 / np.sqrt(mse)) if mse > 0 else 100.0

    return {"mse": mse, "ncc": ncc, "edge_ncc": edge_ncc, "psnr": psnr}


# ============================================================
# Build condition
# ============================================================

def build_condition(lr_up, sht_cond, isht_cond, ms_sht_isht_pairs):
    base = isht_cond(sht_cond(lr_up))
    cond = torch.cat([lr_up, base], dim=1)

    ms_isht = {}
    for L_val, factor in MS_COND:
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
# Main
# ============================================================

def run(gpu=7, epochs=1000):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(OUTPUT_DIR, EXP_NAME)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"[{EXP_NAME}] Loading data...")
    lr_img, hr_img = get_single_image(device)
    lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)

    to_pil(hr_img).save(os.path.join(exp_dir, "hr.png"))
    to_pil(lr_up).save(os.path.join(exp_dir, "bicubic.png"))

    sht_cond, isht_cond = get_sht_isht(device, L_COND)
    ms_sht_isht_pairs = make_ms_sht_isht(device)
    cond, base, ms_isht = build_condition(lr_up, sht_cond, isht_cond, ms_sht_isht_pairs)
    to_pil(base).save(os.path.join(exp_dir, "base.png"))

    # Bicubic baseline metrics
    bicubic_metrics = compute_metrics(lr_up, hr_img)
    base_metrics = compute_metrics(base, hr_img)
    print(f"Bicubic baseline: MSE={bicubic_metrics['mse']:.6f}, PSNR={bicubic_metrics['psnr']:.2f}dB")
    print(f"Base (ISHT L=255): MSE={base_metrics['mse']:.6f}, PSNR={base_metrics['psnr']:.2f}dB")

    # Build DirectUNet with MS ISHT injection + learnable latent
    model = DirectUNet(
        cond_ch=cond.shape[1],
        base_ch=BASE_CH,
        latent_ch=3,
        use_polar_moe=False,
        use_circular_conv=False,
        use_coord_embed=False,
        use_spherical_attn=False,
        hr_size=(H, W),
        ms_injection='add',
        out_ch=3,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Learnable latent + residual target = replicating Phase 4 Exp 2 design
    latent = nn.Parameter(torch.zeros(1, 3, H, W, device=device))
    residual_true = hr_img - base

    # Zero-init final conv → model starts predicting near-zero residual → SR ≈ base
    for m in model.final.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    optimizer = torch.optim.Adam(list(model.parameters()) + [latent], lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"Experiment: {EXP_NAME}")
    print(f"Params: {n_params:,}  |  base_ch={BASE_CH}")
    print(f"Architecture: DirectUNet (no diffusion, no time emb)")
    print(f"Input: latent(3ch) + cond(6ch) + MS ISHT add  |  Output: residual(3ch)")
    print(f"SR = base + residual_pred  |  Loss: MSE on residual")
    print(f"Epochs: {epochs}  |  LR: {LR} (cosine schedule)")
    print(f"Output: {exp_dir}")
    print(f"{'='*60}")

    losses = []
    sample_epochs = []
    sample_metrics = []
    sample_every = max(10, epochs // 20)

    pbar = tqdm(range(1, epochs + 1), desc=f"[{EXP_NAME}]", unit="ep")
    for epoch in pbar:
        model.train()

        pred_residual = model(cond, ms_isht, latent=latent.expand(1, -1, -1, -1))
        loss = F.mse_loss(pred_residual, residual_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if epoch == 1 or epoch % sample_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                pred_residual = model(cond, ms_isht, latent=latent.expand(1, -1, -1, -1))
                sr = base + pred_residual
            metrics = compute_metrics(sr, hr_img)
            sample_epochs.append(epoch)
            sample_metrics.append(metrics)
            to_pil(sr).save(os.path.join(exp_dir, f"e{epoch:04d}.png"))

            sample_mses_for_plot = [m["mse"] for m in sample_metrics]
            update_curves(EXP_NAME, losses, sample_epochs, sample_mses_for_plot,
                          log_dir=os.path.join(OUTPUT_DIR, "logs"))
            make_progression(exp_dir)

            pbar.set_postfix(
                train=f"{loss.item():.6f}",
                mse=f"{metrics['mse']:.6f}",
                psnr=f"{metrics['psnr']:.2f}",
                ncc=f"{metrics['ncc']:.4f}",
            )
        else:
            pbar.set_postfix(train=f"{loss.item():.6f}")

    # Final summary
    best_idx = min(range(len(sample_metrics)), key=lambda i: sample_metrics[i]["mse"])
    best = sample_metrics[best_idx]

    print(f"\n{'='*60}")
    print(f"Experiment {EXP_NAME} Results:")
    print(f"  Final train loss: {losses[-1]:.6f}")
    for ep, m in zip(sample_epochs, sample_metrics):
        marker = " <-- BEST" if ep == sample_epochs[best_idx] else ""
        print(f"  Epoch {ep:4d}: MSE={m['mse']:.6f}  PSNR={m['psnr']:.2f}  "
              f"NCC={m['ncc']:.4f}  Edge-NCC={m['edge_ncc']:.4f}{marker}")

    print(f"\n  Best: epoch {sample_epochs[best_idx]}, "
          f"MSE={best['mse']:.6f}, PSNR={best['psnr']:.2f}dB, "
          f"NCC={best['ncc']:.4f}, Edge-NCC={best['edge_ncc']:.4f}")
    print(f"  vs Bicubic: ΔMSE={best['mse']-bicubic_metrics['mse']:+.5f}, "
          f"ΔPSNR={best['psnr']-bicubic_metrics['psnr']:+.1f}dB")

    if best['mse'] < 0.001:
        print(f"  MSE < 0.001 -> Sanity check PASSED")
    else:
        print(f"  MSE >= 0.001 -> Sanity check FAILED, investigate")
    print(f"{'='*60}")

    return best['mse'], best['psnr']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6 Stage 0: Direct UNet sanity check")
    parser.add_argument("--gpu", "-g", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)
    run(gpu=args.gpu, epochs=args.epochs)
