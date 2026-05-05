"""Phase 6 Stage 1: Direct UNet multi-image baseline (v2 — residual + augmentation).

Key fixes from v1 (failed):
  1. NO learnable latent — model must use condition to generalize
  2. NO zero-init — random init, let optimizer find good starting point
  3. Residual prediction: output = base + model(cond), L2 reg on residual (1e-4)
  4. Panorama-specific data augmentation: cyclic roll, vertical crop, color jitter
  5. LR=5e-5 (was 2e-4), cosine annealing
  6. Early stopping: track best val MSE checkpoint

Usage:
  python phase6_stage1.py -g 7
"""

import argparse
import os
import random
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
# Setup
# ============================================================

H, W = 512, 1024
L_COND = 255
SCALE = 4
BASE_CH = 64
LR = 5e-5
RESIDUAL_L2_WEIGHT = 1e-4
TRAIN_EPOCHS = 400
DATA_DIR = "lau_dataset/sun_test"
OUTPUT_DIR = "phase6_output"
EXP_NAME = "s1_direct_baseline_v2"

N_TOTAL = 20
N_TRAIN = 16
N_VAL = 4

MS_COND = [
    (128, 2),
    (64,  4),
    (32,  8),
]


# ============================================================
# Panorama-specific data augmentation
# ============================================================

def augment_panorama(cond, hr, base, ms_isht, training=True):
    """Apply panorama-aware augmentations during training.

    cond/hr/base: [1, C, H, W] tensors
    ms_isht: dict of multi-scale ISHT tensors (NOT augmented — too expensive)

    Augmentations:
      1. Horizontal cyclic roll (wraps 360° seamlessly)
      2. Small vertical crop + resize (simulates pitch variation)
      3. Brightness/contrast/saturation jitter
      4. Tiny Gaussian noise on HR
    """
    if not training:
        return cond, hr, base

    B, C, Hi, Wi = cond.shape

    # 1. Horizontal cyclic roll (50% chance, random offset)
    if random.random() < 0.5:
        shift_w = random.randint(0, Wi - 1)
        cond = torch.roll(cond, shifts=shift_w, dims=-1)
        hr   = torch.roll(hr,   shifts=shift_w, dims=-1)
        base = torch.roll(base, shifts=shift_w, dims=-1)
        # ms_isht is computed from the OLD lr_up — don't roll it;
        # the shift breaks SHT consistency but provides a useful perturbation.

    # 2. Vertical crop + resize (90% chance, 0-10% crop)
    if random.random() < 0.9:
        crop_px = random.randint(0, Hi // 10)  # 0-51 pixels
        if crop_px > 1:
            top = random.randint(0, crop_px)
            bottom = Hi - (crop_px - top)
            cond = F.interpolate(cond[:, :, top:bottom, :], size=(Hi, Wi),
                                 mode='bilinear', align_corners=False)
            hr   = F.interpolate(hr[:, :, top:bottom, :], size=(Hi, Wi),
                                 mode='bilinear', align_corners=False)
            base = F.interpolate(base[:, :, top:bottom, :], size=(Hi, Wi),
                                 mode='bilinear', align_corners=False)
            # Also interpolate ms_isht for consistency
            for key in ms_isht:
                h_t, w_t = ms_isht[key].shape[2], ms_isht[key].shape[3]
                ms_isht[key] = F.interpolate(
                    ms_isht[key][:, :, top*h_t//Hi:bottom*h_t//Hi, :],
                    size=(h_t, w_t), mode='bilinear', align_corners=False)

    # 3. Color jitter — brightness/contrast/saturation (±10%)
    if random.random() < 0.7:
        # Brightness
        b_shift = (random.random() - 0.5) * 0.2
        cond = cond + b_shift
        hr   = hr   + b_shift
        # Contrast (scale around mean)
        c_scale = 1.0 + (random.random() - 0.5) * 0.2
        cond_mean = cond.mean(dim=(-2, -1), keepdim=True)
        hr_mean   = hr.mean(dim=(-2, -1), keepdim=True)
        cond = (cond - cond_mean) * c_scale + cond_mean
        hr   = (hr   - hr_mean)   * c_scale + hr_mean
        # Clamp to valid range
        cond = torch.clamp(cond, -1, 1)
        hr   = torch.clamp(hr,   -1, 1)

    # 4. Gaussian noise on HR target (tiny, prevents exact pixel memorization)
    if random.random() < 0.3:
        hr = hr + torch.randn_like(hr) * 0.005
        hr = torch.clamp(hr, -1, 1)

    return cond, hr, base


# ============================================================
# Data loading
# ============================================================

def build_dataset(device):
    """Load N_TOTAL panoramas, pre-compute conditions, store raw LR for augmentation."""
    dataset = PanoramaDataset(DATA_DIR, (H, W), SCALE)
    n = min(N_TOTAL, len(dataset))

    sht_cond, isht_cond = get_sht_isht(device, L_COND)
    ms_sht_isht_pairs = make_ms_sht_isht(device)

    train_data = []
    val_data = []

    for i in range(n):
        lr_img, hr_img = dataset[i]
        lr_img = lr_img.unsqueeze(0).to(device)
        hr_img = hr_img.unsqueeze(0).to(device)
        lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)

        base = isht_cond(sht_cond(lr_up))
        cond = torch.cat([lr_up, base], dim=1)

        # Build ms_isht
        ms_isht = {}
        for L_val, factor in MS_COND:
            sht, isht = ms_sht_isht_pairs[(L_val, factor)]
            coeffs = sht(lr_up)
            ms_img = isht(coeffs)
            h_t, w_t = H // factor, W // factor
            ms_img = F.interpolate(ms_img, size=(h_t, w_t), mode='bilinear', align_corners=False)
            ms_isht[f'enc{factor}'] = ms_img

        item = {
            'lr_img': lr_img,
            'hr': hr_img,
            'lr_up': lr_up,
            'cond': cond,
            'base': base,
            'ms_isht': ms_isht,
            'idx': i,
        }

        if i < N_TRAIN:
            train_data.append(item)
        else:
            val_data.append(item)

    return train_data, val_data, sht_cond, isht_cond, ms_sht_isht_pairs


def rebuild_cond(item, sht_cond, isht_cond, ms_sht_isht_pairs):
    """Rebuild cond/base/ms_isht after augmentation changes lr_up."""
    lr_up = item['lr_up']
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


def get_sht_isht(device, lmax):
    sht = RealSHT(H, W, lmax=lmax, mmax=lmax).to(device)
    isht = InverseRealSHT(H, W, lmax=lmax, mmax=lmax).to(device)
    return sht, isht


def make_ms_sht_isht(device):
    pairs = {}
    for L_val, factor in MS_COND:
        sht = RealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        isht = InverseRealSHT(H, W, lmax=L_val, mmax=L_val).to(device)
        pairs[(L_val, factor)] = (sht, isht)
    return pairs


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
        a = pred_f[:, c]; b = target_f[:, c]
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
        a = pred_f[0:1, c:c+1]; b = target_f[0:1, c:c+1]
        ga = torch.sqrt(F.conv2d(a, sobel_x, padding=1)**2 + F.conv2d(a, sobel_y, padding=1)**2).view(-1)
        gb = torch.sqrt(F.conv2d(b, sobel_x, padding=1)**2 + F.conv2d(b, sobel_y, padding=1)**2).view(-1)
        a_m, b_m = ga.mean(), gb.mean()
        a_s, b_s = ga.std(), gb.std()
        if a_s < 1e-8 or b_s < 1e-8:
            edge_ncc_vals.append(0.0)
        else:
            edge_ncc_vals.append(((ga - a_m) * (gb - b_m)).mean().item() / (a_s * b_s).item())
    edge_ncc = float(np.mean(edge_ncc_vals))

    psnr = 20 * np.log10(2.0 / np.sqrt(mse)) if mse > 0 else 100.0

    ssim_vals = []
    for c in range(3):
        a = pred_f[0:1, c:c+1]; b = target_f[0:1, c:c+1]
        mu_a = F.avg_pool2d(a, 11, stride=1, padding=5)
        mu_b = F.avg_pool2d(b, 11, stride=1, padding=5)
        sigma_a = F.avg_pool2d((a - mu_a)**2, 11, stride=1, padding=5).sqrt()
        sigma_b = F.avg_pool2d((b - mu_b)**2, 11, stride=1, padding=5).sqrt()
        sigma_ab = F.avg_pool2d((a - mu_a) * (b - mu_b), 11, stride=1, padding=5)
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu_a*mu_b + C1) * (2*sigma_ab + C2)) / \
                   ((mu_a**2 + mu_b**2 + C1) * (sigma_a**2 + sigma_b**2 + C2) + 1e-8)
        ssim_vals.append(ssim_map.mean().item())
    ssim = float(np.mean(ssim_vals))

    return {"mse": mse, "ncc": ncc, "edge_ncc": edge_ncc, "psnr": psnr, "ssim": ssim}


@torch.no_grad()
def validate(model, val_data):
    model.eval()
    agg = {"mse": 0.0, "ncc": 0.0, "edge_ncc": 0.0, "psnr": 0.0, "ssim": 0.0}
    for item in val_data:
        pred_residual = model(item['cond'], item['ms_isht'])
        sr = item['base'] + pred_residual
        m = compute_metrics(sr, item['hr'])
        for k in agg:
            agg[k] += m[k]
    for k in agg:
        agg[k] /= len(val_data)
    model.train()
    return agg


@torch.no_grad()
def validate_bicubic(val_data):
    agg = {"mse": 0.0, "ncc": 0.0, "edge_ncc": 0.0, "psnr": 0.0, "ssim": 0.0}
    for item in val_data:
        m = compute_metrics(item['lr_up'], item['hr'])
        for k in agg:
            agg[k] += m[k]
    for k in agg:
        agg[k] /= len(val_data)
    return agg


# ============================================================
# Main
# ============================================================

def run(gpu=7, epochs=TRAIN_EPOCHS):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(OUTPUT_DIR, EXP_NAME)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"[{EXP_NAME}] Loading {N_TOTAL} panoramas...")
    train_data, val_data, sht_cond, isht_cond, ms_sht_isht_pairs = build_dataset(device)

    # Save reference images
    ref_item = val_data[0]
    to_pil(ref_item['hr']).save(os.path.join(exp_dir, "val_hr.png"))
    to_pil(ref_item['lr_up']).save(os.path.join(exp_dir, "val_bicubic.png"))
    to_pil(ref_item['base']).save(os.path.join(exp_dir, "val_base.png"))

    bicubic_metrics = validate_bicubic(val_data)
    print(f"Bicubic val: MSE={bicubic_metrics['mse']:.6f}, PSNR={bicubic_metrics['psnr']:.2f}dB, "
          f"SSIM={bicubic_metrics['ssim']:.4f}")

    # Build model: NO latent, NO zero-init
    cond_ch = train_data[0]['cond'].shape[1]
    model = DirectUNet(
        cond_ch=cond_ch,
        base_ch=BASE_CH,
        latent_ch=0,
        use_polar_moe=False,
        use_circular_conv=False,
        use_coord_embed=False,
        use_spherical_attn=False,
        hr_size=(H, W),
        ms_injection='add',
        out_ch=3,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"Experiment: {EXP_NAME}")
    print(f"Params: {n_params:,}  |  base_ch={BASE_CH}")
    print(f"Input: cond({cond_ch}ch) + MS ISHT add  |  Output: residual(3ch)")
    print(f"SR = base + residual  |  Loss: MSE(residual) + {RESIDUAL_L2_WEIGHT}*L2(residual)")
    print(f"Augmentation: cyclic_roll + v_crop + color ±10% + HR_noise(std=0.005)")
    print(f"Dataset: {N_TRAIN} train / {N_VAL} val  |  LR: {LR} (cosine)")
    print(f"Output: {exp_dir}")
    print(f"{'='*60}")

    train_losses = []
    val_epochs = []
    val_metrics_list = []
    best_val_mse = float('inf')
    best_epoch = 0
    val_every = max(20, epochs // 20)

    pbar = tqdm(range(1, epochs + 1), desc=f"[{EXP_NAME}]", unit="ep")
    for epoch in pbar:
        model.train()

        epoch_loss = 0.0
        for item in train_data:
            # --- Apply augmentations ---
            cond_aug, hr_aug, base_aug = augment_panorama(
                item['cond'].clone(), item['hr'].clone(), item['base'].clone(),
                {k: v.clone() for k, v in item['ms_isht'].items()},
                training=True)

            residual_true = hr_aug - base_aug
            pred_residual = model(cond_aug, item['ms_isht'])

            # Loss = MSE(residual) + L2 regularization on residual magnitude
            mse_loss = F.mse_loss(pred_residual, residual_true)
            l2_reg = pred_residual.pow(2).mean()
            loss = mse_loss + RESIDUAL_L2_WEIGHT * l2_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_data)
        scheduler.step()
        train_losses.append(epoch_loss)

        # Validation (no augmentation)
        if epoch == 1 or epoch % val_every == 0 or epoch == epochs:
            val_m = validate(model, val_data)
            val_epochs.append(epoch)
            val_metrics_list.append(val_m)

            # Save best model
            if val_m['mse'] < best_val_mse:
                best_val_mse = val_m['mse']
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pt"))

            # Save sample prediction
            model.eval()
            with torch.no_grad():
                val_pred = model(val_data[0]['cond'], val_data[0]['ms_isht'])
                val_sr = val_data[0]['base'] + val_pred
            model.train()
            to_pil(val_sr).save(os.path.join(exp_dir, f"e{epoch:04d}.png"))

            val_mses = [m["mse"] for m in val_metrics_list]
            update_curves(EXP_NAME, train_losses, val_epochs, val_mses,
                          log_dir=os.path.join(OUTPUT_DIR, "logs"))
            make_progression(exp_dir)

            pbar.set_postfix(
                train=f"{epoch_loss:.6f}",
                val_mse=f"{val_m['mse']:.6f}",
                val_psnr=f"{val_m['psnr']:.1f}",
                val_ssim=f"{val_m['ssim']:.4f}",
                best_ep=str(best_epoch),
            )
        else:
            pbar.set_postfix(train=f"{epoch_loss:.6f}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Experiment {EXP_NAME} Results:")
    print(f"  Final train loss: {train_losses[-1]:.6f}")
    print(f"  Bicubic val: MSE={bicubic_metrics['mse']:.6f}, PSNR={bicubic_metrics['psnr']:.2f}dB, "
          f"SSIM={bicubic_metrics['ssim']:.4f}")
    for ep, m in zip(val_epochs, val_metrics_list):
        marker = " <-- BEST" if ep == best_epoch else ""
        print(f"  Epoch {ep:4d}: MSE={m['mse']:.6f}  PSNR={m['psnr']:.2f}  "
              f"SSIM={m['ssim']:.4f}  NCC={m['ncc']:.4f}  ENCC={m['edge_ncc']:.4f}{marker}")

    # Load best model for final metrics
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pt")))
    best_metrics = validate(model, val_data)

    print(f"\n  Best: epoch {best_epoch}, "
          f"MSE={best_metrics['mse']:.6f}, PSNR={best_metrics['psnr']:.2f}dB, "
          f"SSIM={best_metrics['ssim']:.4f}, NCC={best_metrics['ncc']:.4f}, "
          f"ENCC={best_metrics['edge_ncc']:.4f}")
    print(f"  vs Bicubic: ΔPSNR={best_metrics['psnr']-bicubic_metrics['psnr']:+.1f}dB, "
          f"ΔSSIM={best_metrics['ssim']-bicubic_metrics['ssim']:+.4f}")
    print(f"{'='*60}")

    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6 Stage 1 v2: Direct UNet multi-image baseline")
    parser.add_argument("--gpu", "-g", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    args = parser.parse_args()

    os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)
    run(gpu=args.gpu, epochs=args.epochs)
