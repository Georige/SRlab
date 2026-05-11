"""Phase 6 Stage 2: Re-validate panoramic modules on multi-image generalization.

Each experiment adds ONE module to the Stage 1 baseline (v2 recipe), trains 400 epochs,
and compares against the baseline (s1_direct_baseline_v2) on val metrics.

v2 recipe (matching Stage 1 v2 success):
  - NO learnable latent — model must generalize from condition
  - NO zero-init — random init, let optimizer find good starting point
  - Residual prediction: output = base + model(cond), L2 reg on residual (1e-4)
  - Panorama-specific data augmentation: cyclic roll, vertical crop, color jitter
  - LR=5e-5, cosine annealing, 400 epochs
  - Early stopping: track best val MSE checkpoint

Experiments:
  2.1  +MS ISHT injection (add)      — matches baseline (sanity check)
  2.2  +PolarMoE (bal λ=0.01)        — polar/equatorial expert split
  2.3  +CoordEmbed (24ch)            — spherical position encoding
  2.4  +CircularConv                 — circular W-padding
  2.5  +SphericalAttention           — axial self-attention at bottleneck

Decision: keep if val PSNR improves >0.1dB vs baseline (25.73dB).

Usage:
  python phase6_stage2.py -e 2.1 -g 7
  python phase6_stage2.py -e 2.2 -g 6
  ...
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
# Shared setup
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

N_TOTAL = 20
N_TRAIN = 16
N_VAL = 4

MS_COND = [
    (128, 2),
    (64,  4),
    (32,  8),
]

# Baseline metrics from Stage 1 v2 (best epoch 580)
BASELINE_MSE = 0.010763
BASELINE_PSNR = 25.73
BASELINE_SSIM = 0.6238

MODULE_CONFIGS = {
    "2.1": {
        "name": "s2.1_ms_isht",
        "desc": "Multi-scale ISHT injection (add) — baseline sanity check",
        "use_polar_moe": False,
        "use_circular_conv": False,
        "use_coord_embed": False,
        "use_spherical_attn": False,
        "ms_injection": "add",
        "balance_weight": 0.0,
    },
    "2.2": {
        "name": "s2.2_polar_moe",
        "desc": "PolarMoE with load balancing (λ=0.01)",
        "use_polar_moe": True,
        "use_circular_conv": False,
        "use_coord_embed": False,
        "use_spherical_attn": False,
        "ms_injection": "add",
        "balance_weight": 0.01,
    },
    "2.3": {
        "name": "s2.3_coord_embed",
        "desc": "CoordEmbed (24ch spherical Fourier features)",
        "use_polar_moe": False,
        "use_circular_conv": False,
        "use_coord_embed": True,
        "use_spherical_attn": False,
        "ms_injection": "add",
        "balance_weight": 0.0,
    },
    "2.4": {
        "name": "s2.4_circular_conv",
        "desc": "CircularConv (W-direction circular padding)",
        "use_polar_moe": False,
        "use_circular_conv": True,
        "use_coord_embed": False,
        "use_spherical_attn": False,
        "ms_injection": "add",
        "balance_weight": 0.0,
    },
    "2.5": {
        "name": "s2.5_spherical_attn",
        "desc": "SphericalAttention (axial self-attn at bottleneck)",
        "use_polar_moe": False,
        "use_circular_conv": False,
        "use_coord_embed": False,
        "use_spherical_attn": True,
        "ms_injection": "add",
        "balance_weight": 0.0,
    },
}


# ============================================================
# Panorama-specific data augmentation
# ============================================================

def augment_panorama(cond, hr, base, ms_isht, training=True):
    """Apply panorama-aware augmentations during training.
    Same as Stage 1 v2."""
    if not training:
        return cond, hr, base

    B, C, Hi, Wi = cond.shape

    # 1. Horizontal cyclic roll (50% chance)
    if random.random() < 0.5:
        shift_w = random.randint(0, Wi - 1)
        cond = torch.roll(cond, shifts=shift_w, dims=-1)
        hr   = torch.roll(hr,   shifts=shift_w, dims=-1)
        base = torch.roll(base, shifts=shift_w, dims=-1)

    # 2. Vertical crop + resize (90% chance, 0-10% crop)
    if random.random() < 0.9:
        crop_px = random.randint(0, Hi // 10)
        if crop_px > 1:
            top = random.randint(0, crop_px)
            bottom = Hi - (crop_px - top)
            cond = F.interpolate(cond[:, :, top:bottom, :], size=(Hi, Wi),
                                 mode='bilinear', align_corners=False)
            hr   = F.interpolate(hr[:, :, top:bottom, :], size=(Hi, Wi),
                                 mode='bilinear', align_corners=False)
            base = F.interpolate(base[:, :, top:bottom, :], size=(Hi, Wi),
                                 mode='bilinear', align_corners=False)
            for key in ms_isht:
                h_t, w_t = ms_isht[key].shape[2], ms_isht[key].shape[3]
                ms_isht[key] = F.interpolate(
                    ms_isht[key][:, :, top*h_t//Hi:bottom*h_t//Hi, :],
                    size=(h_t, w_t), mode='bilinear', align_corners=False)

    # 3. Color jitter — brightness/contrast (±10%)
    if random.random() < 0.7:
        b_shift = (random.random() - 0.5) * 0.2
        cond = cond + b_shift
        hr   = hr   + b_shift
        c_scale = 1.0 + (random.random() - 0.5) * 0.2
        cond_mean = cond.mean(dim=(-2, -1), keepdim=True)
        hr_mean   = hr.mean(dim=(-2, -1), keepdim=True)
        cond = (cond - cond_mean) * c_scale + cond_mean
        hr   = (hr   - hr_mean)   * c_scale + hr_mean
        cond = torch.clamp(cond, -1, 1)
        hr   = torch.clamp(hr,   -1, 1)

    # 4. Gaussian noise on HR target (30% chance)
    if random.random() < 0.3:
        hr = hr + torch.randn_like(hr) * 0.005
        hr = torch.clamp(hr, -1, 1)

    return cond, hr, base


# ============================================================
# Data helpers
# ============================================================

def build_dataset(device, n_train=N_TRAIN, n_val=N_VAL):
    dataset = PanoramaDataset(DATA_DIR, (H, W), SCALE)
    n = min(n_train + n_val, len(dataset))

    sht_cond, isht_cond = get_sht_isht(device, L_COND)
    ms_sht_isht_pairs = make_ms_sht_isht(device)

    train_data = []
    val_data = []

    for i in range(n):
        lr_img, hr_img = dataset[i]
        lr_img, hr_img = lr_img.unsqueeze(0).to(device), hr_img.unsqueeze(0).to(device)
        lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)

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

        item = {
            'cond': cond, 'hr': hr_img, 'ms_isht': ms_isht,
            'lr_up': lr_up, 'base': base, 'idx': i,
        }

        if i < n_train:
            train_data.append(item)
        else:
            val_data.append(item)

    return train_data, val_data


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
def validate(model, val_data, ms_injection='add'):
    model.eval()
    agg = {"mse": 0.0, "ncc": 0.0, "edge_ncc": 0.0, "psnr": 0.0, "ssim": 0.0}
    for item in val_data:
        ms_isht = item['ms_isht'] if ms_injection != 'none' else None
        pred_residual = model(item['cond'], ms_isht)
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
# Experiment runner
# ============================================================

def run_experiment(exp_key, gpu, epochs=TRAIN_EPOCHS, resume_ckpt=None, start_epoch=0):
    cfg = MODULE_CONFIGS[exp_key]
    exp_name = cfg["name"]
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(OUTPUT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"[{exp_name}] Loading {N_TRAIN+N_VAL} panoramas...")
    train_data, val_data = build_dataset(device)

    # Save reference images (only if fresh start)
    if start_epoch == 0:
        ref_item = val_data[0]
        to_pil(ref_item['hr']).save(os.path.join(exp_dir, "val_hr.png"))
        to_pil(ref_item['lr_up']).save(os.path.join(exp_dir, "val_bicubic.png"))
        to_pil(ref_item['base']).save(os.path.join(exp_dir, "val_base.png"))

    bicubic_metrics = validate_bicubic(val_data)
    print(f"Bicubic val: MSE={bicubic_metrics['mse']:.6f}, PSNR={bicubic_metrics['psnr']:.2f}dB, "
          f"SSIM={bicubic_metrics['ssim']:.4f}")

    # Build model: v2 recipe — NO latent, NO zero-init
    cond_ch = train_data[0]['cond'].shape[1]
    model = DirectUNet(
        cond_ch=cond_ch,
        base_ch=BASE_CH,
        latent_ch=0,
        use_polar_moe=cfg["use_polar_moe"],
        use_circular_conv=cfg["use_circular_conv"],
        use_coord_embed=cfg["use_coord_embed"],
        use_spherical_attn=cfg["use_spherical_attn"],
        hr_size=(H, W),
        ms_injection=cfg["ms_injection"],
        out_ch=3,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Load checkpoint if resuming
    if resume_ckpt is not None:
        print(f"Resuming from checkpoint: {resume_ckpt}")
        model.load_state_dict(torch.load(resume_ckpt))
        best_val_mse = validate(model, val_data, ms_injection=cfg["ms_injection"])['mse']
        best_epoch = start_epoch
        print(f"Loaded model: start_epoch={start_epoch}, init_val_mse={best_val_mse:.6f}")
    else:
        best_val_mse = float('inf')
        best_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    resume_tag = f" (resume from {start_epoch})" if start_epoch > 0 else ""
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name} — {cfg['desc']}{resume_tag}")
    print(f"Params: {n_params:,}  |  base_ch={BASE_CH}")
    print(f"Modules: PolarMoE={cfg['use_polar_moe']}  "
          f"CoordEmbed={cfg['use_coord_embed']}  "
          f"CircularConv={cfg['use_circular_conv']}  "
          f"SphericalAttn={cfg['use_spherical_attn']}")
    print(f"MS ISHT: {cfg['ms_injection']}  |  Balance weight: {cfg['balance_weight']}")
    print(f"Dataset: {N_TRAIN} train / {N_VAL} val  |  LR: {LR} (cosine, {epochs} epochs)")
    print(f"Loss: MSE(residual) + {RESIDUAL_L2_WEIGHT}*L2(residual)"
          f"{' + bal*load_balance' if cfg['balance_weight'] > 0 else ''}")
    print(f"Augmentation: cyclic_roll + v_crop + color ±10% + HR_noise(std=0.005)")
    print(f"Baseline (Stage 1 v2): PSNR={BASELINE_PSNR:.2f}dB, SSIM={BASELINE_SSIM:.4f}")
    print(f"Output: {exp_dir}")
    print(f"{'='*60}")

    train_losses = []
    val_epochs = []
    val_metrics_list = []
    val_every = max(20, epochs // 20)

    pbar = tqdm(range(1, epochs + 1), desc=f"[{exp_name}]", unit="ep")
    for epoch in pbar:
        global_epoch = start_epoch + epoch

        model.train()

        epoch_loss = 0.0
        epoch_bal = 0.0
        bal_weight = cfg["balance_weight"]
        ms_isht_none = cfg["ms_injection"] == 'none'

        for item in train_data:
            # --- Apply augmentations ---
            cond_aug, hr_aug, base_aug = augment_panorama(
                item['cond'].clone(), item['hr'].clone(), item['base'].clone(),
                {k: v.clone() for k, v in item['ms_isht'].items()},
                training=True)

            residual_true = hr_aug - base_aug
            ms_isht_in = None if ms_isht_none else item['ms_isht']
            pred_residual = model(cond_aug, ms_isht_in)

            # Loss = MSE(residual) + L2 regularization
            mse_loss = F.mse_loss(pred_residual, residual_true)
            l2_reg = pred_residual.pow(2).mean()
            loss = mse_loss + RESIDUAL_L2_WEIGHT * l2_reg

            # PolarMoE load balance loss
            if bal_weight > 0 and cfg["use_polar_moe"]:
                bal_loss = model.load_balance_loss()
                epoch_bal += float(bal_loss)
                loss = loss + bal_weight * bal_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_data)
        if cfg["use_polar_moe"]:
            epoch_bal /= len(train_data)
        train_losses.append(epoch_loss)
        scheduler.step()

        # Validation (no augmentation)
        if epoch == 1 or epoch % val_every == 0 or epoch == epochs:
            val_m = validate(model, val_data, ms_injection=cfg["ms_injection"])
            val_epochs.append(global_epoch)
            val_metrics_list.append(val_m)

            # Save best model (early stopping)
            if val_m['mse'] < best_val_mse:
                best_val_mse = val_m['mse']
                best_epoch = global_epoch
                torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pt"))

            # Save sample prediction
            model.eval()
            with torch.no_grad():
                ms_isht_infer = None if ms_isht_none else val_data[0]['ms_isht']
                val_pred = model(val_data[0]['cond'], ms_isht_infer)
                val_sr = val_data[0]['base'] + val_pred
            model.train()
            to_pil(val_sr).save(os.path.join(exp_dir, f"e{global_epoch:04d}.png"))

            val_mses = [m["mse"] for m in val_metrics_list]
            update_curves(exp_name, train_losses, val_epochs, val_mses,
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
            postfix = {"train": f"{epoch_loss:.6f}"}
            if cfg["use_polar_moe"]:
                postfix["bal"] = f"{epoch_bal:.4f}"
            pbar.set_postfix(postfix)

    # Final summary
    print(f"\n{'='*60}")
    print(f"Experiment {exp_name} Results ({cfg['desc']}):")
    print(f"  Final train loss: {train_losses[-1]:.6f}")
    print(f"  Bicubic val: MSE={bicubic_metrics['mse']:.6f}, PSNR={bicubic_metrics['psnr']:.2f}dB, "
          f"SSIM={bicubic_metrics['ssim']:.4f}")
    for ep, m in zip(val_epochs, val_metrics_list):
        marker = " <-- BEST" if ep == best_epoch else ""
        delta_psnr = f"  ΔPSNR={m['psnr']-BASELINE_PSNR:+.1f}dB" if BASELINE_PSNR else ""
        print(f"  Epoch {ep:4d}: MSE={m['mse']:.6f}  PSNR={m['psnr']:.2f}  "
              f"SSIM={m['ssim']:.4f}  NCC={m['ncc']:.4f}  ENCC={m['edge_ncc']:.4f}"
              f"{delta_psnr}{marker}")

    # Load best model for final metrics
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pt")))
    best_metrics = validate(model, val_data, ms_injection=cfg["ms_injection"])

    print(f"\n  Best checkpoints saved to {os.path.join(exp_dir, 'best_model.pt')}")
    print(f"\n  Best: epoch {best_epoch}, "
          f"MSE={best_metrics['mse']:.6f}, PSNR={best_metrics['psnr']:.2f}dB, "
          f"SSIM={best_metrics['ssim']:.4f}, NCC={best_metrics['ncc']:.4f}, "
          f"ENCC={best_metrics['edge_ncc']:.4f}")

    delta_psnr = best_metrics['psnr'] - BASELINE_PSNR
    delta_ssim = best_metrics['ssim'] - BASELINE_SSIM
    print(f"  vs Baseline (Stage 1 v2): ΔPSNR={delta_psnr:+.1f}dB, ΔSSIM={delta_ssim:+.4f}")
    if delta_psnr > 0.1:
        print(f"  PSNR improved >0.1dB -> KEEP this module")
    elif delta_psnr > -0.1:
        print(f"  PSNR within ±0.1dB -> marginal, consider compute cost")
    else:
        print(f"  PSNR degraded >0.1dB -> DISCARD")
    print(f"{'='*60}")

    return best_metrics


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6 Stage 2: Module re-validation (v2 recipe)")
    parser.add_argument("--exp", "-e", type=str, required=True,
                        choices=list(MODULE_CONFIGS.keys()),
                        help="Experiment: 2.1=MS_ISHT, 2.2=PolarMoE, 2.3=CoordEmbed, "
                             "2.4=CircularConv, 2.5=SphericalAttention")
    parser.add_argument("--gpu", "-g", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from best_model.pt checkpoint")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Epoch offset for resumed training (e.g., 400)")
    args = parser.parse_args()

    os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)

    resume_ckpt = None
    start_epoch = 0
    if args.resume:
        cfg = MODULE_CONFIGS[args.exp]
        ckpt_path = os.path.join(OUTPUT_DIR, cfg["name"], "best_model.pt")
        if os.path.exists(ckpt_path):
            resume_ckpt = ckpt_path
            start_epoch = args.start_epoch or 400
        else:
            print(f"Checkpoint not found: {ckpt_path}")
            exit(1)

    run_experiment(args.exp, gpu=args.gpu, epochs=args.epochs,
                   resume_ckpt=resume_ckpt, start_epoch=start_epoch)
