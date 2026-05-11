"""Minimal TF vs AR test - single image, clear cache aggressively."""
import torch
import torch.nn.functional as F
import numpy as np
import os, sys

torch.backends.cudnn.benchmark = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.center_growing import CenterGrowingUNet
from utility.data import PanoramaDataset

DEVICE = torch.device("cuda:2")
H, W = 1024, 2048
SCALES = [(6, 3), (12, 6), (24, 12), (48, 24), (96, 48),
          (192, 96), (384, 192), (768, 384), (1536, 768),
          (2048, 1024)]

def place_center(canvas, patch):
    _, _, Hc, Wc = canvas.shape
    _, _, Hp, Wp = patch.shape
    start_h = (Hc - Hp) // 2
    start_w = (Wc - Wp) // 2
    canvas[:, :, start_h:start_h + Hp, start_w:start_w + Wp] = patch

@torch.no_grad()
def run_one(model, I_low_up, I_gt, use_tf):
    known_high_res = None
    prev_w, prev_h = 0, 0
    final_sr = None
    for i in range(len(SCALES) - 1):
        w, h = SCALES[i]
        next_w, next_h = SCALES[i + 1]

        local_low = I_low_up.clone()
        start_h = (H - h) // 2
        start_w = (W - w) // 2
        # Zero out outside the central region
        if h < H or w < W:
            zeros_full = torch.zeros_like(local_low)
            zeros_full[:, :, start_h:start_h + h, start_w:start_w + w] = \
                local_low[:, :, start_h:start_h + h, start_w:start_w + w]
            local_low = zeros_full

        if known_high_res is None:
            high_res_known = torch.zeros_like(I_low_up)
            mask = torch.zeros(1, 1, H, W, device=DEVICE)
        else:
            high_res_known = torch.zeros_like(I_low_up)
            place_center(high_res_known, known_high_res)
            mask = torch.zeros(1, 1, H, W, device=DEVICE)
            start_h_prev = (H - prev_h) // 2
            start_w_prev = (W - prev_w) // 2
            mask[:, :, start_h_prev:start_h_prev + prev_h,
                 start_w_prev:start_w_prev + prev_w] = 1.0

        model_input = torch.cat([I_low_up, local_low, high_res_known, mask], dim=1)
        residual = model(model_input)
        sr = I_low_up + residual

        if use_tf:
            gt_crop_h = (H - next_h) // 2
            gt_crop_w = (W - next_w) // 2
            known_high_res = I_gt[:, :, gt_crop_h:gt_crop_h + next_h,
                                  gt_crop_w:gt_crop_w + next_w].clone()
        else:
            pred_crop_h = (H - next_h) // 2
            pred_crop_w = (W - next_w) // 2
            known_high_res = sr[:, :, pred_crop_h:pred_crop_h + next_h,
                                pred_crop_w:pred_crop_w + next_w].clone()
        prev_w, prev_h = next_w, next_h
        final_sr = sr

        # Clear per-iteration temporaries
        del local_low, high_res_known, mask, model_input, residual, sr
        torch.cuda.empty_cache()

    return final_sr

CKPT = "phase7_output/phase7_s3b_noise10/best_model.pt"  # latest S3b

# Load model
print("Loading model...")
model = CenterGrowingUNet(in_ch=10, out_ch=3, base_ch=64).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

# Load data
print("Loading data...")
dataset = PanoramaDataset("lau_dataset/odisr/training", (H, W), 2)
lr_img, hr_img = dataset[0]
lr_img = lr_img.unsqueeze(0)
hr_img = hr_img.unsqueeze(0)
I_low_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False).to(DEVICE)
I_gt = hr_img.to(DEVICE)

# Bicubic
bicubic_mse = F.mse_loss(I_low_up, I_gt).item()
bicubic_psnr = 20 * np.log10(2.0 / np.sqrt(max(bicubic_mse, 1e-10)))
print(f"Bicubic: MSE={bicubic_mse:.6f}, PSNR={bicubic_psnr:.2f}")

# AR
print("AR validation...")
out_ar = run_one(model, I_low_up, I_gt, use_tf=False)
ar_mse = F.mse_loss(out_ar, I_gt).item()
ar_psnr = 20 * np.log10(2.0 / np.sqrt(max(ar_mse, 1e-10)))
print(f"AR: MSE={ar_mse:.6f}, PSNR={ar_psnr:.2f} dB")

# TF
print("TF validation...")
torch.cuda.empty_cache()
out_tf = run_one(model, I_low_up, I_gt, use_tf=True)
tf_mse = F.mse_loss(out_tf, I_gt).item()
tf_psnr = 20 * np.log10(2.0 / np.sqrt(max(tf_mse, 1e-10)))
print(f"TF: MSE={tf_mse:.6f}, PSNR={tf_psnr:.2f} dB")

gap = tf_psnr - ar_psnr
print(f"\nAR→TF gap: {gap:.2f} dB")
print(f"Base_ch=64 TF bound: {tf_psnr:.2f} dB")
print(f"Stage 1 TF bound (base_ch=32): 24.1 dB")
print(f"Improvement: {tf_psnr - 24.1:.1f} dB")
