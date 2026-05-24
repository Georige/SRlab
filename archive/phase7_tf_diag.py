"""Quick TF diagnostic: load best_model.pt and compare AR vs TF validation."""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.center_growing import CenterGrowingUNet
from utility.data import PanoramaDataset

DEVICE = torch.device("cuda:1")
CKPT = "phase7_output/phase7_s2c_basech64/best_model.pt"
H, W = 1024, 2048
SCALES = [(6, 3), (12, 6), (24, 12), (48, 24), (96, 48),
          (192, 96), (384, 192), (768, 384), (1536, 768),
          (2048, 1024)]

def to_pil(t):
    a = t[0].cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip((a + 1) * 127.5, 0, 255).astype(np.uint8))

def crop_center_rect(x, w, h):
    H_, W_ = x.shape[2], x.shape[3]
    start_h = (H_ - h) // 2
    start_w = (W_ - w) // 2
    return x[:, :, start_h:start_h + h, start_w:start_w + w]

def place_center(canvas, patch):
    _, _, Hc, Wc = canvas.shape
    _, _, Hp, Wp = patch.shape
    start_h = (Hc - Hp) // 2
    start_w = (Wc - Wp) // 2
    canvas[:, :, start_h:start_h + Hp, start_w:start_w + Wp] = patch

@torch.no_grad()
def compute_metrics(pred, target):
    pred_f = pred.float()
    target_f = target.float()
    mse = F.mse_loss(pred_f, target_f).item()
    psnr = 20 * np.log10(2.0 / np.sqrt(mse)) if mse > 0 else 100.0
    # NCC
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
    # SSIM
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
    return {"mse": mse, "psnr": psnr, "ncc": ncc, "ssim": ssim}

def run_validation(model, I_low_up, I_gt, use_tf):
    known_high_res = None
    prev_w, prev_h = 0, 0
    final_sr = None
    for i in range(len(SCALES) - 1):
        w, h = SCALES[i]
        next_w, next_h = SCALES[i + 1]

        # Build input
        local_low = torch.zeros_like(I_low_up)
        start_h = (H - h) // 2
        start_w = (W - w) // 2
        local_low[:, :, start_h:start_h + h, start_w:start_w + w] = \
            I_low_up[:, :, start_h:start_h + h, start_w:start_w + w]

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
            known_high_res = crop_center_rect(I_gt, next_w, next_h)
        else:
            pred_crop = crop_center_rect(sr, next_w, next_h)
            known_high_res = pred_crop
        prev_w, prev_h = next_w, next_h
        final_sr = sr

    return final_sr

# Load model
print(f"Loading {CKPT}...")
model = CenterGrowingUNet(in_ch=10, out_ch=3, base_ch=64).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} params (base_ch=64)")

# Load data (image index 0)
dataset = PanoramaDataset("lau_dataset/odisr/training", (H, W), 2)
lr_img, hr_img = dataset[0]
lr_img = lr_img.unsqueeze(0)
hr_img = hr_img.unsqueeze(0)

I_low_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)
I_low_up = I_low_up.to(DEVICE)
I_gt = hr_img.to(DEVICE)

# Bicubic baseline
bicubic_mse = F.mse_loss(I_low_up, I_gt).item()
bicubic_psnr = 20 * np.log10(2.0 / np.sqrt(bicubic_mse)) if bicubic_mse > 0 else 100.0
print(f"\nBicubic: MSE={bicubic_mse:.6f}, PSNR={bicubic_psnr:.2f} dB")

# AR validation
print("\nRunning AR validation...")
out_ar = run_validation(model, I_low_up, I_gt, use_tf=False)
m_ar = compute_metrics(out_ar, I_gt)
print(f"AR: MSE={m_ar['mse']:.6f}, PSNR={m_ar['psnr']:.2f} dB, SSIM={m_ar['ssim']:.4f}, NCC={m_ar['ncc']:.4f}")

# TF validation
print("\nRunning TF validation...")
out_tf = run_validation(model, I_low_up, I_gt, use_tf=True)
m_tf = compute_metrics(out_tf, I_gt)
print(f"TF: MSE={m_tf['mse']:.6f}, PSNR={m_tf['psnr']:.2f} dB, SSIM={m_tf['ssim']:.4f}, NCC={m_tf['ncc']:.4f}")

# Save TF output
outdir = "phase7_output/phase7_s2c_basech64"
to_pil(out_tf).save(os.path.join(outdir, "tf_diag_from_ckpt.png"))
print(f"\nSaved TF output to {outdir}/tf_diag_from_ckpt.png")

# Compute residuals
res_ar = (out_ar - I_gt).abs().mean().item()
res_tf = (out_tf - I_gt).abs().mean().item()
print(f"\nMean abs error ([-1,1]): AR={res_ar:.6f}, TF={res_tf:.6f}")
print(f"TF - AR gap: PSNR delta = {m_tf['psnr'] - m_ar['psnr']:.2f} dB")
