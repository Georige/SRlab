"""Phase 4 x0-prediction baseline: cosine schedule + x0 target + soft truncation.

Key change: model predicts clean HR directly (not noise), eliminating the
division-by-alpha that amplifies noise prediction errors in DDIM.

Usage:
  python phase4_x0pred.py -g 7              # train + test
  python phase4_x0pred.py -g 7 --test-only  # test existing weights
"""
import argparse, os
import torch, torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from utility.data import PanoramaDataset
from utility.schedule import make_cosine_schedule
from model.unet import PixelUNet

H, W = 512, 1024
SCALE = 4
LR = 2e-4
DATA_DIR = "lau_dataset/sun_test"
OUTPUT_DIR = "phase4_output/exp_x0pred"


def to_pil(t):
    a = t[0].cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip((a + 1) * 127.5, 0, 255).astype(np.uint8))


@torch.no_grad()
def ddim_sample_x0(unet, cond, device, sa, ss, T, steps=50):
    """x0-prediction DDIM: model outputs clean image, no division by alpha needed."""
    B = cond.shape[0]
    x = torch.randn(B, 3, H, W, device=device)
    indices = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)

    for i in range(steps):
        t = indices[i]
        tp = indices[i + 1] if i + 1 < steps else -1
        t_norm = torch.full((B,), t.item() / T, device=device)

        # Model predicts x0 directly
        x0_pred = unet(x, cond, t_norm)
        x0_pred = torch.tanh(x0_pred)  # soft truncation

        a_t, s_t = sa[t], ss[t]
        # Recover noise estimate from x0_pred
        eps_est = (x - a_t * x0_pred) / s_t

        if tp >= 0:
            a_p, s_p = sa[tp], ss[tp]
            x = a_p * x0_pred + s_p * eps_est
        else:
            x = x0_pred

    return x


def main(gpu=0, test_only=False):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load single image
    dataset = PanoramaDataset(DATA_DIR, (H, W), SCALE)
    lr_img, hr_img = dataset[0]
    lr_img = lr_img.unsqueeze(0).to(device)
    hr_img = hr_img.unsqueeze(0).to(device)

    to_pil(hr_img).save(os.path.join(OUTPUT_DIR, "hr.png"))
    lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)
    to_pil(lr_up).save(os.path.join(OUTPUT_DIR, "bicubic.png"))

    from torch_harmonics import RealSHT, InverseRealSHT
    sht = RealSHT(H, W, lmax=255, mmax=255).to(device)
    isht = InverseRealSHT(H, W, lmax=255, mmax=255).to(device)
    base = isht(sht(lr_up))
    to_pil(base).save(os.path.join(OUTPUT_DIR, "base.png"))
    cond = torch.cat([lr_up, base], dim=1)

    # Cosine schedule
    T = 1000
    betas = make_cosine_schedule(T).to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sa = torch.sqrt(alphas_cumprod)
    ss = torch.sqrt(1.0 - alphas_cumprod)

    # U-Net
    unet = PixelUNet(in_ch=3, cond_ch=6, base_ch=32, time_dim=256,
                     use_polar_moe=False, use_laplacian=False,
                     use_circular_conv=False, use_coord_embed=False,
                     use_spherical_attn=False, hr_size=(H, W)).to(device)
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"x0-Prediction Baseline  |  {n_params:,} params  |  base_ch=32")
    print(f"Schedule: cosine  |  T={T}  |  Target: x0 (clean HR) + tanh trunc")
    print(f"Cond: bicubic+base (6ch)  |  No innovations")

    if test_only:
        unet.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "unet_e500.pt"), map_location=device))
        unet.eval()
    else:
        # Train 500 epochs
        optimizer = torch.optim.Adam(unet.parameters(), lr=LR)
        pbar = tqdm(range(1, 501), desc="[x0pred]", unit="ep")
        for epoch in pbar:
            unet.train()
            t = torch.randint(0, T, (1,), device=device)
            alpha_t = sa[t].view(1, 1, 1, 1)
            sigma_t = ss[t].view(1, 1, 1, 1)

            noise = torch.randn_like(hr_img)
            noisy = alpha_t * hr_img + sigma_t * noise
            x0_pred = unet(noisy, cond, t.float() / T)
            x0_pred = torch.tanh(x0_pred)

            # ★ Loss on clean image directly (no division)
            loss = F.mse_loss(x0_pred, hr_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"Final train loss: {loss.item():.6f}")
        torch.save(unet.state_dict(), os.path.join(OUTPUT_DIR, "unet_e500.pt"))

    # Per-timestep x0 prediction accuracy (no tanh, raw MSE)
    print("\n=== Per-timestep x0 prediction accuracy (raw, no tanh) ===")
    unet.eval()
    for frac in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
        t_idx = min(int(frac * (T - 1)), T - 1)
        alpha_t, sigma_t = sa[t_idx], ss[t_idx]
        noise = torch.randn_like(hr_img)
        noisy = alpha_t * hr_img + sigma_t * noise
        with torch.no_grad():
            raw_pred = unet(noisy, cond, torch.tensor([t_idx / T], device=device))
            raw_mse = F.mse_loss(raw_pred, hr_img).item()
            tanh_pred = torch.tanh(raw_pred)
            tanh_mse = F.mse_loss(tanh_pred, hr_img).item()
        print(f"  t/T={frac:.2f} (t={t_idx:4d}): alpha={alpha_t.item():.4f}  "
              f"raw_MSE={raw_mse:.4f}  tanh_MSE={tanh_mse:.4f}")

    # Fixed DDIM (x0-prediction)
    print("\n=== x0-DDIM Sampling (t=999, 50 steps) ===")
    unet.eval()
    with torch.no_grad():
        sr = ddim_sample_x0(unet, cond, device, sa, ss, T, steps=50)
    mse = F.mse_loss(sr, hr_img).item()

    # Metrics
    hr_f = hr_img - hr_img.mean()
    sr_f = sr - sr.mean()
    ncc = (hr_f * sr_f).sum() / (hr_f.norm() * sr_f.norm() + 1e-8)

    # Edge NCC
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    def edge_map(img):
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2)

    e_hr = edge_map(hr_img)
    e_sr = edge_map(sr)
    edge_ncc = ((e_hr - e_hr.mean()) * (e_sr - e_sr.mean())).sum() / (e_hr.norm() * e_sr.norm() + 1e-8)

    to_pil(sr).save(os.path.join(OUTPUT_DIR, "recon_x0_ddim.png"))
    print(f"x0-DDIM  MSE={mse:.6f}  NCC={ncc.item():.4f}  Edge-NCC={edge_ncc.item():.4f}")
    for c, cn in enumerate(["R", "G", "B"]):
        print(f"  {cn}: mean={sr[0,c].mean().item():+.4f} std={sr[0,c].std().item():.4f}")

    # Compare with noise-prediction baseline (previous result)
    print(f"\n=== Baseline comparison ===")
    print(f"  Noise-pred + cosine + hard-clip:  MSE=0.166  NCC=0.834  Edge-NCC=0.136")
    print(f"  x0-pred   + cosine + tanh:        MSE={mse:.4f}  NCC={ncc.item():.4f}  Edge-NCC={edge_ncc.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=7)
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    main(gpu=args.gpu, test_only=args.test_only)
