"""Path A: cosine beta schedule + fixed DDIM test.

Usage:
  python phase4_cosine_test.py -g 7
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
OUTPUT_DIR = "phase4_output/exp_cosine"


def to_pil(t):
    a = t[0].cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip((a + 1) * 127.5, 0, 255).astype(np.uint8))


@torch.no_grad()
def ddim_sample_fixed(unet, cond, device, sa, ss, T, steps=50):
    """CORRECTED DDIM: indices high->low, proper iterative denoising."""
    B = cond.shape[0]
    x = torch.randn(B, 3, H, W, device=device)
    indices = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)

    for i in range(steps):
        t = indices[i]
        tp = indices[i + 1] if i + 1 < steps else -1
        t_norm = torch.full((B,), t.item() / T, device=device)
        eps = unet(x, cond, t_norm)
        a_t, s_t = sa[t], ss[t]
        x0_pred = (x - s_t * eps) / a_t
        if tp >= 0:
            a_p, s_p = sa[tp], ss[tp]
            x = a_p * x0_pred + s_p * eps
        else:
            x = x0_pred
    return x


def main(gpu=0):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load single image
    dataset = PanoramaDataset(DATA_DIR, (H, W), SCALE)
    lr_img, hr_img = dataset[0]
    lr_img = lr_img.unsqueeze(0).to(device)
    hr_img = hr_img.unsqueeze(0).to(device)

    # Baselines
    to_pil(hr_img).save(os.path.join(OUTPUT_DIR, "hr.png"))
    lr_up = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)
    to_pil(lr_up).save(os.path.join(OUTPUT_DIR, "bicubic.png"))

    from torch_harmonics import RealSHT, InverseRealSHT
    sht = RealSHT(H, W, lmax=255, mmax=255).to(device)
    isht = InverseRealSHT(H, W, lmax=255, mmax=255).to(device)
    base = isht(sht(lr_up))
    to_pil(base).save(os.path.join(OUTPUT_DIR, "base.png"))
    cond = torch.cat([lr_up, base], dim=1)

    # ★ Cosine schedule
    T = 1000
    betas = make_cosine_schedule(T).to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sa = torch.sqrt(alphas_cumprod)
    ss = torch.sqrt(1.0 - alphas_cumprod)

    # Print key alpha values
    for frac in [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
        t = min(int(frac * (T-1)), T-1)
        print(f"  t={t:4d} alpha={sa[t].item():.4f} sigma={ss[t].item():.4f}")

    # U-Net
    unet = PixelUNet(in_ch=3, cond_ch=6, base_ch=32, time_dim=256,
                     use_polar_moe=False, use_laplacian=False,
                     use_circular_conv=False, use_coord_embed=False,
                     use_spherical_attn=False, hr_size=(H, W)).to(device)
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"\nModel: {n_params:,} params, base_ch=32, cosine schedule, T={T}")
    print(f"Target: HR (noise prediction), Epochs: 500, LR: {LR}")

    optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

    # Train 500 epochs
    losses = []
    pbar = tqdm(range(1, 501), desc="[train-cosine]", unit="ep")
    for epoch in pbar:
        unet.train()
        t = torch.randint(0, T, (1,), device=device)
        alpha_t = sa[t].view(1, 1, 1, 1)
        sigma_t = ss[t].view(1, 1, 1, 1)

        noise = torch.randn_like(hr_img)
        noisy = alpha_t * hr_img + sigma_t * noise
        pred_noise = unet(noisy, cond, t.float() / T)
        if isinstance(pred_noise, tuple):
            pred_noise = pred_noise[0]
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 100 == 0:
            pbar.set_postfix(train=f"{loss.item():.4f}")
    print(f"Final train loss: {losses[-1]:.6f}")

    # Save weights
    torch.save(unet.state_dict(), os.path.join(OUTPUT_DIR, "unet_e500.pt"))

    # Per-timestep noise prediction accuracy
    print("\n=== Per-timestep noise prediction accuracy ===")
    unet.eval()
    for frac in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
        t_idx = min(int(frac * (T-1)), T-1)
        alpha_t, sigma_t = sa[t_idx], ss[t_idx]
        noise = torch.randn_like(hr_img)
        noisy = alpha_t * hr_img + sigma_t * noise
        with torch.no_grad():
            pred = unet(noisy, cond, torch.tensor([t_idx/T], device=device))
        mse = F.mse_loss(pred, noise).item()
        print(f"  t/T={frac:.2f} (t={t_idx}): alpha={alpha_t.item():.4f}, noise_MSE={mse:.6f}")

    # ★ Fixed DDIM from t=999 (50 steps)
    print("\n=== Fixed DDIM Sampling (t=999, 50 steps) ===")
    unet.eval()
    with torch.no_grad():
        sr = ddim_sample_fixed(unet, cond, device, sa, ss, T, steps=50)
    mse = F.mse_loss(sr, hr_img).item()
    to_pil(sr).save(os.path.join(OUTPUT_DIR, "recon_fixed_ddim.png"))
    print(f"Fixed DDIM MSE: {mse:.6f}")

    # Buggy DDIM for comparison
    print("\n=== Buggy DDIM (for reference) ===")
    @torch.no_grad()
    def buggy():
        B, x = 1, torch.randn(1, 3, H, W, device=device)
        idx = torch.linspace(0, T-1, 50, dtype=torch.long, device=device)
        idx_p = torch.cat([idx[1:], torch.tensor([-1], device=device)])
        for i in range(49, -1, -1):
            t, tp = idx[i], idx_p[i]
            eps = unet(x, cond, torch.full((B,), t.item()/T, device=device))
            a_t, s_t = sa[t], ss[t]
            x0p = (x - s_t * eps) / a_t
            x = sa[tp] * x0p + ss[tp] * eps if tp >= 0 else x0p
        return x
    sr_b = buggy()
    mse_b = F.mse_loss(sr_b, hr_img).item()
    to_pil(sr_b).save(os.path.join(OUTPUT_DIR, "recon_buggy_ddim.png"))
    print(f"Buggy DDIM MSE: {mse_b:.6f}")

    # Cross-correlation
    for name, img in [("Fixed DDIM", sr), ("Buggy DDIM", sr_b)]:
        hr_f = hr_img - hr_img.mean()
        img_f = img - img.mean()
        ncc = (hr_f * img_f).sum() / (hr_f.norm() * img_f.norm() + 1e-8)
        print(f"  {name} NCC with HR: {ncc.item():.4f}")

    # Per-channel stats
    print(f"\n=== Output details ===")
    for name, img in [("HR", hr_img), ("Bicubic", lr_up), ("Fixed DDIM", sr)]:
        m = F.mse_loss(img, hr_img).item()
        for c, cn in enumerate(["R", "G", "B"]):
            ch = img[0, c]
            print(f"  {name:12s} {cn}: mean={ch.mean().item():+.4f} std={ch.std().item():.4f}")
        print(f"  {name:12s} MSE={m:.6f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=7)
    args = parser.parse_args()
    main(gpu=args.gpu)
